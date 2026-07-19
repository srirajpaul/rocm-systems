/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA IPC all-reduce.
 * IPC analogue of dda_all_reduce_fabric_ll.cu; reuses the codepath-agnostic
 * ddaAllReduceFlatLL kernel from all_reduce_dda_ll.h. The kernel only needs
 * the peer scratch table (comm->ddaPeerPtrsDev), which the IPC path fills with
 * IPC-mapped peer scratch pointers, so the same kernel backs both paths.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_dda_ll.h"
#include "checks.h"
#include "comm.h"
#include "dda_init_detail.h" // nccl_dda_detail::kDdaFabricLLArMaxBlocks, kDdaNranks
#include "debug.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

using meta::comms::kDdaLLArMaxBytes;
using meta::comms::kDdaLLArSlotStridePkts;
using meta::comms::LLPacket16;
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::kDdaNranks;

// LL scratch footprint: 2 banks * nRanks slots * slotStride packets * 16B.
static inline size_t ddaLLArScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLArSlotStridePkts * sizeof(LLPacket16);
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcLLTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet

  const unsigned threads = 256;
  // LL only serves tiny messages (<= DDA_ALLREDUCE_LL_THRESHOLD) where latency,
  // not occupancy, dominates; cap the grid low so we avoid the launch/sync
  // overhead of a wide grid.
  int nBlocksMax =
      std::min(ddaMaxNBlocksForScratch(), nccl_dda_detail::kDdaFabricLLArMaxBlocks);
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>(
      (nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Dedicated small, high-namespace epoch array for the LL AR tier (see
  // kDdaFabricLLArMaxBlocks / kDdaLLArEpochSeed) so the per-launch epoch reset
  // stays cheap and the derived flag is computed on the device (graph-safe).
  uint32_t* epochDev = comm->ddaLLArEpochDev;
  const int epochLen = comm->ddaLLArEpochLen;

  INFO(
      NCCL_COLL,
      "DDA IPC AllReduce LL: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u",
      nRanks, bytes, nPk, grid.x, block.x);

  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback (up to
  // kDdaMaxNranks). The IPC path is fixed at kDdaNranks ranks.
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllReduceFlatLL<T, 4><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    meta::comms::ddaAllReduceFlatLL<T, 8><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    meta::comms::ddaAllReduceFlatLL<T, 0><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaIpcLLEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // IPC path: requires the IPC handler + scratch + peer table + epoch array.
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr ||
      comm->ddaPeerPtrsDev == nullptr || comm->ddaLLArEpochDev == nullptr) {
    return false;
  }
  // IPC kernels are single-node and fixed at kDdaNranks ranks.
  if (comm->nNodes != 1 || comm->nRanks != kDdaNranks) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);
  // Payload is staged as 8-byte LL packets, so it must be a whole number of
  // packets.
  if (bytes % 8 != 0) {
    return false;
  }
  if (bytes > kDdaLLArMaxBytes) {
    return false;
  }
  if (ddaLLArScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllReduceDdaIpcLL(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaIpcLLTyped<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaIpcLLTyped<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaIpcLLTyped<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
