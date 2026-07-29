/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA IPC all-reduce. The
 * ddaAllReduceFlatLL kernel in all_reduce_dda_ll.h is codepath-agnostic -- it
 * reaches peers only through the scratch pointer table -- so this file is the IPC
 * counterpart of dda_all_reduce_fabric_ll.cu and shares that kernel. The only
 * difference is where the table comes from: hipIpcOpenMemHandle here, imported
 * fabric handles there.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_dda_ll.h"
#include "checks.h"
#include "comm.h"
#include "dda_init_detail.h" // nccl_dda_detail::ddaMaxNBlocksForScratch, kDdaNranks
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
static ncclResult_t ncclAllReduceDdaIpcLLTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                               cudaStream_t stream) {
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet

  const unsigned threads = 256;
  // The IPC path caps every DDA kernel at DDA_IPC_MAXBLOCKS, which is already the
  // narrow grid this latency-bound tier wants. blockIdx.x indexes the epoch array,
  // so the grid can never outgrow it either.
  int nBlocksMax = std::min(ddaMaxNBlocksForScratch(), comm->ddaLLArEpochLen);
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLArEpochDev;
  const int epochLen = comm->ddaLLArEpochLen;

  INFO(NCCL_COLL, "DDA IPC AllReduce LL: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u", comm->nRanks, bytes, nPk,
       grid.x, block.x);

  // Unlike the fabric launcher there is no runtime-nRanks instantiation: the IPC
  // path only ever runs at kDdaNranks, so the reduce loop is always unrolled.
  meta::comms::ddaAllReduceFlatLL<T, kDdaNranks><<<grid, block, 0, stream>>>(
    peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, comm->nRanks, epochDev,
    epochLen);

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaIpcLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // IPC path: requires the IPC handler + scratch + peer table. The barrier state
  // the Simple IPC kernels need is not checked -- LL syncs through its flags.
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  // The tier's epoch array carries the flag/bank derivation; without it the
  // kernel cannot sync.
  if (comm->ddaLLArEpochDev == nullptr || comm->ddaLLArEpochLen < 1) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != kDdaNranks) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);
  // Payload is staged as 8-byte LL packets, so it must be a whole number of
  // packets. Note this is looser than the Simple IPC tier's 16-byte requirement.
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

ncclResult_t ncclAllReduceDdaIpcLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaIpcLLTyped<float>(sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaIpcLLTyped<half>(sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaIpcLLTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
