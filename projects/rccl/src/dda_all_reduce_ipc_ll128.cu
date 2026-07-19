/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA IPC all-reduce.
 * IPC analogue of dda_all_reduce_ipc_ll.cu with a 128B line (15 payload words +
 * 1 flag word, 16 lanes cooperative) for 93.75% staging efficiency. Uses the
 * codepath-agnostic ddaAllReduceFlatLL128 kernel from all_reduce_dda_ll128.h and
 * reuses the same scratch/peer table/epoch array as the 16B LL IPC path (only
 * one all-reduce protocol runs per call).
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_dda_ll128.h"
#include "checks.h"
#include "comm.h"
#include "dda_init_detail.h" // nccl_dda_detail::kDdaNranks
#include "debug.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

using meta::comms::kDdaLL128ArMaxBytes;
using meta::comms::kDdaLL128DataElems;
using meta::comms::kDdaLL128Lanes;
using meta::comms::LLLine128;
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::kDdaNranks;

// Per-call slot stride in 128B lines. Matches the message exactly (compact
// layout) so small all-reduces keep their scratch slots close together.
static inline size_t ddaLL128ArSlotLines(size_t numLines) {
  return numLines;
}

// LL128 scratch for this call: 2 banks * nRanks slots * slotLines * 128B.
static inline size_t ddaLL128ArScratchSize(int nRanks, size_t numLines) {
  return (size_t)2 * (size_t)nRanks * ddaLL128ArSlotLines(numLines) *
         sizeof(LLLine128);
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcLL128Typed(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nWords = bytes >> 3;
  const size_t numLines =
      (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;
  const size_t slotStrideLines = ddaLL128ArSlotLines(numLines);

  // 1D grid over line-groups; each block has threads/16 groups.
  const unsigned threads = 1024; // multiple of 16 (lanes/line)
  const size_t groups = threads / (unsigned)kDdaLL128Lanes;
  int nBlocksMax = ddaMaxNBlocksForScratch();
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>(
      (numLines + groups - 1) / groups, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  // flatBlockId (blockIdx.x) must stay within the device epoch array.
  if ((int)blocks > comm->ddaLLArEpochLen) {
    blocks = (unsigned)comm->ddaLLArEpochLen;
    if (blocks == 0) blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Reuse the LL AR epoch array (shared monotonic flag namespace; only one
  // all-reduce protocol runs per call, so LL and LL128 never race on it).
  uint32_t* epochDev = comm->ddaLLArEpochDev;
  const int epochLen = comm->ddaLLArEpochLen;

  INFO(
      NCCL_COLL,
      "DDA IPC AllReduce LL128: nRanks=%d bytes=%zu numLines=%zu grid=%u block=%u",
      nRanks, bytes, numLines, grid.x, block.x);

  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback. IPC is fixed at
  // kDdaNranks ranks.
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllReduceFlatLL128<T, 4><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen, slotStrideLines);
    break;
  case 8:
    meta::comms::ddaAllReduceFlatLL128<T, 8><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen, slotStrideLines);
    break;
  default:
    meta::comms::ddaAllReduceFlatLL128<T, 0><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen, slotStrideLines);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaIpcLL128Eligible(
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
  // Payload is staged as 8-byte words, so it must be a whole number of words.
  if (bytes % 8 != 0) {
    return false;
  }
  if (bytes > kDdaLL128ArMaxBytes) {
    return false;
  }
  // Scratch is sized from the actual message (compact per-call slot stride), so
  // eligibility is bounded by the runtime scratch capacity for this size.
  const size_t nWords = bytes >> 3;
  const size_t numLines =
      (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;
  if (ddaLL128ArScratchSize(comm->nRanks, numLines) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllReduceDdaIpcLL128(
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
    return ncclAllReduceDdaIpcLL128Typed<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaIpcLL128Typed<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaIpcLL128Typed<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
