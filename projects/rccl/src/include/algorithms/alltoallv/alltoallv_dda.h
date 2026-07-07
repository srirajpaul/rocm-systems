/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh.
 * Adapted for the alltoallv (variable count) collective operation.
 * Includes use *.h names so RCCL hipify output (src/include/...) resolves correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "ipc_gpu_barrier.h"
#include "algorithms/CollCommon.h"

#include <cstddef>
#include <cstdint>

namespace meta::comms {

// Copy nbytes from src to dst using every thread in the grid. A 16-byte
// (uint4) fast path is used when both pointers are 16-byte aligned; otherwise a
// byte-wise copy is used so arbitrary displacements/counts are handled safely.
template <typename T>
static inline __device__ void ddaCopyLinear(
    T* __restrict__ dst,
    const T* __restrict__ src,
    size_t nbytes,
    size_t gtIdx,
    size_t nThreads) {
  const uintptr_t misalign = reinterpret_cast<uintptr_t>(dst) |
      reinterpret_cast<uintptr_t>(src);
  if ((misalign & 0xF) == 0) {
    const size_t nVec = nbytes >> 4; // number of 16-byte chunks
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    for (size_t i = gtIdx; i < nVec; i += nThreads) {
      d4[i] = s4[i];
    }
    for (size_t i = (nVec << 4) + gtIdx; i < nbytes; i += nThreads) {
      dst[i] = src[i];
    }
  } else {
    for (size_t i = gtIdx; i < nbytes; i += nThreads) {
      dst[i] = src[i];
    }
  }
}

// alltoallv via IPC scratch, using a fixed "slot" layout so that no remote
// displacement metadata needs to be exchanged:
//   * every rank's scratch buffer is divided into NRANKS equally sized slots of
//     slotStride bytes; slot d holds the chunk destined for rank d.
//   * phase 1: each rank scatters its send chunks into its own scratch slots.
//   * phase 2: after a barrier, each rank gathers, from every peer's slot that
//     targets this rank, into its recvbuff at the user displacement.
// The sender writes chunk-for-d at offset d*slotStride; the receiver (rank d)
// therefore reads peer r's slot at offset d*slotStride. Both sides derive the
// same offset from the (globally identical) slotStride, so only local
// counts/displacements are required.
template <typename T, int NRANKS>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
    __global__ void ddaAllToAllvIpc(
        T* const* __restrict__ ipcbuffs,
        const T* __restrict__ sendbuff,
        const std::array<size_t, NRANKS> sendcounts,
        const std::array<size_t, NRANKS> sdispls,
        T* __restrict__ recvbuff,
        const std::array<size_t, NRANKS> recvcounts,
        const std::array<size_t, NRANKS> rdispls,
        int selfRank,
        size_t slotStride,
        IpcGpuBarrier barrier) {
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch
  // TODO: we should be able to deal with left over as well
  const size_t count = sendcounts[0];
  const size_t countPerRank = count;
  constexpr auto countPerThread = 1; //sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t copyCount = count * NRANKS;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  //copyFromSrcToDest<T>(
  //    sendbuff, ipcbuffs[selfRank], idxStart, copyCount, idxStride);
#pragma unroll NRANKS
  for (int r = 0; r < NRANKS; ++r) {
      copyFromSrcToDest1<T>(
          sendbuff + sdispls[r], ipcbuffs[selfRank] + r * slotStride, idxStart, sendcounts[r], idxStride);
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

#pragma unroll NRANKS
  for (int r = 0; r < NRANKS; ++r) {
      copyFromSrcToDest1<T>(
          ipcbuffs[r] + selfRank * slotStride, recvbuff + rdispls[r], idxStart, recvcounts[r], idxStride);

    //for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    //  int srcRank = r;
    //  int srcIdx = idx + selfRank * idxEnd;
    //  int destIdx = idx + r * idxEnd;
    //  *reinterpret_cast<uint4*>(&recvbuff[destIdx]) =
    //      reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    //}
  }

  // barrier to ensure remote ranks won't free/reuse their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
