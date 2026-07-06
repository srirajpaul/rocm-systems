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

// Per-call layout description passed by value into the kernel.
// All quantities are expressed in *bytes* (the kernel operates on int8_t so it
// is datatype agnostic). Arrays are indexed by peer rank.
template <int NRANKS>
struct DdaAllToAllvArgs {
  size_t sendcounts[NRANKS]; // bytes this rank sends to peer d
  size_t sdispls[NRANKS];    // byte offset into sendbuff of the chunk for d
  size_t recvcounts[NRANKS]; // bytes this rank receives from peer r
  size_t rdispls[NRANKS];    // byte offset into recvbuff of the chunk from r
};

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
        T* __restrict__ recvbuff,
        const T* __restrict__ sendbuff,
        int selfRank,
        size_t slotStride,
        DdaAllToAllvArgs<NRANKS> args,
        IpcGpuBarrier barrier) {
  const size_t gtIdx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  const size_t nThreads = static_cast<size_t>(gridDim.x) * blockDim.x;

  // Phase 1: scatter local send chunks into this rank's own scratch slots.
  T* myscratch = ipcbuffs[selfRank];
#pragma unroll
  for (int d = 0; d < NRANKS; ++d) {
    ddaCopyLinear<T>(
        &myscratch[static_cast<size_t>(d) * slotStride],
        &sendbuff[args.sdispls[d]],
        args.sendcounts[d],
        gtIdx,
        nThreads);
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  // Phase 2: gather from every peer's slot that targets this rank.
  const size_t selfSlotOff = static_cast<size_t>(selfRank) * slotStride;
#pragma unroll
  for (int r = 0; r < NRANKS; ++r) {
    ddaCopyLinear<T>(
        &recvbuff[args.rdispls[r]],
        &ipcbuffs[r][selfSlotOff],
        args.recvcounts[r],
        gtIdx,
        nThreads);
  }

  // barrier to ensure remote ranks won't free/reuse their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
