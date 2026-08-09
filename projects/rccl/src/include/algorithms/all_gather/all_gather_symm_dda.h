/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Symmetric-memory DDA all-gather kernel (IPC path). Unlike the copy-based
 * ddaAllGatherIpc, this variant takes only the symmetric send/recv pointers and
 * derives this rank's local base via localPtr() and each peer's base via
 * lsaPtr(r) on device -- no host-resolved pointer array is needed.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "ipc_gpu_barrier.h"
#include "algorithms/CollCommon.h"
#include "nccl_device/ptr.h"

namespace meta::comms {

// symmetric-memory DDA all-gather kernel (IPC path)
template <typename T, int NRANKS, int inplace>
__device__ void gather_symm_dda(
    T* __restrict__ (&recvPtr)[NRANKS],
    size_t count,
    T* __restrict__ (&sendPtr)[NRANKS],
    int selfRank) {
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch.
  static_assert(sizeof(T) == 1);
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  #if 1
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    v4u tmp[NRANKS];
    #pragma unroll NRANKS
    for (int r = inplace; r < NRANKS; ++r) {
      int peer = (selfRank + r) % NRANKS;
      tmp[r] = *(v4u_gptr)(&sendPtr[peer][idx]);
    }
    #pragma unroll NRANKS
    for (int r = inplace; r < NRANKS; ++r) {
      int peer = (selfRank + r) % NRANKS;
      size_t dstIdx = peer * count + idx;
      *(v4u_gptr)(&recvPtr[selfRank][dstIdx]) = tmp[r];
    }
  }
  #else
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    #pragma unroll NRANKS
    for (int r = inplace; r < NRANKS; ++r) {
        int peer = (selfRank + r) % NRANKS;
        size_t dstIdx = peer * count + idx;
        *(v4u_gptr)(&recvPtr[selfRank][dstIdx]) = *(v4u_gptr)(&sendPtr[peer][idx]);
    }
  }
  #endif
}

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
    __global__ void ddaAllGatherIpcSymm(
        ncclSymPtr<T> recvSymPtr,
        size_t count,
        const ncclSymPtr<T> sendSymPtr,
        int selfRank,
        IpcGpuBarrier barrier) {

  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  static_assert(sizeof(T) == 1);

  T* __restrict__ recvPtr[NRANKS];
  T* __restrict__ sendPtr[NRANKS];
  const int inplace = sendSymPtr == recvSymPtr + count * selfRank;
  if (inplace) {
    #pragma unroll
    for (int i = 0; i < NRANKS; i++) {
        recvPtr[i] = recvSymPtr.lsaPtr(i);
        sendPtr[i] = recvSymPtr.lsaPtr(i) + count * i;
    }
    gather_symm_dda<T, NRANKS, 1>(recvPtr, count, sendPtr, selfRank);
  }
  else {
    #pragma unroll
    for (int i = 0; i < NRANKS; i++) {
        recvPtr[i] = recvSymPtr.lsaPtr(i);
        sendPtr[i] = sendSymPtr.lsaPtr(i);
    }
    gather_symm_dda<T, NRANKS, 0>(recvPtr, count, sendPtr, selfRank);
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
