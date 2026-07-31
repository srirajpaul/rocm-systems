/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh.
 * Adapted for alltoall collective operation.
 * Includes use *.h names so RCCL hipify output (src/include/...) resolves correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "ipc_gpu_barrier.h"
#include "algorithms/CollCommon.h"
#include "nccl_device/ptr.h"
#include "nccl_device/rccl_ptr.h"

namespace meta::comms {

// The alltoall copy is routed through these helpers so the local base pointer
// is passed as a __restrict__ function parameter. Clang only applies the
// noalias guarantee to function parameters (not to __restrict__ locals), so
// this lets the compiler prove the stores don't alias the loads and keep
// multiple memory ops in flight, even when the base is derived on-device via
// localPtr(). __forceinline__ preserves the noalias info at the call site.
template <typename T, int NRANKS>
__device__ __forceinline__ void ddaAllToAllGather(
    T* __restrict__ dst,
    const std::array<T*, NRANKS>& src,
    size_t idxStart, size_t idxEnd, size_t idxStride, int selfRank) {
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
    for (int r = 0; r < NRANKS; ++r) {
      int srcIdx = idx + selfRank * idxEnd;
      int destIdx = idx + r * idxEnd;
      *(v4u_gptr)(&dst[destIdx]) = *(v4u_gptr)(&src[r][srcIdx]);
      //__attribute__((address_space(1))) v4u* dst1 = (__attribute__((address_space(1))) v4u*)&dst[destIdx];
      //__attribute__((address_space(1))) v4u* src1 = (__attribute__((address_space(1))) v4u*)&src[r][srcIdx];
      //*dst1 = *src1;
    }
  }
}

template <typename T, int NRANKS>
__device__ __forceinline__ void ddaAllToAllScatter(
    const std::array<T*, NRANKS>& dst,
    const T* __restrict__ src,
    size_t idxStart, size_t idxEnd, size_t idxStride, int selfRank) {
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
    for (int r = 0; r < NRANKS; ++r) {
      int srcIdx = idx + selfRank * idxEnd;
      int destIdx = idx + r * idxEnd;
      *(v4u_gptr)(&dst[r][srcIdx]) = *(v4u_gptr)(&src[destIdx]);
    }
  }
}

template <typename T, int NRANKS, bool hasAcc, bool useHostPtr = true>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
    __global__ void ddaAllToAllIpc(
        T* const* __restrict__ ipcbuffs,
        ncclSymPtr<T> recvSymPtr,
        T* __restrict__ recvbuff1,
        std::array<T*, NRANKS> recvbuffs,
        size_t count,
        const ncclSymPtr<T> sendSymPtr,
        const T* __restrict__ sendbuff1,
        std::array<T*, NRANKS> sendbuffs,
        int selfRank,
        IpcGpuBarrier barrier) {
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch
  // TODO: we should be able to deal with left over as well
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t copyCount = count * NRANKS;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // Select how this rank obtains its own local send/recv base pointers:
  //  - useHostPtr == true : use raw pointers resolved on the host (sendbuff1/recvbuff1)
  //  - useHostPtr == false: derive them on device from the symmetric pointers
  const T* __restrict__ mysendbuff;
  T* __restrict__ myrecvbuff;
  if constexpr (useHostPtr) {
    mysendbuff = sendbuff1;
    myrecvbuff = recvbuff1;
  } else {
    mysendbuff = sendSymPtr.localPtr();
    myrecvbuff = recvSymPtr.localPtr();
  }

  barrier.syncOnSameBlockIdx<
      false /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  static_assert(sizeof(T) == 1);
  bool useRead = true;

  if (useRead) {
    ddaAllToAllGather<T, NRANKS>(
        myrecvbuff, sendbuffs, idxStart, idxEnd, idxStride, selfRank);
  } else {
    ddaAllToAllScatter<T, NRANKS>(
        recvbuffs, mysendbuff, idxStart, idxEnd, idxStride, selfRank);
  }
  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms

