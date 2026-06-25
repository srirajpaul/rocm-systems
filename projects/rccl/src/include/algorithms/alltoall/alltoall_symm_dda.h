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

namespace meta::comms {

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
    __global__ void ddaAllToAllIpc(
        T* const* __restrict__ ipcbuffs,
        ncclSymPtr<T> recvbuff,
        T* __restrict__ recvbuff1,
        size_t count,
        const ncclSymPtr<T> sendbuff,
        const T* __restrict__ sendbuff1,
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

  const T *src[NRANKS];
  T *dst[NRANKS];
#if USE_WRITE
  const T* srcPtr = sendbuff.localPtr();
  ncclSymPtr<T> dstSlice = recvbuff + selfRank * countPerRank;
#pragma unroll NRANKS
  for (int r = 0; r < NRANKS; ++r) {
    src[r] = (T*)srcPtr + r * countPerRank;
    dst[r] = dstSlice.lsaPtr(r);
  }
#else
  //ncclSymPtr<T> srcSlice = sendbuff + selfRank * countPerRank;
  //T *dstPtr = recvbuff.localPtr();
  //#pragma unroll NRANKS
  //for (int r = 0; r < NRANKS; ++r) {
  //  src[r] = srcSlice.lsaPtr(r);
  //  dst[r] = dstPtr + r * countPerRank;
  //}

  //const T* __restrict__ sendbuffPtr = sendbuff.localPtr();
  //T* __restrict__ recvbuffPtr = recvbuff.localPtr();
#endif

  // It is expensive to launch hipMemcpyAsync on ROCm
  // Move data copy here. Each block copies part of sendbuff data
  copyFromSrcToDest<T>(
      sendbuff1, ipcbuffs[selfRank], idxStart, copyCount, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  //bool inPlace = (sendbuff == recvbuff);
  static_assert(sizeof(T) == 1);

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
    for (int r = 0; r < NRANKS; ++r) {
      //const size_t peer = (selfRank + r) % NRANKS;
      //if (peer == selfRank && inPlace) continue;

      //*(reinterpret_cast<uint4*>(dst[r] + idx)) =
      //(reinterpret_cast<const uint4*>(src[r] + idx))[0];

      int srcRank = r;
      int srcIdx = idx + selfRank * idxEnd;
      int destIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff1[destIdx]) =
          reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }
  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms

