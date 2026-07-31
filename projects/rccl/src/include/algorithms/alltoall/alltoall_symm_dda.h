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

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll NRANKS
    for (int r = 0; r < NRANKS; ++r) {
      int srcRank = r;
      int srcIdx = idx + selfRank * idxEnd;
      int destIdx = idx + r * idxEnd;
      //read
      if (useRead) {
        *reinterpret_cast<uint4*>(&myrecvbuff[destIdx]) =
            *reinterpret_cast<const uint4*>(&sendbuffs[r][srcIdx]);
      }
      else {
        //write
        *reinterpret_cast<uint4*>(&recvbuffs[r][srcIdx]) =
	          *reinterpret_cast<const uint4*>(&mysendbuff[destIdx]);
      }
    }
  }
  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms

