/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh.
 * Includes use *.h names so RCCL hipify output (src/include/...) resolves correctly.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "ipc_gpu_barrier.h"
#include "algorithms/CollCommon.h"

namespace meta::comms {

struct ddaAllReduceFlatIpcLLArgs {
    size_t counter;
    size_t scratchBytes;
};

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
__global__ void ddaAllReduceFlatIpcLL(
    T* const* __restrict__ ipcbuffs,
    T* __restrict__ recvbuff,
    size_t count,
    const T* __restrict__ sendbuff,
    int selfRank,
    IpcGpuBarrier barrier,
    ddaAllReduceFlatIpcLLArgs args,
    const T* __restrict__ acc) {
  // TODO: support types other than 4 bytes also
  assert(sizeof(T) == sizeof(uint32_t));

  constexpr auto countPerThread = sizeof(uint64_t) / sizeof(T);
  const auto gIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const auto gStride = gridDim.x * blockDim.x;

  const auto idxStart = gIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gStride * countPerThread;

  const size_t numBanks = 2;
  const size_t counter = args.counter;
  const size_t bankBytes = args.scratchBytes / numBanks;
  const size_t rankBytes = bankBytes / NRANKS;
  const size_t bankOffset = (counter % numBanks) * bankBytes;
  const size_t dataOffset = bankOffset + selfRank * rankBytes;
  const size_t num_packets = count / countPerThread;
  for (int r = 0; r < NRANKS; r++) {
      ll_pack_and_write((LLPacket16*)ipcbuffs[r] + (dataOffset / sizeof(LLPacket16)), (uint32_t*)sendbuff,
                        num_packets, counter, gIdx, gStride);

    //copyFromSrcToDest<T>(
    //    sendbuff,
    //    ipcbuffs[r] + selfRank * count,
    //    idxStart,
    //    idxEnd,
    //    idxStride);
  }

  __threadfence();

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  //for (int r = 0; r < NRANKS; r++) {
      //volatile LLPacket16* my_slot_from_p = my_recv_base + p * slot_stride_pkts + bank_offset;
      //uint32_t* out_slot = my_output_base + p * output_stride_words;
      //ll_poll_and_unpack(out_slot, my_slot_from_p,
      //                   num_packets, flag_val, tid, nthreads);
  //}

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
      for (int r = 0; r < NRANKS; r++) {
          const size_t dataOffset = bankOffset + r * rankBytes;
          LLPacket16 *src = (LLPacket16*)ipcbuffs[selfRank] + (dataOffset / sizeof(LLPacket16));
          //read data from src by converting to LLstruct
          LLPacket16 pkt = src[idx / countPerThread];
          //get the two data and add to recv buffer
          recvbuff[idx] += pkt.data0;
          recvbuff[idx + 1] += pkt.data1;
      }
  }


  //localReduceLL<T, NRANKS, hasAcc>(
  //    ipcbuffs[selfRank], recvbuff, acc, idxStart, idxEnd, idxStride);

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
