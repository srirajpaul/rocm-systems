/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA alltoall kernel for the fabric/VMM path, using FabricGpuBarrier.
 *
 * Templated on a compile-time rank count NRANKS_CT (matching the all-reduce
 * fabric design):
 *   - NRANKS_CT > 0  : specialized for that clique size (unrolled peer loop).
 *   - NRANKS_CT == 0 : runtime fallback driven by the nRanks argument, so a
 *                      single instantiation covers any other clique size up to
 *                      kDdaMaxNranks.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

constexpr int kDdaA2APushThreadsPerBlock = 1024;

// Pull alltoall: each rank stages its sendbuff into its own scratch, then reads its slot out of every
// peer's scratch into recvbuff. Every access is system-scope, so neither barrier needs a cache fence.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(kDdaA2APushThreadsPerBlock)
#endif
  __global__ void ddaAllToAllFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                    const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                    FabricGpuBarrier barrier) {
  const uint32_t flag = barrier.beginEpoch();

  // use uint4 to do 16-byte loads to maximize memory efficiency. We assume
  // that count % countPerThread == 0, enforced before kernel launch.
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  // Fully unroll when the clique size is a compile-time constant; otherwise
  // partially unroll 8-wide for the runtime fallback (Option A).
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // Stage every slot of sendbuff into our scratch. Thread gtIdx on each peer later reads exactly the lines this
  // thread writes (same grid on every rank), so the per-block barrier covers them. The previous op's closing
  // barrier guarantees no peer is still reading our scratch.
  T* selfScratch = ipcbuffs[selfRank];
  ddaSysCopyLines<kUnroll>(
    nRanksEff, idxStart, idxEnd, idxStride, [&](int r) { return sendbuff + r * idxEnd; },
    [&](int r) { return selfScratch + r * idxEnd; });

  ddaFabricWaitMemOps();
  // Publish the staged scratch; the reads below are system-scope, so no acquire is needed.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

  const size_t selfSlot = selfRank * idxEnd;
  ddaSysCopyLines<kUnroll>(
    nRanksEff, idxStart, idxEnd, idxStride, [&](int r) { return ipcbuffs[r] + selfSlot; },
    [&](int r) { return recvbuff + r * idxEnd; });

  // Peers must not overwrite their scratch until every rank has read its slot; nothing is published here.
  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */>(flag + 2);
  barrier.endEpoch(flag + 2);
}

template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(kDdaA2APushThreadsPerBlock)
#endif
  __global__ void ddaAllToAllFabricPush(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                        const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                        FabricGpuBarrier barrier) {
  const uint32_t flag = barrier.beginEpoch();
  // Only local sendbuff is read before the next barrier, so no acquire is needed here.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // Every access is system-scope so none of it goes through L2. The line helpers are compiler memory
  // barriers, so issue every load of the group before its stores.
  const size_t destIdx = selfRank * idxEnd;
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    for (int r0 = 0; r0 < nRanksEff; r0 += kUnroll) {
      uint4 v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (r0 + u < nRanksEff) {
          uint32_t t0, t1, t2, t3;
          ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(&sendbuff[idx + (r0 + u) * idxEnd]), t0, t1, t2, t3);
          v[u] = make_uint4(t0, t1, t2, t3);
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (r0 + u < nRanksEff) {
          ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&ipcbuffs[r0 + u][destIdx + idx]), v[u].x, v[u].y, v[u].z,
                             v[u].w);
        }
      }
    }
  }

  // TODO : move these inside barrier
  __syncthreads();

#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if defined(__gfx1250__)
  asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0");
#else
  asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
#endif
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#else
  __threadfence_system();
#endif

  // Publish our remote writes. The drain below reads only with system-scope loads, so no acquire is needed.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 2);

  // Drain our scratch into recvbuff. Peer r's thread gtIdx wrote exactly the positions this thread reads
  // from slot r, since every rank launches the same grid. As above, every access is system-scope and the
  // loads of a group are issued before its stores.
  const T* selfScratch = ipcbuffs[selfRank];
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    for (int r0 = 0; r0 < nRanksEff; r0 += kUnroll) {
      uint32_t w[kUnroll][4];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (r0 + u < nRanksEff) {
          ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(&selfScratch[(r0 + u) * idxEnd + idx]), w[u][0], w[u][1],
                            w[u][2], w[u][3]);
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (r0 + u < nRanksEff) {
          ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&recvbuff[(r0 + u) * idxEnd + idx]), w[u][0], w[u][1], w[u][2],
                             w[u][3]);
        }
      }
    }
  }
  barrier.endEpoch(flag + 2);
}

} // namespace dda::common
