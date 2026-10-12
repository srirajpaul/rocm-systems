/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA all-reduce kernels for the fabric/VMM path, using FabricGpuBarrier.
 *
 * The kernels are templated on a compile-time rank count NRANKS_CT:
 *   - NRANKS_CT > 0  : specialized for that clique size; the unified CollCommon
 *                      reduceScatter/allGather fully unroll the peer loop
 *                      (matching the IPC fast path). The host launcher
 *                      instantiates this for the common sizes (e.g. 4, 8).
 *   - NRANKS_CT == 0 : runtime fallback; the rank count is passed via the nRanks
 *                      argument and the unified helpers partially unroll 8-wide,
 *                      so a single instantiation covers any other clique size
 *                      up to kDdaMaxNranks.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

// Writes to dst[idx] the sum of line idx at srcOff in every rank's scratch (plus acc), all with system-scope
// b128 accesses. Ranks are added in order, as reduceScatter does. The line helpers are compiler memory
// barriers, so every load of a group is issued before its adds.
template <typename T, int kUnroll, bool hasAcc>
__device__ __forceinline__ void ddaArSysReduceLines(T* const* __restrict__ ipcbuffs, int nRanks, size_t srcOff,
                                                    T* __restrict__ dst, const T* __restrict__ acc, size_t idxStart,
                                                    size_t idxEnd, size_t idxStride) {
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    uint4 sum{0, 0, 0, 0};
    if constexpr (hasAcc) {
      sum = reinterpret_cast<const uint4*>(&acc[srcOff + idx])[0];
    }
    for (int r0 = 0; r0 < nRanks; r0 += kUnroll) {
      uint32_t w[kUnroll][4];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (r0 + u < nRanks) {
          ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(&ipcbuffs[r0 + u][srcOff + idx]), w[u][0], w[u][1],
                            w[u][2], w[u][3]);
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (r0 + u < nRanks) sum = vecElementAdd<T>(sum, make_uint4(w[u][0], w[u][1], w[u][2], w[u][3]));
      }
    }
    ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[idx]), sum.x, sum.y, sum.z, sum.w);
  }
}

// One-shot: stage sendbuff into our scratch, then every rank reduces the whole buffer from all scratches into
// its recvbuff. Every access is system-scope, so neither barrier needs a cache fence.
template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceFlatFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                         const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                         FabricGpuBarrier barrier, const T* __restrict__ acc) {
  const uint32_t flag = barrier.beginEpoch();

  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // Thread gtIdx on each peer later reads exactly the lines this thread stages (same grid on every rank). The
  // previous op's closing barrier guarantees no peer is still reading our scratch.
  T* selfScratch = ipcbuffs[selfRank];
  ddaSysCopyLines<1>(
    1, idxStart, idxEnd, idxStride, [&](int) { return sendbuff; }, [&](int) { return selfScratch; });

  ddaFabricWaitMemOps();
  // Publish the staged scratch; the reads below are system-scope, so no acquire is needed.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

  ddaArSysReduceLines<T, kUnroll, hasAcc>(ipcbuffs, nRanksEff, 0, recvbuff, acc, idxStart, idxEnd, idxStride);

  // Peers must not overwrite their scratch until every rank has read it; nothing is published here.
  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */>(flag + 2);
  barrier.endEpoch(flag + 2);
}

// Two-shot: stage sendbuff into our scratch, reduce this rank's shard from every scratch into our own scratch,
// then gather every rank's reduced shard into recvbuff. Every access is system-scope, so no barrier needs a
// cache fence.
template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceTreeFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                         const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                         FabricGpuBarrier barrier, const T* __restrict__ acc) {
  const uint32_t flag = barrier.beginEpoch();

  // Use the compile-time rank count as the divisor when specialized.
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t countPerRank = count / nRanksEff;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;

  // Stage every shard of sendbuff. Peer r's thread gtIdx reduces shard r of our scratch at exactly the lines
  // this thread writes (same grid on every rank). The previous op's closing barrier guarantees no peer is still
  // reading our scratch.
  T* selfScratch = ipcbuffs[selfRank];
  ddaSysCopyLines<kUnroll>(
    nRanksEff, idxStart, idxEnd, idxStride, [&](int r) { return sendbuff + r * countPerRank; },
    [&](int r) { return selfScratch + r * countPerRank; });

  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

  // Reduce our shard into our own scratch. Peers only read their own shard of our scratch in this phase.
  const size_t selfShard = selfRank * countPerRank;
  ddaArSysReduceLines<T, kUnroll, hasAcc>(ipcbuffs, nRanksEff, selfShard, selfScratch + selfShard, acc, idxStart,
                                          idxEnd, idxStride);

  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 2);

  ddaSysCopyLines<kUnroll>(
    nRanksEff, idxStart, idxEnd, idxStride, [&](int r) { return ipcbuffs[r] + r * countPerRank; },
    [&](int r) { return recvbuff + r * countPerRank; });

  // Peers must not overwrite their scratch until every rank has read it; nothing is published here.
  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */>(flag + 3);
  barrier.endEpoch(flag + 3);
}

} // namespace dda::common
