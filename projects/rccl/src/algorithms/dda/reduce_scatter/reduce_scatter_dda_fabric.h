/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA reduce-scatter kernel for the fabric/VMM path, using FabricGpuBarrier.
 *
 * Templated on a compile-time rank count NRANKS_CT (matching the all-reduce
 * fabric design):
 *   - NRANKS_CT > 0  : specialized for that clique size; the unified CollCommon
 *                      reduceScatter (pattern 0, one-shot) fully unrolls the
 *                      peer loop.
 *   - NRANKS_CT == 0 : runtime fallback; the rank count is passed via nRanks and
 *                      the unified helper partially unrolls 8-wide, so a single
 *                      instantiation covers any other clique size up to
 *                      kDdaMaxNranks.
 *
 * The flat path copies sendbuff to scratch before launch. The eight-rank
 * hierarchy stages only the sibling's shards inside its kernel.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "algorithms/dda/reduce_scatter/dda_rs_hierarchy.h"

namespace dda::common {

// Every wave drains its system-scope scratch writes before the block publishes
// barrier tokens. Retain the existing release/acquire flag protocol.
__device__ __forceinline__ void publishHierarchicalRsData(FabricGpuBarrier& barrier) {
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0" ::: "memory");
#else
  barrier.orderDestWrites();
#endif
  barrier.syncOnSameBlockIdx<true, true>();
}

// With Stage=true, stage 0 publishes only the opposite rail's input shards.
// Stage 1 reduces paired tiles for the destination shards on this tile's rail.
// Those shards stay at their original offsets in our scratch. The sibling reads
// and writes the OTHER rail's shards, so neither tile overwrites input still
// needed by its sibling. No additional scratch allocation is required.
//
// Both stages assign the same offset within EVERY shard to the same block. The
// intermediate per-block barrier therefore orders all producers of a scaleup
// load without requiring a grid-wide barrier or a second kernel launch.
template <typename T, int ScaleUp = 0, bool Stage = false>
__launch_bounds__(1024)
__global__ void ddaReduceScatterFabricHierarchical(
    T* const* __restrict__ scratch, T* recvbuff, size_t count, int self,
    nccl_dda_detail::DdaRsHierarchy<kDdaMaxNranks> hierarchy, FabricGpuBarrier barrier,
    const T* sendbuff) {
  const int nScaleUp = ScaleUp > 0 ? ScaleUp : hierarchy.nScaleUp;

  constexpr size_t elements = sizeof(uint4) / sizeof(T);
  const size_t start = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * elements;
  const size_t stride = size_t(gridDim.x) * blockDim.x * elements;
  if constexpr (Stage) {
    static_assert(ScaleUp == 4, "In-kernel staging is specialized for eight ranks");
    // Publish only the four shards our sibling needs. Our own contributions
    // are read directly from sendbuff during the pair reduction below.
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      if (d == hierarchy.scaleUpRanks[0] || d == hierarchy.scaleUpRanks[1] ||
          d == hierarchy.scaleUpRanks[2] || d == hierarchy.scaleUpRanks[3]) continue;
      const size_t shard = size_t(d) * count;
      for (size_t i = start; i < count; i += stride) {
        const uint4 value = *reinterpret_cast<const uint4*>(sendbuff + shard + i);
        ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(scratch[self] + shard + i),
                           value.x, value.y, value.z, value.w);
      }
    }
  }
  if constexpr (Stage) {
    publishHierarchicalRsData(barrier);
  } else {
    barrier.syncOnSameBlockIdx<true, true>();
  }

#pragma unroll
  for (int d = 0; d < nScaleUp; ++d) {
    const size_t shard = size_t(hierarchy.scaleUpRanks[d]) * count;
    for (size_t i = start; i < count; i += stride) {
      const size_t offset = shard + i;
      uint4 a, b;
      if constexpr (Stage) {
        a = *reinterpret_cast<const uint4*>(sendbuff + offset);
      } else {
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(scratch[self] + offset), a.x, a.y, a.z, a.w);
      }
      ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(scratch[hierarchy.sibling] + offset), b.x, b.y, b.z, b.w);
      const uint4 sum = vecElementAdd<T>(a, b);
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(scratch[self] + offset), sum.x, sum.y, sum.z, sum.w);
    }
  }
  publishHierarchicalRsData(barrier);

  for (size_t i = start; i < count; i += stride) {
    const size_t offset = size_t(self) * count + i;
    uint4 sum{0, 0, 0, 0};
    if constexpr (ScaleUp == 4) {
      // Issue all four peer loads before their dependent arithmetic.
      uint4 values[4];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(scratch[hierarchy.scaleUpRanks[p]] + offset),
                         values[p].x, values[p].y, values[p].z, values[p].w);
      }
#pragma unroll
      for (int p = 0; p < 4; ++p) sum = vecElementAdd<T>(sum, values[p]);
    } else {
#pragma unroll
      for (int p = 0; p < nScaleUp; ++p) {
        uint4 value;
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(scratch[hierarchy.scaleUpRanks[p]] + offset),
                         value.x, value.y, value.z, value.w);
        sum = vecElementAdd<T>(sum, value);
      }
    }
    *reinterpret_cast<uint4*>(recvbuff + i) = sum;
  }
  // Every peer must finish reading before a later collective reuses scratch.
  barrier.syncOnSameBlockIdx<true, false>();
}

template <typename T, int NRANKS_CT, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                         const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                         FabricGpuBarrier barrier, const T* __restrict__ acc) {
  // Release-acquire barrier ensures the stream-ordered scratch write is visible to peers.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  reduceScatter<T, NRANKS_CT, hasAcc>(ipcbuffs, recvbuff, acc, selfRank, nRanks, idxStart, idxEnd, idxStride, 0);

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace dda::common
