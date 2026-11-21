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
 * The flat path copies sendbuff to scratch before launch. The eight- and
 * sixteen-rank hierarchies stage only the sibling's shards inside the kernel.
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

// Interleave (destination, wave-sized offset) tasks within a block's existing
// offset ownership. Each task is one coalesced vector per lane. For sequential
// RS, begin/stride select this block's grid-stride tiles; for pipeline RS they
// select consecutive tiles of this producer's chunk. Input and landing bases
// differ only for the pipeline's reusable banks. No task crosses block owners.
template <typename T, int ScaleUp>
__device__ __forceinline__ void pushHierarchicalRsInterleaved(
    T* const* scratch, const T* sendbuff, size_t count, int self,
    const nccl_dda_detail::DdaRsHierarchy<kDdaMaxNranks>& hierarchy, int firstPeer,
    size_t begin, size_t end, size_t stride, size_t inputBase, size_t landingBase) {
  static_assert(ScaleUp == 4 || ScaleUp == 8, "Interleaving requires a specialized hierarchy");
  constexpr size_t elements = sizeof(uint4) / sizeof(T);
  const int wave = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int waves = blockDim.x / warpSize;
  for (size_t tile = begin; tile < end; tile += stride) {
#pragma unroll
    for (int step = 0; step < ScaleUp; ++step) {
      const int task = wave + step * waves;
      const int destination = hierarchy.scaleUpRanks[(firstPeer + task % ScaleUp) % ScaleUp];
      const size_t i = tile + size_t(task / ScaleUp * warpSize + lane) * elements;
      if (i < end) {
        const size_t offset = size_t(destination) * count + inputBase + i;
        const uint4 a = *reinterpret_cast<const uint4*>(sendbuff + offset);
        uint4 b;
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(scratch[hierarchy.sibling] + offset),
                         b.x, b.y, b.z, b.w);
        const uint4 sum = vecElementAdd<T>(a, b);
        ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(scratch[destination] + size_t(self) * count + landingBase + i),
                           sum.x, sum.y, sum.z, sum.w);
      }
    }
  }
}

// With Stage=true, stage 0 publishes only the opposite rail's input shards.
// Stage 1 reduces paired tiles for the destination shards on this tile's rail.
// Each producer pushes its sum to the destination's scratch slot indexed by the
// producer rank. These slots belong to the destination's own rail, disjoint from
// the opposite-rail input shards its sibling still reads. Own contributions come
// from sendbuff, so incoming sums cannot overwrite unread input. Stage 2 reduces
// the local slots. No additional scratch allocation is required.
// Stage=false retains the generic local-store/remote-read path: its full-input
// scratch copy cannot safely receive sums while pair reductions still read it.
//
// Both stages assign the same offset within EVERY shard to the same block. The
// intermediate per-block barrier therefore orders all partial-sum producers
// before the final reduction, without a grid-wide barrier or a second launch.
template <typename T, int ScaleUp = 0, bool Stage = false>
__launch_bounds__(1024)
__global__ void ddaReduceScatterFabricHierarchical(
    T* const* __restrict__ scratch, T* recvbuff, size_t count, int self,
    nccl_dda_detail::DdaRsHierarchy<kDdaMaxNranks> hierarchy, FabricGpuBarrier barrier,
    const T* sendbuff, int schedule) {
  const int nScaleUp = ScaleUp > 0 ? ScaleUp : hierarchy.nScaleUp;
  int firstPeer = 0;
  if constexpr (Stage) {
#pragma unroll
    for (int p = 0; p < ScaleUp; ++p)
      if (hierarchy.scaleUpRanks[p] == self) firstPeer = (p + 1) % ScaleUp;
  }

  constexpr size_t elements = sizeof(uint4) / sizeof(T);
  const size_t start = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * elements;
  const size_t stride = size_t(gridDim.x) * blockDim.x * elements;
  if constexpr (Stage) {
    static_assert(ScaleUp == 4 || ScaleUp == 8, "In-kernel staging requires eight or sixteen ranks");
    // Publish only the opposite rail's shards that our sibling needs. Our own contributions
    // are read directly from sendbuff during the pair reduction below.
#pragma unroll
    for (int d = 0; d < 2 * ScaleUp; ++d) {
      if (d == hierarchy.scaleUpRanks[0] || d == hierarchy.scaleUpRanks[1] ||
          d == hierarchy.scaleUpRanks[2] || d == hierarchy.scaleUpRanks[3]) continue;
      if constexpr (ScaleUp == 8) {
        if (d == hierarchy.scaleUpRanks[4] || d == hierarchy.scaleUpRanks[5] ||
            d == hierarchy.scaleUpRanks[6] || d == hierarchy.scaleUpRanks[7]) continue;
      }
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

  if constexpr (Stage) {
    if (schedule == 2) {
      pushHierarchicalRsInterleaved<T, ScaleUp>(scratch, sendbuff, count, self, hierarchy, firstPeer,
        size_t(blockIdx.x) * blockDim.x * elements, count, stride, 0, 0);
    }
  }
  if (!Stage || schedule != 2) {
#pragma unroll
    for (int d = 0; d < nScaleUp; ++d) {
      const int destination = hierarchy.scaleUpRanks[schedule == 0 ? d : (firstPeer + d) % nScaleUp];
      const size_t shard = size_t(destination) * count;
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
        T* partial;
        if constexpr (Stage) {
          partial = scratch[destination] + size_t(self) * count + i;
        } else {
          partial = scratch[self] + offset;
        }
        ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(partial), sum.x, sum.y, sum.z, sum.w);
      }
    }
  }
  publishHierarchicalRsData(barrier);

  for (size_t i = start; i < count; i += stride) {
    const size_t offset = size_t(self) * count + i;
    uint4 sum{0, 0, 0, 0};
    if constexpr (ScaleUp == 4 || ScaleUp == 8) {
      // Issue every specialized partial-sum load before dependent arithmetic.
      uint4 values[ScaleUp];
#pragma unroll
      for (int p = 0; p < ScaleUp; ++p) {
        const T* partial;
        if constexpr (Stage) {
          partial = scratch[self] + size_t(hierarchy.scaleUpRanks[p]) * count + i;
        } else {
          partial = scratch[hierarchy.scaleUpRanks[p]] + offset;
        }
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(partial),
                         values[p].x, values[p].y, values[p].z, values[p].w);
      }
#pragma unroll
      for (int p = 0; p < ScaleUp; ++p) sum = vecElementAdd<T>(sum, values[p]);
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
  // Every rank must finish reading before a later collective reuses scratch.
  barrier.syncOnSameBlockIdx<true, false>();
}

// Double-buffered version of the fused-staging push hierarchy. The first half
// of the CTAs produces pair sums, the second half consumes them concurrently.
// CTA b owns the same chunk offsets in every input shard. Its two scratch banks
// occupy [(2*b)*chunk, (2*b+2)*chunk) inside each producer's existing rail slot.
// The host ensures that all banks fit and the total grid stays within the
// communicator's CU-derived cap, so waiting CTAs cannot starve their producers.
template <typename T, int ScaleUp>
__launch_bounds__(1024)
__global__ void ddaReduceScatterFabricHierarchicalPipeline(
    T* const* __restrict__ scratch, T* recvbuff, size_t count, int self,
    nccl_dda_detail::DdaRsHierarchy<kDdaMaxNranks> hierarchy, FabricGpuBarrier barrier,
    const T* sendbuff, size_t chunk, int schedule) {
  static_assert(ScaleUp == 4 || ScaleUp == 8, "Pipeline requires fused staging");
  constexpr size_t elements = sizeof(uint4) / sizeof(T);
  const unsigned producers = gridDim.x / 2;
  const bool producer = blockIdx.x < producers;
  const unsigned owner = producer ? blockIdx.x : blockIdx.x - producers;
  const size_t stride = size_t(producers) * chunk;
  const unsigned rounds = unsigned((count + stride - 1) / stride);
  const uint32_t epoch = barrier.epochBegin();
  int firstPeer = 0;
#pragma unroll
  for (int p = 0; p < ScaleUp; ++p)
    if (hierarchy.scaleUpRanks[p] == self) firstPeer = (p + 1) % ScaleUp;

  if (producer) {
#pragma unroll
    for (int d = 0; d < 2 * ScaleUp; ++d) {
      bool ownRail = false;
#pragma unroll
      for (int p = 0; p < ScaleUp; ++p) ownRail |= d == hierarchy.scaleUpRanks[p];
      if (ownRail) continue;
      const size_t shard = size_t(d) * count;
      for (size_t base = size_t(owner) * chunk; base < count; base += stride) {
        for (size_t i = size_t(threadIdx.x) * elements; i < chunk && base + i < count;
             i += size_t(blockDim.x) * elements) {
          const uint4 value = *reinterpret_cast<const uint4*>(sendbuff + shard + base + i);
          ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(scratch[self] + shard + base + i),
                             value.x, value.y, value.z, value.w);
        }
      }
    }
  }
  // Producer b pairs only with sibling b's staged offsets. Consumer rows have
  // no staging work and may begin waiting for ready tokens immediately.
  publishHierarchicalRsData(barrier);

  for (unsigned round = 0; round < rounds; ++round) {
    const size_t base = size_t(owner) * chunk + size_t(round) * stride;
    const size_t slot = (size_t(2) * owner + (round & 1)) * chunk;
    if (producer) {
      // First two banks are free after the preceding collective's closing
      // barrier. Later reuse waits for every destination to finish round-2.
      if (round >= 2)
        barrier.epochWaitRanks<ScaleUp>(hierarchy.scaleUpRanks, owner + producers, epoch + round);
      if (schedule == 2) {
        const size_t remaining = base < count ? count - base : 0;
        const size_t end = chunk < remaining ? chunk : remaining;
        pushHierarchicalRsInterleaved<T, ScaleUp>(scratch, sendbuff, count, self, hierarchy, firstPeer,
          0, end, size_t(blockDim.x) * elements, base, slot);
      } else {
#pragma unroll
        for (int d = 0; d < ScaleUp; ++d) {
          const int destination = hierarchy.scaleUpRanks[schedule == 0 ? d : (firstPeer + d) % ScaleUp];
          const size_t shard = size_t(destination) * count;
          for (size_t i = size_t(threadIdx.x) * elements; i < chunk && base + i < count;
               i += size_t(blockDim.x) * elements) {
            const uint4 a = *reinterpret_cast<const uint4*>(sendbuff + shard + base + i);
            uint4 b;
            ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(scratch[hierarchy.sibling] + shard + base + i),
                             b.x, b.y, b.z, b.w);
            const uint4 sum = vecElementAdd<T>(a, b);
            ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(scratch[destination] + size_t(self) * count + slot + i),
                               sum.x, sum.y, sum.z, sum.w);
          }
        }
      }
      barrier.epochPostRanks<ScaleUp>(hierarchy.scaleUpRanks, epoch + 2 + round);
    } else {
      barrier.epochWaitRanks<ScaleUp>(hierarchy.scaleUpRanks, owner, epoch + 2 + round);
      for (size_t i = size_t(threadIdx.x) * elements; i < chunk && base + i < count;
           i += size_t(blockDim.x) * elements) {
        uint4 values[ScaleUp];
#pragma unroll
        for (int p = 0; p < ScaleUp; ++p) {
          const T* partial = scratch[self] + size_t(hierarchy.scaleUpRanks[p]) * count + slot + i;
          ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(partial),
                           values[p].x, values[p].y, values[p].z, values[p].w);
        }
        uint4 sum{0, 0, 0, 0};
#pragma unroll
        for (int p = 0; p < ScaleUp; ++p) sum = vecElementAdd<T>(sum, values[p]);
        *reinterpret_cast<uint4*>(recvbuff + base + i) = sum;
      }
      barrier.epochPostRanks<ScaleUp>(hierarchy.scaleUpRanks, epoch + 2 + round);
    }
  }
  barrier.syncOnSameBlockIdx<true, false>();
  // Advance every allocated epoch cell uniformly, including unused rows, so
  // changing counts/grid sizes and graph replays cannot accept stale tokens.
  barrier.epochEnd(epoch, rounds);
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
