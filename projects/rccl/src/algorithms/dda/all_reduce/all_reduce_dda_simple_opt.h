/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE.txt for license information.
 ************************************************************************/
#pragma once
#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

struct DdaSimpleRegs {
  uint4 a{}, b{}, c{}, d{};
  template <int I>
  __device__ __forceinline__ uint4& at(std::integral_constant<int, I>) {
    if constexpr (I == 0) return a;
    else if constexpr (I == 1) return b;
    else if constexpr (I == 2) return c;
    else return d;
  }
};

template <int... I, typename F>
__device__ __forceinline__ void ddaSimpleVectors(std::integer_sequence<int, I...>, F f) {
  (f(std::integral_constant<int, I>{}), ...);
}

template <bool FULL>
__device__ __forceinline__ bool ddaSimpleValid(size_t v, size_t shard) {
  if constexpr (FULL) return true;
  else return v < shard;
}

// Full batches contain no per-vector predicates: that permits the compiler to
// issue independent loads together. A separate tail path retains bounds checks.
template <int NRANKS_CT, int BATCH, bool FULL>
__device__ __forceinline__ void ddaSimpleStageTile(const uint4* __restrict__ input,
    uint4* __restrict__ local, int nRanksRuntime, size_t shard, size_t base) {
  const int nRanks = NRANKS_CT > 0 ? NRANKS_CT : nRanksRuntime;
  constexpr int unroll = NRANKS_CT > 0 ? NRANKS_CT : 8;
#pragma unroll unroll
  for (int p = 0; p < nRanks; ++p) {
    ddaSimpleVectors(std::make_integer_sequence<int, BATCH>{}, [&](auto i) {
      const size_t v = base + size_t(i) * blockDim.x;
      if (ddaSimpleValid<FULL>(v, shard)) local[size_t(p) * shard + v] = input[size_t(p) * shard + v];
    });
  }
}

template <typename T, int NRANKS_CT, int BATCH, bool FULL>
__device__ __forceinline__ void ddaSimpleReduceTile(T* const* __restrict__ scratch,
    uint4* local, int selfRank, int nRanksRuntime, size_t shard, size_t base) {
  const int nRanks = NRANKS_CT > 0 ? NRANKS_CT : nRanksRuntime;
  constexpr int unroll = NRANKS_CT > 0 ? NRANKS_CT : 8;
  DdaSimpleRegs sums;
#pragma unroll unroll
  for (int p = 0; p < nRanks; ++p) {
    DdaSimpleRegs values;
    ddaSimpleVectors(std::make_integer_sequence<int, BATCH>{}, [&](auto i) {
      const size_t v = base + size_t(i) * blockDim.x;
      if (ddaSimpleValid<FULL>(v, shard)) values.at(i) = reinterpret_cast<const uint4*>(scratch[p])[size_t(selfRank) * shard + v];
    });
    ddaSimpleVectors(std::make_integer_sequence<int, BATCH>{}, [&](auto i) {
      sums.at(i) = vecElementAdd<T>(sums.at(i), values.at(i));
    });
  }
  ddaSimpleVectors(std::make_integer_sequence<int, BATCH>{}, [&](auto i) {
    const size_t v = base + size_t(i) * blockDim.x;
    if (ddaSimpleValid<FULL>(v, shard)) local[size_t(selfRank) * shard + v] = sums.at(i);
  });
}

template <int NRANKS_CT, int BATCH, bool FULL>
__device__ __forceinline__ void ddaSimpleGatherTile(uint4* const* __restrict__ scratch,
    uint4* __restrict__ dest, int selfRank, int nRanksRuntime, size_t shard, size_t base) {
  const int nRanks = NRANKS_CT > 0 ? NRANKS_CT : nRanksRuntime;
  constexpr int unroll = NRANKS_CT > 0 ? NRANKS_CT : 8;
#pragma unroll unroll
  for (int r = 0; r < nRanks; ++r) {
    const int peer = (selfRank + r) % nRanks;
    ddaSimpleVectors(std::make_integer_sequence<int, BATCH>{}, [&](auto i) {
      const size_t v = base + size_t(i) * blockDim.x;
      if (ddaSimpleValid<FULL>(v, shard)) dest[size_t(peer) * shard + v] = scratch[peer][size_t(peer) * shard + v];
    });
  }
}

// Keep original Simple's pull reduce-scatter/all-gather and rank-order
// arithmetic. A CTA owns the same vectors in every shard and phase; fused
// staging and in-place output therefore cannot race another CTA.
template <typename T, int NRANKS_CT, int BATCH, bool FUSED>
__launch_bounds__(512)
__global__ void ddaAllReduceSimpleOpt(T* const* __restrict__ scratch, T* __restrict__ recvbuff,
    const T* sendbuff, size_t count, int selfRank, int nRanksRuntime, FabricGpuBarrier barrier) {
  const int nRanks = NRANKS_CT > 0 ? NRANKS_CT : nRanksRuntime;
  const size_t shard = count * sizeof(T) / (nRanks * sizeof(uint4));
  const size_t start = size_t(blockIdx.x) * blockDim.x * BATCH + threadIdx.x;
  const size_t stride = size_t(gridDim.x) * blockDim.x * BATCH;
  // Null sendbuff denotes in-place: accesses to that user allocation then
  // derive from recvbuff, preserving its restrict contract and load scheduling.
  const uint4* input = reinterpret_cast<const uint4*>(sendbuff ? sendbuff : recvbuff);
  uint4* local = reinterpret_cast<uint4*>(scratch[selfRank]);

  if constexpr (FUSED) {
    for (size_t base = start; base < shard; base += stride) {
      if (base + size_t(BATCH - 1) * blockDim.x < shard)
        ddaSimpleStageTile<NRANKS_CT, BATCH, true>(input, local, nRanksRuntime, shard, base);
      else
        ddaSimpleStageTile<NRANKS_CT, BATCH, false>(input, local, nRanksRuntime, shard, base);
    }
  }
  barrier.syncOnSameBlockIdx<true, true>();
  for (size_t base = start; base < shard; base += stride) {
    if (base + size_t(BATCH - 1) * blockDim.x < shard)
      ddaSimpleReduceTile<T, NRANKS_CT, BATCH, true>(scratch, local, selfRank, nRanksRuntime, shard, base);
    else
      ddaSimpleReduceTile<T, NRANKS_CT, BATCH, false>(scratch, local, selfRank, nRanksRuntime, shard, base);
  }
  barrier.syncOnSameBlockIdx<true, true>();
  for (size_t base = start; base < shard; base += stride) {
    if (base + size_t(BATCH - 1) * blockDim.x < shard)
      ddaSimpleGatherTile<NRANKS_CT, BATCH, true>(reinterpret_cast<uint4* const*>(scratch),
          reinterpret_cast<uint4*>(recvbuff), selfRank, nRanksRuntime, shard, base);
    else
      ddaSimpleGatherTile<NRANKS_CT, BATCH, false>(reinterpret_cast<uint4* const*>(scratch),
          reinterpret_cast<uint4*>(recvbuff), selfRank, nRanksRuntime, shard, base);
  }
  barrier.syncOnSameBlockIdx<true, false>();
}

// Peer groups exchange only LDS partial sums here. Fabric stores are ordered
// once by the existing phase barrier after reduction, not on every tile.
__device__ __forceinline__ void ddaSimplePeerGroupLdsBarrier() {
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
  __syncthreads();
#endif
}

template <typename T, int GROUPS, bool FULL>
__device__ __forceinline__ void ddaSimplePeerGroupReduceStep(
    T* const* __restrict__ scratch, uint4* local, uint4* partials,
    size_t window, size_t shard, size_t v, int selfRank, int team, int group, int lane) {
  static_assert(GROUPS == 2 || GROUPS == 4);
  constexpr int wave = 32;
  uint4 sum{};
  if (ddaSimpleValid<FULL>(v, shard)) {
    const size_t offset = size_t(selfRank) * shard + v;
    DdaSimpleRegs operands;
    // Four independent operands, held in named registers, for this warp's
    // consecutive peer group. Only the partial sum crosses the warp boundary.
    ddaSimpleVectors(std::make_integer_sequence<int, 4>{}, [&](auto p) {
      operands.at(p) = reinterpret_cast<const uint4*>(scratch[group * 4 + int(p)])[offset];
    });
    ddaSimpleVectors(std::make_integer_sequence<int, 4>{}, [&](auto p) {
      sum = vecElementAdd<T>(sum, operands.at(p));
    });
  }
  const size_t teamSlot = window + size_t(team) * (GROUPS - 1) * wave + lane;
  if (group != 0) partials[teamSlot + (group - 1) * wave] = sum;
  ddaSimplePeerGroupLdsBarrier();
  if (group == 0 && ddaSimpleValid<FULL>(v, shard)) {
#pragma unroll
    for (int g = 1; g < GROUPS; ++g)
      sum = vecElementAdd<T>(sum, partials[teamSlot + (g - 1) * wave]);
    local[size_t(selfRank) * shard + v] = sum;
  }
}

// V2-style reduction only: one warp per four peers. Staging, pull gather,
// scratch layout, and all three fabric barriers match the BATCH=1 tuned path.
// Each CTA still owns blockDim.x consecutive vectors per grid-stride tile;
// peer grouping changes only which threads cooperate on those same vectors.
// This ownership must match staging/gather to protect fused in-place calls.
// Grouped partial sums change floating-point association at 8/16 ranks.
template <typename T, int NRANKS, bool FUSED>
__launch_bounds__(512)
__global__ void ddaAllReduceSimplePeerGroups(T* const* __restrict__ scratch, T* __restrict__ recvbuff,
    const T* sendbuff, size_t count, int selfRank, int nRanksRuntime, FabricGpuBarrier barrier) {
  static_assert(NRANKS == 4 || NRANKS == 8 || NRANKS == 16);
  constexpr int groups = NRANKS / 4;
  constexpr int wave = 32;
  // Preserve the existing runtime-rank staging/gather path at sixteen ranks;
  // only the reduction uses the new compile-time peer-group specialization.
  constexpr int stageGatherRanks = NRANKS == 16 ? 0 : NRANKS;
  const size_t shard = count * sizeof(T) / (nRanksRuntime * sizeof(uint4));
  const size_t start = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  const uint4* input = reinterpret_cast<const uint4*>(sendbuff ? sendbuff : recvbuff);
  uint4* local = reinterpret_cast<uint4*>(scratch[selfRank]);

  if constexpr (FUSED) {
    for (size_t base = start; base < shard; base += stride)
      ddaSimpleStageTile<stageGatherRanks, 1, true>(input, local, nRanksRuntime, shard, base);
  }
  barrier.syncOnSameBlockIdx<true, true>();
  if constexpr (groups == 1) {
    // At four ranks every warp already covers all peers; no LDS exchange.
    for (size_t base = start; base < shard; base += stride)
      ddaSimpleReduceTile<T, 4, 1, true>(scratch, local, selfRank, nRanksRuntime, shard, base);
  } else {
    // The launcher requires complete warp teams, including after caller caps.
    const int lane = threadIdx.x % wave;
    const int warp = threadIdx.x / wave;
    const int group = warp % groups;
    const int team = warp / groups;
    const int teams = blockDim.x / (groups * wave);
    const size_t vectorsPerStep = size_t(teams) * wave;
    const size_t windowSlots = size_t(teams) * (groups - 1) * wave;
    extern __shared__ uint4 partials[];
    unsigned windowIndex = 0;
    for (size_t tile = size_t(blockIdx.x) * blockDim.x; tile < shard; tile += stride) {
#pragma unroll
      for (int part = 0; part < groups; ++part) {
        const size_t step = tile + size_t(part) * vectorsPerStep;
        // Both conditions are block-uniform: every thread, including invalid
        // tail lanes, participates in each LDS barrier.
        if (step < shard) {
          const size_t v = step + size_t(team) * wave + lane;
          const size_t window = size_t(windowIndex) * windowSlots;
          if (step + vectorsPerStep <= shard)
            ddaSimplePeerGroupReduceStep<T, groups, true>(
                scratch, local, partials, window, shard, v, selfRank, team, group, lane);
          else
            ddaSimplePeerGroupReduceStep<T, groups, false>(
                scratch, local, partials, window, shard, v, selfRank, team, group, lane);
          // The intervening barrier finishes every reader of one window
          // before any warp can overwrite that window two steps later.
          windowIndex ^= 1;
        }
      }
    }
  }
  barrier.syncOnSameBlockIdx<true, true>();
  for (size_t base = start; base < shard; base += stride)
    ddaSimpleGatherTile<stageGatherRanks, 1, true>(reinterpret_cast<uint4* const*>(scratch),
        reinterpret_cast<uint4*>(recvbuff), selfRank, nRanksRuntime, shard, base);
  barrier.syncOnSameBlockIdx<true, false>();
}

} // namespace dda::common
