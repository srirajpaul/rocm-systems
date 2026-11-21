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
} // namespace dda::common
