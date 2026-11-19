/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

// Each CTA owns the same offsets in every shard in both scratch regions.
// Matching-CTA release/acquire barriers publish the two pushes and protect
// scratch reuse, including in-place output and graph replay. No grid-wide
// synchronization or new barrier state is needed.
// First shot transposes input into each reducer's local scratch. The second
// shot pushes its reduced shard to a separate output region on every peer.
// Keeping the regions disjoint prevents an early reducer overwriting a peer's
// input. The host requires 2 * messageBytes of existing scratch capacity.
template <typename T>
__launch_bounds__(512)
__global__ void ddaAllReduceTreeFabricPush4(T* const* __restrict__ scratch,
                                           T* __restrict__ recvbuff, size_t count,
                                           const T* __restrict__ sendbuff, int selfRank,
                                           FabricGpuBarrier barrier) {
  constexpr size_t elementsPerVector = sizeof(uint4) / sizeof(T);
  const size_t shardVectors = count / (4 * elementsPerVector);
  const size_t outputBase = 4 * shardVectors;
  const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const auto* input = reinterpret_cast<const uint4*>(sendbuff);
  const auto* local = reinterpret_cast<const uint4*>(scratch[selfRank]);
  for (size_t v = first; v < shardVectors; v += stride) {
    uint4 values[4];
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) values[peer] = input[static_cast<size_t>(peer) * shardVectors + v];
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      auto* dest = reinterpret_cast<uint4*>(scratch[peer]) + static_cast<size_t>(selfRank) * shardVectors + v;
      const auto value = values[peer];
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(dest), value.x, value.y, value.z, value.w);
    }
  }
  barrier.orderDestWrites();
  barrier.syncOnSameBlockIdx<true, true>();
  for (size_t v = first; v < shardVectors; v += stride) {
    uint4 values[4];
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      const auto* src = reinterpret_cast<const uint32_t*>(local + static_cast<size_t>(peer) * shardVectors + v);
      ddaLLLoadLineB128(src, values[peer].x, values[peer].y, values[peer].z, values[peer].w);
    }
    const uint4 sum = vecElementAdd<T>(vecElementAdd<T>(values[0], values[1]),
                                      vecElementAdd<T>(values[2], values[3]));
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      auto* dest = reinterpret_cast<uint4*>(scratch[peer]) + outputBase + static_cast<size_t>(selfRank) * shardVectors + v;
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(dest), sum.x, sum.y, sum.z, sum.w);
    }
  }
  barrier.orderDestWrites();
  barrier.syncOnSameBlockIdx<true, true>();
  auto* output = reinterpret_cast<uint4*>(recvbuff);
  for (size_t v = first; v < shardVectors; v += stride) {
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      const size_t offset = static_cast<size_t>(peer) * shardVectors + v;
      uint4 value;
      ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(local + outputBase + offset),
                         value.x, value.y, value.z, value.w);
      output[offset] = value;
    }
  }
  barrier.syncOnSameBlockIdx<true, false>();
}

} // namespace dda::common
