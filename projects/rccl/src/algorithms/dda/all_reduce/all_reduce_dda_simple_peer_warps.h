/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

// Simple-protocol counterpart of ddaAllReduceTwoShotLL128PeerParallel:
// a group of four wave32 warps owns a slice across every shard, one warp per
// peer. Output vectors are distributed across the four warps for register-tree
// reduction, then shared in LDS for the peer pushes. Simple uses unpadded
// 16-byte payloads and the existing matching-CTA barriers instead of LL flags.
//
// Every input read and output write for a slice stays in the same CTA. All
// groups execute the same LDS barriers, including inactive tail groups.
// Self input is read directly and the self result goes directly to recvbuff;
// only remote contributions/results use the two disjoint scratch regions.
// BatchRounds retains this CTA's strided slice ownership in all phases.
// Only the reduction's LDS exchange is batched; cross-rank completion and
// scratch reuse keep the original matching-CTA barrier sequence.
template <typename T, int BatchRounds = 1>
__launch_bounds__(512)
__global__ void ddaAllReduceTwoShotSimplePeerParallel(
    T* const* __restrict__ scratch, T* __restrict__ recvbuff, size_t count,
    const T* __restrict__ sendbuff, int selfRank, FabricGpuBarrier barrier) {
  static_assert(BatchRounds == 1 || BatchRounds == 2 || BatchRounds == 4);
  constexpr int peers = 4;
  constexpr int wave = 32;
  constexpr int vectorsPerLane = 4;
  constexpr size_t vectorsPerSlice = wave * vectorsPerLane;
  const int lane = threadIdx.x % wave;
  const int warp = threadIdx.x / wave;
  const int group = warp / peers;
  const int peerWarp = warp % peers;
  const int peer = (selfRank + peerWarp) % peers;
  const int groups = blockDim.x / (peers * wave);
  const size_t shardVectors = count / (peers * (sizeof(uint4) / sizeof(T)));
  const size_t slices = (shardVectors + vectorsPerSlice - 1) / vectorsPerSlice;
  const size_t firstSlice = static_cast<size_t>(blockIdx.x) * groups;
  const size_t sliceStride = static_cast<size_t>(gridDim.x) * groups;
  const size_t outputBase = peers * shardVectors;
  const auto* input = reinterpret_cast<const uint4*>(sendbuff);
  auto* output = reinterpret_cast<uint4*>(recvbuff);
  const auto* local = reinterpret_cast<const uint4*>(scratch[selfRank]);
  auto* remote = reinterpret_cast<uint4*>(scratch[peer]);
  __shared__ uint4 reduced[BatchRounds][4][vectorsPerLane][wave];

  if (peerWarp != 0) {
    for (size_t base = firstSlice; base < slices; base += sliceStride) {
      const size_t firstVector = (base + group) * vectorsPerSlice + lane;
      uint4 values[vectorsPerLane];
#pragma unroll
      for (int u = 0; u < vectorsPerLane; ++u) {
        const size_t v = firstVector + u * wave;
        if (v < shardVectors) values[u] = input[static_cast<size_t>(peer) * shardVectors + v];
      }
#pragma unroll
      for (int u = 0; u < vectorsPerLane; ++u) {
        const size_t v = firstVector + u * wave;
        if (v < shardVectors) {
          auto* dst = reinterpret_cast<uint32_t*>(remote + static_cast<size_t>(selfRank) * shardVectors + v);
          ddaLLStoreLineB128(dst, values[u].x, values[u].y, values[u].z, values[u].w);
        }
      }
    }
  }
  barrier.orderDestWrites();
  barrier.syncOnSameBlockIdx<true, true>();

  for (size_t base = firstSlice; base < slices; base += sliceStride * BatchRounds) {
    // Populate disjoint LDS slots for several rounds before publishing them
    // to the peer waves. Inactive tail rounds still take both CTA barriers.
#pragma unroll
    for (int batch = 0; batch < BatchRounds; ++batch) {
      const size_t firstVector = (base + batch * sliceStride + group) * vectorsPerSlice + lane;
      const size_t v = firstVector + peerWarp * wave;
      uint4 operands[peers] = {};
      if (v < shardVectors) {
        operands[0] = input[static_cast<size_t>(selfRank) * shardVectors + v];
#pragma unroll
        for (int p = 1; p < peers; ++p) {
          const int sourceRank = (selfRank + p) % peers;
          const auto* src = reinterpret_cast<const uint32_t*>(local + static_cast<size_t>(sourceRank) * shardVectors + v);
          ddaLLLoadLineB128(src, operands[p].x, operands[p].y, operands[p].z, operands[p].w);
        }
      }
      reduced[batch][group][peerWarp][lane] =
          vecElementAdd<T>(vecElementAdd<T>(operands[0], operands[2]),
                           vecElementAdd<T>(operands[1], operands[3]));
    }
    __syncthreads();

#pragma unroll
    for (int batch = 0; batch < BatchRounds; ++batch) {
      const size_t firstVector = (base + batch * sliceStride + group) * vectorsPerSlice + lane;
      uint4 values[vectorsPerLane];
#pragma unroll
      for (int u = 0; u < vectorsPerLane; ++u) values[u] = reduced[batch][group][u][lane];
#pragma unroll
      for (int u = 0; u < vectorsPerLane; ++u) {
        const size_t v = firstVector + u * wave;
        if (v < shardVectors) {
          const size_t offset = static_cast<size_t>(selfRank) * shardVectors + v;
          if (peerWarp == 0) {
            output[offset] = values[u];
          } else {
            auto* dst = reinterpret_cast<uint32_t*>(remote + outputBase + offset);
            ddaLLStoreLineB128(dst, values[u].x, values[u].y, values[u].z, values[u].w);
          }
        }
      }
    }
    __syncthreads(); // All peer waves consumed every LDS slot before reuse.
  }
  barrier.orderDestWrites();
  barrier.syncOnSameBlockIdx<true, true>();

  if (peerWarp != 0) {
    for (size_t base = firstSlice; base < slices; base += sliceStride) {
      const size_t firstVector = (base + group) * vectorsPerSlice + lane;
#pragma unroll
      for (int u = 0; u < vectorsPerLane; ++u) {
        const size_t v = firstVector + u * wave;
        if (v < shardVectors) {
          const size_t offset = static_cast<size_t>(peer) * shardVectors + v;
          uint4 value;
          ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(local + outputBase + offset),
                           value.x, value.y, value.z, value.w);
          output[offset] = value;
        }
      }
    }
  }
  barrier.syncOnSameBlockIdx<true, false>();
}

} // namespace dda::common
