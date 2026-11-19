/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/all_reduce/all_reduce_dda_ll128.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_twoshot.h"

namespace dda::common {

// One CTA owns a full-message slice, with one warp per peer. Even a single
// slice therefore has independent transfers and polls at every rank count.
// All input reads for a slice stay in its owning CTA for in-place safety.
// The partial array is indexed by absolute rank: every GPU must use exactly
// the same reduction tree, unlike two-shot where one GPU reduces each shard.
template <typename T, int N>
__launch_bounds__(N * kDdaLL128Warp)
__global__ void ddaAllReduceOneShotLL128PeerParallel(
    T* const* __restrict__ peerScratch, T* __restrict__ recvbuff,
    const T* __restrict__ sendbuff, size_t bytes, int selfRank, int nRanksRt,
    uint32_t* __restrict__ epochDev, int epochLen, size_t slicesTotal,
    size_t bankSize, DdaKernelParams params) {
  static_assert(N == 4 || N == 8 || N == 16 || N == 32, "one warp per peer, power-of-two tree");
  (void)nRanksRt;
  const int lane = threadIdx.x % kDdaLL128Warp;
  const int peer = threadIdx.x / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);
  __shared__ uint64_t partial[N][kDdaLL128WordsPerThread][kDdaLL128Warp];
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | flag32;
  const size_t slotWords = ddaLL128ArSlotWords(bankSize, N);
  const size_t bankWords = (size_t)(flag32 & 1u) * (bankSize / sizeof(uint64_t));
  const uint64_t* self = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;
  uint64_t* remote = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords;
  const int8_t* src = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dst = reinterpret_cast<int8_t*>(recvbuff);

  // Publish all slices assigned to this CTA before waiting for peer inputs.
  if (peer != selfRank) {
    for (size_t s = blockIdx.x; s < slicesTotal; s += gridDim.x) {
      const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem = bytes - sliceByte;
      const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
      uint64_t regs[kDdaLL128WordsPerThread];
      ddaLL128LoadRegsBatched<int8_t>(regs, src + sliceByte, elt, lane, flagLane);
      ddaLL128StoreWire(remote + (size_t)selfRank * slotWords + s * kDdaLL128WireWordsPerSlice + 2 * lane,
                       regs, flag, flagLane);
    }
  }
  __syncthreads();

  for (size_t s = blockIdx.x; s < slicesTotal; s += gridDim.x) {
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = bytes - sliceByte;
    const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread] = {};
    bool landed = true;
    if (peer == selfRank) {
      ddaLL128LoadRegsBatched<int8_t>(regs, src + sliceByte, elt, lane, flagLane);
    } else {
      landed = ddaLL128PollWire(self + (size_t)peer * slotWords + s * kDdaLL128WireWordsPerSlice + 2 * lane,
                               regs, flag, lane, params);
    }
#pragma unroll
    for (int u = 0; u < kDdaLL128WordsPerThread; ++u) partial[peer][u][lane] = regs[u];
    if (__syncthreads_or(!landed)) break;

    // Each output word has one reducer. Load every operand before evaluating
    // its register tree, avoiding a CTA barrier at every tree level.
#pragma unroll
    for (int u = peer; u < kDdaLL128WordsPerThread; u += N) {
      uint64_t values[N];
#pragma unroll
      for (int p = 0; p < N; ++p) values[p] = partial[p][u][lane];
#pragma unroll
      for (int step = N / 2; step > 0; step /= 2) {
#pragma unroll
        for (int p = 0; p < step; ++p) values[p] = ddaLL128AddWord<T>(values[p], values[p + step]);
      }
      partial[0][u][lane] = values[0];
    }
    __syncthreads();
    if (peer == 0) {
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; ++u) regs[u] = partial[0][u][lane];
      ddaLL128StoreRegs<int8_t>(dst + sliceByte, regs, elt, lane, flagLane);
    }
    __syncthreads(); // Finish the unpack before reusing partial for the next slice.
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

// Each group of N warps owns one slice across all shards, with a warp per
// peer for transfers and polling. THREADS sizes the complete groups in the
// CTA, its slice stride, and its LDS allocation together. Defaults retain
// 512 threads for 4/8/16 ranks and 1024 threads for 32 ranks.
// Reduction words are distributed across the group and summed in register trees.
//
// A slice and all of its input reads/output writes belong to the same CTA.
// This preserves in-place operation without a dependency on another local CTA.
// All groups execute the same LDS barriers, including tail groups and aborts.
// Gather warps converge at the final CTA barrier before advancing the epoch.
template <typename T, int N, int THREADS = (N == 32 ? 1024 : 512)>
__launch_bounds__(THREADS)
__global__ void ddaAllReduceTwoShotLL128PeerParallel(
    T* const* __restrict__ peerScratch, T* __restrict__ recvbuff,
    const T* __restrict__ sendbuff, size_t shardBytes, int selfRank, int nRanksRt,
    uint32_t* __restrict__ epochDev, int epochLen, size_t slicesTotal,
    size_t bankSize, DdaKernelParams params) {
  static_assert(N == 4 || N == 8 || N == 16 || N == 32, "one warp per peer, power-of-two tree");
  (void)nRanksRt;
  const int lane = threadIdx.x % kDdaLL128Warp;
  const int warp = threadIdx.x / kDdaLL128Warp;
  const int group = warp / N;
  const int peerWarp = warp % N;
  const int peer = (selfRank + peerWarp) % N;
  static_assert(THREADS >= N * kDdaLL128Warp && THREADS % (N * kDdaLL128Warp) == 0,
                "blocks contain complete peer groups");
  constexpr int groups = THREADS / (N * kDdaLL128Warp);
  const size_t firstSlice = (size_t)blockIdx.x * groups;
  const size_t sliceStride = (size_t)gridDim.x * groups;
  const bool flagLane = ddaLL128IsFlagLane(lane);
  __shared__ uint64_t partial[groups][N][kDdaLL128WordsPerThread][kDdaLL128Warp];

  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | flag32;
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(bankSize, N);
  const size_t bankWords = (size_t)(flag32 & 1u) * (bankSize / sizeof(uint64_t));
  const size_t nextWords = bankWords + (size_t)N * slotWords;
  const uint64_t* self = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);
  uint64_t* remote = reinterpret_cast<uint64_t*>(peerScratch[peer]);
  const int8_t* src = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dst = reinterpret_cast<int8_t*>(recvbuff);

  // Publish every slice owned by this CTA before waiting on inputs. Keeping
  // the three phases separate amortizes fabric latency over a large message;
  // a complete exchange per slice would serialize two round trips per pass.
  if (peerWarp != 0) {
    for (size_t base = firstSlice; base < slicesTotal; base += sliceStride) {
      const size_t s = base + group;
      if (s >= slicesTotal) continue;
      const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem = shardBytes - sliceByte;
      const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
      const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
      uint64_t regs[kDdaLL128WordsPerThread];
      ddaLL128LoadRegsBatched<int8_t>(regs, src + (size_t)peer * shardBytes + sliceByte, elt, lane, flagLane);
      ddaLL128StoreWire(remote + bankWords + (size_t)selfRank * slotWords + wireOff, regs, flag, flagLane);
    }
  }
  __syncthreads();

  bool aborted = false;
  for (size_t base = firstSlice; base < slicesTotal; base += sliceStride) {
    const size_t s = base + group;
    const bool active = s < slicesTotal;
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = active ? shardBytes - sliceByte : 0;
    const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;

    // Each group's first warp supplies self; the others poll peers in parallel.
    uint64_t acc[kDdaLL128WordsPerThread] = {};
    bool landed = true;
    if (active && peerWarp == 0) {
      ddaLL128LoadRegsBatched<int8_t>(acc, src + (size_t)selfRank * shardBytes + sliceByte, elt, lane, flagLane);
    } else if (active) {
      landed = ddaLL128PollWire(self + bankWords + (size_t)peer * slotWords + wireOff,
                                acc, flag, lane, params);
    }
#pragma unroll
    for (int u = 0; u < kDdaLL128WordsPerThread; ++u) partial[group][peerWarp][u][lane] = acc[u];
    if (__syncthreads_or(!landed)) {
      aborted = true;
      break;
    }

    // Partition output words across the group. Each word has one reducer,
    // with all peer operands loaded before its register tree is evaluated.
    // This avoids a CTA barrier and an LDS round trip at each tree level.
#pragma unroll
    for (int u = peerWarp; u < kDdaLL128WordsPerThread; u += N) {
      uint64_t values[N];
#pragma unroll
      for (int p = 0; p < N; ++p) values[p] = partial[group][p][u][lane];
#pragma unroll
      for (int step = N / 2; step > 0; step /= 2) {
#pragma unroll
        for (int p = 0; p < step; ++p) values[p] = ddaLL128AddWord<T>(values[p], values[p + step]);
      }
      partial[group][0][u][lane] = values[0];
    }
    __syncthreads();

#pragma unroll
    for (int u = 0; u < kDdaLL128WordsPerThread; ++u) acc[u] = partial[group][0][u][lane];

    // Publish before the local unpack, which changes the flag-lane registers.
    if (active && peerWarp != 0) {
      ddaLL128StoreWire(remote + nextWords + (size_t)selfRank * slotWords + wireOff, acc, flag, flagLane);
    } else if (active) {
      ddaLL128StoreRegs<int8_t>(dst + (size_t)selfRank * shardBytes + sliceByte, acc, elt, lane, flagLane);
    }
    __syncthreads(); // Every warp consumed partial[group][0] before the next slice reuses LDS.
  }

  if (!aborted) {
    for (size_t base = firstSlice; base < slicesTotal; base += sliceStride) {
      const size_t s = base + group;
      const bool active = s < slicesTotal;
      const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem = active ? shardBytes - sliceByte : 0;
      const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
      const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
      bool landed = true;
      if (active && peerWarp != 0) {
        uint64_t regs[kDdaLL128WordsPerThread];
        landed = ddaLL128PollWire(self + nextWords + (size_t)peer * slotWords + wireOff,
                                  regs, flag, lane, params);
        if (landed) {
          ddaLL128StoreRegs<int8_t>(dst + (size_t)peer * shardBytes + sliceByte, regs, elt, lane, flagLane);
        }
      }
      if (!landed) break; // No cross-warp dependencies remain in the gather phase.
    }
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

} // namespace dda::common
