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

// Eight warps can reduce all eight words of a slice in parallel. The optional
// partial-sum mode uses at most this many warps per slice, leaving two
// independent slice groups in a 512-thread CTA at higher rank counts. An
// explicit peer factor instead assigns 1/2/4/8 contributions to each warp.
__host__ __device__ constexpr int ddaLL128ArTwoShotPeerWarps(int nRanks, bool partialReduce, int peersPerWarp = 0) {
  if (peersPerWarp > 0) return nRanks >= peersPerWarp ? nRanks / peersPerWarp : 1;
  return partialReduce && nRanks > kDdaLL128WordsPerThread ? kDdaLL128WordsPerThread : nRanks;
}

// Each group of N warps owns one slice across all shards, with a warp per
// peer for transfers and polling. A 512-thread CTA has 16/N independent groups
// for 4/8/16 ranks; 32 ranks use a 1024-thread CTA with one complete group.
// Reduction words are distributed across the group and summed in register trees.
// With PARTIAL_REDUCE, groups have at most eight warps. Each warp transfers
// and accumulates peers peerWarp, peerWarp + peerWarps, ... before the group
// reduces its partials. Blocks stay at 512 threads and retain multiple slices.
// PEERS_PER_WARP explicitly selects a factor at 4/8/16 ranks. The factor
// includes the local contribution; only remote contributions are polled.
// BLOCK_THREADS permits explicit block-size experiments with complete slice
// groups. Host geometry uses the same peer-warp helper and block size.
// Grouping additions can change floating-point rounding; the shard owner
// publishes the same reduced value to every rank.
//
// A slice and all of its input reads/output writes belong to the same CTA.
// This preserves in-place operation without a dependency on another local CTA.
// All groups execute the same LDS barriers, including tail groups and aborts.
// Gather warps converge at the final CTA barrier before advancing the epoch.
template <typename T, int N, bool PARTIAL_REDUCE = false, int PEERS_PER_WARP = 0,
          int BLOCK_THREADS = (!PARTIAL_REDUCE && PEERS_PER_WARP == 0 && N == 32 ? 1024 : 512)>
__launch_bounds__(BLOCK_THREADS)
__global__ void ddaAllReduceTwoShotLL128PeerParallel(
    T* const* __restrict__ peerScratch, T* __restrict__ recvbuff,
    const T* __restrict__ sendbuff, size_t shardBytes, int selfRank, int nRanksRt,
    uint32_t* __restrict__ epochDev, int epochLen, size_t slicesTotal,
    size_t bankSize, DdaKernelParams params) {
  static_assert(N == 4 || N == 8 || N == 16 || N == 32 || (PARTIAL_REDUCE && N == 12),
                "supported ranks, power-of-two partial tree");
  static_assert(PEERS_PER_WARP == 0 ||
                    ((N == 4 || N == 8 || N == 16) &&
                     (PEERS_PER_WARP == 1 || PEERS_PER_WARP == 2 || PEERS_PER_WARP == 4 || PEERS_PER_WARP == 8)),
                "explicit peer factors require supported power-of-two ranks");
  (void)nRanksRt;
  const int lane = threadIdx.x % kDdaLL128Warp;
  const int warp = threadIdx.x / kDdaLL128Warp;
  constexpr int peerWarps = ddaLL128ArTwoShotPeerWarps(N, PARTIAL_REDUCE, PEERS_PER_WARP);
  const int group = warp / peerWarps;
  const int peerWarp = warp % peerWarps;
  static_assert(BLOCK_THREADS >= peerWarps * kDdaLL128Warp &&
                BLOCK_THREADS % (peerWarps * kDdaLL128Warp) == 0, "complete slice groups per block");
  constexpr int groups = BLOCK_THREADS / kDdaLL128Warp / peerWarps;
  const size_t firstSlice = (size_t)blockIdx.x * groups;
  const size_t sliceStride = (size_t)gridDim.x * groups;
  const bool flagLane = ddaLL128IsFlagLane(lane);
  __shared__ uint64_t partial[groups][peerWarps][kDdaLL128WordsPerThread][kDdaLL128Warp];

  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | flag32;
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(bankSize, N);
  const size_t bankWords = (size_t)(flag32 & 1u) * (bankSize / sizeof(uint64_t));
  const size_t nextWords = bankWords + (size_t)N * slotWords;
  const uint64_t* self = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);
  const int8_t* src = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dst = reinterpret_cast<int8_t*>(recvbuff);

  // Publish every slice owned by this CTA before waiting on inputs. Keeping
  // the three phases separate amortizes fabric latency over a large message;
  // a complete exchange per slice would serialize two round trips per pass.
  for (int r = peerWarp; r < N; r += peerWarps) {
    if (r == 0) continue;
    const int peer = (selfRank + r) % N;
    uint64_t* remote = reinterpret_cast<uint64_t*>(peerScratch[peer]);
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

    // Each warp forms a register partial over its assigned peers. In the
    // default specialization this loop has one iteration, as before.
    uint64_t acc[kDdaLL128WordsPerThread] = {};
    bool landed = true;
    if (active) {
      for (int r = peerWarp; r < N; r += peerWarps) {
        const int peer = (selfRank + r) % N;
        uint64_t regs[kDdaLL128WordsPerThread] = {};
        if (r == 0) {
          ddaLL128LoadRegsBatched<int8_t>(regs, src + (size_t)selfRank * shardBytes + sliceByte, elt, lane, flagLane);
        } else {
          landed = ddaLL128PollWire(self + bankWords + (size_t)peer * slotWords + wireOff,
                                    regs, flag, lane, params);
          if (!landed) break;
        }
#pragma unroll
        for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
          acc[u] = r == peerWarp ? regs[u] : ddaLL128AddWord<T>(acc[u], regs[u]);
        }
      }
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
    for (int u = peerWarp; u < kDdaLL128WordsPerThread; u += peerWarps) {
      uint64_t values[peerWarps];
#pragma unroll
      for (int p = 0; p < peerWarps; ++p) values[p] = partial[group][p][u][lane];
#pragma unroll
      for (int step = peerWarps / 2; step > 0; step /= 2) {
#pragma unroll
        for (int p = 0; p < step; ++p) values[p] = ddaLL128AddWord<T>(values[p], values[p + step]);
      }
      partial[group][0][u][lane] = values[0];
    }
    __syncthreads();

#pragma unroll
    for (int u = 0; u < kDdaLL128WordsPerThread; ++u) acc[u] = partial[group][0][u][lane];

    // Publish before the local unpack, which changes the flag-lane registers.
    if (active) {
      for (int r = peerWarp; r < N; r += peerWarps) {
        if (r == 0) continue;
        const int peer = (selfRank + r) % N;
        uint64_t* remote = reinterpret_cast<uint64_t*>(peerScratch[peer]);
        ddaLL128StoreWire(remote + nextWords + (size_t)selfRank * slotWords + wireOff, acc, flag, flagLane);
      }
      if (peerWarp == 0) {
        ddaLL128StoreRegs<int8_t>(dst + (size_t)selfRank * shardBytes + sliceByte, acc, elt, lane, flagLane);
      }
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
      if (active) {
        for (int r = peerWarp; r < N; r += peerWarps) {
          if (r == 0) continue;
          const int peer = (selfRank + r) % N;
          uint64_t regs[kDdaLL128WordsPerThread];
          landed = ddaLL128PollWire(self + nextWords + (size_t)peer * slotWords + wireOff,
                                    regs, flag, lane, params);
          if (!landed) break;
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

// Poll two slices of the same peer together. Issue both sets of cooperative
// loads before checking flags, and retain a completed slice while the other
// is still pending. Validity and completion are uniform within each warp.
__device__ __forceinline__ bool ddaLL128ArPollSlicePair(
    const uint64_t* wire0, const uint64_t* wire1,
    uint64_t (&regs)[2][kDdaLL128WordsPerThread], bool valid0, bool valid1,
    uint64_t flag, int lane, DdaKernelParams params) {
  bool done0 = !valid0;
  bool done1 = !valid1;
  const bool flagLane = ddaLL128IsFlagLane(lane);
  int spins = 0;
  while (!done0 || !done1) {
    if (params.abortSpinTick(spins)) return false;
#pragma unroll
    for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
      if (!done0) ddaLL128LoadAtomic16B(wire0 + u * kDdaLL128Warp, regs[0][u], regs[0][u + 1]);
      if (!done1) ddaLL128LoadAtomic16B(wire1 + u * kDdaLL128Warp, regs[1][u], regs[1][u + 1]);
    }
    bool need0 = false;
    bool need1 = false;
#pragma unroll
    for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
      need0 |= !done0 && flagLane && regs[0][u + 1] != flag;
      need1 |= !done1 && flagLane && regs[1][u + 1] != flag;
    }
    done0 = !__any(need0);
    done1 = !__any(need1);
  }
#if !RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
  // Match the existing polling helper's payload reload on the 16B fallback.
#pragma unroll
  for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
    if (valid0) ddaLL128LoadAtomic16B(wire0 + u * kDdaLL128Warp, regs[0][u], regs[0][u + 1]);
    if (valid1) ddaLL128LoadAtomic16B(wire1 + u * kDdaLL128Warp, regs[1][u], regs[1][u + 1]);
  }
#endif
  return true;
}

// Two consecutive iterations of the original grid-stride reduction are
// processed as a pair. Each warp keeps one peer; grid size, slice ownership,
// scatter, and gather remain the same as the 512-thread one-peer kernel.
// Both slices share the three reduction barriers. This requires two LDS
// planes and can increase register pressure; selection is explicitly opt-in.
template <typename T, int N>
__launch_bounds__(512)
__global__ void ddaAllReduceTwoShotLL128PeerParallelSlicePair(
    T* const* __restrict__ peerScratch, T* __restrict__ recvbuff,
    const T* __restrict__ sendbuff, size_t shardBytes, int selfRank, int nRanksRt,
    uint32_t* __restrict__ epochDev, int epochLen, size_t slicesTotal,
    size_t bankSize, DdaKernelParams params) {
  static_assert(N == 4 || N == 8 || N == 16, "one peer per warp, 512 threads");
  (void)nRanksRt;
  const int lane = threadIdx.x % kDdaLL128Warp;
  const int warp = threadIdx.x / kDdaLL128Warp;
  const int group = warp / N;
  const int peerWarp = warp % N;
  const int peer = (selfRank + peerWarp) % N;
  constexpr int groups = 16 / N;
  const size_t firstSlice = (size_t)blockIdx.x * groups;
  const size_t sliceStride = (size_t)gridDim.x * groups;
  const bool flagLane = ddaLL128IsFlagLane(lane);
  __shared__ uint64_t partial[2][groups][N][kDdaLL128WordsPerThread][kDdaLL128Warp];

  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | flag32;
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(bankSize, N);
  const size_t bankWords = (size_t)(flag32 & 1u) * (bankSize / sizeof(uint64_t));
  const size_t nextWords = bankWords + (size_t)N * slotWords;
  const uint64_t* self = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);
  uint64_t* remote = reinterpret_cast<uint64_t*>(peerScratch[peer]);
  const int8_t* src = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dst = reinterpret_cast<int8_t*>(recvbuff);

  // Publish all input slices before polling, with the original assignment.
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
  for (size_t base = firstSlice; base < slicesTotal; base += 2 * sliceStride) {
    bool active[2];
    size_t sliceByte[2];
    size_t wireOff[2];
    int elt[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const size_t s = base + group + k * sliceStride;
      active[k] = s < slicesTotal;
      sliceByte[k] = s * (size_t)kDdaLL128DataBytesPerSlice;
      wireOff[k] = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
      const size_t rem = active[k] ? shardBytes - sliceByte[k] : 0;
      elt[k] = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    }
    uint64_t acc[2][kDdaLL128WordsPerThread] = {};
    bool landed = true;
    if (peerWarp == 0) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        if (active[k]) ddaLL128LoadRegsBatched<int8_t>(acc[k], src + (size_t)selfRank * shardBytes + sliceByte[k],
                                                       elt[k], lane, flagLane);
      }
    } else {
      const uint64_t* wire0 = active[0] ? self + bankWords + (size_t)peer * slotWords + wireOff[0] : nullptr;
      const uint64_t* wire1 = active[1] ? self + bankWords + (size_t)peer * slotWords + wireOff[1] : nullptr;
      landed = ddaLL128ArPollSlicePair(wire0, wire1, acc, active[0], active[1], flag, lane, params);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; ++u) partial[k][group][peerWarp][u][lane] = acc[k][u];
    }
    if (__syncthreads_or(!landed)) {
      aborted = true;
      break;
    }

#pragma unroll
    for (int k = 0; k < 2; ++k) {
#pragma unroll
      for (int u = peerWarp; u < kDdaLL128WordsPerThread; u += N) {
        uint64_t values[N];
#pragma unroll
        for (int p = 0; p < N; ++p) values[p] = partial[k][group][p][u][lane];
#pragma unroll
        for (int step = N / 2; step > 0; step /= 2) {
#pragma unroll
          for (int p = 0; p < step; ++p) values[p] = ddaLL128AddWord<T>(values[p], values[p + step]);
        }
        partial[k][group][0][u][lane] = values[0];
      }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < 2; ++k) {
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; ++u) acc[k][u] = partial[k][group][0][u][lane];
      if (active[k] && peerWarp != 0) {
        ddaLL128StoreWire(remote + nextWords + (size_t)selfRank * slotWords + wireOff[k], acc[k], flag, flagLane);
      } else if (active[k]) {
        ddaLL128StoreRegs<int8_t>(dst + (size_t)selfRank * shardBytes + sliceByte[k], acc[k], elt[k], lane, flagLane);
      }
    }
    __syncthreads(); // Both LDS planes have been consumed before the next pair.
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
        landed = ddaLL128PollWire(self + nextWords + (size_t)peer * slotWords + wireOff, regs, flag, lane, params);
        if (landed) ddaLL128StoreRegs<int8_t>(dst + (size_t)peer * shardBytes + sliceByte, regs, elt, lane, flagLane);
      }
      if (!landed) break;
    }
  }
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

} // namespace dda::common
