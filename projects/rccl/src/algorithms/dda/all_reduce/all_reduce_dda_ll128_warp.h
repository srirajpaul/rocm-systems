/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_twoshot.h"

namespace dda::common {

constexpr int kDdaLL128WarpPeers = 4;
constexpr int kDdaLL128WarpThreads = 512;

template <typename T, int NRANKS_CT, int VERSION, bool SKIP_REDUCE>
__device__ __forceinline__ void ddaAllReduceTwoShotLL128Warp(
    T* const* __restrict__ peerScratch, T* recvbuff, const T* sendbuff, size_t shardBytes,
    int selfRank, int nRanksRt, uint32_t* epochDev, int epochLen,
    size_t slicesTotal, size_t bankSize, DdaKernelParams params) {
  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int groups = (nRanks + kDdaLL128WarpPeers - 1) / kDdaLL128WarpPeers;
  const int lane = threadIdx.x % kDdaLL128Warp;
  const int warp = threadIdx.x / kDdaLL128Warp;
  const int teams = (blockDim.x / kDdaLL128Warp) / groups;
  const int team = warp / groups;
  const int group = warp % groups;
  const int first = group * kDdaLL128WarpPeers;
  const bool flagLane = ddaLL128IsFlagLane(lane);
  const size_t start = (size_t)blockIdx.x * teams;
  const size_t stride = (size_t)gridDim.x * teams;
  const int ldsSlots = VERSION == 1 ? teams : teams * groups;

  extern __shared__ uint64_t ddaLL128WarpLds[];
  __shared__ int blockAbort;
  if (threadIdx.x == 0) blockAbort = 0;
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | flag32;
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(bankSize, nRanks);
  const size_t bank = (size_t)(flag32 & 1u) * (bankSize / sizeof(uint64_t));
  const size_t nextBank = bank + (size_t)nRanks * slotWords;
  const uint64_t* local = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);
  const int8_t* input = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* output = reinterpret_cast<int8_t*>(recvbuff);

  // Each team owns a slice; each warp owns up to four peers.
  for (size_t base = start; base < slicesTotal; base += stride) {
    const size_t s = base + team;
    if (team >= teams || s >= slicesTotal) continue;
    const size_t byte = s * kDdaLL128DataBytesPerSlice;
    const int bytes = (int)min(shardBytes - byte, (size_t)kDdaLL128DataBytesPerSlice);
    const size_t off = s * kDdaLL128WireWordsPerSlice + 2 * lane;
    uint64_t data[kDdaLL128WarpPeers][kDdaLL128WordsPerThread] = {};
#pragma unroll
    for (int i = 0; i < kDdaLL128WarpPeers; ++i) {
      const int r = first + i;
      const int peer = (selfRank + 1 + r) % nRanks;
      if (r < nRanks && peer != selfRank)
        ddaLL128LoadRegs<int8_t>(data[i], input + (size_t)peer * shardBytes + byte, bytes, lane, flagLane);
    }
#pragma unroll
    for (int i = 0; i < kDdaLL128WarpPeers; ++i) {
      const int r = first + i;
      const int peer = (selfRank + 1 + r) % nRanks;
      if (r < nRanks && peer != selfRank) {
        uint64_t* dst = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bank + (size_t)selfRank * slotWords + off;
        ddaLL128StoreWire(dst, data[i], flag, flagLane);
      }
    }
  }

  int iteration = 0;
  for (size_t base = start; base < slicesTotal; base += stride, ++iteration) {
    const size_t s = base + team;
    const bool active = team < teams && s < slicesTotal;
    const size_t byte = s * kDdaLL128DataBytesPerSlice;
    const int bytes = active ? (int)min(shardBytes - byte, (size_t)kDdaLL128DataBytesPerSlice) : 0;
    const size_t off = s * kDdaLL128WireWordsPerSlice + 2 * lane;
    uint64_t sum[kDdaLL128WordsPerThread] = {};

    if constexpr (SKIP_REDUCE) {
      if (active)
        ddaLL128LoadRegs<int8_t>(sum, input + (size_t)selfRank * shardBytes + byte, bytes, lane, flagLane);
    } else {
      if (active && (VERSION == 2 || group == 0)) {
        const bool serial = VERSION == 1 || groups == 1;
        const int begin = serial ? 0 : first;
        const int end = serial ? nRanks : min(first + kDdaLL128WarpPeers, nRanks);
        for (int r = begin; r < end; ++r) {
          const int peer = (selfRank + r + (serial ? 0 : 1)) % nRanks;
          uint64_t data[kDdaLL128WordsPerThread] = {};
          if (peer == selfRank) {
            ddaLL128LoadRegs<int8_t>(data, input + (size_t)selfRank * shardBytes + byte, bytes, lane, flagLane);
          } else if (!ddaLL128PollWire(local + bank + (size_t)peer * slotWords + off, data, flag, lane, params)) {
            atomicExch(&blockAbort, 1);
            break;
          }
#pragma unroll
          for (int u = 0; u < kDdaLL128WordsPerThread; ++u)
            sum[u] = r == begin ? data[u] : ddaLL128AddWord<T>(sum[u], data[u]);
        }
      }

      if (groups > 1) {
        const size_t window = (size_t)(iteration & 1) * ldsSlots * kDdaLL128WireWordsPerSlice;
        const int slot = VERSION == 1 ? team : warp;
        if (active && (VERSION == 2 || group == 0)) {
#pragma unroll
          for (int u = 0; u < kDdaLL128WordsPerThread; ++u)
            ddaLL128WarpLds[window + (size_t)slot * kDdaLL128WireWordsPerSlice + u * kDdaLL128Warp + lane] = sum[u];
        }
        // Alternating windows protect reuse with one barrier per iteration, including inactive teams.
        __syncthreads();
        if (active) {
#pragma unroll
          for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
            const size_t index = window + (size_t)(VERSION == 1 ? team : team * groups) * kDdaLL128WireWordsPerSlice
                                 + u * kDdaLL128Warp + lane;
            sum[u] = ddaLL128WarpLds[index];
            if constexpr (VERSION == 2) {
              for (int g = 1; g < groups; ++g)
                sum[u] = ddaLL128AddWord<T>(sum[u], ddaLL128WarpLds[index + (size_t)g * kDdaLL128WireWordsPerSlice]);
            }
          }
        }
      }
    }
    if (__atomic_load_n(&blockAbort, __ATOMIC_RELAXED) || !active) continue;
#pragma unroll
    for (int i = 0; i < kDdaLL128WarpPeers; ++i) {
      const int r = first + i;
      const int peer = (selfRank + 1 + r) % nRanks;
      if (r < nRanks && peer != selfRank) {
        uint64_t* dst = reinterpret_cast<uint64_t*>(peerScratch[peer]) + nextBank + (size_t)selfRank * slotWords + off;
        ddaLL128StoreWire(dst, sum, flag, flagLane);
      }
    }
    if (group == 0)
      ddaLL128StoreRegs<int8_t>(output + (size_t)selfRank * shardBytes + byte, sum, bytes, lane, flagLane);
  }

  if (__atomic_load_n(&blockAbort, __ATOMIC_RELAXED)) {
    ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
    return;
  }

  // Gather the same peer shards this warp read in phase 1, preserving in-place ordering.
  for (size_t base = start; base < slicesTotal; base += stride) {
    const size_t s = base + team;
    if (team >= teams || s >= slicesTotal) continue;
    const size_t byte = s * kDdaLL128DataBytesPerSlice;
    const int bytes = (int)min(shardBytes - byte, (size_t)kDdaLL128DataBytesPerSlice);
    const size_t off = s * kDdaLL128WireWordsPerSlice + 2 * lane;
    for (int i = 0; i < kDdaLL128WarpPeers; ++i) {
      const int r = first + i;
      const int peer = (selfRank + 1 + r) % nRanks;
      if (r >= nRanks || peer == selfRank) continue;
      uint64_t data[kDdaLL128WordsPerThread];
      if (!ddaLL128PollWire(local + nextBank + (size_t)peer * slotWords + off, data, flag, lane, params)) {
        ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
        return;
      }
      ddaLL128StoreRegs<int8_t>(output + (size_t)peer * shardBytes + byte, data, bytes, lane, flagLane);
    }
  }
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

template <typename T, int NRANKS_CT, bool SKIP_REDUCE = false>
__launch_bounds__(kDdaLL128WarpThreads)
__global__ void ddaAllReduceTwoShotLL128WarpV1(
    T* const* __restrict__ peerScratch, T* recvbuff, const T* sendbuff, size_t shardBytes,
    int selfRank, int nRanksRt, uint32_t* epochDev, int epochLen,
    size_t slicesTotal, size_t bankSize, DdaKernelParams params) {
  ddaAllReduceTwoShotLL128Warp<T, NRANKS_CT, 1, SKIP_REDUCE>(peerScratch, recvbuff, sendbuff, shardBytes,
      selfRank, nRanksRt, epochDev, epochLen, slicesTotal, bankSize, params);
}

template <typename T, int NRANKS_CT, bool SKIP_REDUCE = false>
__launch_bounds__(kDdaLL128WarpThreads)
__global__ void ddaAllReduceTwoShotLL128WarpV2(
    T* const* __restrict__ peerScratch, T* recvbuff, const T* sendbuff, size_t shardBytes,
    int selfRank, int nRanksRt, uint32_t* epochDev, int epochLen,
    size_t slicesTotal, size_t bankSize, DdaKernelParams params) {
  ddaAllReduceTwoShotLL128Warp<T, NRANKS_CT, 2, SKIP_REDUCE>(peerScratch, recvbuff, sendbuff, shardBytes,
      selfRank, nRanksRt, epochDev, epochLen, slicesTotal, bankSize, params);
}

} // namespace dda::common
