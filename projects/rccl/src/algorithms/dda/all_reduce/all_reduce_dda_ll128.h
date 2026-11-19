/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-reduce device kernel for the DDA path following RCCL LL128
 * protocol. All-reduce counterpart of all_gather_dda_fabric_ll128.h: same slice /
 * wire geometry and the same CollCommon_ll128.h pack-poll-unpack helpers, but
 * every block fans out to all peers instead of owning one peer column, because a
 * reduced output word depends on every rank.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/device/CollCommon_ll128.h"
#include "device/uniform.h"

namespace dda::common {

// Slot geometry, derived from the scratch bank the host picked. The bank splits
// evenly across ranks and the per-rank stride is floored to a whole number of
// slices, so a slot always begins on a slice boundary and the 16B wire accesses
// stay aligned however nRanks divides the bank
constexpr size_t ddaLL128ArSlotWords(size_t bankSize, int nRanks) {
  return ddaLL128SlotWords(bankSize, nRanks, 1);
}

// Batch independent peer reads before inspecting flags, as in LL128 ReduceScatter.
// Invalid entries must still address a complete in-bounds scratch slice. They are
// read but neither waited on nor folded. The non-cooperative fallback must reload
// payload after flags arrive, just like ddaLL128PollWire.
template <int Batch>
__device__ __forceinline__ bool ddaLL128ArWaitBatch(
    const uint64_t* const (&lines)[Batch], const bool (&valid)[Batch], int lane,
    uint64_t flag, bool flagLane, DdaKernelParams params,
    uint64_t (&vr)[Batch][kDdaLL128WordsPerThread]) {
  int spins = 0;
  while (true) {
    if (params.abortSpinTick(spins)) return false;
#pragma unroll
    for (int k = 0; k < Batch; ++k) {
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
        ddaLL128LoadAtomic16B(lines[k] + 2 * lane + u * kDdaLL128Warp, vr[k][u], vr[k][u + 1]);
      }
    }
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_sched_barrier)
    __builtin_amdgcn_sched_barrier(0);
#endif
    bool stale = false;
#pragma unroll
    for (int k = 0; k < Batch; ++k) {
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
        stale |= valid[k] && flagLane && vr[k][u + 1] != flag;
      }
    }
    if (!__any(stale)) {
#if !RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
#pragma unroll
      for (int k = 0; k < Batch; ++k) {
#pragma unroll
        for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
          ddaLL128LoadAtomic16B(lines[k] + 2 * lane + u * kDdaLL128Warp, vr[k][u], vr[k][u + 1]);
        }
      }
#endif
      return true;
    }
  }
}

// fp8 conversion and FP32 allreduce folds need more registers. Four entries
// avoid FP32 spills at eight/runtime ranks; half and bfloat16 use eight.
template <typename T>
inline constexpr int kDdaLL128ArPollBatch = (is_fp8_v<T> || sizeof(T) == 4) ? 4 : 8;

// LL128 flat all-reduce kernel. One warp owns one slice at a time; the grid is
// 1D over slices and each block fans out to every remote peer.
//
// Phase 1 (publish): pack this rank's slice into registers once, then push it
// onto the wire in every peer's scratch at slot selfRank, flag word embedded.
// Phase 2 (reduce): for each rank in rank order, poll that rank's slot in our own
// scratch and fold the arriving words into an accumulator, then unpack the
// accumulator into recvbuff. Flag polling supplies the cross-rank ordering, so
// no GPU barrier is used.
//
// Self does not round-trip through scratch; its contribution is packed straight
// from sendbuff. Scratch is double buffered: bank = flag & 1.
template <typename T, int NRANKS_CT, bool PEER_BATCHED = false>
__launch_bounds__(512)
  __global__ void ddaAllReduceFlatLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                        T* __restrict__ recvbuff, // local user output
                                        const T* __restrict__ sendbuff, // local user input
                                        size_t bytes, // full-message payload; multiple of 16
                                        int selfRank, int nRanksRt,
                                        uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                        int epochLen, // number of cells in epochDev
                                        size_t slicesTotal, // slices this call actually uses
                                        size_t bankSize, // scratch bank size (from host)
                                        DdaKernelParams params) {

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  // ddaBankSize() floors the bank to a multiple of 128B, so it is always a whole
  // number of 8B words and the slot stride below is a whole number of slices.
  const size_t slotWords = ddaLL128ArSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));

  // Slices stride by warp across the whole grid. The bound has to be the slice
  // count, not the warp count: the grid is capped at ddaFabricMaxBlocks, so a
  // large message has more slices than warps and each warp must take several.
  const size_t gwarp = (size_t)blockIdx.x * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.x * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* gatherBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

  // Phase 1: pack each slice once, then push it to every peer's slot[selfRank].
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = bytes - dataByte;
    const int eltInSlice =
      rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread] = {};
    if constexpr (PEER_BATCHED) {
      ddaLL128LoadRegsBatched<int8_t>(regs, srcBytes + dataByte, eltInSlice, lane, flagLane);
    } else {
      ddaLL128LoadRegs<int8_t>(regs, srcBytes + dataByte, eltInSlice, lane, flagLane);
    }

#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords +
        (uint64_t)selfRank * (uint64_t)slotWords + s * (size_t)kDdaLL128WireWordsPerSlice;
      if constexpr (PEER_BATCHED) scatterSlot = wave::uniform(scatterSlot);
      ddaLL128StoreWire(scatterSlot + 2 * lane, regs, flag, flagLane);
    }
  }

  bool aborted = false;
  // Phase 2: poll every peer's slot for the same slices and fold them in.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = bytes - dataByte;
    const int eltInSlice =
      rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;

    // Fold every rank's payload in rank order, so each rank rounds the same
    // partial sums and all get the same result. Our own payload is packed from
    // sendbuff into the same register layout the wire delivers.
    uint64_t acc[kDdaLL128WordsPerThread] = {};
    if constexpr (PEER_BATCHED) {
      constexpr int Batch = NRANKS_CT && NRANKS_CT < kDdaLL128ArPollBatch<T>
                              ? NRANKS_CT : kDdaLL128ArPollBatch<T>;
      uint64_t local[kDdaLL128WordsPerThread];
      ddaLL128LoadRegsBegin<int8_t>(local, srcBytes + dataByte, eltInSlice, lane, flagLane);
      for (int base = 0; base < nRanks; base += Batch) {
        const uint64_t* lines[Batch];
        bool valid[Batch];
#pragma unroll
        for (int k = 0; k < Batch; ++k) {
          const int peer = base + k < nRanks ? base + k : selfRank;
          valid[k] = base + k < nRanks && peer != selfRank;
          // The self slot is allocated but never published; it is a safe dummy
          // address for self/tail entries, whose flags are ignored.
          lines[k] = wave::uniform(gatherBase + (uint64_t)peer * slotWords +
                                   s * (size_t)kDdaLL128WireWordsPerSlice);
        }
        uint64_t vr[Batch][kDdaLL128WordsPerThread];
        if (!ddaLL128ArWaitBatch(lines, valid, lane, flag, flagLane, params, vr)) {
          aborted = true;
          break;
        }
        if (base == 0) ddaLL128LoadRegsFinish(local, flagLane);
        // All ranks must fold 0,1,...,N-1, inserting self at its absolute rank.
        // Seeding with self then cycling peers (as RS does) would round differently.
#pragma unroll
        for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
#pragma unroll
          for (int k = 0; k < Batch; ++k) {
            const int peer = base + k;
            const uint64_t value = peer == selfRank ? local[u] : vr[k][u];
            const uint64_t sum = peer == 0 ? value : ddaLL128AddWord<T>(acc[u], value);
            acc[u] = peer < nRanks ? sum : acc[u];
          }
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_sched_barrier)
          if constexpr (is_fp8_v<T>) __builtin_amdgcn_sched_barrier(0x20);
#endif
        }
      }
    } else {
      for (int peer = 0; peer < nRanks; ++peer) {
        uint64_t vr[kDdaLL128WordsPerThread] = {};
        if (peer == selfRank) {
          ddaLL128LoadRegs<int8_t>(vr, srcBytes + dataByte, eltInSlice, lane, flagLane);
        } else {
          const uint64_t* gatherSlot = gatherBase + (uint64_t)peer * (uint64_t)slotWords;
          if (!ddaLL128PollWire(gatherSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, vr, flag, lane,
                                params)) {
            ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
            return;
          }
        }
        // On a flag lane the odd words hold flags rather than payload, so the sums
        // landing there are meaningless -- ddaLL128StoreRegs re-derives those slots
        // from the even ones and never writes them out, so folding blind is cheaper
        // than predicating the loop.
#pragma unroll
        for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
          acc[u] = peer == 0 ? vr[u] : ddaLL128AddWord<T>(acc[u], vr[u]);
        }
      }
    }
    if (aborted) break;

    ddaLL128StoreRegs<int8_t>(dstBytes + dataByte, acc, eltInSlice, lane, flagLane);
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

} // namespace dda::common
