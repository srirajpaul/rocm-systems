/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Two-shot tier of the LL128-protocol all-reduce for the DDA path.
 *
 * Same relation to all_reduce_dda_ll128.h as all_reduce_dda_ll_twoshot.h has to
 * all_reduce_dda_ll.h: each rank owns one shard of count/nRanks elements and only
 * ever transports that shard, so every rank moves nRanks-1 shards instead of
 * nRanks-1 whole messages. Three phases, all flag-ordered like the one-shot, so
 * still no GPU barrier:
 *
 *   1. publish: send each peer the shard that peer owns,
 *   2. reduce:  sum the copies of my own shard that arrived, publish the reduced
 *               shard back to every peer, and write it to my part of recvbuff,
 *   3. writeback: collect the peers' reduced shards into the rest of recvbuff.
 *
 * The wire format, slice geometry and pack/poll/unpack helpers are the one-shot
 * tier's, unchanged; only the unit of work (a shard rather than the whole
 * message) and the extra staging area differ.
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
// Slice/wire geometry is shared with the one-shot tier, not redefined here.
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128.h"

namespace dda::common {

// Slot geometry for the two-shot tier. A bank has to hold two staging areas
// (publish + write-back) of nRanks slots each, so a slot is half the one-shot's
// and the bank stride is unchanged. Floored to a whole number of slices for the
// same reason as the one-shot: an unrounded stride can come out an odd number of
// 8B words and misalign every slot but rank 0's.
constexpr size_t ddaLL128ArTwoShotSlotWords(size_t bankSize, int nRanks) {
  return ddaLL128SlotWords(bankSize, nRanks, 2);
}

// LL128 two-shot all-reduce kernel. One warp owns one slice of a shard at a time,
// so the launcher sizes the grid on the shard's slices, not the message's.
//
// Phase 1 (publish): send peer p the shard p owns, into p's scratch at slot
// selfRank in the first staging area, as LL128 lines carrying the epoch flag.
// Phase 2 (reduce and publish): poll my slot for each peer's copy of my shard,
// sum with my own, publish the result to every peer's second staging area, and
// unpack it into my part of recvbuff.
// Phase 3 (gather): poll for the reduced shards the peers published and unpack
// each into that peer's part of recvbuff.
//
// Flag polling provides all the cross-rank ordering, so no GPU barrier is used.
// Scratch is double buffered: bank = flag & 1, with bankWordsNext naming the
// second staging area inside that bank. Self does not round-trip through scratch;
// its contribution seeds the accumulator straight from sendbuff.
//
// A warp owns the same slice index in every shard and phase. It finishes the
// input reads for those slices before overwriting any corresponding output, so
// an in-place call never overwrites another warp's unread input.
//
// WIRE (DdaLL128Wire) picks the phase-1 and phase-2 writer; staged, it takes ddaLL128StageBytes(blockDim.x) of LDS.
template <typename T, int NRANKS_CT, int WIRE = kDdaLL128WireCoop, bool PEER_BATCHED = false>
__launch_bounds__(512)
  __global__ void ddaAllReduceTwoShotLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                           T* __restrict__ recvbuff, // local user output
                                           const T* __restrict__ sendbuff, // local user input
                                           size_t shardBytes, // per-rank shard payload; multiple of 16
                                           int selfRank, int nRanksRt,
                                           uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                           int epochLen, // number of cells in epochDev
                                           size_t slicesTotal, // slices per shard this call uses
                                           size_t bankSize, // scratch bank size (from host)
                                           DdaKernelParams params) {

  static_assert(!PEER_BATCHED || WIRE == kDdaLL128WireCoop, "peer batching uses the cooperative writer");
  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;

  // Two staging areas per bank; the second starts one full fan-out in.
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const uint64_t bankWordsNext = bankWords + (uint64_t)nRanks * (uint64_t)slotWords;

  // Slices stride by warp across the whole grid. The bound has to be the slice
  // count, not the warp count: the grid is capped at ddaFabricMaxBlocks, so a
  // large shard has more slices than warps and each warp must take several.
  const size_t gwarp = (size_t)blockIdx.x * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.x * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* selfBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);

  DdaLL128Stage stage{};
  if constexpr (ddaLL128Staged<WIRE>()) {
    extern __shared__ __attribute__((aligned(128))) uint8_t ddaLL128ArStageLds[];
    stage = ddaLL128StageInit(ddaLL128ArStageLds, warp);
  }

  // Start of the wire slice holding wireOff in my slot of the peer's staging area `area` words into its scratch.
  auto scatterSlice = [&](int peer, uint64_t area, size_t wireOff) {
    uint64_t* slice = reinterpret_cast<uint64_t*>(peerScratch[peer]) + area +
                      (uint64_t)selfRank * (uint64_t)slotWords + (wireOff - 2 * lane);
    if constexpr (PEER_BATCHED) slice = wave::uniform(slice);
    return slice;
  };

  // Phase 1: send each peer the shard that peer owns, kBatch peers per round.
  // Staged: a push returns once issued, so loading one peer ahead hides the load; a batch of 8 would spill.
  // Coop: load a batch of 8 peers, then store it.
  constexpr int kAhead = ddaLL128Staged<WIRE>() ? 1 : 0;
  constexpr int kBatch = kAhead ? 1 : kDdaLLPeerBatch;
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice =
      rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;

    // With look-ahead, the round's batch was loaded by the round before.
    uint64_t cur[kBatch][kDdaLL128WordsPerThread] = {};
    if constexpr (kAhead) {
      ddaLL128LoadRegs<int8_t>(cur[0], srcBytes + (size_t)((selfRank + 1) % nRanks) * shardBytes + sliceByte,
                               eltInSlice, lane, flagLane);
    }
#pragma unroll 1
    for (int base = 1; base < nRanks; base += kBatch) {
      uint64_t next[kBatch][kDdaLL128WordsPerThread] = {};
      // do-while: a batch of one leaves no inner loop, which keeps the staged code as tight as a plain peer loop.
      int i = 0;
#pragma unroll
      do {
        const int r = base + i;
        if ((PEER_BATCHED && NRANKS_CT == 0) || r + kAhead < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          const int src = kAhead ? (peer + 1) % nRanks : peer;
          if constexpr (PEER_BATCHED) {
            // Runtime tail entries wrap to an in-bounds shard but are not sent.
            ddaLL128LoadRegsBegin<int8_t>(next[i], srcBytes + (size_t)src * shardBytes + sliceByte,
                                          eltInSlice, lane, flagLane);
          } else {
            ddaLL128LoadRegs<int8_t>(next[i], srcBytes + (size_t)src * shardBytes + sliceByte, eltInSlice, lane,
                                     flagLane);
          }
        }
      } while (++i < kBatch);
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_sched_barrier)
      if constexpr (PEER_BATCHED) __builtin_amdgcn_sched_barrier(0);
#endif
      uint64_t (&send)[kBatch][kDdaLL128WordsPerThread] = kAhead ? cur : next;
      i = 0;
#pragma unroll
      do {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          if constexpr (PEER_BATCHED) ddaLL128LoadRegsFinish(send[i], flagLane);
          ddaLL128StageFill<WIRE>(stage, send[i], flag, flagLane, lane);
          ddaLL128StagePush<WIRE>(stage, scatterSlice(peer, bankWords, wireOff), send[i], flag, flagLane, lane);
        }
      } while (++i < kBatch);
      if constexpr (kAhead) {
#pragma unroll
        for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
          cur[0][u] = next[0][u];
        }
      }
    }
  }

  // Phase 2: fold every copy of my shard, hand the result back out, then keep it.
  // The staged kernel folds over the runtime count: an unrolled fold on top of the staging state raises register use.
  bool aborted = false;
  const int foldRanks = ddaLL128Staged<WIRE>() ? nRanksRt : nRanks;
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice =
      rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;

    // Seed with my own shard; peers are folded on top.
    uint64_t acc[kDdaLL128WordsPerThread] = {};
    if constexpr (PEER_BATCHED) {
      constexpr int Batch = NRANKS_CT && NRANKS_CT - 1 < kDdaLL128ArPollBatch<T>
                              ? NRANKS_CT - 1 : kDdaLL128ArPollBatch<T>;
      ddaLL128LoadRegsBegin<int8_t>(acc, srcBytes + (size_t)selfRank * shardBytes + sliceByte,
                                    eltInSlice, lane, flagLane);
      for (int base = 1; base < nRanks; base += Batch) {
        const uint64_t* lines[Batch];
        bool valid[Batch];
#pragma unroll
        for (int k = 0; k < Batch; ++k) {
          valid[k] = base + k < nRanks;
          const int peer = valid[k] ? (selfRank + base + k) % nRanks : selfRank;
          lines[k] = wave::uniform(selfBase + bankWords + (uint64_t)peer * slotWords +
                                   s * (size_t)kDdaLL128WireWordsPerSlice);
        }
        uint64_t vr[Batch][kDdaLL128WordsPerThread];
        if (!ddaLL128ArWaitBatch(lines, valid, lane, flag, flagLane, params, vr)) {
          aborted = true;
          break;
        }
        if (base == 1) ddaLL128LoadRegsFinish(acc, flagLane);
        // Preserve the existing self,+1,+2,... fold order for this shard.
#pragma unroll
        for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
#pragma unroll
          for (int k = 0; k < Batch; ++k) {
            const uint64_t sum = ddaLL128AddWord<T>(acc[u], vr[k][u]);
            acc[u] = valid[k] ? sum : acc[u];
          }
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_sched_barrier)
          if constexpr (is_fp8_v<T>) __builtin_amdgcn_sched_barrier(0x20);
#endif
        }
      }
    } else {
      ddaLL128LoadRegs<int8_t>(acc, srcBytes + (size_t)selfRank * shardBytes + sliceByte, eltInSlice, lane, flagLane);

      for (int r = 1; r < foldRanks; ++r) {
        const int peer = (selfRank + r) % nRanks;
        const uint64_t* gatherSlot = selfBase + bankWords + (uint64_t)peer * (uint64_t)slotWords;
        uint64_t vr[kDdaLL128WordsPerThread];
        if (!ddaLL128PollWire(gatherSlot + wireOff, vr, flag, lane, params)) {
          ddaLL128StageDrain<WIRE>();
          ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
          return;
        }
        // On a flag lane the odd words hold flags rather than payload, so the sums
        // landing there are meaningless -- ddaLL128StoreRegs re-derives those slots
        // from the even ones and never writes them out, so folding blind is cheaper
        // than predicating the loop.
#pragma unroll
        for (int u = 0; u < kDdaLL128WordsPerThread; ++u) {
          acc[u] = ddaLL128AddWord<T>(acc[u], vr[u]);
        }
      }
    }
    if (aborted) break;

    // Publish before the local unpack: it is what the peers are spinning on, and
    // ddaLL128StoreRegs rewrites acc in place when it reverses the flag-lane shuffle.
    if constexpr (ddaLL128Staged<WIRE>()) {
      // The abort return leaves lastOps in a VGPR; an EXEC-masked wait ladder rung would wait to 0, so keep it scalar.
      stage.lastOps = __builtin_amdgcn_readfirstlane(stage.lastOps);
    }
    ddaLL128StageFill<WIRE>(stage, acc, flag, flagLane, lane);
#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      ddaLL128StagePush<WIRE>(stage, scatterSlice(peer, bankWordsNext, wireOff), acc, flag, flagLane, lane);
    }

    ddaLL128StoreRegs<int8_t>(dstBytes + (size_t)selfRank * shardBytes + sliceByte, acc, eltInSlice, lane, flagLane);
  }

  // Phase 3: collect the peers' reduced shards into the rest of recvbuff.
  for (size_t s = gwarp; !aborted && s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice =
      rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;

    if constexpr (PEER_BATCHED) {
      constexpr int Batch = NRANKS_CT && NRANKS_CT - 1 < kDdaLL128ArPollBatch<T>
                              ? NRANKS_CT - 1 : kDdaLL128ArPollBatch<T>;
      for (int base = 1; base < nRanks; base += Batch) {
        const uint64_t* lines[Batch];
        bool valid[Batch];
        int peers[Batch];
#pragma unroll
        for (int k = 0; k < Batch; ++k) {
          valid[k] = base + k < nRanks;
          peers[k] = valid[k] ? (selfRank + base + k) % nRanks : selfRank;
          lines[k] = wave::uniform(selfBase + bankWordsNext + (uint64_t)peers[k] * slotWords +
                                   s * (size_t)kDdaLL128WireWordsPerSlice);
        }
        uint64_t vr[Batch][kDdaLL128WordsPerThread];
        if (!ddaLL128ArWaitBatch(lines, valid, lane, flag, flagLane, params, vr)) {
          aborted = true;
          break;
        }
#pragma unroll
        for (int k = 0; k < Batch; ++k) {
          if (valid[k]) {
            ddaLL128StoreRegs<int8_t>(dstBytes + (size_t)peers[k] * shardBytes + sliceByte,
                                      vr[k], eltInSlice, lane, flagLane);
          }
        }
      }
    } else {
      for (int r = 1; r < nRanks; ++r) {
        const int peer = (selfRank + r) % nRanks;
        const uint64_t* gatherSlot = selfBase + bankWordsNext + (uint64_t)peer * (uint64_t)slotWords;
        uint64_t vr[kDdaLL128WordsPerThread];
        if (!ddaLL128PollWire(gatherSlot + wireOff, vr, flag, lane, params)) {
          ddaLL128StageDrain<WIRE>();
          ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
          return;
        }
        ddaLL128StoreRegs<int8_t>(dstBytes + (size_t)peer * shardBytes + sliceByte, vr, eltInSlice, lane, flagLane);
      }
    }
  }

#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  ddaLL128StageDrain<WIRE>();

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

} // namespace dda::common
