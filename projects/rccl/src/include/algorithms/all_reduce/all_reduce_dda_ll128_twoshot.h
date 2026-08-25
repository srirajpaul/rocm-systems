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

#include "algorithms/CollCommon.h"
#include "algorithms/CollCommon_ll128.h"
// Slice/wire geometry is shared with the one-shot tier, not redefined here.
#include "algorithms/all_reduce/all_reduce_dda_ll128.h"
#include "algorithms/ll128_pack.h"

namespace meta::comms {

// All LL/LL128 all-reduce tiers share one scratch and one epoch counter, so bank
// 1 has to start at the same byte offset for every one of them. A slot here holds
// only a shard, but a bank spans two of them (publish + write-back), so the slot
// is half the one-shot's and the bank stride comes out identical.
// dda_all_reduce_fabric_ll.cu static_asserts the halving.
constexpr size_t kDdaLL128ArTwoShotSlotWords = kDdaLL128ArSlotWords / 2;

// Slices one two-shot slot can hold, and hence the largest shard this tier can
// stage. The whole message may be nRanks times this.
inline size_t ddaLL128ArTwoShotMaxSlices(int warpSize) {
  return kDdaLL128ArTwoShotSlotWords / (size_t)ddaLL128ArWireWordPerSlice(warpSize);
}

// Fixed-width peer staging for phase 1: unlike the one-shot, each peer receives a
// different shard, so the pack cannot be hoisted out of the peer loop. Loading a
// batch before storing it keeps the per-peer streams overlapped -- interleaved,
// every store waits on the next pack. 8 covers rank counts 4 and 8 in one pass;
// the rank test is a predicate on a fixed trip count so both loops unroll and the
// unused entries fold away when NRANKS_CT is known.
constexpr int kDdaLL128ArTwoShotPeerBatch = 8;

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
// A warp owns the same slice index in every phase, and phase 1 only ever reads
// peer shards while phase 2 only ever touches the self shard, so an in-place call
// (sendbuff aliasing recvbuff) never has one warp's write land on a range another
// warp still has to read.
template <typename T, int NRANKS_CT, int kWordsPerThread>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
__global__ void ddaAllReduceTwoShotLL128(
    T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
    T* __restrict__ recvbuff,              // local user output
    const T* __restrict__ sendbuff,        // local user input
    size_t shardBytes,                     // per-rank shard payload; multiple of 16
    int selfRank,
    int nRanksRt,
    uint32_t* __restrict__ epochDev,       // per-block LL epoch cells
    int epochLen,                          // number of cells in epochDev
    size_t slicesTotal,                    // slices per shard this call uses
    size_t slotWords,                      // fixed per-slot stride, from host
    int wireWordPerSlice,                  // u64 words on the wire per slice
    int dataBytesPerSlice) {               // payload bytes per slice

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % warpSize;
  const int warp = tid / warpSize;
  const int nwarps = nthreads / warpSize;
  const bool flagLane = ll128::isFlagLane(lane);

  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const uint32_t bank = flag32 & 1u;

  // Two staging areas per bank, so the bank stride is twice a slot's fan-out.
  const uint64_t bankWords =
      (uint64_t)bank * (uint64_t)nRanks * (uint64_t)slotWords * 2;
  const uint64_t bankWordsNext = bankWords + (uint64_t)nRanks * (uint64_t)slotWords;

  // Slices stride by warp across the whole grid. The bound has to be the slice
  // count, not the warp count: the grid is capped at ddaFabricMaxBlocks, so a
  // large shard has more slices than warps and each warp must take several.
  const size_t gwarp = (size_t)blockIdx.x * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.x * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* selfBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);

  // Phase 1: send each peer the shard that peer owns.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)dataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice =
        rem < (size_t)dataBytesPerSlice ? (int)rem : dataBytesPerSlice;
    const size_t wireOff = s * (size_t)wireWordPerSlice + 2 * lane;

    for (int base = 1; base < nRanks; base += kDdaLL128ArTwoShotPeerBatch) {
      uint64_t regs[kDdaLL128ArTwoShotPeerBatch][kWordsPerThread] = {};
#pragma unroll
      for (int i = 0; i < kDdaLL128ArTwoShotPeerBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          ll128::loadRegs<int8_t>(regs[i], srcBytes + (size_t)peer * shardBytes + sliceByte,
                                  eltInSlice, lane, flagLane);
        }
      }
#pragma unroll
      for (int i = 0; i < kDdaLL128ArTwoShotPeerBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) +
              bankWords + (uint64_t)selfRank * slotWords;
          ll128::storeWire(scatterSlot + wireOff, regs[i], flag, flagLane);
        }
      }
    }
  }

  // Phase 2: fold every copy of my shard, hand the result back out, then keep it.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)dataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice =
        rem < (size_t)dataBytesPerSlice ? (int)rem : dataBytesPerSlice;
    const size_t wireOff = s * (size_t)wireWordPerSlice + 2 * lane;

    // Seed with my own shard; peers are folded on top.
    uint64_t acc[kWordsPerThread] = {};
    ll128::loadRegs<int8_t>(acc, srcBytes + (size_t)selfRank * shardBytes + sliceByte,
                            eltInSlice, lane, flagLane);

    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      const uint64_t* gatherSlot = selfBase + bankWords + (uint64_t)peer * slotWords;
      uint64_t vr[kWordsPerThread];
      ll128::pollWire(gatherSlot + wireOff, vr, flag, lane);
      // On a flag lane the odd words hold flags rather than payload, so the sums
      // landing there are meaningless -- storeRegs re-derives those slots from
      // the even ones and never writes them out, so folding blind is cheaper
      // than predicating the loop.
#pragma unroll
      for (int u = 0; u < kWordsPerThread; ++u) {
        acc[u] = ddaLL128AddWord<T>(acc[u], vr[u]);
      }
    }

    // Publish before the local unpack: it is what the peers are spinning on, and
    // storeRegs rewrites acc in place when it reverses the flag-lane shuffle.
#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) +
          bankWordsNext + (uint64_t)selfRank * slotWords;
      ll128::storeWire(scatterSlot + wireOff, acc, flag, flagLane);
    }

    ll128::storeRegs<int8_t>(dstBytes + (size_t)selfRank * shardBytes + sliceByte,
                             acc, eltInSlice, lane, flagLane);
  }

  // Phase 3: collect the peers' reduced shards into the rest of recvbuff.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t sliceByte = s * (size_t)dataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice =
        rem < (size_t)dataBytesPerSlice ? (int)rem : dataBytesPerSlice;
    const size_t wireOff = s * (size_t)wireWordPerSlice + 2 * lane;

    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      const uint64_t* gatherSlot = selfBase + bankWordsNext + (uint64_t)peer * slotWords;
      uint64_t vr[kWordsPerThread];
      ll128::pollWire(gatherSlot + wireOff, vr, flag, lane);
      ll128::storeRegs<int8_t>(dstBytes + (size_t)peer * shardBytes + sliceByte,
                               vr, eltInSlice, lane, flagLane);
    }
  }

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

} // namespace meta::comms
