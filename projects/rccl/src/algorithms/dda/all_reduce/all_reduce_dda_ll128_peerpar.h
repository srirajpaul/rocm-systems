/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Peer-parallel LL128 one-shot and two-shot all-reduce kernels for the DDA
 * fabric path.
 *
 * The flat tiers (all_reduce_dda_ll128.h, all_reduce_dda_ll128_twoshot.h) give
 * a slice to one warp, which then walks every peer in turn: nRanks - 1 stores
 * and, worse, nRanks - 1 flag polls back to back, each a full fabric round trip.
 * Their per-slice latency therefore grows linearly with nRanks, while all-gather
 * and all-to-all, which give each peer its own warps, stay flat.
 *
 * Here a slice gets a group of groupWarps = ddaLL128ArPeerWarps() warps instead
 * and the peer loops are split across the group: warp groupWarp sends to, and
 * polls, the ranks congruent to groupWarp mod groupWarps, so a group pays one poll
 * round trip per groupWarps ranks. The polled copies meet in LDS and are folded
 * there, still in rank order 0..nRanks-1, so every rank rounds identically.
 *
 * A group carries a unit of SLICES consecutive slices at once: each warp polls
 * its ranks' copies of all of them in one wait and the group folds each in turn,
 * so a round trip and a barrier pair cover SLICES slices.
 *
 * Wire format, slot layout, epoch/bank handling and the pack/poll/unpack helpers
 * are the flat tiers', unchanged, so these kernels are drop-in replacements.
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
// Slot geometry is shared with the flat tiers, not redefined here.
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_twoshot.h"

namespace dda::common {

// Widest block the peer-parallel kernels take; it bounds the fold's unrolled round.
constexpr int kDdaLL128ArPeerMaxThreads = 512;
constexpr int kDdaLL128ArPeerMaxWarps = kDdaLL128ArPeerMaxThreads / kDdaLL128Warp;
// One 16B wire pair (lo, hi); 16B-aligned so LDS accesses are 16B wide.
struct alignas(16) DdaLL128Pair {
  uint64_t lo;
  uint64_t hi;
};
// 16B pairs of one slice.
constexpr int kDdaLL128PairsPerSlice = kDdaLL128WireWordsPerSlice / 2;
// Pairs of a slice one thread folds. Blocks come in 64-thread granules and nRanks >= 2, so a group is at least two
// warps.
constexpr int kDdaLL128ArPeerFoldPairs = kDdaLL128PairsPerSlice / (2 * kDdaLL128Warp);

// Dynamic LDS of a peer-parallel block of `threads` carrying `slices` slices per group: one wire slice per warp and
// slice, for the copies it parks each round.
__host__ __device__ constexpr size_t ddaLL128ArPeerLdsBytes(unsigned threads, int slices = 1) {
  return (size_t)(threads / kDdaLL128Warp) * (size_t)slices * kDdaLL128PairsPerSlice * sizeof(DdaLL128Pair);
}

// Payload bytes of slice `i` of a unit that starts `bytesLeft` bytes before the end of the payload.
__device__ __forceinline__ int ddaLL128ArPeerEltIn(size_t bytesLeft, int i) {
  const size_t at = (size_t)i * kDdaLL128DataBytesPerSlice;
  if (at >= bytesLeft) {
    return 0;
  }
  return bytesLeft - at < (size_t)kDdaLL128DataBytesPerSlice ? (int)(bytesLeft - at) : kDdaLL128DataBytesPerSlice;
}

// Slices of a unit that start `first` slices into `slicesTotal`.
__device__ __forceinline__ int ddaLL128ArPeerLiveSlices(size_t first, size_t slicesTotal, int slices) {
  if (first >= slicesTotal) {
    return 0;
  }
  return slicesTotal - first < (size_t)slices ? (int)(slicesTotal - first) : slices;
}

// The block's dynamic LDS as wire slices, SLICES per warp; the launcher passes ddaLL128ArPeerLdsBytes.
__device__ __forceinline__ DdaLL128Pair (*ddaLL128ArPeerLds())[kDdaLL128PairsPerSlice] {
  extern __shared__ DdaLL128Pair ddaLL128ArPeerLdsSlices[];
  return reinterpret_cast<DdaLL128Pair (*)[kDdaLL128PairsPerSlice]>(ddaLL128ArPeerLdsSlices);
}

// Warps per slice group: one per rank, up to the whole block.
__host__ __device__ constexpr int ddaLL128ArPeerWarps(int nRanks, int nwarps) {
  return nRanks < nwarps ? nRanks : nwarps;
}

// This thread's warp. When a warp is one wave its index is wave-uniform, so readfirstlane keeps the scratch
// addresses derived from it scalar; on a wider wave two warps share it, and the index stays per lane.
__device__ __forceinline__ int ddaLL128ArWarp() {
  const int warp = threadIdx.x / kDdaLL128Warp;
  return __builtin_amdgcn_wavefrontsize() == kDdaLL128Warp ? __builtin_amdgcn_readfirstlane(warp) : warp;
}

// Packed add of two 16B pairs, word by word.
template <typename T>
__device__ __forceinline__ DdaLL128Pair ddaLL128ArAddPair(const DdaLL128Pair& a, const DdaLL128Pair& b) {
  return {ddaLL128AddWord<T>(a.lo, b.lo), ddaLL128AddWord<T>(a.hi, b.hi)};
}

// Poll one slice. With cooperative loads, all of the slice's loads issue before any flag is checked and the payload
// lands with its flag, so there is no re-read; without them the plain poll re-reads once the flags are current.
__device__ __forceinline__ bool ddaLL128ArPeerPoll(const uint64_t* wire, uint64_t (&vr)[kDdaLL128WordsPerThread],
                                                   uint64_t flag, int lane, DdaKernelParams params) {
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
  return ddaLL128PollWireBatched(wire, vr, flag, lane, params);
#else
  return ddaLL128PollWire(wire, vr, flag, lane, params);
#endif
}

// Poll the first `n` (wave-uniform) of SLICES wires together: every load issues before any flag is checked, so their
// round trips overlap. Without cooperative loads the payload is read again once the flags are current.
template <int SLICES>
__device__ __forceinline__ bool ddaLL128ArPeerPollN(const uint64_t* const (&wire)[SLICES], int n,
                                                    uint64_t (&vr)[SLICES][kDdaLL128WordsPerThread], uint64_t flag,
                                                    int lane, DdaKernelParams params) {
  const bool flagLane = ddaLL128IsFlagLane(lane);
  auto loadAll = [&]() {
#pragma unroll
    for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
#pragma unroll
      for (int i = 0; i < SLICES; ++i) {
        if (i < n) {
          ddaLL128LoadAtomic16B(wire[i] + u * kDdaLL128Warp, vr[i][u], vr[i][u + 1]);
        }
      }
    }
  };
  int spins = 0;
  bool needReload;
  do {
    if (params.abortSpinTick(spins)) {
      return false;
    }
    needReload = false;
    loadAll();
#pragma unroll
    for (int i = 0; i < SLICES; ++i) {
#pragma unroll
      for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
        needReload |= i < n && flagLane && vr[i][u + 1] != flag;
      }
    }
  } while (__any(needReload));
#if !RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
  loadAll();
#endif
  return true;
}

// Rank-order fold of a unit of liveSlices (at most SLICES) consecutive slices
// across a warp group, through LDS. In rounds of groupWarps ranks, warp groupWarp
// parks rank base + groupWarp's copies of the unit's slices (its own from `self`,
// a peer's polled together from that peer's slot under `gather`) in its LDS
// slices, lds[warp * SLICES + i] for slice i, and after a barrier the group's
// threads fold their pairs of each slice's parked copies in rank order, so each
// rank rounds the same partial sums. Slice i's sum lands in the store warp's LDS
// slice i, lds[(warp - groupWarp + (nRanks - 1) % groupWarps) * SLICES + i], until
// that warp parks again.
//
// The barriers are block-wide, so every warp of the block calls this the same
// number of times; liveSlices is 0 for a warp with no slice. Only LDS crosses
// them, so they are ddaLdsBarrier: on gfx1250 they do not wait for the fabric
// stores in flight. Returns false, on every warp alike, if a poll saw the abort
// flag.
template <typename T, int NRANKS_CT, int SLICES>
__device__ __forceinline__ bool ddaLL128ArPeerFold(DdaLL128Pair (*lds)[kDdaLL128PairsPerSlice], int& aborted,
                                                   int liveSlices, int nRanks, int selfRank, int groupWarps, int warp,
                                                   const int8_t* self, size_t bytesLeft, const uint64_t* gather,
                                                   size_t slotWords, uint64_t flag, DdaKernelParams params) {
  const int lane = threadIdx.x % kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);
  const int group = warp / groupWarps; // group index
  const int groupWarp = warp % groupWarps; // warp's index in group
  const int storeWarp = (nRanks - 1) % groupWarps;
  const int groupThread = groupWarp * kDdaLL128Warp + lane; // thread's index in group
  const int groupThreads = groupWarps * kDdaLL128Warp; // threads per group

  DdaLL128Pair acc[SLICES][kDdaLL128ArPeerFoldPairs] = {};
  for (int base = 0; base < nRanks; base += groupWarps) {
    const int rank = base + groupWarp;
    if (liveSlices > 0 && rank < nRanks) {
      uint64_t vr[SLICES][kDdaLL128WordsPerThread] = {};
      if (rank == selfRank) {
#pragma unroll
        for (int i = 0; i < SLICES; ++i) {
          if (i < liveSlices) {
            ddaLL128LoadRegsBatched<int8_t>(vr[i], self + (size_t)i * kDdaLL128DataBytesPerSlice,
                                            ddaLL128ArPeerEltIn(bytesLeft, i), lane, flagLane);
          }
        }
      } else {
        const uint64_t* wires[SLICES];
#pragma unroll
        for (int i = 0; i < SLICES; ++i) {
          wires[i] = gather + (uint64_t)rank * (uint64_t)slotWords + (size_t)i * kDdaLL128WireWordsPerSlice + 2 * lane;
        }
        if (!ddaLL128ArPeerPollN<SLICES>(wires, liveSlices, vr, flag, lane, params)) {
          aborted = 1;
        }
      }
#pragma unroll
      for (int i = 0; i < SLICES; ++i) {
        if (i < liveSlices) {
#pragma unroll
          for (int u = 0; u < kDdaLL128Pairs; ++u) {
            lds[warp * SLICES + i][u * kDdaLL128Warp + lane] = {vr[i][2 * u], vr[i][2 * u + 1]};
          }
        }
      }
    }
    ddaLdsBarrier();
    if (aborted) {
      return false;
    }

    // On a flag lane the copies hold a flag in each pair's hi word, not payload; that
    // sum is never stored, since ddaLL128StoreRegs re-derives those slots.
    if (liveSlices > 0) {
      const int nIn = nRanks - base < groupWarps ? nRanks - base : groupWarps;
      constexpr int kRoundMax =
        NRANKS_CT != 0 && NRANKS_CT < kDdaLL128ArPeerMaxWarps ? NRANKS_CT : kDdaLL128ArPeerMaxWarps;
#pragma unroll
      for (int i = 0; i < SLICES; ++i) {
        if (i < liveSlices) {
#pragma unroll
          for (int k = 0; k < kDdaLL128ArPeerFoldPairs; ++k) {
            const int pairPos = groupThread + k * groupThreads; // 16B pair of the slice this thread folds
            if (pairPos < kDdaLL128PairsPerSlice) {
#pragma unroll
              for (int j = 0; j < kRoundMax; ++j) {
                if (j < nIn) {
                  const DdaLL128Pair v = lds[(group * groupWarps + j) * SLICES + i][pairPos];
                  acc[i][k] = (base == 0 && j == 0) ? v : ddaLL128ArAddPair<T>(acc[i][k], v);
                }
              }
              // Only this thread reads pairPos, and the store warp parks nothing more until it has read the sum.
              if (base + groupWarps >= nRanks) {
                lds[(group * groupWarps + storeWarp) * SLICES + i][pairPos] = acc[i][k];
              }
            }
          }
        }
      }
    }
    // Ends the round: the next one overwrites lds, and the caller reads the sum.
    ddaLdsBarrier();
  }
  return true;
}

// LL128 peer-parallel one-shot all-reduce kernel. Same contract and scratch
// layout as ddaAllReduceFlatLL128, but a group of groupWarps warps owns a unit of
// SLICES consecutive slices at a time; the grid is 1D over unit groups.
//
// Phase 1 (publish): pack each slice of the unit into registers once; warp
// groupWarp of the group pushes it to peers selfRank + 1 + groupWarp, selfRank +
// 1 + groupWarp + groupWarps, ...
// Phase 2 (reduce): in rounds of groupWarps ranks, warp groupWarp parks rank
// base + groupWarp's copies of the unit's slices in its LDS slices (its own from
// sendbuff, a peer's by polling our scratch), and after a barrier the group's
// threads fold their pairs of the parked copies in rank order. The sums land in
// the store warp's LDS slices, which that warp unpacks into recvbuff.
//
// The barriers are block-wide, so every warp runs the same number of phase-2
// iterations. A unit belongs to one group in both phases, and only that group
// writes it, after all of its reads, so an in-place call is safe.
template <typename T, int NRANKS_CT, int SLICES = 1>
__launch_bounds__(kDdaLL128ArPeerMaxThreads)
  __global__ void ddaAllReduceFlatLL128PeerPar(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: scratch bases
                                               T* __restrict__ recvbuff, // local user output
                                               const T* __restrict__ sendbuff, // local user input
                                               size_t bytes, // full-message payload; multiple of 16
                                               int selfRank, int nRanksRt,
                                               uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                               int epochLen, // number of cells in epochDev
                                               size_t slicesTotal, // slices this call actually uses
                                               size_t bankSize, // scratch bank size (from host)
                                               DdaKernelParams params) {
  // SLICES wire slices per warp: the copies it parked this round.
  DdaLL128Pair (*lds)[kDdaLL128PairsPerSlice] = ddaLL128ArPeerLds();
  __shared__ int aborted;

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int tid = threadIdx.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = ddaLL128ArWarp();
  const int nwarps = blockDim.x / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  // Warp groupWarp of group `group` handles ranks congruent to groupWarp mod groupWarps. Leftover warps, when
  // groupWarps does not divide the block, only join the barriers.
  const int groupWarps = ddaLL128ArPeerWarps(nRanks, nwarps); // warps per group (ranks per round)
  const int groups = nwarps / groupWarps; // number of groups
  const int group = warp / groupWarps; // group index
  const int groupWarp = warp % groupWarps; // warp's index in group
  const bool inGroup = group < groups;
  // The warp that unpacks the group's sum: the one with the fewest peers to publish to, so its recvbuff stores do
  // not queue behind phase 1's.
  const int storeWarp = (nRanks - 1) % groupWarps;

  if (tid == 0) {
    aborted = 0;
  }
  // Its barrier also publishes `aborted`.
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128ArSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const size_t units = (slicesTotal + SLICES - 1) / SLICES;
  const size_t unitGroups = (units + groups - 1) / groups;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* gatherBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

  // Phase 1: pack each slice of the unit once, then push it to this warp's share of the peers' slot[selfRank].
  for (size_t g = blockIdx.x; g < unitGroups; g += gridDim.x) {
    const size_t first = (g * (size_t)groups + (size_t)group) * SLICES; // unit's first slice
    if (!inGroup || first >= slicesTotal || 1 + groupWarp >= nRanks) {
      break;
    }
#pragma unroll
    for (int i = 0; i < SLICES; ++i) {
      const size_t s = first + (size_t)i;
      if (s >= slicesTotal) {
        break;
      }
      const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
      uint64_t regs[kDdaLL128WordsPerThread] = {};
      ddaLL128LoadRegsBatched<int8_t>(regs, srcBytes + dataByte, ddaLL128ArPeerEltIn(bytes - dataByte, 0), lane,
                                      flagLane);
      for (int r = 1 + groupWarp; r < nRanks; r += groupWarps) {
        const int peer = (selfRank + r) % nRanks;
        uint64_t* scatterSlot =
          reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords + (uint64_t)selfRank * (uint64_t)slotWords;
        ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, regs, flag, flagLane);
      }
    }
  }

  // Phase 2: gather every rank's copies of the unit's slices groupWarps at a time and fold them in.
  for (size_t g = blockIdx.x; g < unitGroups; g += gridDim.x) {
    const size_t first = (g * (size_t)groups + (size_t)group) * SLICES; // unit's first slice
    const int liveSlices = inGroup ? ddaLL128ArPeerLiveSlices(first, slicesTotal, SLICES) : 0;
    const size_t dataByte = first * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t bytesLeft = liveSlices > 0 ? bytes - dataByte : 0;

    if (!ddaLL128ArPeerFold<T, NRANKS_CT, SLICES>(lds, aborted, liveSlices, nRanks, selfRank, groupWarps, warp,
                                                  srcBytes + dataByte, bytesLeft,
                                                  gatherBase + first * (size_t)kDdaLL128WireWordsPerSlice, slotWords,
                                                  flag, params)) {
      ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
      return;
    }

    if (groupWarp == storeWarp) {
#pragma unroll
      for (int i = 0; i < SLICES; ++i) {
        if (i < liveSlices) {
          uint64_t out[kDdaLL128WordsPerThread];
#pragma unroll
          for (int u = 0; u < kDdaLL128Pairs; ++u) {
            const DdaLL128Pair v = lds[warp * SLICES + i][u * kDdaLL128Warp + lane];
            out[2 * u] = v.lo;
            out[2 * u + 1] = v.hi;
          }
          ddaLL128StoreRegs<int8_t>(dstBytes + dataByte + (size_t)i * kDdaLL128DataBytesPerSlice, out,
                                    ddaLL128ArPeerEltIn(bytesLeft, i), lane, flagLane);
        }
      }
    }
  }

  ddaLL128WaitStores();

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

// LL128 peer-parallel two-shot all-reduce kernel. Same contract and scratch
// layout as ddaAllReduceTwoShotLL128 (cooperative writer).
//
// Phase 1 (publish): one warp per (slice, peer) item across the whole grid sends
// the peer that slice of the shard it owns.
// Phase 2 (reduce and publish): a group of groupWarps warps per unit of SLICES
// slices folds every rank's copy of my shard's slices through LDS in rank order
// (ddaLL128ArPeerFold); each warp hands the sums to its share of the peers, and
// the store warp unpacks them into my part of recvbuff.
// Phase 3 (gather): the phase-1 items again, now polling the peer's reduced
// shard and unpacking it.
//
// In-place safety: a (slice, peer) item maps to the same warp in phases 1 and 3,
// so sendbuff shard `peer` is read before recvbuff shard `peer` is written, in
// program order; my own shard is read before phase 2's fold barrier and written
// after it.
template <typename T, int NRANKS_CT, int SLICES = 1>
__launch_bounds__(kDdaLL128ArPeerMaxThreads)
  __global__ void ddaAllReduceTwoShotLL128PeerPar(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: scratch bases
                                                  T* __restrict__ recvbuff, // local user output
                                                  const T* __restrict__ sendbuff, // local user input
                                                  size_t shardBytes, // per-rank shard payload; multiple of 16
                                                  int selfRank, int nRanksRt,
                                                  uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                                  int epochLen, // number of cells in epochDev
                                                  size_t slicesTotal, // slices per shard this call uses
                                                  size_t bankSize, // scratch bank size (from host)
                                                  DdaKernelParams params) {
  // SLICES wire slices per warp: the copies it parked this round.
  DdaLL128Pair (*lds)[kDdaLL128PairsPerSlice] = ddaLL128ArPeerLds();
  __shared__ int aborted;

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int tid = threadIdx.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = ddaLL128ArWarp();
  const int nwarps = blockDim.x / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  // Phase 2 layout, as in the one-shot: warp groupWarp of group `group` handles ranks congruent to groupWarp mod
  // groupWarps.
  const int groupWarps = ddaLL128ArPeerWarps(nRanks, nwarps); // warps per group (ranks per round)
  const int groups = nwarps / groupWarps; // number of groups
  const int group = warp / groupWarps; // group index
  const int groupWarp = warp % groupWarps; // warp's index in group
  const bool inGroup = group < groups;
  const int storeWarp = (nRanks - 1) % groupWarps;

  if (tid == 0) {
    aborted = 0;
  }
  // Its barrier also publishes `aborted`.
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  // Two staging areas per bank; the second starts one full fan-out in.
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const uint64_t bankWordsNext = bankWords + (uint64_t)nRanks * (uint64_t)slotWords;
  const size_t units = (slicesTotal + SLICES - 1) / SLICES;
  const size_t unitGroups = (units + groups - 1) / groups;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* selfBase = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]);

  // Phases 1 and 3 need no reduction, so they skip the groups and spread their (slice, peer) items over every warp:
  // item i is slice i / (nRanks - 1) for peer selfRank + 1 + i % (nRanks - 1).
  const size_t items = slicesTotal * (size_t)(nRanks - 1);
  const size_t gwarp = (size_t)blockIdx.x * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.x * (size_t)nwarps;

  // Phase 1: send each item's peer the slice of the shard that peer owns.
  for (size_t i = gwarp; i < items; i += wstride) {
    const size_t s = i / (size_t)(nRanks - 1);
    const int peer = (selfRank + 1 + (int)(i - s * (size_t)(nRanks - 1))) % nRanks;
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread] = {};
    ddaLL128LoadRegsBatched<int8_t>(regs, srcBytes + (size_t)peer * shardBytes + sliceByte, eltInSlice, lane,
                                    flagLane);
    uint64_t* scatterSlot =
      reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords + (uint64_t)selfRank * (uint64_t)slotWords;
    ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, regs, flag, flagLane);
  }

  // Phase 2: fold every copy of my shard's unit of slices, hand the sums back out, then keep them.
  for (size_t g = blockIdx.x; g < unitGroups; g += gridDim.x) {
    const size_t first = (g * (size_t)groups + (size_t)group) * SLICES; // unit's first slice
    const int liveSlices = inGroup ? ddaLL128ArPeerLiveSlices(first, slicesTotal, SLICES) : 0;
    const size_t sliceByte = first * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t bytesLeft = liveSlices > 0 ? shardBytes - sliceByte : 0;

    if (!ddaLL128ArPeerFold<T, NRANKS_CT, SLICES>(lds, aborted, liveSlices, nRanks, selfRank, groupWarps, warp,
                                                  srcBytes + (size_t)selfRank * shardBytes + sliceByte, bytesLeft,
                                                  selfBase + bankWords + first * (size_t)kDdaLL128WireWordsPerSlice,
                                                  slotWords, flag, params)) {
      ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
      return;
    }
    uint64_t acc[SLICES][kDdaLL128WordsPerThread];
#pragma unroll
    for (int i = 0; i < SLICES; ++i) {
      if (i < liveSlices) {
#pragma unroll
        for (int u = 0; u < kDdaLL128Pairs; ++u) {
          const DdaLL128Pair v = lds[(group * groupWarps + storeWarp) * SLICES + i][u * kDdaLL128Warp + lane];
          acc[i][2 * u] = v.lo;
          acc[i][2 * u + 1] = v.hi;
        }
      }
    }
    // Every warp of the group has read the sums before the store warp parks its next copies over them.
    ddaLdsBarrier();
#pragma unroll
    for (int i = 0; i < SLICES; ++i) {
      if (i < liveSlices) {
        const size_t s = first + (size_t)i;
        // Publish before the local unpack: it is what the peers are spinning on, and
        // ddaLL128StoreRegs rewrites acc in place when it reverses the flag-lane shuffle.
        for (int r = 1 + groupWarp; r < nRanks; r += groupWarps) {
          const int peer = (selfRank + r) % nRanks;
          uint64_t* scatterSlot =
            reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWordsNext + (uint64_t)selfRank * (uint64_t)slotWords;
          ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, acc[i], flag, flagLane);
        }
        if (groupWarp == storeWarp) {
          ddaLL128StoreRegs<int8_t>(dstBytes + (size_t)selfRank * shardBytes + s * (size_t)kDdaLL128DataBytesPerSlice,
                                    acc[i], ddaLL128ArPeerEltIn(bytesLeft, i), lane, flagLane);
        }
      }
    }
  }

  // Phase 3: collect each item's peer's reduced shard slice into the rest of recvbuff.
  for (size_t i = gwarp; i < items; i += wstride) {
    const size_t s = i / (size_t)(nRanks - 1);
    const int peer = (selfRank + 1 + (int)(i - s * (size_t)(nRanks - 1))) % nRanks;
    const size_t sliceByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = shardBytes - sliceByte;
    const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const uint64_t* gatherSlot = selfBase + bankWordsNext + (uint64_t)peer * (uint64_t)slotWords;
    uint64_t vr[kDdaLL128WordsPerThread];
    if (!ddaLL128ArPeerPoll(gatherSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, vr, flag, lane,
                            params)) {
      break;
    }
    ddaLL128StoreRegs<int8_t>(dstBytes + (size_t)peer * shardBytes + sliceByte, vr, eltInSlice, lane, flagLane);
  }

  ddaLL128WaitStores();

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag32);
}

} // namespace dda::common
