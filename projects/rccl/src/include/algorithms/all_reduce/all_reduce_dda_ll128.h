/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-reduce device kernel for the DDA path following RCCL LL128
 * protocol. All-reduce counterpart of all_gather_dda_ll128.h: same slice / wire
 * geometry and the same ll128_pack.h pack-poll-unpack helpers, but every block
 * fans out to all peers instead of owning one peer column, because a reduced
 * output word depends on every rank.
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
#include "algorithms/ll128_pack.h"

namespace meta::comms {

// Host-side mirrors of the wire layout, the LL128 counterparts of the LL path's
// kDdaLLArSlotStridePkts. Pass comm->WarpSize / comm->ll128LineElems: the device
// macros must not be used for host logic, they default to the gfx9 values there.
inline int ddaLL128ArWireWordPerSlice(int warpSize) {
  return warpSize * ll128::kWordsPerThread;
}

// One word per line carries the flag, so a slice's payload is its wire size
// scaled by (lineElems - 1) / lineElems: 16/15 on gfx1250, 8/7 on gfx9.
inline int ddaLL128ArDataBytesPerSlice(int warpSize, int lineElems) {
  const int wire = ddaLL128ArWireWordPerSlice(warpSize);
  return (wire - wire / lineElems) * 8;
}

inline size_t ddaLL128ArSlices(size_t bytes, int warpSize, int lineElems) {
  const size_t d = (size_t)ddaLL128ArDataBytesPerSlice(warpSize, lineElems);
  return (bytes + d - 1) / d;
}

// Words per slot, fixed for the comm rather than derived from a call's slice
// count, so consecutive epochs always occupy disjoint banks whatever the message
// size. With a size-dependent stride a small call's bank can land inside a large
// call's region, and because a rank may run a whole epoch ahead of a peer, its
// stores then overwrite flag words that peer is still polling, wedging it.
//
// The value is kDdaLLMaxBytes worth of words on purpose: this tier shares one
// scratch and one epoch counter with the two LL all-reduce tiers, so bank 1 has
// to start at the same byte offset (nRanks * kDdaLLMaxBytes) for all three.
constexpr size_t kDdaLL128ArSlotWords = kDdaLLMaxBytes / sizeof(uint64_t);

// Slices one slot can hold, and hence the largest message this tier can stage.
inline size_t ddaLL128ArMaxSlices(int warpSize) {
  return kDdaLL128ArSlotWords / (size_t)ddaLL128ArWireWordPerSlice(warpSize);
}

// LL128 flat all-reduce kernel. One warp owns one slice at a time; the grid is
// 1D over slices and each block fans out to every remote peer.
//
// Phase 1 (publish): pack this rank's slice into registers once, then push it
// onto the wire in every peer's scratch at slot selfRank, flag word embedded.
// Phase 2 (reduce): for each peer, poll that peer's slot in our own scratch and
// fold the arriving words into an accumulator seeded with our own payload, then
// unpack the accumulator into recvbuff. Flag polling supplies the cross-rank
// ordering, so no GPU barrier is used.
//
// Self does not round-trip through scratch; its contribution seeds the
// accumulator straight from sendbuff. Scratch is double buffered: bank = flag & 1.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
__global__ void ddaAllReduceFlatLL128(
    T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
    T* __restrict__ recvbuff,              // local user output
    const T* __restrict__ sendbuff,        // local user input
    size_t bytes,                          // full-message payload; multiple of 16
    int selfRank,
    int nRanksRt,
    uint32_t* __restrict__ epochDev,       // per-block LL epoch cells
    int epochLen,                          // number of cells in epochDev
    size_t slicesTotal,                    // slices this call actually uses
    size_t slotWords,                      // fixed per-slot stride, from host
    int wireWordPerSlice,                  // u64 words on the wire per slice
    int dataBytesPerSlice) {               // payload bytes per slice

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % ll128::kWarp;
  const int warp = tid / ll128::kWarp;
  const int nwarps = nthreads / ll128::kWarp;
  const bool flagLane = ll128::isFlagLane(lane);

  const int flatBlockId = (int)blockIdx.x;
  const int total = (int)gridDim.x;
  uint32_t f = epochDev[flatBlockId] + 1u;
  if (f == 0u) f = 2u;                     // skip 0 sentinel; keep bank parity
  const uint32_t flag32 = f;
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const uint32_t bank = flag32 & 1u;

  const uint64_t bankWords =
      (uint64_t)bank * (uint64_t)nRanks * (uint64_t)slotWords;

  // Slices stride by warp across the whole grid.
  const size_t gwarp = (size_t)blockIdx.x * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.x * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff);
  const uint64_t* gatherBase =
      reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;

  // Phase 1: pack each slice once, then push it to every peer's slot[selfRank].
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t dataByte = s * (size_t)dataBytesPerSlice;
    const size_t rem = bytes - dataByte;
    const int eltInSlice =
        rem < (size_t)dataBytesPerSlice ? (int)rem : dataBytesPerSlice;
    uint64_t regs[ll128::kWordsPerThread] = {};
    ll128::loadRegs<int8_t>(regs, srcBytes + dataByte, eltInSlice, lane, flagLane);

#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) +
          bankWords + (uint64_t)selfRank * slotWords;
      ll128::storeWire(scatterSlot + s * (size_t)wireWordPerSlice + 2 * lane,
                       regs, flag, flagLane);
    }
  }

  // Phase 2: poll every peer's slot for the same slices and fold them in.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t dataByte = s * (size_t)dataBytesPerSlice;
    const size_t rem = bytes - dataByte;
    const int eltInSlice =
        rem < (size_t)dataBytesPerSlice ? (int)rem : dataBytesPerSlice;

    // Seed with our own payload; peers are folded on top.
    uint64_t acc[ll128::kWordsPerThread] = {};
    ll128::loadRegs<int8_t>(acc, srcBytes + dataByte, eltInSlice, lane, flagLane);

    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      const uint64_t* gatherSlot = gatherBase + (uint64_t)peer * slotWords;
      uint64_t vr[ll128::kWordsPerThread];
      ll128::pollWire(gatherSlot + s * (size_t)wireWordPerSlice + 2 * lane,
                      vr, flag, lane);
      // On a flag lane the odd words hold flags rather than payload, so the sums
      // landing there are meaningless -- storeRegs re-derives those slots from
      // the even ones and never writes them out, so folding blind is cheaper
      // than predicating the loop.
#pragma unroll
      for (int u = 0; u < ll128::kWordsPerThread; ++u) {
        acc[u] = ddaLL128AddWord<T>(acc[u], vr[u]);
      }
    }

    ll128::storeRegs<int8_t>(dstBytes + dataByte, acc, eltInSlice, lane, flagLane);
  }

  __syncthreads();
  for (int e = flatBlockId + tid * total; e < epochLen; e += total * nthreads) {
    epochDev[e] = flag32;
  }
}

} // namespace meta::comms
