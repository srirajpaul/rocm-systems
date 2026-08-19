/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-reduce device kernel for the DDA path. Each 16-byte line
 * holds 8 bytes of payload and two 4-byte flags; the flags carry cross-rank
 * sync, so no GPU barrier is needed. Staging uses the DDA scratch
 * (comm->ddaScratch, reachable via comm->ddaPeerPtrsDev).
 *
 * This is the all-reduce analogue of all_gather_dda_fabric_ll128.h. The kernel is
 * codepath-agnostic (it only needs the peer scratch table), so the same kernel
 * backs the fabric launcher (dda_all_reduce_fabric_ll.cu) and can equally back
 * an IPC launcher.
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

namespace meta::comms {

using PackType = uint64_t;

// LL128 is for mid-size messages, so the full payload is well under the staging cap.
// each packet takes 16 bytes.
constexpr size_t kDdaLL128ArSlotStridePkts = kDdaLLMaxBytes / sizeof(PackType);

// LL128 flat all-reduce kernel. 1D grid over packets (8B payload each).
//
// Phase 1 (publish): rank selfRank writes its full sendbuff into every peer's
// scratch at slot selfRank, as LL lines carrying the epoch flag.
// Phase 2 (reduce): rank selfRank polls its own scratch slot for each other
// rank (waiting on the flag), and sums those with its own sendbuff into
// recvbuff. The flag polling provides the cross-rank ordering, so no GPU
// barrier is used.
//
// Scratch is double-buffered: bank = epoch & 1, selected via bankOffsetPkts.
// Self does not round-trip through scratch; its contribution is read directly
// from sendbuff in the reduce.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllReduceFlatLL128(T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
                                        T* __restrict__ recvbuff,              // local user output
                                        const T* __restrict__ sendbuff,        // local user input
                                        size_t count,                          // full-message element count
                                        int selfRank, int nRanksRt,
                                        uint32_t* __restrict__ epochDev,       // per-block LL epoch cells (shared AG+AR)
                                        int epochLen) {                        // number of cells in epochDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = count * sizeof(T);
  const size_t nWords = bytes / sizeof(PackType);           // 8B payload words
  const size_t numLines = ddaLL128NumLines(nWords); // 128B lines this size
  const size_t slot = kDdaLL128ArSlotStridePkts;

  const uint32_t flag = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (size_t)nRanks * slot;


  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  // 16 lanes cooperate on one 128B line; grid-stride over line-groups.
  const int group = threadIdx.x / kDdaLL128Lanes;
  const int lane = threadIdx.x % kDdaLL128Lanes;
  const bool isDataLane = lane < kDdaLL128DataElems;
  const int groups = blockDim.x / kDdaLL128Lanes;
  const size_t groupBase = (size_t)blockIdx.x * (size_t)groups + (size_t)group;
  const size_t groupStride = (size_t)gridDim.x * (size_t)groups;

  const PackType* in = reinterpret_cast<const PackType*>(sendbuff);
  PackType* out = reinterpret_cast<PackType*>(recvbuff);

  // Phase 1: publish my payload into every peer's slot[selfRank], flag-last.
  for (size_t ln = groupBase; ln < numLines; ln += groupStride) {
    const size_t base = ln * (size_t)kDdaLL128DataElems;
    const size_t e = base + (size_t)lane;
    const PackType v = (isDataLane && e < nWords) ? in[e]
                       : (lane == kDdaLL128FlagElem) ? static_cast<PackType>(flag) : 0ull;

#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      PackType* dst_pack = reinterpret_cast<PackType*>(peerScratch[peer]) + bankOffsetPkts + (size_t)selfRank * slot;
      LLLine128* dst = reinterpret_cast<LLLine128*>(dst_pack);
      ddaLL128StoreWord(&dst[ln].w[lane], v);
    }
  }

  // Phase 2: poll my slots for the other ranks, reduce with my own data.
  PackType* myBase = reinterpret_cast<PackType*>(peerScratch[selfRank]) + bankOffsetPkts;
  for (size_t ln = groupBase; ln < numLines; ln += groupStride) {
    const size_t base = ln * (size_t)kDdaLL128DataElems;
    const size_t e = base + (size_t)lane;
    const bool hasWord = isDataLane && (e < nWords);
    PackType acc = hasWord ? in[e] : 0ull;
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      PackType* src_pack = myBase + (size_t)peer * slot;
      LLLine128* src = reinterpret_cast<LLLine128*>(src_pack);
      // All 16 lanes poll the shared flag word (broadcast); unfenced.
      while (ddaLL128LoadWord(&src[ln].w[kDdaLL128FlagElem]) != (uint64_t)flag) {
      }
      const uint64_t d = ddaLL128LoadWord(&src[ln].w[lane]);
      acc = ddaLL128AddWord<T>(acc, d);
    }
    if (hasWord) {
      out[e] = acc;
    }
  }

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
}

} // namespace meta::comms
