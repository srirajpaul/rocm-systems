/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Two-shot (reduce-scatter + all-gather) LL-protocol all-reduce device kernel
 * for the DDA path. Each 16-byte line holds 8 bytes of payload and two 4-byte
 * flags; the flags carry cross-rank sync, so no GPU barrier is needed. Staging
 * uses the DDA scratch (comm->ddaScratch, reachable via comm->ddaPeerPtrsDev).
 *
 * Where the one-shot algorithm publishes the full message to every peer (egress
 * 2 * S * (nRanks-1) bytes per rank, counting the 8B->16B line expansion), this one
 * exchanges only shards: egress drops to 4 * S * (nRanks-1) / nRanks, a factor of
 * nRanks/2 less. It pays a second serialized publish/poll dependency, which is why
 * it takes the larger half of the LL size range and the one-shot tier keeps the
 * latency-bound small messages (the crossover lives in the dispatch in
 * collectives.cc). The saving is only 1x at 2 ranks and 2x at 4, so the trade needs
 * a reasonably wide clique to pay off.
 *
 * Scratch layout: bank (flag & 1) selects one of two banks, and each bank holds two
 * stages of nRanks slots -- stage 0 for the scatter, stage 1 for the reduced shards.
 * A slot is kDdaLLArTwoShotSlotStridePkts packets wide whatever the message size, so
 * the footprint is the same for every call and a launcher can check it once.
 *
 * Like the other DDA LL kernels this one is codepath-agnostic (it only needs the
 * peer scratch table), so it can back either an IPC or a fabric launcher. It lays
 * its lines out differently from the one-shot kernel over the same scratch bytes,
 * so a launcher must drive both from one epoch counter -- the monotonic flag is
 * what stops a leftover line from either layout matching the other's flag.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "algorithms/CollCommon.h"
// ddaLLEpochBegin / ddaLLEpochEnd, shared with the LL128 kernels. They derive the
// same flag the hand-rolled epoch code in all_reduce_dda_ll.h does, so both
// all-reduce variants can share one epoch array.
#include "algorithms/CollCommon_ll128.h"

namespace meta::comms {

// Hard per-message cap, well above the one-shot variant's kDdaLLArMaxBytes: the
// two-shot saving grows with the message, so this tier serves the band the one-shot
// tier cannot.
constexpr size_t kDdaLLArTwoShotMaxBytes = 4194304; // 4 MiB
// Slot stride in packets, sized so a slot holds the largest shard this tier accepts:
// at 2 ranks that is half of kDdaLLArTwoShotMaxBytes, or maxBytes/16 packets. Fixed
// rather than per call, so every rank lays scratch out the same way without being
// told the message size. A launcher must reject anything whose shard exceeds it.
constexpr size_t kDdaLLArTwoShotSlotStridePkts = kDdaLLArTwoShotMaxBytes / 16;   // 262144

// LL all-reduce kernel for the two-shot tier. 1D grid over the packets (8B payload
// each) of one shard -- count / nRanks elements -- with the peer fan-out inside each
// phase, so the grid is sized on the shard and not on the whole message.
//
// Phase 1 (scatter): rank selfRank writes each peer's shard of its own sendbuff into
// that peer's stage-0 slot selfRank, as LL lines carrying the epoch flag.
// Phase 2 (reduce-scatter + publish): selfRank polls its own stage-0 slots for the
// other ranks, sums them with its own shard of sendbuff, writes the result to its
// shard of recvbuff -- already the final home -- and republishes it into every
// peer's stage-1 slot selfRank.
// Phase 3 (all-gather): selfRank polls its stage-1 slots and copies each peer's
// reduced shard into that peer's place in recvbuff.
//
// The flag polling provides all the cross-rank ordering, so no GPU barrier is used,
// and self never round-trips through scratch. Scratch is double-buffered: bank =
// flag & 1, and each bank spans both stages (bankOffsetPkts, then that plus
// nRanks * slot).
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceTwoShotLL(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                        T* __restrict__ recvbuff,           // local user output
                                        const T* __restrict__ sendbuff,     // local user input
                                        size_t count,                       // full-message element count
                                        int selfRank, int nRanksRt,
                                        uint32_t* __restrict__ epochDev, // per-block LL epoch cells (LL AR tier)
                                        int epochLen) {                  // number of cells in epochDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet
  const size_t nPk_rank = nPk / nRanks;
  const size_t slot = kDdaLLArTwoShotSlotStridePkts;
  assert(slot >= nPk_rank);

  // On-device, graph-safe flag/bank derivation (1D grid: flatBlockId=blockIdx.x).
  // tid 0 reads this block's epoch cell (all cells hold the same value) and
  // broadcasts the launch's flag, so nothing is baked into a HIP graph capture.
  const int flatBlockId = blockIdx.x;
  const int total = gridDim.x;
  __shared__ uint32_t s_flag;
  const uint32_t flag = ddaLLEpochBegin(epochDev, flatBlockId, s_flag);
  // TODO: this not how it should be done, we should use the same calculation in all collectives
  // multiply by 2 since have use different buffers for rs and ag phases
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (size_t)nRanks * slot * 2;
  const size_t bankOffsetPkts_next = bankOffsetPkts + (size_t)nRanks * slot;

  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  // The user buffers carry no flags, so they are indexed as bare 8B payload units --
  // one per LL line, which keeps the packet index the same on both sides and lets a
  // line's payload move in a single 64-bit access.
  const Packet8* in = reinterpret_cast<const Packet8*>(sendbuff) + selfRank * nPk_rank;
  Packet8* out = reinterpret_cast<Packet8*>(recvbuff) + selfRank * nPk_rank;

  // Phase 1: publish my payload into every peer's slot[selfRank].
  for (size_t pk = gtid; pk < nPk_rank; pk += stride) {
    #pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      const Packet8* peer_in = reinterpret_cast<const Packet8*>(sendbuff) + peer * nPk_rank;
      const Packet8 v = peer_in[pk];
      LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts + (size_t)selfRank * slot;
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), v.d0, flag, v.d1, flag);
    }
  }

  // Phase 2: poll my slots for the other ranks, reduce with my own data and write back.
  LLPacket16* myBase = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts;
  for (size_t pk = gtid; pk < nPk_rank; pk += stride) {
    const Packet8 mine = in[pk];
    uint32_t acc0 = mine.d0;
    uint32_t acc1 = mine.d1;
    #pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      volatile LLPacket16* src = myBase + (size_t)peer * slot;
      uint32_t d0, f0, d1, f1;
      do {
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(const_cast<LLPacket16*>(&src[pk])), d0, f0, d1, f1);
      } while (f0 != flag || f1 != flag);
      acc0 = vecElementAdd<T>(acc0, d0);
      acc1 = vecElementAdd<T>(acc1, d1);
    }
    // Assigned as a whole unit rather than field by field, so the payload leaves in
    // one 64-bit store.
    Packet8 acc;
    acc.d0 = acc0;
    acc.d1 = acc1;
    out[pk] = acc;

    #pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts_next + (size_t)selfRank * slot;
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), acc0, flag, acc1, flag);
    }
  }

  // Phase 3: poll my slots for the other ranks, read data and write to output.
  myBase = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts_next;
  for (size_t pk = gtid; pk < nPk_rank; pk += stride) {
    #pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      Packet8* peer_out = reinterpret_cast<Packet8*>(recvbuff) + peer * nPk_rank;
      volatile LLPacket16* src = myBase + (size_t)peer * slot;
      uint32_t d0, f0, d1, f1;
      do {
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(const_cast<LLPacket16*>(&src[pk])), d0, f0, d1, f1);
      } while (f0 != flag || f1 != flag);
      // write to output array without reduction
      Packet8 v;
      v.d0 = d0;
      v.d1 = d1;
      peer_out[pk] = v;
    }
  }

  ddaLLEpochEnd(epochDev, flatBlockId, total, epochLen, flag);
}

} // namespace meta::comms
