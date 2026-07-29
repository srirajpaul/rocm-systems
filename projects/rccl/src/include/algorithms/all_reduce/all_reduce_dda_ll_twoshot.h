/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Two-shot (reduce-scatter + all-gather) LL-protocol all-reduce device kernel
 * for the DDA path. Each 16-byte line holds 8 bytes of payload and two 4-byte
 * flags; the flags carry cross-rank sync, so no GPU barrier is needed. Staging
 * uses the DDA scratch (comm->ddaScratch, reachable via comm->ddaPeerPtrsDev).
 *
 * Where the one-shot kernel in all_reduce_dda_ll.h publishes the full message
 * to every peer (egress 2 * S * (nRanks-1) bytes per rank, counting the 8B->16B
 * line expansion), this kernel exchanges only shards: egress drops to
 * 4 * S * (nRanks-1) / nRanks, a factor of nRanks/2 less. It pays a second
 * serialized publish/poll dependency, so it is selected only above a size
 * threshold and only for nRanks >= 4 (see dda_all_reduce_fabric_ll.cu).
 *
 * The kernel is a fusion of ddaReduceScatterFabricLL and the non-self half of
 * ddaAllGatherFabricLL. It needs no intermediate buffer: the reduced shard is
 * already part of the final output, so it is written straight to
 * recvbuff[selfRank] and read back from there when publishing stage B.
 *
 * Like the other DDA LL kernels this one is codepath-agnostic (it only needs
 * the peer scratch table), so the same kernel can back an IPC launcher.
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
// ddaLLEpochBegin / ddaLLEpochEnd are shared by all LL/LL128 kernels.
#include "algorithms/CollCommon_ll128.h"

namespace meta::comms {

// Hard per-message cap for the two-shot LL tier, matching the one-shot cap in
// all_reduce_dda_ll.h. The slot stride is derived per call from the shard size
// (not from this cap), so small messages keep a compact scratch layout.
constexpr size_t kDdaLLArTwoShotMaxBytes = 131072; // 128 KiB

// Scratch stages: 0 == reduce-scatter, 1 == all-gather. Both stages of a launch
// carry the same flag value, so they must occupy disjoint scratch regions.
constexpr int kDdaLLArTwoShotStages = 2;

// At 2 ranks two-shot moves exactly as many bytes as one-shot (the nRanks/2
// saving is 1) while paying a second round trip, so it never pays off there.
constexpr int kDdaLLArTwoShotMinRanks = 4;

// Two-shot LL all-reduce kernel. shardPkts == (bytes / 8) / nRanks, guaranteed
// whole by the eligibility check.
//
// Sweep 1 (scatter): publish this rank's chunk-for-peer, sendbuff[peer], into
// peer's stage-0 slot selfRank.
// Sweep 2 (reduce + publish): seed with the own chunk sendbuff[selfRank], poll
// the local stage-0 slots of the other ranks, and write the fully reduced shard
// to recvbuff[selfRank] -- which is its final location -- then publish that same
// shard into every peer's stage-1 slot selfRank.
// Sweep 3 (gather): poll the local stage-1 slots and unpack them into
// recvbuff[peer].
//
// A shard is nRanks times smaller than the whole message, so a grid sized for
// shardPkts would leave the kernel far narrower than the one-shot variant for
// the same message. Sweeps 1 and 3 therefore spread (peer, packet) pairs over
// the grid instead of packets alone, which restores roughly the one-shot issue
// width; only sweep 2 is one thread per packet, because the accumulator for a
// packet lives in that thread's registers.
//
// Sweep 1 completes every publish before the first poll, and stage-1 publishes
// only happen after stage 0 has resolved, so the global order is
// publish(0) -> poll(0) -> publish(1) -> poll(1) with no circular wait.
//
// Scratch is double-buffered: bank = flag & 1.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceTwoShotLL(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                        T* __restrict__ recvbuff,           // local user output (count elems)
                                        const T* __restrict__ sendbuff,     // local user input (count elems)
                                        size_t count,                       // full-message element count
                                        int selfRank, int nRanksRt,
                                        uint32_t* __restrict__ epochDev, // per-block LL epoch cells (LL AR tier)
                                        int epochLen,                    // number of cells in epochDev
                                        size_t slotStridePkts) {         // per-call packets/slot (from host)

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3;              // 8 payload bytes per packet
  const size_t shardPkts = nPk / (size_t)nRanks;
  const size_t slot = slotStridePkts;         // packets per slot (per-call)
  const size_t chunkWords = shardPkts * 2;    // uint32 words per shard
  const size_t regionPkts = (size_t)nRanks * slot; // one stage: nRanks slots

  // On-device, graph-safe flag/bank derivation (1D grid: flatBlockId=blockIdx.x).
  const int flatBlockId = blockIdx.x;
  const int total = gridDim.x;
  __shared__ uint32_t s_flag;
  const uint32_t flag = ddaLLEpochBegin(epochDev, flatBlockId, s_flag);
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (size_t)kDdaLLArTwoShotStages * regionPkts;

  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff);
  uint32_t* out = reinterpret_cast<uint32_t*>(recvbuff);

  // (peer, packet) pairs handled by sweeps 1 and 3; peer index r runs 1..nRanks-1
  // and consecutive threads take consecutive packets of one peer, so the stores
  // and loads stay coalesced.
  const size_t pairs = (size_t)(nRanks - 1) * shardPkts;

  // Sweep 1: scatter my chunk-for-peer (sendbuff[peer]) into peer's stage-0 slot.
  for (size_t it = gtid; it < pairs; it += stride) {
    const int r = (int)(it / shardPkts) + 1;
    const size_t pk = it - (size_t)(r - 1) * shardPkts;
    const int peer = (selfRank + r) % nRanks;
    const uint32_t* inPeer = in + (size_t)peer * chunkWords;
    const uint32_t d0 = inPeer[2 * pk];
    const uint32_t d1 = inPeer[2 * pk + 1];
    LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts + (size_t)selfRank * slot;
    ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), d0, flag, d1, flag);
  }

  // Sweep 2: reduce my shard out of the stage-0 slots, store it to its final
  // place in recvbuff, then publish it into every peer's stage-1 slot.
  const uint32_t* inSelf = in + (size_t)selfRank * chunkWords;
  uint32_t* outSelf = out + (size_t)selfRank * chunkWords;
  LLPacket16* myStage0 = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts;
  for (size_t pk = gtid; pk < shardPkts; pk += stride) {
    uint32_t acc0 = inSelf[2 * pk];
    uint32_t acc1 = inSelf[2 * pk + 1];
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      volatile LLPacket16* src = myStage0 + (size_t)peer * slot;
      uint32_t d0, f0, d1, f1;
      do {
        ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(const_cast<LLPacket16*>(&src[pk])), d0, f0, d1, f1);
      } while (f0 != flag || f1 != flag);
      acc0 = vecElementAdd<T>(acc0, d0);
      acc1 = vecElementAdd<T>(acc1, d1);
    }
    outSelf[2 * pk] = acc0;
    outSelf[2 * pk + 1] = acc1;

#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      LLPacket16* dst =
        reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts + regionPkts + (size_t)selfRank * slot;
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), acc0, flag, acc1, flag);
    }
  }

  // Sweep 3: gather the peers' reduced shards out of the stage-1 slots.
  LLPacket16* myStage1 = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts + regionPkts;
  for (size_t it = gtid; it < pairs; it += stride) {
    const int r = (int)(it / shardPkts) + 1;
    const size_t pk = it - (size_t)(r - 1) * shardPkts;
    const int peer = (selfRank + r) % nRanks;
    volatile LLPacket16* src = myStage1 + (size_t)peer * slot;
    uint32_t* outPeer = out + (size_t)peer * chunkWords;
    uint32_t d0, f0, d1, f1;
    do {
      ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(const_cast<LLPacket16*>(&src[pk])), d0, f0, d1, f1);
    } while (f0 != flag || f1 != flag);
    outPeer[2 * pk] = d0;
    outPeer[2 * pk + 1] = d1;
  }

  ddaLLEpochEnd(epochDev, flatBlockId, total, epochLen, flag);
}

} // namespace meta::comms
