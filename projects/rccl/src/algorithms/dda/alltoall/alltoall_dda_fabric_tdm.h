/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "tdm/tdmCopy.h"

namespace dda::common {

// gfx1250 is wave32. One wave owns a 16 KiB LDS window and fills it with UnrollPeers tiles before the
// stores go out, so 16 waves of a 512-thread block use 256 KiB of dynamic LDS.
constexpr int kDdaA2ATdmWave = 32;
constexpr int kDdaA2ATdmMaxWarpsPerBlock = 16;
constexpr int kDdaA2ATdmWindowBytes = 16 << 10;
constexpr int kDdaA2ATdmMaxSmemBytes = kDdaA2ATdmMaxWarpsPerBlock * kDdaA2ATdmWindowBytes;
constexpr size_t kDdaA2ATdmAlign = 256;
constexpr CachePolicy kDdaA2ATdmPolicy = createCachePolicy(TemporalHint::NT, MemScope::SYS);

// Push the same exchange as ddaAllToAllFabricPush, but each tile moves through TDM: global -> this wave's
// LDS window -> the peer's scratch slot. Items are numbered item = tileIdx * nRanks + k and item k goes to
// peer (selfRank + k) % nRanks. A rank that drains slot s therefore sees the item index its writer used, and
// since the grid depends only on bytesPerRank the same block owns it on both ends, so the per-block barrier
// pairs the write with the read. There is no closing barrier: the drain is local and the next kernel's
// opening barrier covers scratch reuse.
template <int TileBytes, int UnrollPeers>
__global__ void __launch_bounds__(kDdaA2ATdmMaxWarpsPerBlock* kDdaA2ATdmWave)
ddaAllToAllFabricTdmPush(uint8_t* const* __restrict__ peerScratch, uint8_t* __restrict__ recvbuff,
                          const uint8_t* __restrict__ sendbuff, size_t bytesPerRank, int selfRank, int nRanks,
                          FabricGpuBarrier barrier) {
  static_assert(TileBytes * UnrollPeers == kDdaA2ATdmWindowBytes, "tiles must fill one wave window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
#if TDM_SUPPORTED
  extern __shared__ __attribute__((aligned(128))) uint8_t ddaA2ATdmSmem[];
  // TDM descriptors live in SGPRs, so the wave id that selects the LDS window must be wave-uniform.
  const int warp = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  const size_t warps = blockDim.x / warpSize;
  uint8_t* window = ddaA2ATdmSmem + static_cast<size_t>(warp) * kDdaA2ATdmWindowBytes;

  const size_t tilesPerPeer = (bytesPerRank + TileBytes - 1) / TileBytes;
  const size_t nItems = tilesPerPeer * static_cast<size_t>(nRanks);
  const size_t stride = static_cast<size_t>(gridDim.x) * warps * UnrollPeers;
  const size_t first = (static_cast<size_t>(blockIdx.x) * warps + warp) * UnrollPeers;
  const size_t selfSlot = static_cast<size_t>(selfRank) * bytesPerRank;

  // The previous collective's readers must be done with every scratch slot this block is about to overwrite.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  for (size_t base = first; base < nItems; base += stride) {
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int peer = (selfRank + static_cast<int>(item % nRanks)) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
            sendbuff + static_cast<size_t>(peer) * bytesPerRank + off, window + u * TileBytes, n);
      }
    }
    tdm::tdmWait();
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int peer = (selfRank + static_cast<int>(item % nRanks)) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
            window + u * TileBytes, peerScratch[peer] + selfSlot + off, n);
      }
    }
    tdm::tdmWait();
  }

  // tdmWait is per wave. Drain the whole block's stores before any peer is told to read its scratch.
  barrier.orderDestWrites();
  __syncthreads();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  const uint8_t* selfScratch = peerScratch[selfRank];
  for (size_t base = first; base < nItems; base += stride) {
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int src = (selfRank - static_cast<int>(item % nRanks) + nRanks) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
            selfScratch + static_cast<size_t>(src) * bytesPerRank + off, window + u * TileBytes, n);
      }
    }
    tdm::tdmWait();
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int src = (selfRank - static_cast<int>(item % nRanks) + nRanks) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
            window + u * TileBytes, recvbuff + static_cast<size_t>(src) * bytesPerRank + off, n);
      }
    }
    tdm::tdmWait();
  }
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch;
  (void)recvbuff;
  (void)sendbuff;
  (void)bytesPerRank;
  (void)selfRank;
  (void)nRanks;
  (void)barrier;
  __builtin_trap();
#endif
}

// Peer-major TDM push: the grid is nRanks * blocksPerPeer blocks, and block b works only on peer
// k = b / blocksPerPeer (rank (selfRank + k) % nRanks), so its tiles stream to one remote GPU. Slice
// j = b % blocksPerPeer of the group interleaves over the peer's chunk: the group's waves take UnrollPeers
// consecutive tiles at a time, round-robin. The block drains the slot of rank (selfRank - k) % nRanks, which that
// rank's block b wrote with the same tiles, so the two-rank handshake between block b on the destination and on the
// source pairs the write with the read, as in ddaAllToAllFabricPushPeer. Out of place only: a block drains into
// recvbuff slots that other blocks of this rank may still be reading from sendbuff.
// pieces >= 1 (the host never passes 0): a wave goes through nRounds rounds of UnrollPeers tiles, and a piece is a
// run of those rounds. Step p pushes piece p, waits for the source's piece p - 1 and drains it, then posts piece p.
// The flags are the barrier's epoch flags (see FabricGpuBarrier): a post never waits for the previous piece to be
// consumed, and one thread polls the flag for the block. The pieces land at their place in the chunk, so the scratch
// needed is unchanged. One piece pushes the whole share, posts, waits and drains: no overlap.
// Block group 0 (k == 0) copies its own chunk straight from sendbuff to recvbuff (no scratch, no flags). It then drains
// the pieces 0 .. moved - 1 of every other source's slot, where moved = nPieces - keep: the push blocks (k != 0) drain only
// the last keep pieces themselves, so they keep pushing. Block j of group 0 and block k * blocksPerPeer + j of group k
// cover the same tiles. keep == 0 moves every drain to group 0.
// drain > 0: drain * (nRanks - 1) drain blocks follow the push blocks, as in ddaAllToAllFabricPushPeer; drain block d
// of source offset r drains slices d, d + drain, ... of every piece of rank (self - r) % nRanks. The host then passes
// keep == 0 and group 0 only copies, so the push blocks only push and post.
//
// Tuning notes (these are not the defaults; re-check them on each machine). TransferBench's TDM all-to-all, a useful
// reference for the push rate the fabric can sustain, did best with about 32 thread blocks (sub-executors) per peer
// transfer, 256 to 1024 threads per block, and the device maximum of dynamic LDS per block (TDM_LDS_BYTES=0, split evenly
// over the waves, so every wave stages a large window). A small LDS budget per wave was much slower. The matching knobs
// here are the blocks per peer picked by the host (ddaAllToAllFabricPeerGeom, bounded by the block cap), the waves per
// block, and the wave window (TileBytes * UnrollPeers, at most the device's LDS limit divided by the waves). About 32
// blocks per peer, fewer waves per block and a larger window per wave raised the alltoall rate on one machine.
template <int TileBytes, int UnrollPeers>
__global__ void __launch_bounds__(kDdaA2ATdmMaxWarpsPerBlock* kDdaA2ATdmWave)
ddaAllToAllFabricTdmPushPeer(uint8_t* const* __restrict__ peerScratch, uint8_t* __restrict__ recvbuff,
                              const uint8_t* __restrict__ sendbuff, size_t bytesPerRank, int selfRank, int nRanks,
                              int blocksPerPeer, unsigned pieces, unsigned keep, unsigned drain,
                              FabricGpuBarrier barrier) {
  static_assert(TileBytes * UnrollPeers == kDdaA2ATdmWindowBytes, "tiles must fill one wave window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
#if TDM_SUPPORTED
  extern __shared__ __attribute__((aligned(128))) uint8_t ddaA2ATdmSmem[];
  // TDM descriptors live in SGPRs, so the wave id that selects the LDS window must be wave-uniform.
  const int warp = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  const size_t warps = blockDim.x / warpSize;
  uint8_t* window = ddaA2ATdmSmem + static_cast<size_t>(warp) * kDdaA2ATdmWindowBytes;

  const size_t nTiles = (bytesPerRank + TileBytes - 1) / TileBytes;
  const size_t stride = static_cast<size_t>(blocksPerPeer) * warps * UnrollPeers;
  const unsigned pushBlocks = static_cast<unsigned>(nRanks * blocksPerPeer);

  // Rounds a wave goes through and the pieces they are cut into. Every wave of every rank derives the same piece
  // count, since it depends only on bytesPerRank, blocksPerPeer and blockDim.
  const size_t nRounds = (nTiles + stride - 1) / stride;
  const size_t perPiece = (nRounds + pieces - 1) / pieces;
  const size_t nPieces = (nRounds + perPiece - 1) / perPiece;

  // Copies the rounds [p * perPiece, (p + 1) * perPiece) of the wave whose first tile is `first` of a bytesPerRank
  // chunk from src to dst through this wave's LDS window.
  auto passFrom = [&](size_t first, const uint8_t* src, uint8_t* dst, size_t p) {
    const size_t limit = first + (p + 1) * perPiece * stride;
    for (size_t base = first + p * perPiece * stride; base < nTiles && base < limit; base += stride) {
#pragma unroll
      for (int u = 0; u < UnrollPeers; ++u) {
        const size_t t = base + u;
        if (t < nTiles) {
          const size_t off = t * TileBytes;
          const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
          tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(src + off,
                                                                                      window + u * TileBytes, n);
        }
      }
      tdm::tdmWait();
#pragma unroll
      for (int u = 0; u < UnrollPeers; ++u) {
        const size_t t = base + u;
        if (t < nTiles) {
          const size_t off = t * TileBytes;
          const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
          tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(window + u * TileBytes,
                                                                                         dst + off, n);
        }
      }
      tdm::tdmWait();
    }
  };

  const uint32_t epoch = barrier.epochBegin();
  if (blockIdx.x >= pushBlocks) {
    const unsigned idx = blockIdx.x - pushBlocks;
    const int r = 1 + static_cast<int>(idx / drain);
    const int fromRank = (selfRank - r + nRanks) % nRanks;
    const size_t fromOff = static_cast<size_t>(fromRank) * bytesPerRank;
    for (size_t p = 0; p < nPieces; ++p) {
      for (unsigned j = idx % drain; j < static_cast<unsigned>(blocksPerPeer); j += drain) {
        barrier.epochWaitPiece(barrier.epochFlagFrom(fromRank, r * blocksPerPeer + static_cast<int>(j)), epoch,
                               static_cast<unsigned>(p));
        passFrom((static_cast<size_t>(j) * warps + warp) * UnrollPeers, peerScratch[selfRank] + fromOff,
                 recvbuff + fromOff, p);
      }
    }
    barrier.epochEnd(epoch, pieces);
    return;
  }

  const int k = static_cast<int>(blockIdx.x) / blocksPerPeer;
  const size_t slice = static_cast<size_t>(blockIdx.x) % blocksPerPeer;
  const size_t first = (slice * warps + warp) * UnrollPeers;
  auto pass = [&](const uint8_t* s, uint8_t* d, size_t p) { passFrom(first, s, d, p); };

  const int peer = (selfRank + k) % nRanks;
  const int from = (selfRank - k + nRanks) % nRanks;

  // Entry handshake instead of a full barrier: only peer reads the slot this block writes, so wait for peer alone.
  if (k != 0) barrier.epochSignalWait(from, peer, epoch);

  uint8_t* const dst = peerScratch[peer] + static_cast<size_t>(selfRank) * bytesPerRank;
  const uint8_t* const src = sendbuff + static_cast<size_t>(peer) * bytesPerRank;
  const uint8_t* const slot = peerScratch[selfRank] + static_cast<size_t>(from) * bytesPerRank;
  uint8_t* const out = recvbuff + static_cast<size_t>(from) * bytesPerRank;
  // Pieces 0 .. moved - 1 of the slots are drained by group 0, the last `keep` by the push blocks.
  const size_t moved = nPieces > keep ? nPieces - keep : 0;
  if (k == 0) {
    // Own chunk: peer == from == selfRank, so this is sendbuff -> recvbuff through the LDS window. No scratch, no flags.
    for (size_t p = 0; p < nPieces; ++p) pass(src, out, p);
    // Then drain what the push blocks leave: piece p of rank (selfRank - r)'s slot, once that rank's block
    // r * blocksPerPeer + slice has posted it.
    for (size_t p = 0; p < moved && drain == 0; ++p) {
      for (int r = 1; r < nRanks; ++r) {
        const int rank = (selfRank - r + nRanks) % nRanks;
        barrier.epochWaitPiece(barrier.epochFlagFrom(rank, r * blocksPerPeer + static_cast<int>(slice)), epoch,
                               static_cast<unsigned>(p));
        pass(peerScratch[selfRank] + static_cast<size_t>(rank) * bytesPerRank,
             recvbuff + static_cast<size_t>(rank) * bytesPerRank, p);
      }
    }
  } else {
    // Every TDM store of a piece has drained (tdmWait) before epochPostPiece, which also waits for the wave's stores and
    // syncs the block before thread 0 posts.
    const FabricGpuBarrier::FlagType* const ready = barrier.epochFlagFrom(from);
    for (size_t p = 0; p <= nPieces; ++p) {
      if (p < nPieces) pass(src, dst, p);
      // Piece p - 1 is drained here only if it is one of the last `keep` pieces.
      if (p > moved) {
        barrier.epochWaitPiece(ready, epoch, static_cast<unsigned>(p - 1));
        pass(slot, out, p - 1);
      }
      if (p < nPieces) barrier.epochPostPiece(peer, epoch, static_cast<unsigned>(p));
    }
  }
  barrier.epochEnd(epoch, pieces);
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch;
  (void)recvbuff;
  (void)sendbuff;
  (void)bytesPerRank;
  (void)selfRank;
  (void)nRanks;
  (void)blocksPerPeer;
  (void)pieces;
  (void)keep;
  (void)drain;
  (void)barrier;
  __builtin_trap();
#endif
}

} // namespace dda::common
