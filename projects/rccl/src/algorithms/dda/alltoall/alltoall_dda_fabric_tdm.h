/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * DDA alltoall kernels for the fabric/VMM path that move data through LDS with
 * the gfx1250 Tensor Data Mover instead of per-lane vector loads and stores.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "tdm/tdmCopy.h"

#include <cstddef>
#include <cstdint>

namespace dda::common {

constexpr int kDdaA2ATdmMaxWarpsPerBlock = 16;
// Every tile shape fills the same window, so the LDS grant depends only on the warp count.
constexpr int kDdaA2ATdmWindowBytes = 16 << 10;
// Chunk offsets and buffers must be this aligned so every transfer is whole 256 B rows.
constexpr size_t kDdaA2ATdmAlign = 256;

// Each tile is read once and not reread on this GPU, so bypass the caches; system scope because peers are reached
// over the fabric.
constexpr CachePolicy kDdaA2ATdmPolicy = createCachePolicy(TemporalHint::NT, MemScope::SYS);

#if TDM_SUPPORTED
// Copies tiles [first, end) of a bytes-long chunk from src to dst through this warp's LDS window: UnrollTiles
// consecutive tiles per batch, batches stride tiles apart. Every argument must be wave-uniform.
template <int TileBytes, int UnrollTiles>
__device__ __forceinline__ void ddaA2ATdmCopyTiles(uint8_t* dst, const uint8_t* src, size_t bytes, size_t first,
                                                   size_t end, size_t stride, uint8_t* win) {
  for (size_t base = first; base < end; base += stride) {
#pragma unroll
    for (int u = 0; u < UnrollTiles; ++u) {
      const size_t tile = base + u;
      if (tile < end) {
        const size_t off = tile * TileBytes;
        const size_t n = bytes - off < TileBytes ? bytes - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(src + off, win + u * TileBytes, n);
      }
    }
    tdm::tdmWait();
#pragma unroll
    for (int u = 0; u < UnrollTiles; ++u) {
      const size_t tile = base + u;
      if (tile < end) {
        const size_t off = tile * TileBytes;
        const size_t n = bytes - off < TileBytes ? bytes - off : TileBytes;
        tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(win + u * TileBytes, dst + off,
                                                                                      n);
      }
    }
    tdm::tdmWait();
  }
}
#endif

// Peer-major TDM push: every block serves a single peer, as ddaAllToAllFabricPushPeer does with vector copies.
// With P = gridDim.x / nRanks blocks per peer, block b serves peer offset k = b / P and tiles
// [j * partTiles, (j + 1) * partTiles) of that peer's chunk, j = b % P; each warp moves UnrollTiles consecutive tiles
// per batch. The block pushes to peer (self + k), then syncs with just those two peers and drains the slot of peer
// (self - k), which that peer's block b wrote. k == 0 copies sendbuff straight to recvbuff and drains nothing.
// With pieces > 1 each block's part is cut into that many pieces of whole tiles and the sync becomes a pipeline: step p
// pushes piece p, waits for the source's piece p - 1 and drains it, then posts piece p. A wave drains its LDS window
// before the next transfer, so it does not overlap its own push with its drain; the gain is that each drain waits
// only for the matching piece while other waves and blocks are in other phases. The pieces reuse the block's flag
// slot, so a post first waits for the destination to consume the previous piece. Every TDM access bypasses the
// caches at system scope and tdmWait completes the stores, so no fence is needed per piece.
// gridDim.x must be a multiple of nRanks. Out of place only: a block drains into recvbuff slots that other blocks of
// this rank may still be reading from sendbuff.
template <int TileBytes, int UnrollTiles>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricTdmPushPeer(uint8_t* const* __restrict__ peerScratch, uint8_t* __restrict__ recvbuff,
                                               const uint8_t* __restrict__ sendbuff, size_t bytesPerRank, int selfRank,
                                               int nRanks, FabricGpuBarrier barrier, size_t partTiles, int pieces) {
  static_assert(TileBytes * UnrollTiles <= kDdaA2ATdmWindowBytes, "tiles overflow the per-warp LDS window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
#if TDM_SUPPORTED
  // A peer's scratch is free to overwrite once that peer has reached this op.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  extern __shared__ __attribute__((aligned(128))) uint8_t ddaA2ATdmSmem[];
  // TDM descriptors live in SGPRs, so every address below must be wave-uniform.
  const int lw = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  uint8_t* win = ddaA2ATdmSmem + lw * kDdaA2ATdmWindowBytes;

  const size_t nParts = gridDim.x / nRanks;
  const int k = static_cast<int>(blockIdx.x / nParts);
  const size_t part = blockIdx.x % nParts;
  const size_t tilesPerPeer = (bytesPerRank + TileBytes - 1) / TileBytes;
  const size_t tileBegin = part * partTiles;
  const size_t tileEnd = tileBegin + partTiles < tilesPerPeer ? tileBegin + partTiles : tilesPerPeer;
  const size_t first = tileBegin + static_cast<size_t>(lw) * UnrollTiles;
  const size_t stride = (blockDim.x / warpSize) * UnrollTiles;

  const int peer = (selfRank + k) % nRanks;
  uint8_t* dst = (k == 0 ? recvbuff : peerScratch[peer]) + static_cast<size_t>(selfRank) * bytesPerRank;
  const uint8_t* src = sendbuff + static_cast<size_t>(peer) * bytesPerRank;
  if (k == 0) {
    ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(dst, src, bytesPerRank, first, tileEnd, stride, win);
    return;
  }

  const int srcRank = (selfRank - k + nRanks) % nRanks;
  const size_t srcOff = static_cast<size_t>(srcRank) * bytesPerRank;
  uint8_t* drainDst = recvbuff + srcOff;
  const uint8_t* drainSrc = peerScratch[selfRank] + srcOff;

  if (pieces <= 1) {
    ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(dst, src, bytesPerRank, first, tileEnd, stride, win);
    // Point-to-point closing sync: release our writes to the peer we pushed to and acquire the writes of the peer
    // whose slot we drain, block b on both sides. Blocks of this rank's own chunk neither wrote remotely nor drain,
    // so they skip it.
    barrier.signalWaitPeer(peer, srcRank, blockIdx.x);
    ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(drainDst, drainSrc, bytesPerRank, first, tileEnd, stride, win);
    return;
  }

  // Every rank derives the same part and pieces for block b, so posts and waits stay paired even for empty pieces.
  const size_t warpOff = static_cast<size_t>(lw) * UnrollTiles;
  const size_t partLen = tileEnd > tileBegin ? tileEnd - tileBegin : 0;
  const size_t pieceTiles = (partLen + pieces - 1) / pieces;
  auto pieceBegin = [&](int p) {
    const size_t b = tileBegin + static_cast<size_t>(p) * pieceTiles;
    return b < tileEnd ? b : tileEnd;
  };
  for (int p = 0; p <= pieces; ++p) {
    if (p < pieces) {
      ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(dst, src, bytesPerRank, pieceBegin(p) + warpOff, pieceBegin(p + 1),
                                                 stride, win);
    }
    if (p > 0) {
      if (threadIdx.x == 0) barrier.waitPeerRelaxed(srcRank);
      __syncthreads();
      ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(drainDst, drainSrc, bytesPerRank, pieceBegin(p - 1) + warpOff,
                                                 pieceBegin(p), stride, win);
    }
    if (p < pieces) {
      __syncthreads();
      if (threadIdx.x == 0) barrier.postPeerRelaxed(peer, blockIdx.x);
    }
  }
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch, (void)recvbuff, (void)sendbuff, (void)bytesPerRank, (void)selfRank, (void)nRanks, (void)barrier,
    (void)partTiles, (void)pieces;
  __builtin_trap();
#endif
}

} // namespace dda::common
