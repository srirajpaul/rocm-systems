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
// per batch. The block pushes to peer (self + k) and later drains the slot of peer (self - k), which that peer's
// block b wrote, so the per-block barrier pairs the write with the read. k == 0 copies sendbuff straight to recvbuff
// and drains nothing. gridDim.x must be a multiple of nRanks. Out of place only: a block drains into recvbuff slots
// that other blocks of this rank may still be reading from sendbuff.
template <int TileBytes, int UnrollTiles>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricTdmPushPeer(uint8_t* const* __restrict__ peerScratch, uint8_t* __restrict__ recvbuff,
                                               const uint8_t* __restrict__ sendbuff, size_t bytesPerRank, int selfRank,
                                               int nRanks, FabricGpuBarrier barrier, size_t partTiles) {
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
  ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(dst, sendbuff + static_cast<size_t>(peer) * bytesPerRank, bytesPerRank,
                                             first, tileEnd, stride, win);

  // Release our remote writes and acquire every peer's writes into our scratch.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  if (k != 0) {
    const size_t src = static_cast<size_t>((selfRank - k + nRanks) % nRanks) * bytesPerRank;
    ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(recvbuff + src, peerScratch[selfRank] + src, bytesPerRank, first,
                                               tileEnd, stride, win);
  }
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch, (void)recvbuff, (void)sendbuff, (void)bytesPerRank, (void)selfRank, (void)nRanks, (void)barrier,
    (void)partTiles;
  __builtin_trap();
#endif
}

} // namespace dda::common
