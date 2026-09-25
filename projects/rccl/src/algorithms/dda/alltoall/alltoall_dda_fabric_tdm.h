/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * DDA alltoall kernel for the fabric/VMM path that moves each peer's chunk
 * through LDS with the gfx1250 Tensor Data Mover instead of per-lane vector
 * loads. Same pull model and FabricGpuBarrier protocol as ddaAllToAllFabric:
 * this rank reads its slot out of every peer's scratch into recvbuff.
 *
 * Work is split into (tile, peer) items. Each warp stages UnrollPeers tiles,
 * one per peer, into its own LDS window, drains, stores them to recvbuff and
 * drains again. Peers vary fastest and are rotated by selfRank so concurrent
 * warps, and different ranks, read from different peers.
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
// Chunk offsets and recvbuff must be this aligned so every transfer is whole 256 B rows.
constexpr size_t kDdaA2ATdmAlign = 256;

// Peer scratch is read once and the landing tile is not reread on this GPU, so bypass
// the caches; system scope because peers are reached over the fabric.
constexpr CachePolicy kDdaA2ATdmPolicy = createCachePolicy(TemporalHint::NT, MemScope::SYS);

template <int TileBytes, int UnrollPeers>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricTdm(uint8_t* const* __restrict__ peerScratch, uint8_t* __restrict__ recvbuff,
                                       size_t bytesPerRank, int selfRank, int nRanks, FabricGpuBarrier barrier) {
  static_assert(TileBytes * UnrollPeers <= kDdaA2ATdmWindowBytes, "tiles overflow the per-warp LDS window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
#if TDM_SUPPORTED
  // Release-acquire barrier ensures the stream-ordered scratch write is visible to peers.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  extern __shared__ __attribute__((aligned(128))) uint8_t ddaA2ATdmSmem[];
  // TDM descriptors live in SGPRs, so every address below must be wave-uniform.
  const int lw = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  uint8_t* win = ddaA2ATdmSmem + lw * kDdaA2ATdmWindowBytes;

  const size_t tilesPerPeer = (bytesPerRank + TileBytes - 1) / TileBytes;
  const size_t nItems = tilesPerPeer * nRanks;
  const size_t warpsPerBlock = blockDim.x / warpSize;
  const size_t nWarps = gridDim.x * warpsPerBlock;
  const size_t srcBase = static_cast<size_t>(selfRank) * bytesPerRank;

  for (size_t base = (blockIdx.x * warpsPerBlock + lw) * UnrollPeers; base < nItems; base += nWarps * UnrollPeers) {
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int peer = (selfRank + static_cast<int>(item % nRanks)) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(peerScratch[peer] + srcBase + off,
                                                                                   win + u * TileBytes, n);
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
          win + u * TileBytes, recvbuff + static_cast<size_t>(peer) * bytesPerRank + off, n);
      }
    }
    tdm::tdmWait();
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch, (void)recvbuff, (void)bytesPerRank, (void)selfRank, (void)nRanks, (void)barrier;
  __builtin_trap();
#endif
}

} // namespace dda::common
