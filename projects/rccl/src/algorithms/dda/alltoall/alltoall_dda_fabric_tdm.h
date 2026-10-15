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
                                       const uint8_t* __restrict__ sendbuff, size_t bytesPerRank, int selfRank,
                                       int nRanks, FabricGpuBarrier barrier) {
  static_assert(TileBytes * UnrollPeers <= kDdaA2ATdmWindowBytes, "tiles overflow the per-warp LDS window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
#if TDM_SUPPORTED
  const uint32_t flag = barrier.beginEpoch();

  extern __shared__ __attribute__((aligned(128))) uint8_t ddaA2ATdmSmem[];
  // TDM descriptors live in SGPRs, so every address below must be wave-uniform.
  const int lw = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  uint8_t* win = ddaA2ATdmSmem + lw * kDdaA2ATdmWindowBytes;

  const size_t tilesPerPeer = (bytesPerRank + TileBytes - 1) / TileBytes;
  const size_t nItems = tilesPerPeer * nRanks;
  const size_t warpsPerBlock = blockDim.x / warpSize;
  const size_t nWarps = gridDim.x * warpsPerBlock;
  const size_t firstItem = (blockIdx.x * warpsPerBlock + lw) * UnrollPeers;
  const size_t itemStride = nWarps * UnrollPeers;
  const size_t srcBase = static_cast<size_t>(selfRank) * bytesPerRank;

  // Stage sendbuff into our scratch. Item k writes slot (self - k), the tile peer (self - k) reads from us
  // under the same item index, so the per-block barrier pairs the write with the read. The previous op's
  // closing barrier guarantees no peer is still reading our scratch.
  uint8_t* selfScratch = peerScratch[selfRank];
  for (size_t base = firstItem; base < nItems; base += itemStride) {
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const size_t slot = static_cast<size_t>((selfRank - static_cast<int>(item % nRanks) + nRanks) % nRanks);
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(sendbuff + slot * bytesPerRank + off,
                                                                                   win + u * TileBytes, n);
      }
    }
    tdm::tdmWait();
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const size_t slot = static_cast<size_t>((selfRank - static_cast<int>(item % nRanks) + nRanks) % nRanks);
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
          win + u * TileBytes, selfScratch + slot * bytesPerRank + off, n);
      }
    }
    tdm::tdmWait();
  }

  ddaFabricWaitMemOps();
  // Publish the staged scratch; the reads below go through TDM, which bypasses the cache, so no acquire.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

  for (size_t base = firstItem; base < nItems; base += itemStride) {
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

  // Peers must not overwrite their scratch until every rank has read its slot; nothing is published here.
  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */>(flag + 2);
  barrier.endEpoch(flag + 2);
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch, (void)recvbuff, (void)sendbuff, (void)bytesPerRank, (void)selfRank, (void)nRanks, (void)barrier;
  __builtin_trap();
#endif
}

// Push variant: this rank writes its chunk for every peer into that peer's scratch, then drains its own
// scratch into recvbuff. Same (tile, peer) item split and per-warp LDS window as the pull kernel.
//
// Item k pushes to peer (self + k) and drains the slot of peer (self - k). Peer p's item k pushes to
// (p + k), so the tile this rank drains under item k was written by the same item index on the peer;
// the grid depends only on count, so that item sits in the same block and the per-block barrier pairs
// the write with the read.
template <int TileBytes, int UnrollPeers>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricTdmPush(uint8_t* const* __restrict__ peerScratch, uint8_t* __restrict__ recvbuff,
                                           const uint8_t* __restrict__ sendbuff, size_t bytesPerRank, int selfRank,
                                           int nRanks, FabricGpuBarrier barrier) {
  static_assert(TileBytes * UnrollPeers <= kDdaA2ATdmWindowBytes, "tiles overflow the per-warp LDS window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
#if TDM_SUPPORTED
  const uint32_t flag = barrier.beginEpoch();

  // Only local sendbuff is read before the next barrier, so no acquire is needed here.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

  extern __shared__ __attribute__((aligned(128))) uint8_t ddaA2ATdmSmem[];
  // TDM descriptors live in SGPRs, so every address below must be wave-uniform.
  const int lw = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  uint8_t* win = ddaA2ATdmSmem + lw * kDdaA2ATdmWindowBytes;

  const size_t tilesPerPeer = (bytesPerRank + TileBytes - 1) / TileBytes;
  const size_t nItems = tilesPerPeer * nRanks;
  const size_t warpsPerBlock = blockDim.x / warpSize;
  const size_t nWarps = gridDim.x * warpsPerBlock;
  const size_t firstItem = (blockIdx.x * warpsPerBlock + lw) * UnrollPeers;
  const size_t itemStride = nWarps * UnrollPeers;
  const size_t selfSlot = static_cast<size_t>(selfRank) * bytesPerRank;

  for (size_t base = firstItem; base < nItems; base += itemStride) {
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int peer = (selfRank + static_cast<int>(item % nRanks)) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
          sendbuff + static_cast<size_t>(peer) * bytesPerRank + off, win + u * TileBytes, n);
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
        tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(win + u * TileBytes,
                                                                                      peerScratch[peer] + selfSlot + off, n);
      }
    }
    tdm::tdmWait();
  }

  // TODO : move these inside barrier
  __syncthreads();

#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if defined(__gfx1250__)
  asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0");
#else
  asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
#endif
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#else
  __threadfence_system();
#endif

  // Every TDM store above has drained (tdmWait), and they bypass L2, so no release fence is needed; the
  // drain below reads only through TDM, which bypasses the cache too.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 2);

  const uint8_t* selfScratch = peerScratch[selfRank];
  for (size_t base = firstItem; base < nItems; base += itemStride) {
#pragma unroll
    for (int u = 0; u < UnrollPeers; ++u) {
      const size_t item = base + u;
      if (item < nItems) {
        const int src = (selfRank - static_cast<int>(item % nRanks) + nRanks) % nRanks;
        const size_t off = (item / nRanks) * TileBytes;
        const size_t n = bytesPerRank - off < TileBytes ? bytesPerRank - off : TileBytes;
        tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaA2ATdmPolicy, /*Aligned=*/true>(
          selfScratch + static_cast<size_t>(src) * bytesPerRank + off, win + u * TileBytes, n);
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
          win + u * TileBytes, recvbuff + static_cast<size_t>(src) * bytesPerRank + off, n);
      }
    }
    tdm::tdmWait();
  }
  barrier.endEpoch(flag + 2);
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch, (void)recvbuff, (void)sendbuff, (void)bytesPerRank, (void)selfRank, (void)nRanks, (void)barrier;
  __builtin_trap();
#endif
}

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
  const uint32_t flag = barrier.beginEpoch();
  // Only local sendbuff is read before the next barrier, so no acquire is needed here.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

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

  ddaFabricWaitMemOps();
  // Every TDM store above has drained (tdmWait), and they bypass L2, so no release fence is needed; the drain below
  // reads only through TDM, which bypasses the cache too.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 2);

  if (k != 0) {
    const size_t src = static_cast<size_t>((selfRank - k + nRanks) % nRanks) * bytesPerRank;
    ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(recvbuff + src, peerScratch[selfRank] + src, bytesPerRank, first,
                                               tileEnd, stride, win);
  }
  barrier.endEpoch(flag + 2);
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch, (void)recvbuff, (void)sendbuff, (void)bytesPerRank, (void)selfRank, (void)nRanks, (void)barrier,
    (void)partTiles;
  __builtin_trap();
#endif
}

// Pipelined peer-major TDM push: the block roles and parts of ddaAllToAllFabricTdmPushPeer, with each part cut into
// `pieces` pieces of whole tiles and a ready flag per piece in place of the barrier between push and drain, as
// ddaAllToAllFabricPushPeerPipe does with vector copies. Step i pushes piece i, then waits for the source's piece
// i - 1 and drains it, then posts piece i's flag; a final step drains the last piece. The source is the peer this
// block pushes to, (self + k): the block drains the slot that peer's block with offset nRanks - k (same part) wrote,
// and each warp drains exactly the tiles it pushed, so the kernel runs in place. Piece i's flag is epoch
// flag + 2 + i in the slot this block owns at the peer; every wave polls the flag itself. The entry barrier stays a
// full barrier unless dbuf chains it (double-buffered scratch, as in ddaAllToAllFabricPushPeerPipe; the two kernels
// share the records and the half offset, so either can follow the other). gridDim.x must be a multiple of nRanks.
template <int TileBytes, int UnrollTiles>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricTdmPushPeerPipe(uint8_t* const* __restrict__ peerScratch, uint8_t* recvbuff,
                                                   const uint8_t* sendbuff, size_t bytesPerRank, int selfRank,
                                                   int nRanks, FabricGpuBarrier barrier, size_t partTiles,
                                                   int pieces, size_t halfOff, bool dbuf) {
  static_assert(TileBytes * UnrollTiles <= kDdaA2ATdmWindowBytes, "tiles overflow the per-warp LDS window");
  static_assert(TileBytes % kDdaA2ATdmAlign == 0, "tiles must be whole 256 B rows");
#if TDM_SUPPORTED
  const uint32_t flag = barrier.beginEpoch();
  uint32_t lastHalf = 0;
  const bool chained = dbuf && barrier.lastEndedWithRecord(flag, &lastHalf);
  const uint32_t half = chained ? (lastHalf ^ 1u) : 0u;
  const size_t dataOff = static_cast<size_t>(half) * halfOff;
  // Only local sendbuff is read before the next barrier, so no acquire is needed here.
  if (!chained) {
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                               true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);
  }
  auto finish = [&]() {
    if (dbuf) {
      barrier.endEpochRecord(flag + 1 + pieces, half);
    } else {
      barrier.endEpoch(flag + 1 + pieces);
    }
  };

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
  const size_t warpOff = static_cast<size_t>(lw) * UnrollTiles;
  const size_t stride = (blockDim.x / warpSize) * UnrollTiles;

  const int peer = (selfRank + k) % nRanks;
  uint8_t* dst = (k == 0 ? recvbuff : peerScratch[peer] + dataOff) + static_cast<size_t>(selfRank) * bytesPerRank;
  const uint8_t* src = sendbuff + static_cast<size_t>(peer) * bytesPerRank;

  if (k == 0 || tileBegin >= tileEnd) {
    // Own chunk (no remote write, no drain; in place it is already in recvbuff) or an empty part: nothing to signal
    // or wait for.
    if (tileBegin < tileEnd && dst != src) {
      ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(dst, src, bytesPerRank, tileBegin + warpOff, tileEnd, stride, win);
    }
    finish();
    return;
  }

  const size_t pieceTiles = (tileEnd - tileBegin + pieces - 1) / pieces;
  auto pieceBegin = [&](int i) {
    const size_t b = tileBegin + static_cast<size_t>(i) * pieceTiles;
    return b < tileEnd ? b : tileEnd;
  };
  const size_t srcOff = static_cast<size_t>(peer) * bytesPerRank;
  const uint8_t* drainSrc = peerScratch[selfRank] + dataOff + srcOff;
  uint8_t* drainDst = recvbuff + srcOff;
  const FabricGpuBarrier::FlagType* ready = barrier.peerFlag(peer, static_cast<int>((nRanks - k) * nParts + part));

  for (int i = 0; i <= pieces; ++i) {
    if (i < pieces) {
      ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(dst, src, bytesPerRank, pieceBegin(i) + warpOff, pieceBegin(i + 1),
                                                 stride, win);
    }
    if (i > 0) {
      const FabricGpuBarrier::FlagType want = flag + 1 + i;
      while (static_cast<int32_t>(__scoped_atomic_load_n(ready, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) - want) < 0) {
      }
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      ddaA2ATdmCopyTiles<TileBytes, UnrollTiles>(drainDst, drainSrc, bytesPerRank, pieceBegin(i - 1) + warpOff,
                                                 pieceBegin(i), stride, win);
    }
    if (i < pieces) {
      // Every warp's TDM stores of piece i have drained (tdmWait); finish its other memory ops, then publish.
      ddaFabricWaitMemOps();
      __syncthreads();
      if (threadIdx.x == 0) barrier.postPeer(peer, flag + 2 + i);
    }
  }
  finish();
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)peerScratch, (void)recvbuff, (void)sendbuff, (void)bytesPerRank, (void)selfRank, (void)nRanks, (void)barrier,
    (void)partTiles, (void)pieces, (void)halfOff, (void)dbuf;
  __builtin_trap();
#endif
}

} // namespace dda::common
