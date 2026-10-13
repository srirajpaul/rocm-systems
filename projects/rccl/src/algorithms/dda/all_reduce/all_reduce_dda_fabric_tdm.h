/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * DDA all-reduce kernels for the fabric/VMM path that move data through LDS with
 * the gfx1250 Tensor Data Mover instead of per-lane vector accesses. Same phases
 * and FabricGpuBarrier protocol as ddaAllReduceFlatFabric / ddaAllReduceTreeFabric.
 *
 * Work is split into tiles. A warp takes one tile index at a time and keeps one
 * tile per rank in its own LDS window, so TileBytes * nRanks must fit the window.
 * Copies are a TDM load into LDS and a TDM store out of it; a reduction loads
 * every rank's tile, adds them in rank order from LDS in registers (as the vector
 * kernels do, so results are bitwise identical), and TDM-stores the sum.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "tdm/tdmCopy.h"

#include <cstddef>
#include <cstdint>

namespace dda::common {

constexpr int kDdaArTdmMaxWarpsPerBlock = 16;
constexpr int kDdaArTdmWindowBytes = 16 << 10;
constexpr int kDdaArTdmMaxDynSmem = kDdaArTdmMaxWarpsPerBlock * kDdaArTdmWindowBytes;
// Buffer sizes, offsets and pointers must be this aligned so every transfer is whole 256 B rows.
constexpr size_t kDdaArTdmAlign = 256;
// Every access bypasses the caches at system scope, as the vector kernels' b128 accesses do.
constexpr CachePolicy kDdaArTdmPolicy = createCachePolicy(TemporalHint::NT, MemScope::SYS);

#if TDM_SUPPORTED
// Sums the nRanks tiles of n bytes at win + r * TileBytes into win, in rank order. Each lane owns whole 16 B
// chunks, so reading and overwriting slot 0 never races. The DS wait makes the sum visible to a following
// TDM store, which reads LDS asynchronously.
template <typename T, int TileBytes>
__device__ __forceinline__ void ddaArTdmReduceWindow(uint8_t* win, int nRanks, size_t n) {
  asm volatile("" ::: "memory");
  const int lane = threadIdx.x % warpSize;
  for (size_t c = lane * sizeof(uint4); c < n; c += warpSize * sizeof(uint4)) {
    uint4 sum{0, 0, 0, 0};
    for (int r = 0; r < nRanks; ++r) {
      sum = vecElementAdd<T>(sum, *reinterpret_cast<const uint4*>(win + static_cast<size_t>(r) * TileBytes + c));
    }
    *reinterpret_cast<uint4*>(win + c) = sum;
  }
  asm volatile("s_wait_dscnt 0x0" ::: "memory");
}
#endif

// One-shot: stage sendbuff into our scratch, then reduce tile t of every rank's scratch into recvbuff. Item t
// stages and reduces the same tile, so the per-block barrier pairs each peer's write with this rank's read.
template <typename T, int TileBytes>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceFlatFabricTdm(uint8_t* const* __restrict__ ipcbuffs, uint8_t* __restrict__ recvbuff,
                                            size_t bytes, const uint8_t* __restrict__ sendbuff, int selfRank,
                                            int nRanks, FabricGpuBarrier barrier) {
  static_assert(TileBytes % kDdaArTdmAlign == 0, "tiles must be whole 256 B rows");
#if TDM_SUPPORTED
  const uint32_t flag = barrier.beginEpoch();

  extern __shared__ __attribute__((aligned(128))) uint8_t ddaArTdmSmem[];
  // TDM descriptors live in SGPRs, so every address below must be wave-uniform.
  const int lw = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  uint8_t* win = ddaArTdmSmem + lw * kDdaArTdmWindowBytes;

  const size_t nTiles = (bytes + TileBytes - 1) / TileBytes;
  const size_t warpsPerBlock = blockDim.x / warpSize;
  const size_t firstTile = blockIdx.x * warpsPerBlock + lw;
  const size_t tileStride = gridDim.x * warpsPerBlock;
  uint8_t* selfScratch = ipcbuffs[selfRank];

  for (size_t t = firstTile; t < nTiles; t += tileStride) {
    const size_t off = t * TileBytes;
    const size_t n = bytes - off < TileBytes ? bytes - off : TileBytes;
    tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaArTdmPolicy, /*Aligned=*/true>(sendbuff + off, win, n);
    tdm::tdmWait();
    tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaArTdmPolicy, /*Aligned=*/true>(win, selfScratch + off, n);
    tdm::tdmWait();
  }

  ddaFabricWaitMemOps();
  // Publish the staged scratch; the reads below go through TDM, which bypasses the cache, so no acquire.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

  for (size_t t = firstTile; t < nTiles; t += tileStride) {
    const size_t off = t * TileBytes;
    const size_t n = bytes - off < TileBytes ? bytes - off : TileBytes;
    for (int r = 0; r < nRanks; ++r) {
      tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaArTdmPolicy, /*Aligned=*/true>(
        ipcbuffs[r] + off, win + static_cast<size_t>(r) * TileBytes, n);
    }
    tdm::tdmWait();
    ddaArTdmReduceWindow<T, TileBytes>(win, nRanks, n);
    tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaArTdmPolicy, /*Aligned=*/true>(win, recvbuff + off, n);
    tdm::tdmWait();
  }

  // Peers must not overwrite their scratch until every rank has read it; nothing is published here.
  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */>(flag + 2);
  barrier.endEpoch(flag + 2);
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)ipcbuffs, (void)recvbuff, (void)bytes, (void)sendbuff, (void)selfRank, (void)nRanks, (void)barrier;
  __builtin_trap();
#endif
}

// Two-shot over shards of shardBytes = bytes / nRanks. Item t is tile t of a shard in every phase: staging
// writes tile t of every shard into our scratch, which is what peer r's item t reduces from shard r; the
// reduced tile t of our shard is what every peer's item t gathers.
template <typename T, int TileBytes>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceTreeFabricTdm(uint8_t* const* __restrict__ ipcbuffs, uint8_t* __restrict__ recvbuff,
                                            size_t bytes, const uint8_t* __restrict__ sendbuff, int selfRank,
                                            int nRanks, FabricGpuBarrier barrier) {
  static_assert(TileBytes % kDdaArTdmAlign == 0, "tiles must be whole 256 B rows");
#if TDM_SUPPORTED
  const uint32_t flag = barrier.beginEpoch();

  extern __shared__ __attribute__((aligned(128))) uint8_t ddaArTdmSmem[];
  // TDM descriptors live in SGPRs, so every address below must be wave-uniform.
  const int lw = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / warpSize));
  uint8_t* win = ddaArTdmSmem + lw * kDdaArTdmWindowBytes;

  const size_t shardBytes = bytes / nRanks;
  const size_t nTiles = (shardBytes + TileBytes - 1) / TileBytes;
  const size_t warpsPerBlock = blockDim.x / warpSize;
  const size_t firstTile = blockIdx.x * warpsPerBlock + lw;
  const size_t tileStride = gridDim.x * warpsPerBlock;
  uint8_t* selfScratch = ipcbuffs[selfRank];
  const size_t selfShard = static_cast<size_t>(selfRank) * shardBytes;

  // The previous op's closing barrier guarantees no peer is still reading our scratch.
  for (size_t t = firstTile; t < nTiles; t += tileStride) {
    const size_t off = t * TileBytes;
    const size_t n = shardBytes - off < TileBytes ? shardBytes - off : TileBytes;
    for (int s = 0; s < nRanks; ++s) {
      tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaArTdmPolicy, /*Aligned=*/true>(
        sendbuff + static_cast<size_t>(s) * shardBytes + off, win + static_cast<size_t>(s) * TileBytes, n);
    }
    tdm::tdmWait();
    for (int s = 0; s < nRanks; ++s) {
      tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaArTdmPolicy, /*Aligned=*/true>(
        win + static_cast<size_t>(s) * TileBytes, selfScratch + static_cast<size_t>(s) * shardBytes + off, n);
    }
    tdm::tdmWait();
  }

  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

  // Reduce our shard into our own scratch. Peers only read their own shard of our scratch in this phase.
  for (size_t t = firstTile; t < nTiles; t += tileStride) {
    const size_t off = t * TileBytes;
    const size_t n = shardBytes - off < TileBytes ? shardBytes - off : TileBytes;
    for (int r = 0; r < nRanks; ++r) {
      tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaArTdmPolicy, /*Aligned=*/true>(
        ipcbuffs[r] + selfShard + off, win + static_cast<size_t>(r) * TileBytes, n);
    }
    tdm::tdmWait();
    ddaArTdmReduceWindow<T, TileBytes>(win, nRanks, n);
    tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaArTdmPolicy, /*Aligned=*/true>(win, selfScratch + selfShard + off,
                                                                                 n);
    tdm::tdmWait();
  }

  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 2);

  for (size_t t = firstTile; t < nTiles; t += tileStride) {
    const size_t off = t * TileBytes;
    const size_t n = shardBytes - off < TileBytes ? shardBytes - off : TileBytes;
    for (int r = 0; r < nRanks; ++r) {
      tdm::asyncLoadToLDS<SyncPolicy::Async, kDdaArTdmPolicy, /*Aligned=*/true>(
        ipcbuffs[r] + static_cast<size_t>(r) * shardBytes + off, win + static_cast<size_t>(r) * TileBytes, n);
    }
    tdm::tdmWait();
    for (int r = 0; r < nRanks; ++r) {
      tdm::asyncStoreFromLDS<SyncPolicy::Async, kDdaArTdmPolicy, /*Aligned=*/true>(
        win + static_cast<size_t>(r) * TileBytes, recvbuff + static_cast<size_t>(r) * shardBytes + off, n);
    }
    tdm::tdmWait();
  }

  // Peers must not overwrite their scratch until every rank has read it; nothing is published here.
  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */>(flag + 3);
  barrier.endEpoch(flag + 3);
#else
  // Only dispatched on gfx1250; every other arch still has to compile this kernel.
  (void)ipcbuffs, (void)recvbuff, (void)bytes, (void)sendbuff, (void)selfRank, (void)nRanks, (void)barrier;
  __builtin_trap();
#endif
}

} // namespace dda::common
