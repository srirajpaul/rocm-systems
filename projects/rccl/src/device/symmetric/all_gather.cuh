/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#if defined(__HIP_PLATFORM_AMD__)
#include "symmetric/kernel.h"
#include "symmetric/primitives.h"
#else
#include "kernel.cuh"
#include "primitives.cuh"
#endif

// TileAligned: the caller's tier guarantees every tile address is ncclSymkTileLine-aligned, which
// lets the staging wrappers drop TDM's head peel.
template <int BytePerPack, int UnrollPacks, int UnrollPeers, bool EnableTma, bool TileAligned = false>
static __device__ void bcastDeep(ncclSymkArgsHandler const& handler, int tn, int t, bool waitNeeded,
                                 ncclLsaBarrierSession<ncclCoopCta>& bar, ncclSymPtr<char> input,
                                 ncclSymPtr<char> output, bool inPlace, int nIters) {
  using Pack = BytePack<BytePerPack>;
  int wn = tn / WARP_SIZE;
  int w = t / WARP_SIZE;
  int lane = t % WARP_SIZE;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  Pack* inpPacks = (Pack*)input.localPtr() + intptr_t(w) * UnrollPacks * WARP_SIZE +
                   (
#if NCCL_SYMK_ASYNC_TILE
                     EnableTma ? 0 :
#endif
                                 lane);

  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w) * UnrollPacks * WARP_SIZE +
                              (
#if NCCL_SYMK_ASYNC_TILE
                                EnableTma ? 0 :
#endif
                                            lane);

  Pack tmp[UnrollPacks];

#if NCCL_SYMK_ASYNC_TILE
  int lw = threadIdx.x / WARP_SIZE;
  using tmaSmemStruct_t = tmaSmemStruct<Pack, UnrollPacks>;
  tmaSmemStruct_t* tmaSmem = ncclSymkTileSmem<tmaSmemStruct_t>(lw);
  constexpr size_t tileSize = UnrollPacks * WARP_SIZE * BytePerPack;
#endif
  bool skip = false; // all lanes issue loads/stores

#if NCCL_SYMK_ASYNC_TILE
  size_t pending = 0;
  if NCCL_IF_CONSTEXPR (EnableTma) {
    ncclSymkTileBarInit(&tmaSmem->bar, /*arrivers=*/1, lane);
    // TMA issues from lane 0, so the rest of the warp has nothing left to do.
    // TDM needs the whole warp to build the descriptor, so every lane stays in.
    skip = NCCL_SYMK_TILE_TMA && lane != 0;
  }
#endif

  nIters -= w;
  if (0 < nIters) {
#if NCCL_SYMK_ASYNC_TILE
    if NCCL_IF_CONSTEXPR (EnableTma) {
      ncclSymkTileLoad<TileAligned>(tmaSmem->buff[0], inpPacks, tileSize, tmaSmem->bar, pending, lane);
      ncclSymkTileLoadWait</*Arrivers=*/1>(tmaSmem->bar, pending, lane);
    } else
#endif
    {
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < UnrollPacks; u++) {
        tmp[u] = inpPacks[u * WARP_SIZE];
      }
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);

  if (0 < nIters) {
    while (true) {
      int dr = inPlace ? 1 : 0;
      int r = rank + dr;
      if (r == nRanks) r = 0;
      NVCC_PRAGMA_UNROLL(2)
      for (int partial = 0; partial <= 1 && !skip; partial++) {
        NVCC_PRAGMA_UNROLL_DISABLED
        for (int i = 0; partial ? i < 1 : (dr + UnrollPeers <= nRanks); partial ? i++ : (dr += UnrollPeers)) {
          NVCC_PRAGMA_UNROLL_AUTO
          for (int ur = 0; ur < UnrollPeers - partial; ur++) {
            if (partial && dr + ur == nRanks) break;
#if NCCL_SYMK_ASYNC_TILE
            if NCCL_IF_CONSTEXPR (EnableTma) {
              ncclSymkTileStore<TileAligned>(outPacks.lsaPtr(r), tmaSmem->buff[0], tileSize, lane);
            } else
#endif
            {
              NVCC_PRAGMA_UNROLL(UnrollPacks)
              for (int u = 0; u < UnrollPacks; u++) {
                outPacks.lsaPtr(r)[u * WARP_SIZE] = tmp[u];
              }
            }
            if (++r == nRanks) r = 0;
          }
#if NCCL_SYMK_ASYNC_TILE
          if NCCL_IF_CONSTEXPR (EnableTma) {
            ncclSymkTileStoreWait(lane);
          }
#endif
        }
      }
      inpPacks += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      outPacks += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;
#if NCCL_SYMK_ASYNC_TILE
      if NCCL_IF_CONSTEXPR (EnableTma) {
        ncclSymkTileLoad<TileAligned>(tmaSmem->buff[0], inpPacks, tileSize, tmaSmem->bar, pending, lane);
        ncclSymkTileLoadWait</*Arrivers=*/1>(tmaSmem->bar, pending, lane);
      } else
#endif
      {
        NVCC_PRAGMA_UNROLL_AUTO
        for (int u = 0; u < UnrollPacks; u++) {
          tmp[u] = inpPacks[u * WARP_SIZE];
        }
      }
    }
  }
}

template <int UnrollPeers, typename T>
static __device__ void bcastEnds(ncclSymkArgsHandler const& handler, int tn, int t, ncclSymPtr<T> input,
                                 ncclSymPtr<T> output, bool inPlace, size_t nElts, uint32_t nPreElts, size_t nSufElts) {
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  BytePack<sizeof(T)>* inpPacks = (BytePack<sizeof(T)>*)input.localPtr();
  ncclSymPtr<BytePack<sizeof(T)>> outPacks = (ncclSymPtr<BytePack<sizeof(T)>>)output;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t i = t; i < nPreElts + nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts - nPreElts - nSufElts + i;
    BytePack<sizeof(T)> tmp = inpPacks[elt];
    int dr = inPlace ? 1 : 0;
    int r = rank + dr;
    if (r == nRanks) r = 0;
    NVCC_PRAGMA_UNROLL_DISABLED
    for (; dr + UnrollPeers <= nRanks; dr += UnrollPeers) {
      NVCC_PRAGMA_UNROLL(UnrollPeers)
      for (int u = 0; u < UnrollPeers; u++) {
        outPacks.lsaPtr(r)[elt] = tmp;
        if (++r == nRanks) r = 0;
      }
    }
    NVCC_PRAGMA_UNROLL(UnrollPeers)
    for (int u = 0; u < UnrollPeers; u++) {
      if (dr + u == nRanks) break;
      outPacks.lsaPtr(r)[elt] = tmp;
      if (++r == nRanks) r = 0;
    }
  }
}

template <typename T, bool EnableTma>
static __device__ void bcast(ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks, bool waitNeeded,
                             ncclLsaBarrierSession<ncclCoopCta>& bar, ncclSymPtr<T> input, ncclSymPtr<T> output,
                             size_t nElts) {
  bool inPlace = (input == output);
  size_t nBytes = nElts * sizeof(T);

#if defined(__gfx950__)
  // Engage on a floor instead of trimming, so the partial final wave is kept rather than handed to
  // the per-byte tail. The floor is the old trim modulus, so the deep path engages where it did.
  uint32_t const chunkFloor = uint32_t(nBlocks);
#else
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);
  uint32_t const chunkFloor = 1;
#endif

  uint32_t alignment = uint32_t(input.offset - output.offset);
  uint32_t nPreBytes =
#if NCCL_SYMK_ASYNC_TILE
    (EnableTma && alignment % 256 == 0) ? (256 - input.offset) % 256 :
#endif
                                          (16 - input.offset) % 16;

  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

#if NCCL_SYMK_ASYNC_TILE
  if NCCL_IF_CONSTEXPR (EnableTma) {
    if (alignment % 256 == 0) {
      constexpr int BytePerPack = ncclSymkBytePerPack, UnrollPacks = ncclSymkAlign256BDeepUnrollPacks, UnrollPeers = 2;
      constexpr int BytePerChunk = ncclSymkAlign256BDeepBytePerChunk;
      uint32_t chunks = (nBytes - cursor) / BytePerChunk;
      chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
      if (chunks != 0) {
        uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
        // Both sides start on a 256 B boundary here (nPreBytes peels input to one, alignment % 256
        // carries output with it, window bases are page-aligned) and every tile advances by a
        // multiple of 256, so the tiles are ncclSymkTileLine-aligned and the peel can go.
        bcastDeep<BytePerPack, UnrollPacks, UnrollPeers, EnableTma, /*TileAligned=*/true>(
          handler, tn, t, waitNeeded, bar, (ncclSymPtr<char>)input + cursor,
          (ncclSymPtr<char>)output + cursor, inPlace, chunks * ncclSymkMinWarpsPerBlock);
        cursor = cursorAfter;
        waitNeeded = false;
      }
    }
  }
#endif

  if (alignment % 16 == 0) {
#if defined(__gfx950__)
    // Dropping to one pack cuts BytePerChunk to a quarter, so mid sizes reach this tier's floor instead
    // of falling to the 4-byte and per-byte paths. One pack per peer lets UnrollPeers batch them all.
    constexpr int UnrollPacks = 1, UnrollPeers = 8;
#else
    constexpr int UnrollPacks = ncclSymkUnrollPacks, UnrollPeers = 2;
#endif
    constexpr int BytePerPack = ncclSymkBytePerPack;
    constexpr int BytePerChunk = ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, UnrollPacks);
    uint32_t chunks = (nBytes - cursor) / BytePerChunk;
#if !defined(__gfx950__)
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
#endif
    if (chunks >= chunkFloor) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
      bcastDeep<BytePerPack, UnrollPacks, UnrollPeers, EnableTma>(handler, tn, t, waitNeeded, bar,
                                                                  (ncclSymPtr<char>)input + cursor,
                                                                  (ncclSymPtr<char>)output + cursor, inPlace,
                                                                  chunks * ncclSymkMinWarpsPerBlock);
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && alignment % 4 == 0)) {
#if defined(__gfx950__)
    // Only reached by 16-byte misaligned buffers, since the tier above shares this chunk size.
    constexpr int UnrollPeers = 8;
#else
    constexpr int UnrollPeers = 4;
#endif
    constexpr int BytePerPack = 4, UnrollPacks = 4;
    constexpr int BytePerChunk = ncclSymkMinWarpsPerBlock * UnrollPacks * WARP_SIZE * BytePerPack;
    uint32_t chunks = (nBytes - cursor) / BytePerChunk;
#if !defined(__gfx950__)
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
#endif
    if (chunks >= chunkFloor) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
      bcastDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, false>(
        handler, tn, t, waitNeeded, bar, (ncclSymPtr<char>)input + cursor, (ncclSymPtr<char>)output + cursor, inPlace,
        chunks * ncclSymkMinWarpsPerBlock);
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes - cursor) / sizeof(T);
  bcastEnds<UnrollPeers>(handler, tn, t, input, output, inPlace, nElts, nPreBytes / sizeof(T), nSufElts);
}

template <bool EnableProfiler, bool EnableTma>
__device__ __forceinline__ void ncclSymkRun_AllGather_ST_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};
  int const& rank = handler.comm.rank;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);
  if NCCL_IF_CONSTEXPR (EnableProfiler) {
    // Finish the opening barrier here so AFTER_OPEN marks the end of the peer sync.
    // Same barrier ops as the default variant (which fuses the wait into bcast), so
    // the two stay barrier-compatible when peers disagree on profiling.
    bar.wait(ncclCoopCta(), cuda::memory_order_acquire);
    ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);
  }

  bool waitNeeded = !EnableProfiler;
  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
        // Threads numbered over rank.
    int bt =
      flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, block, nBlocks, threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int btn = nBlocks * blockDim.x;
    bcast<char, EnableTma>(handler, btn, bt, nBlocks, waitNeeded, bar, input, output + rank * nAllElts, nElts);
    waitNeeded = false;
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_ST(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_ST_impl<EnableProfiler, /*EnableTma=*/false>(args);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_TmaST(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_ST_impl<EnableProfiler, /*EnableTma=*/true>(args);
}

#if defined(__HIP_PLATFORM_AMD__)
// Global-aperture pointer, so accesses compile to global_* rather than flat_* instructions. Flat accesses
// also count against lgkmcnt, so any wait on one drains every load still in flight.
template <typename T>
using ncclSymkGlobalPtr = __attribute__((address_space(1))) T*;
// Plain scalar and vector types: BytePack's copy/assignment only bind generic-address-space references,
// so it cannot be loaded or stored through a ncclSymkGlobalPtr.
template <int Bytes>
struct ncclSymkGatherWord;
template <>
struct ncclSymkGatherWord<1> {
  using Type = uint8_t;
};
template <>
struct ncclSymkGatherWord<4> {
  using Type = uint32_t;
};
template <>
struct ncclSymkGatherWord<16> {
  using Type = v4u;
};
#else
template <typename T>
using ncclSymkGlobalPtr = T*;
template <int Bytes>
struct ncclSymkGatherWord {
  using Type = BytePack<Bytes>;
};
#endif

// Resolves the bases of up to UnrollPeers peers starting dr past our rank: src[u] is where peer r's
// contribution sits in its own window and dst[u] is its slot in our output. Done once per batch, ahead of
// any store: the window lookups behind lsaPtr()/localPtr() are plain loads that the compiler cannot move
// past our stores, so left inside a copy loop they serialize every access behind a full wait.
template <int UnrollPeers, typename T>
static __device__ __forceinline__ int gatherPeerBases(ncclSymkArgsHandler const& handler, int dr,
                                                      ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nAllElts,
                                                      bool inPlace, ncclSymkGlobalPtr<char> (&src)[UnrollPeers],
                                                      ncclSymkGlobalPtr<char> (&dst)[UnrollPeers]) {
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  int nPeers = min(UnrollPeers, nRanks - dr);
  T* out = output.localPtr();
  NVCC_PRAGMA_UNROLL(UnrollPeers)
  for (int u = 0; u < UnrollPeers; u++) {
    int r = rank + dr + (u < nPeers ? u : 0);
    if (r >= nRanks) r -= nRanks;
    // In-place send buffers sit at each rank's own slot, so an in-place peer's contribution
    // is read out of its output rather than from our input's offset.
    src[u] = (ncclSymkGlobalPtr<char>)(inPlace ? output + r * nAllElts : input).lsaPtr(r);
    dst[u] = (ncclSymkGlobalPtr<char>)(out + r * nAllElts);
  }
  return nPeers;
}

// Copies nIters warp tiles from each of the first nPeers src[] bases to the matching dst[]. Dispatches down
// to NPeers == nPeers so the copy loop has a compile-time peer count: runtime per-peer guards split it into
// one block per access, which serializes them.
template <int BytePerPack, int UnrollPacks, int NPeers, int UnrollPeers>
static __device__ __forceinline__ void gatherDeepPeers(int nPeers, ncclSymkGlobalPtr<char> const (&src)[UnrollPeers],
                                                       ncclSymkGlobalPtr<char> const (&dst)[UnrollPeers], int tn,
                                                       int t, int nIters) {
  if NCCL_IF_CONSTEXPR (NPeers > 1) {
    if (nPeers < NPeers) {
      gatherDeepPeers<BytePerPack, UnrollPacks, NPeers - 1, UnrollPeers>(nPeers, src, dst, tn, t, nIters);
      return;
    }
  }
  using Word = typename ncclSymkGatherWord<BytePerPack>::Type;
  constexpr size_t TileBytes = size_t(UnrollPacks) * WARP_SIZE * BytePerPack;
  int wn = tn / WARP_SIZE;
  int w = t / WARP_SIZE;
  int lane = t % WARP_SIZE;
  // One per-thread byte cursor added to fixed bases; stepping each peer's pointer instead lets the compiler
  // advance every base separately, roughly doubling the address arithmetic per iteration.
  size_t cur = size_t(w) * TileBytes + size_t(lane) * BytePerPack;
  size_t const step = size_t(wn) * TileBytes;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int i = w; i < nIters; i += wn, cur += step) {
    Word tmp[NPeers][UnrollPacks];
    NVCC_PRAGMA_UNROLL(NPeers)
    for (int u = 0; u < NPeers; u++) {
      NVCC_PRAGMA_UNROLL(UnrollPacks)
      for (int p = 0; p < UnrollPacks; p++) {
        tmp[u][p] = *(ncclSymkGlobalPtr<Word>)(src[u] + cur + p * WARP_SIZE * BytePerPack);
      }
    }
    NVCC_PRAGMA_UNROLL(NPeers)
    for (int u = 0; u < NPeers; u++) {
      NVCC_PRAGMA_UNROLL(UnrollPacks)
      for (int p = 0; p < UnrollPacks; p++) {
        *(ncclSymkGlobalPtr<Word>)(dst[u] + cur + p * WARP_SIZE * BytePerPack) = tmp[u][p];
      }
    }
  }
}

#if NCCL_SYMK_ASYNC_TILE
// Staged counterpart of gatherDeepPeers: each warp pulls its tile from up to UnrollPeers peers into LDS
// through the tile engine, then writes them out to our own output.
template <int BytePerPack, int UnrollPacks, int UnrollPeers, bool TileAligned>
static __device__ void gatherDeepTile(ncclSymkArgsHandler const& handler, int tn, int t, ncclSymPtr<char> input,
                                      ncclSymPtr<char> output, size_t nAllBytes, bool inPlace, int nIters) {
  using Pack = BytePack<BytePerPack>;
  int wn = tn / WARP_SIZE;
  int w = t / WARP_SIZE;
  int lane = t % WARP_SIZE;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  size_t nAllPacks = nAllBytes / BytePerPack;

  ncclSymPtr<Pack> inpPacks = (ncclSymPtr<Pack>)input + intptr_t(w) * UnrollPacks * WARP_SIZE;
  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w) * UnrollPacks * WARP_SIZE;

  int lw = threadIdx.x / WARP_SIZE;
  using tmaSmemStruct_t = tmaSmemStruct<Pack, UnrollPacks, UnrollPeers>;
  tmaSmemStruct_t* tmaSmem = ncclSymkTileSmem<tmaSmemStruct_t>(lw);
  constexpr size_t tileSize = UnrollPacks * WARP_SIZE * BytePerPack;
  size_t pending = 0;
  ncclSymkTileBarInit(&tmaSmem->bar, /*arrivers=*/1, lane);
  // TMA issues from lane 0, so the rest of the warp has nothing left to do.
  // TDM needs the whole warp to build the descriptor, so every lane stays in.
  bool skip = NCCL_SYMK_TILE_TMA && lane != 0;

  nIters -= w;
  if (0 < nIters) {
    while (true) {
      int dr = inPlace ? 1 : 0;
      int r = rank + dr;
      if (r == nRanks) r = 0;
      NVCC_PRAGMA_UNROLL(2)
      for (int partial = 0; partial <= 1 && !skip; partial++) {
        NVCC_PRAGMA_UNROLL_DISABLED
        for (int i = 0; partial ? i < 1 : (dr + UnrollPeers <= nRanks); partial ? i++ : (dr += UnrollPeers)) {
          if (partial && dr == nRanks) break;
          int rBatch = r;
          NVCC_PRAGMA_UNROLL_AUTO
          for (int ur = 0; ur < UnrollPeers - partial; ur++) {
            if (partial && ur != 0 && dr + ur == nRanks) break;
            ncclSymPtr<Pack> srcPacks = inPlace ? outPacks + r * nAllPacks : inpPacks;
            ncclSymkTileLoad<TileAligned>(tmaSmem->buff[ur], srcPacks.lsaPtr(r), tileSize, tmaSmem->bar, pending,
                                          lane);
            if (++r == nRanks) r = 0;
          }
          ncclSymkTileLoadWait</*Arrivers=*/1>(tmaSmem->bar, pending, lane);
          r = rBatch;
          NVCC_PRAGMA_UNROLL_AUTO
          for (int ur = 0; ur < UnrollPeers - partial; ur++) {
            if (partial && ur != 0 && dr + ur == nRanks) break;
            ncclSymkTileStore<TileAligned>(outPacks.localPtr() + r * nAllPacks, tmaSmem->buff[ur], tileSize, lane);
            if (++r == nRanks) r = 0;
          }
          ncclSymkTileStoreWait(lane);
        }
      }
      inpPacks += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      outPacks += intptr_t(wn) * UnrollPacks * WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;
    }
  }
}
#endif

// Pull counterpart of bcastDeep: each warp loads its tile out of every peer's input and stores it into
// our own output at that peer's slot, so all remote traffic is loads. `output` addresses peer 0's slot;
// peer r's slot is nAllBytes further along.
template <int BytePerPack, int UnrollPacks, int UnrollPeers, bool EnableTma, bool TileAligned = false>
static __device__ void gatherDeep(ncclSymkArgsHandler const& handler, int tn, int t, bool waitNeeded,
                                  ncclLsaBarrierSession<ncclCoopCta>& bar, ncclSymPtr<char> input,
                                  ncclSymPtr<char> output, size_t nAllBytes, bool inPlace, int nIters) {
  // Unlike bcastDeep there is no tile to prefetch ahead of the barrier: every load reads a peer.
  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);
#if NCCL_SYMK_ASYNC_TILE
  if NCCL_IF_CONSTEXPR (EnableTma) {
    gatherDeepTile<BytePerPack, UnrollPacks, UnrollPeers, TileAligned>(handler, tn, t, input, output, nAllBytes,
                                                                      inPlace, nIters);
  } else
#endif
  {
    int const& nRanks = handler.comm.nRanks;
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int dr = inPlace ? 1 : 0; dr < nRanks; dr += UnrollPeers) {
      ncclSymkGlobalPtr<char> src[UnrollPeers];
      ncclSymkGlobalPtr<char> dst[UnrollPeers];
      int nPeers = gatherPeerBases<UnrollPeers>(handler, dr, input, output, nAllBytes, inPlace, src, dst);
      gatherDeepPeers<BytePerPack, UnrollPacks, UnrollPeers, UnrollPeers>(nPeers, src, dst, tn, t, nIters);
    }
  }
}

// Per-element counterpart of gatherDeepPeers for the unaligned head and the tail.
template <typename Word, int NPeers, int UnrollPeers>
static __device__ __forceinline__ void gatherEndsPeers(int nPeers, ncclSymkGlobalPtr<char> const (&src)[UnrollPeers],
                                                       ncclSymkGlobalPtr<char> const (&dst)[UnrollPeers], int tn,
                                                       int t, size_t nElts, uint32_t nPreElts, size_t nSufElts) {
  if NCCL_IF_CONSTEXPR (NPeers > 1) {
    if (nPeers < NPeers) {
      gatherEndsPeers<Word, NPeers - 1, UnrollPeers>(nPeers, src, dst, tn, t, nElts, nPreElts, nSufElts);
      return;
    }
  }
  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t i = t; i < nPreElts + nSufElts; i += tn) {
    size_t cur = (i < nPreElts ? i : nElts - nPreElts - nSufElts + i) * sizeof(Word);
    Word tmp[NPeers];
    NVCC_PRAGMA_UNROLL(NPeers)
    for (int u = 0; u < NPeers; u++) tmp[u] = *(ncclSymkGlobalPtr<Word>)(src[u] + cur);
    NVCC_PRAGMA_UNROLL(NPeers)
    for (int u = 0; u < NPeers; u++) *(ncclSymkGlobalPtr<Word>)(dst[u] + cur) = tmp[u];
  }
}

template <int UnrollPeers, typename T>
static __device__ void gatherEnds(ncclSymkArgsHandler const& handler, int tn, int t, ncclSymPtr<T> input,
                                  ncclSymPtr<T> output, bool inPlace, size_t nElts, size_t nAllElts, uint32_t nPreElts,
                                  size_t nSufElts) {
  using Word = typename ncclSymkGatherWord<sizeof(T)>::Type;
  int const& nRanks = handler.comm.nRanks;
  if (nPreElts + nSufElts == 0) return;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int dr = inPlace ? 1 : 0; dr < nRanks; dr += UnrollPeers) {
    ncclSymkGlobalPtr<char> src[UnrollPeers];
    ncclSymkGlobalPtr<char> dst[UnrollPeers];
    int nPeers = gatherPeerBases<UnrollPeers>(handler, dr, input, output, nAllElts, inPlace, src, dst);
    gatherEndsPeers<Word, UnrollPeers, UnrollPeers>(nPeers, src, dst, tn, t, nElts, nPreElts, nSufElts);
  }
}

// `output` addresses peer 0's slot, as in gatherDeep.
template <typename T, bool EnableTma>
static __device__ void gather(ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks, bool waitNeeded,
                              ncclLsaBarrierSession<ncclCoopCta>& bar, ncclSymPtr<T> input, ncclSymPtr<T> output,
                              size_t nElts, size_t nAllElts) {
  int const& rank = handler.comm.rank;
  bool inPlace = (input == output + rank * nAllElts);
  size_t nBytes = nElts * sizeof(T);
  size_t nAllBytes = nAllElts * sizeof(T);

#if defined(__gfx950__)
  // Unlike bcast, engage the deep tiers on any whole chunk rather than on one per block: the per-element
  // tail costs a remote load per byte per peer, far more than leaving some warps idle for a pass.
  uint32_t const chunkFloor = 1;
#else
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);
  uint32_t const chunkFloor = 1;
#endif

  // Loads land at our input's offset on every peer but stores land nAllBytes apart, one slot per peer,
  // so each tier also needs nAllBytes aligned for its test on slot 0 to cover every slot.
  uint32_t alignment = uint32_t(input.offset - output.offset);
  uint32_t nPreBytes =
#if NCCL_SYMK_ASYNC_TILE
    (EnableTma && alignment % 256 == 0 && nAllBytes % 256 == 0) ? (256 - input.offset) % 256 :
#endif
                                                                 (16 - input.offset) % 16;

  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

#if NCCL_SYMK_ASYNC_TILE
  if NCCL_IF_CONSTEXPR (EnableTma) {
    if (alignment % 256 == 0 && nAllBytes % 256 == 0) {
      // One peer per batch: a 256 B-deep tile already fills most of the warp's LDS window.
      constexpr int BytePerPack = ncclSymkBytePerPack, UnrollPacks = ncclSymkAlign256BDeepUnrollPacks, UnrollPeers = 1;
      constexpr int BytePerChunk = ncclSymkAlign256BDeepBytePerChunk;
      uint32_t chunks = (nBytes - cursor) / BytePerChunk;
      chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
      if (chunks != 0) {
        uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
        // Same reasoning as bcast's 256 B tier; the nAllBytes test extends it to every output slot.
        gatherDeep<BytePerPack, UnrollPacks, UnrollPeers, EnableTma, /*TileAligned=*/true>(
          handler, tn, t, waitNeeded, bar, (ncclSymPtr<char>)input + cursor, (ncclSymPtr<char>)output + cursor,
          nAllBytes, inPlace, chunks * ncclSymkMinWarpsPerBlock);
        cursor = cursorAfter;
        waitNeeded = false;
      }
    }
  }
#endif

  if (alignment % 16 == 0 && nAllBytes % 16 == 0) {
#if defined(__gfx950__)
    // Same shape as bcast's gfx950 tier: one pack per peer, with up to 8 peers' loads issued per batch.
    constexpr int UnrollPacks = 1, UnrollPeers = 8;
#else
    constexpr int UnrollPacks = ncclSymkUnrollPacks, UnrollPeers = 2;
#endif
    constexpr int BytePerPack = ncclSymkBytePerPack;
    constexpr int BytePerChunk = ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, UnrollPacks);
    uint32_t chunks = (nBytes - cursor) / BytePerChunk;
#if !defined(__gfx950__)
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
#endif
    if (chunks >= chunkFloor) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
      gatherDeep<BytePerPack, UnrollPacks, UnrollPeers, EnableTma>(handler, tn, t, waitNeeded, bar,
                                                                   (ncclSymPtr<char>)input + cursor,
                                                                   (ncclSymPtr<char>)output + cursor, nAllBytes,
                                                                   inPlace, chunks * ncclSymkMinWarpsPerBlock);
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && alignment % 4 == 0 && nAllBytes % 4 == 0)) {
#if defined(__gfx950__)
    constexpr int UnrollPeers = 8;
#else
    constexpr int UnrollPeers = 4;
#endif
    constexpr int BytePerPack = 4, UnrollPacks = 4;
    constexpr int BytePerChunk = ncclSymkMinWarpsPerBlock * UnrollPacks * WARP_SIZE * BytePerPack;
    uint32_t chunks = (nBytes - cursor) / BytePerChunk;
#if !defined(__gfx950__)
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
#endif
    if (chunks >= chunkFloor) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks) * BytePerChunk;
      gatherDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, false>(
        handler, tn, t, waitNeeded, bar, (ncclSymPtr<char>)input + cursor, (ncclSymPtr<char>)output + cursor,
        nAllBytes, inPlace, chunks * ncclSymkMinWarpsPerBlock);
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_acquire);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes - cursor) / sizeof(T);
  gatherEnds<UnrollPeers>(handler, tn, t, input, output, inPlace, nElts, nAllElts, nPreBytes / sizeof(T), nSufElts);
}

template <bool EnableProfiler, bool EnableTma>
__device__ __forceinline__ void ncclSymkRun_AllGather_LD_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);
  if NCCL_IF_CONSTEXPR (EnableProfiler) {
    // Finish the opening barrier here so AFTER_OPEN marks the end of the peer sync.
    // Same barrier ops as the default variant (which fuses the wait into gather), so
    // the two stay barrier-compatible when peers disagree on profiling.
    bar.wait(ncclCoopCta(), cuda::memory_order_acquire);
    ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);
  }

  bool waitNeeded = !EnableProfiler;
  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
    // Block-contiguous numbering: each block's warps take adjacent tiles, so a block streams one
    // contiguous stretch per iteration.
    int bt = block * blockDim.x + threadIdx.x;
    int btn = nBlocks * blockDim.x;
    gather<char, EnableTma>(handler, btn, bt, nBlocks, waitNeeded, bar, input, output, nElts, nAllElts);
    waitNeeded = false;
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  // Every store was local, so unlike ST there is nothing to release to peers; the closing sync only
  // keeps each rank from reusing its input while peers may still be loading from it.
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LD(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_LD_impl<EnableProfiler, /*EnableTma=*/false>(args);
}

template <bool EnableProfiler, bool EnableTma>
__device__ __forceinline__ void ncclSymkRun_AllGather_STMC_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar(ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x, /*multimem=*/true);
  int const& rank = handler.comm.rank;

  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);

  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
        // Round robin memory to blocks.
    int t =
      flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, block, nBlocks, threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int tn = nBlocks * blockDim.x;
    bcastMultimem<char, EnableTma>(handler, tn, t, input, output + rank * nAllElts, nElts);
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_STMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_STMC_impl<EnableProfiler, /*EnableTma=*/false>(args);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_TmaSTMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_STMC_impl<EnableProfiler, /*EnableTma=*/true>(args);
}

template <bool EnableProfiler, typename EltType>
static __device__ void allgather_LL_body(ncclSymkDevWorkArgs const* args, ncclSymkArgsHandler& handler,
                                         ncclLLA2ASession<ncclCoopCta>& lla2a, EltType* input, EltType* output,
                                         int nElts, int nPacks, int nStrideElts) {
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8 / sizeof(EltType);
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  int t = threadIdx.x;
  // The round-downs below mask with -(Unroll * tn), so the launch width must be a power of two.
  int tn = blockDim.x;

  // LL fuses the peer sync into the first epoch, so AFTER_OPEN is stamped once, at the
  // first endEpoch below (see ncclDevProfilerPhases in device.h); BEGIN marks the start.
  [[maybe_unused]] bool profilerPhase1Done = false;
  NVCC_PRAGMA_UNROLL_DISABLED
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    if (t < nIterPacks) {
      Pack x = loadPack<Pack>(input, t * EltPerPack, nElts);
      lla2a.bcast(/*slot=*/nIterPacks * rank + t, x);
    }

    int tn_div_nPacks = tn / nIterPacks;
    int tn_mod_nPacks = tn % nIterPacks;
    int peer = t / nIterPacks;
    int pack = t % nIterPacks;
#if 1
    // NOTE: Unrolling speedup on eos nranks=8 size=64K: 5.7us vs 6.7us
    constexpr int Unroll = 4;
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = t; i < (nRanks * nIterPacks & -(Unroll * tn)); i += Unroll * tn) {
      Pack got[Unroll];
      lla2a.template recvUnrolled<Unroll, Unroll>(i, Unroll, tn, /*&*/ got);
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < Unroll; u++) {
        storePack<Pack>(output + peer * nStrideElts, pack * EltPerPack, nElts, got[u]);
        peer += tn_div_nPacks;
        pack += tn_mod_nPacks;
        if (nIterPacks <= pack) {
          peer += 1;
          pack -= nIterPacks;
        }
      }
    }

    int i = (nRanks * nIterPacks & -(Unroll * tn)) + t;
    int n = (nRanks * nIterPacks) / tn % Unroll;
    if (i + n * tn < nRanks * nIterPacks) n += 1;
    if (n != 0) {
      Pack got[Unroll];
      lla2a.template recvUnrolled<1, Unroll>(i, n, tn, /*&*/ got);
      NVCC_PRAGMA_UNROLL_AUTO
      for (int u = 0; u < Unroll; u++) {
        if (u != 0 && u == n) break;
        storePack(output + peer * nStrideElts, pack * EltPerPack, nElts, got[u]);
        peer += tn_div_nPacks;
        pack += tn_mod_nPacks;
        if (nIterPacks <= pack) {
          peer += 1;
          pack -= nIterPacks;
        }
      }
    }
#else
    // The non-unrolled but "obviously correct" implementation for reference.
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = t; i < nRanks * nIterPacks; i += tn) {
      Pack got = lla2a.template recv<Pack>(i);
      storePack(output + peer * nStrideElts, pack * EltPerPack, nElts, got);
      peer += tn_div_nPacks;
      pack += tn_mod_nPacks;
      if (nIterPacks <= pack) {
        peer += 1;
        pack -= nIterPacks;
      }
    }
#endif

    lla2a.endEpoch(ncclCoopCta());
    if NCCL_IF_CONSTEXPR (EnableProfiler) {
      if (!profilerPhase1Done) {
        ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);
        profilerPhase1Done = true;
      }
    }

    input += tn * EltPerPack;
    output += tn * EltPerPack;
    nElts -= tn * EltPerPack;
    nPacks -= tn;
  }
  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
}

template <bool EnableProfiler>
static __device__ void ncclSymkRun_AllGather_LL_impl(ncclSymkDevWorkArgs const* args, bool multimem) {
  ncclSymkArgsHandler handler{args};
  ncclLLA2ASession<ncclCoopCta> lla2a(ncclCoopCta(), handler.comm, ncclTeamLsa(handler.comm), handler.lsaLLA2A,
                                      blockIdx.x, /*maxElts=*/ncclSymkMaxThreads, multimem, handler.comm.lsaMultimem);

  using Pack = BytePack<8>;
  constexpr int BytePerPack = 8;

  handler.singleWork<char>([&] __device__(int nElts, int nAllElts, ncclSymPtr<char> input, ncclSymPtr<char> output) {
    int nPacks = divUp(nElts, BytePerPack);

    char* blockInput = input.localPtr();
    char* blockOutput = output.localPtr();

    uint32_t lowBits = nAllElts;
    lowBits |= (uintptr_t)blockInput;
    lowBits |= (uintptr_t)blockOutput;
    if (__builtin_expect(lowBits % 8 == 0, true)) {
          // NOTE: Specializing for 8-byte alignment in one case help at size=65K: 8.9us vs 5.6us
      allgather_LL_body<EnableProfiler>(args, handler, lla2a, (BytePack<8>*)blockInput, (BytePack<8>*)blockOutput,
                                        nElts / 8, nPacks, nAllElts / 8);
    } else {
      allgather_LL_body<EnableProfiler>(args, handler, lla2a, blockInput, blockOutput, nElts, nPacks, nAllElts);
    }
  });
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LL(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_LL_impl<EnableProfiler>(args, /*multimem=*/false);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LLMC(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_LL_impl<EnableProfiler>(args, /*multimem=*/true);
}
