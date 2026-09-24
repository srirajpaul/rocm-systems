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
            if (partial && dr == nRanks) break;
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
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);

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
    constexpr int BytePerPack = ncclSymkBytePerPack, UnrollPacks = ncclSymkUnrollPacks, UnrollPeers = 2;
    constexpr int BytePerChunk = ncclSymkBytePerChunk;
    uint32_t chunks = (nBytes - cursor) / BytePerChunk;
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
    if (chunks != 0) {
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
    constexpr int BytePerPack = 4, UnrollPacks = 4, UnrollPeers = 4;
    constexpr int BytePerChunk = ncclSymkMinWarpsPerBlock * UnrollPacks * WARP_SIZE * BytePerPack;
    uint32_t chunks = (nBytes - cursor) / BytePerChunk;
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
    if (chunks != 0) {
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

#if defined(__HIP_PLATFORM_AMD__)
// Global-aperture pointer, so accesses compile to global_* rather than flat_* instructions.
template <typename T>
using ncclSymkGlobalPtr = __attribute__((address_space(1))) T*;
#else
template <typename T>
using ncclSymkGlobalPtr = T*;
#endif

constexpr int ncclSymkGatherUnrollPeers = 8;

// Copies pack i of each of the first nPeers src[] to dst[]. Dispatches down to NPeers == nPeers so the
// copy loop has a compile-time peer count: runtime per-peer guards split it into one block per access,
// which serializes the stores.
template <int NPeers, typename Pack>
static __device__ __forceinline__ void gatherPeers(int nPeers,
                                                   ncclSymkGlobalPtr<Pack> const (&src)[ncclSymkGatherUnrollPeers],
                                                   ncclSymkGlobalPtr<Pack> const (&dst)[ncclSymkGatherUnrollPeers],
                                                   int tn, int t, size_t nPacks) {
  if NCCL_IF_CONSTEXPR (NPeers > 1) {
    if (nPeers < NPeers) {
      gatherPeers<NPeers - 1, Pack>(nPeers, src, dst, tn, t, nPacks);
      return;
    }
  }
  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t i = t; i < nPacks; i += tn) {
    Pack tmp[NPeers];
    NVCC_PRAGMA_UNROLL(NPeers)
    for (int u = 0; u < NPeers; u++) tmp[u] = src[u][i];
    NVCC_PRAGMA_UNROLL(NPeers)
    for (int u = 0; u < NPeers; u++) dst[u][i] = tmp[u];
  }
}

// Pull counterpart of bcast: load every peer's contribution out of its window and
// store it into our own output, so all remote traffic is reads.
// `output` addresses peer 0's slot; peer r's slot is `nAllPacks` further along.
template <typename Pack>
static __device__ void gatherPacks(int rank, int nRanks, bool inPlace, int tn, int t, ncclSymPtr<Pack> input,
                                   ncclSymPtr<Pack> output, size_t nAllPacks, size_t nPacks) {
  constexpr int UnrollPeers = ncclSymkGatherUnrollPeers;
  Pack* outPacks = output.localPtr();
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int dr = inPlace ? 1 : 0; dr < nRanks; dr += UnrollPeers) {
    int nPeers = min(UnrollPeers, nRanks - dr);
    ncclSymkGlobalPtr<Pack> src[UnrollPeers];
    ncclSymkGlobalPtr<Pack> dst[UnrollPeers];
    NVCC_PRAGMA_UNROLL(UnrollPeers)
    for (int u = 0; u < UnrollPeers; u++) {
      int r = rank + dr + (u < nPeers ? u : 0);
      if (r >= nRanks) r -= nRanks;
      // In-place send buffers sit at a different offset on every rank (their own slot),
      // so an in-place peer's contribution has to be read out of its output window.
      src[u] = (ncclSymkGlobalPtr<Pack>)(inPlace ? output + r * nAllPacks : input).lsaPtr(r);
      dst[u] = (ncclSymkGlobalPtr<Pack>)(outPacks + r * nAllPacks);
    }
    gatherPeers<UnrollPeers, Pack>(nPeers, src, dst, tn, t, nPacks);
  }
}

static __device__ void gather(ncclSymkArgsHandler const& handler, int tn, int t, ncclSymPtr<char> input,
                              ncclSymPtr<char> output, size_t nBytes, size_t nAllBytes) {
#if defined(__HIP_PLATFORM_AMD__)
  // A plain vector type: BytePack's copy/assignment only bind generic-address-space references,
  // so it cannot be loaded or stored through the global pointers in gatherPacks.
  using Pack = v4u;
#else
  using Pack = BytePack<ncclSymkBytePerPack>;
#endif
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  bool inPlace = (input == output + rank * nAllBytes);

  // Loads land at the same offset on every peer while stores land r*nAllBytes apart,
  // so the per-rank stride has to be packed too for one alignment test to cover all peers.
  uint32_t alignment = uint32_t(input.offset - output.offset);
  size_t nPreBytes = 0, nPacks = 0;
  if (alignment % sizeof(Pack) == 0 && nAllBytes % sizeof(Pack) == 0) {
    nPreBytes = min((size_t)((sizeof(Pack) - input.offset) % sizeof(Pack)), nBytes);
    nPacks = (nBytes - nPreBytes) / sizeof(Pack);
  }
  size_t cursor = nPreBytes + nPacks * sizeof(Pack);

  gatherPacks<char>(rank, nRanks, inPlace, tn, t, input, output, nAllBytes, nPreBytes);
  gatherPacks<Pack>(rank, nRanks, inPlace, tn, t, (ncclSymPtr<Pack>)(input + nPreBytes),
                    (ncclSymPtr<Pack>)(output + nPreBytes), nAllBytes / sizeof(Pack), nPacks);
  gatherPacks<char>(rank, nRanks, inPlace, tn, t, input + cursor, output + cursor, nAllBytes, nBytes - cursor);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LD(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};

  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);

  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
    int t =
      flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, block, nBlocks, threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int tn = nBlocks * blockDim.x;
    gather(handler, tn, t, input, output, nElts, nAllElts);
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}

#if 0
// symmetric-memory DDA all-gather kernel (IPC path)
template <typename T, int NRANKS, int inplace>
__device__ void gather_symm_dda(
    T* __restrict__ (&recvPtr)[NRANKS],
    size_t count,
    T* __restrict__ (&sendPtr)[NRANKS],
    int selfRank) {
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch.
  static_assert(sizeof(T) == 1);
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  T* __restrict__ (&sendPtr_use)[NRANKS] = inplace ? recvPtr : sendPtr;

  #if 1
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    v4u tmp[NRANKS];
    #pragma unroll NRANKS
    for (int r = inplace; r < NRANKS; ++r) {
      int peer = (selfRank + r) % NRANKS;
      size_t srcIdx = inplace * count * peer + idx;
      tmp[r] = *(v4u_gptr)(&sendPtr_use[peer][srcIdx]);
    }
    #pragma unroll NRANKS
    for (int r = inplace; r < NRANKS; ++r) {
      int peer = (selfRank + r) % NRANKS;
      size_t dstIdx = peer * count + idx;
      *(v4u_gptr)(&recvPtr[selfRank][dstIdx]) = tmp[r];
    }
  }
  #else
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    #pragma unroll NRANKS
    for (int r = inplace; r < NRANKS; ++r) {
        int peer = (selfRank + r) % NRANKS;
        size_t dstIdx = peer * count + idx;
        size_t srcIdx = inplace * count * peer + idx;
        *(v4u_gptr)(&recvPtr[selfRank][dstIdx]) = *(v4u_gptr)(&sendPtr_use[peer][srcIdx]);
    }
  }
  #endif
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LD(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};

  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);

  bool waitNeeded = true;
  handler.forEachWork<char>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                           ncclSymPtr<char> input, ncclSymPtr<char> output) {
        // Threads numbered over rank.
    int bt =
      flattenIx(threadIdx.x % WARP_SIZE, WARP_SIZE, block, nBlocks, threadIdx.x / WARP_SIZE, blockDim.x / WARP_SIZE);
    int btn = nBlocks * blockDim.x;

    int const& rank = handler.comm.rank;
    int const& nRanks = handler.comm.nRanks;
    const size_t count = nElts;
    const int inplace = input == output + count * rank;

    constexpr int NR = 8;
    char* __restrict__ recvPtr[NR];
    char* __restrict__ sendPtr[NR];

    #pragma unroll
    for (int i = 0; i < nRanks; i++) {
        recvPtr[i] = output.lsaPtr(i);
        sendPtr[i] = input.lsaPtr(i);
    }

    assert(nRanks == 8);
    if (inplace) {
      gather_symm_dda<char, NR, 1>(recvPtr, count, sendPtr, rank);
    }
    else {
      gather_symm_dda<char, NR, 0>(recvPtr, count, sendPtr, rank);
    }

    //gather<char>(handler, btn, bt, nBlocks, waitNeeded, bar, input, output, nElts, nAllElts);
  });

  if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
}
#endif

template <bool EnableProfiler, typename EltType>
static __device__ void allgather_LL_body(ncclSymkDevWorkArgs const* args, ncclSymkArgsHandler& handler,
                                         ncclLLA2ASession<ncclCoopCta>& lla2a, EltType* input, EltType* output,
                                         int nElts, int nPacks, int nStrideElts) {
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8 / sizeof(EltType);
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  int t = threadIdx.x;
  constexpr int tn = ncclSymkMaxThreads;

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
