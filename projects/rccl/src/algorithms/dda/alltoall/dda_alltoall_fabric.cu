/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/dda_alltoall.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/alltoall/alltoall_dda_fabric.h"
#include "algorithms/dda/alltoall/alltoall_dda_fabric_tdm.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "param.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

// Stage the fabric alltoall through LDS with the gfx1250 Tensor Data Mover.
// Opt-in while the path is being measured. Env: RCCL_DDA_A2A_TDM.
RCCL_PARAM(DdaA2ATdm, "DDA_A2A_TDM", 0);
// TDM tile size in KiB: 8 (x2 peers per warp, default), 16 (x1) or 4 (x4). Other values use 8.
RCCL_PARAM(DdaA2ATdmTileKB, "DDA_A2A_TDM_TILE_KB", 8);
// TDM warps per block before the block cap is reached (1..16). Fewer spreads the same warps over more CUs.
RCCL_PARAM(DdaA2ATdmWarps, "DDA_A2A_TDM_WARPS", 16);
// Push model: each rank writes its chunks into peers' scratch instead of reading from them. Env: RCCL_DDA_A2A_PUSH.
RCCL_PARAM(DdaA2APush, "DDA_A2A_PUSH", 0);
// Vector push/pull threads per block before the block cap is reached (one warp .. 1024). Fewer spreads the same
// threads over more CUs.
RCCL_PARAM(DdaA2AThreads, "DDA_A2A_THREADS", 256);
// Peer-major vector push: each block serves one peer (out of place only). Env: RCCL_DDA_A2A_PUSH_PEER.
RCCL_PARAM(DdaA2APushPeer, "DDA_A2A_PUSH_PEER", 0);
// Peer-major push threads per block (one warp .. 1024).
RCCL_PARAM(DdaA2APushPeerThreads, "DDA_A2A_PUSH_PEER_THREADS", 256);
// Peer-major push blocks per peer; 0 = as many as the block cap allows.
RCCL_PARAM(DdaA2APushPeerBlocks, "DDA_A2A_PUSH_PEER_BLOCKS", 0);// Peer-major push: each block's part of a peer chunk is rounded up to this many KiB (0 = 16 B lines only). Parts
// that start off such a boundary run the push up to 35% slower.
RCCL_PARAM(DdaA2APushPeerAlignKB, "DDA_A2A_PUSH_PEER_ALIGN_KB", 64);

namespace {

using nccl_dda_detail::DdaFabricBarrierState;
using dda::common::kDdaA2ATdmAlign;
using dda::common::kDdaA2ATdmMaxWarpsPerBlock;
using dda::common::kDdaA2ATdmWindowBytes;

// Vector push and pull kernels: one 16 B vector per thread per peer. Blocks are sized for the
// RCCL_DDA_A2A_THREADS target until the block cap is reached, then filled up to kDdaA2APushThreadsPerBlock.
// The kernels are instantiated for int8_t, so `bytes` is the per-rank chunk. Depends only on count, so every
// rank launches the same grid.
static inline std::pair<dim3, dim3> ddaAllToAllFabricVectorGeom(ncclComm* comm, size_t bytes) {
  const size_t vecs = dda::common::divRoundUp(bytes, sizeof(uint4));
  const int64_t t = rcclParamDdaA2AThreads();
  const uint32_t target = (t >= comm->WarpSize && t <= dda::common::kDdaA2APushThreadsPerBlock) ?
                            static_cast<uint32_t>(t) :
                            static_cast<uint32_t>(dda::common::kDdaA2APushThreadsPerBlock);
  const uint32_t blocks = dda::common::calcBlockCount(vecs, target, comm->ddaFabricMaxBlocks);
  const size_t warps = dda::common::divRoundUp(dda::common::divRoundUp(vecs, blocks), comm->WarpSize);
  const uint32_t threads =
    std::min<uint32_t>(dda::common::kDdaA2APushThreadsPerBlock, static_cast<uint32_t>(warps * comm->WarpSize));
  return std::make_pair(dim3(blocks, 1, 1), dim3(threads, 1, 1));
}

// Peer-major push grid: nRanks groups of blocks, one group per peer offset, within the barrier's block cap.
// Returns false when the cap cannot give every peer a block. Depends only on count, so every rank launches the
// same grid.
static inline bool ddaAllToAllFabricPushPeerGeom(ncclComm* comm, size_t bytes, dim3* grid, dim3* block) {
  const uint32_t maxPerPeer = static_cast<uint32_t>(comm->ddaFabricMaxBlocks) / comm->nRanks;
  if (maxPerPeer == 0) return false;
  const int64_t t = rcclParamDdaA2APushPeerThreads();
  const uint32_t threadsCap = (t >= comm->WarpSize && t <= dda::common::kDdaA2APushThreadsPerBlock) ?
                                static_cast<uint32_t>(t) :
                                static_cast<uint32_t>(dda::common::kDdaA2APushThreadsPerBlock);
  const size_t lines = dda::common::divRoundUp(bytes, sizeof(uint4));
  const int64_t b = rcclParamDdaA2APushPeerBlocks();
  uint32_t perPeer = (b >= 1 && b <= maxPerPeer) ? static_cast<uint32_t>(b) : maxPerPeer;
  perPeer = std::min<uint32_t>(perPeer, static_cast<uint32_t>(dda::common::divRoundUp(lines, threadsCap)));
  perPeer = std::max<uint32_t>(perPeer, 1);
  const size_t warps = dda::common::divRoundUp(dda::common::divRoundUp(lines, perPeer), comm->WarpSize);
  const uint32_t threads = std::min<uint32_t>(threadsCap, static_cast<uint32_t>(warps * comm->WarpSize));
  *grid = dim3(perPeer * comm->nRanks, 1, 1);
  *block = dim3(threads, 1, 1);
  return true;
}

// Units of unitBytes per part of a peer chunk split over perPeer blocks, rounded up to RCCL_DDA_A2A_PUSH_PEER_ALIGN_KB
// (and to whole units). The alignment is halved while rounding would leave more than a quarter of the blocks without
// a part (small messages).
static inline size_t ddaAllToAllFabricPushPeerPartUnits(size_t units, uint32_t perPeer, size_t unitBytes) {
  const int64_t alignKB = rcclParamDdaA2APushPeerAlignKB();
  const size_t alignBytes = alignKB > 0 ? static_cast<size_t>(alignKB) << 10 : unitBytes;
  size_t alignUnits = std::max<size_t>(1, alignBytes / unitBytes);
  const size_t even = dda::common::divRoundUp(units, perPeer);
  auto partUnits = [&]() { return dda::common::divRoundUp(even, alignUnits) * alignUnits; };
  while (alignUnits > 1 && 4 * dda::common::divRoundUp(units, partUnits()) < 3 * static_cast<size_t>(perPeer)) {
    alignUnits /= 2;
  }
  return partUnits();
}

static inline int ddaAllToAllFabricTdmTileBytes() {
  switch (rcclParamDdaA2ATdmTileKB()) {
  case 4:
    return 4 << 10;
  case 16:
    return 16 << 10;
  default:
    return 8 << 10;
  }
}

// Everything but recvbuff alignment, which the block-count report cannot see.
static bool ddaAllToAllFabricTdmEligible(ncclComm* comm, size_t bytesPerRank) {
  if (!rcclParamDdaA2ATdm()) return false;
  if (!TDM_TOOLCHAIN_AVAILABLE || comm->archName == nullptr || !IsArchMatch(comm->archName, "gfx1250")) return false;
  // The kernel is bounded to 512 threads, i.e. 16 wave32 warps.
  if (comm->WarpSize * kDdaA2ATdmMaxWarpsPerBlock > 512) return false;
  if (bytesPerRank % kDdaA2ATdmAlign != 0) return false;
  return reinterpret_cast<uintptr_t>(comm->ddaScratch) % kDdaA2ATdmAlign == 0;
}

static inline uint32_t ddaAllToAllFabricTdmWarpsTarget() {
  const int64_t w = rcclParamDdaA2ATdmWarps();
  return (w >= 1 && w <= kDdaA2ATdmMaxWarpsPerBlock) ? static_cast<uint32_t>(w) : kDdaA2ATdmMaxWarpsPerBlock;
}

// Grid covers one warp per UnrollPeers (tile, peer) items. Blocks are sized for the
// RCCL_DDA_A2A_TDM_WARPS target until the barrier's block cap is reached, then filled up to 16
// warps. Depends only on count, so every rank launches the same grid.
static inline std::pair<dim3, dim3> ddaAllToAllFabricTdmGeom(ncclComm* comm, size_t bytesPerRank, int tileBytes) {
  const size_t unrollPeers = kDdaA2ATdmWindowBytes / tileBytes;
  const size_t tilesPerPeer = (bytesPerRank + tileBytes - 1) / tileBytes;
  const size_t warpsNeeded = dda::common::divRoundUp(tilesPerPeer * comm->nRanks, unrollPeers);
  const uint32_t blocks =
    dda::common::calcBlockCount(warpsNeeded, ddaAllToAllFabricTdmWarpsTarget(), comm->ddaFabricMaxBlocks);
  const uint32_t warpsPerBlock =
    std::min<uint32_t>(kDdaA2ATdmMaxWarpsPerBlock, dda::common::divRoundUp(warpsNeeded, blocks));
  return std::make_pair(dim3(blocks, 1, 1), dim3(warpsPerBlock * comm->WarpSize, 1, 1));
}

// Raise the kernel's dynamic LDS cap once per device. Returns false if the device cannot grant it.
template <auto Kernel>
static bool ddaAllToAllFabricTdmPrepare(int cudaDev) {
  constexpr int kMaxDevs = 64;
  static std::atomic<int8_t> state[kMaxDevs]; // 0 unknown, 1 ready, -1 unavailable
  const bool cached = cudaDev >= 0 && cudaDev < kMaxDevs;
  if (cached && state[cudaDev].load(std::memory_order_acquire) != 0) return state[cudaDev].load() > 0;

  constexpr int kMaxDynSmem = kDdaA2ATdmMaxWarpsPerBlock * kDdaA2ATdmWindowBytes;
  int optin = 0;
  bool ok = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev) == cudaSuccess &&
            optin >= kMaxDynSmem &&
            cudaFuncSetAttribute(reinterpret_cast<const void*>(Kernel), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kMaxDynSmem) == cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    WARN("DDA fabric AllToAll: TDM path needs %d B of dynamic LDS (device opt-in %d B); using the vector kernel",
         kMaxDynSmem, optin);
  }
  if (cached) state[cudaDev].store(ok ? 1 : -1, std::memory_order_release);
  return ok;
}

template <int TileBytes, bool Push>
static ncclResult_t ncclAllToAllDdaFabricTdmLaunch(const void* sendbuff, void* recvbuff, size_t bytesPerRank,
                                                   ncclComm* comm, dda::common::FabricGpuBarrier barrierHost,
                                                   cudaStream_t stream, bool* launched) {
  constexpr int kUnrollPeers = kDdaA2ATdmWindowBytes / TileBytes;
  *launched = false;
  bool ready;
  if constexpr (Push) {
    ready = ddaAllToAllFabricTdmPrepare<&dda::common::ddaAllToAllFabricTdmPush<TileBytes, kUnrollPeers>>(comm->cudaDev);
  } else {
    ready = ddaAllToAllFabricTdmPrepare<&dda::common::ddaAllToAllFabricTdm<TileBytes, kUnrollPeers>>(comm->cudaDev);
  }
  if (!ready) return ncclSuccess;

  auto gridBlock = ddaAllToAllFabricTdmGeom(comm, bytesPerRank, TileBytes);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;
  const size_t dynSmem = (block.x / comm->WarpSize) * kDdaA2ATdmWindowBytes;

  INFO(NCCL_COLL,
       "DDA fabric AllToAll: launching TDM %s kernel: nRanks=%d count=%zu tile=%d x%d grid=%u block=%u lds=%zu",
       Push ? "push" : "pull", comm->nRanks, bytesPerRank, TileBytes, kUnrollPeers, grid.x, block.x, dynSmem);

  auto* peerScratch = reinterpret_cast<uint8_t* const*>(comm->ddaPeerPtrsDev);
  if constexpr (Push) {
    dda::common::ddaAllToAllFabricTdmPush<TileBytes, kUnrollPeers><<<grid, block, dynSmem, stream>>>(
      peerScratch, static_cast<uint8_t*>(recvbuff), static_cast<const uint8_t*>(sendbuff), bytesPerRank, comm->rank,
      comm->nRanks, barrierHost);
  } else {
    dda::common::ddaAllToAllFabricTdm<TileBytes, kUnrollPeers><<<grid, block, dynSmem, stream>>>(
      peerScratch, static_cast<uint8_t*>(recvbuff), static_cast<const uint8_t*>(sendbuff), bytesPerRank, comm->rank,
      comm->nRanks, barrierHost);
  }
  *launched = true;
  return ncclSuccess;
}

// Picks the tile shape from RCCL_DDA_A2A_TDM_TILE_KB. *launched stays false if the device cannot grant the LDS.
template <bool Push>
static ncclResult_t ncclAllToAllDdaFabricTdmDispatch(const void* sendbuff, void* recvbuff, size_t bytesPerRank,
                                                     ncclComm* comm, dda::common::FabricGpuBarrier barrierHost,
                                                     cudaStream_t stream, bool* launched) {
  switch (ddaAllToAllFabricTdmTileBytes()) {
  case 4 << 10:
    return ncclAllToAllDdaFabricTdmLaunch<4 << 10, Push>(sendbuff, recvbuff, bytesPerRank, comm, barrierHost, stream,
                                                         launched);
  case 16 << 10:
    return ncclAllToAllDdaFabricTdmLaunch<16 << 10, Push>(sendbuff, recvbuff, bytesPerRank, comm, barrierHost, stream,
                                                          launched);
  default:
    return ncclAllToAllDdaFabricTdmLaunch<8 << 10, Push>(sendbuff, recvbuff, bytesPerRank, comm, barrierHost, stream,
                                                         launched);
  }
}

static inline bool ddaA2ATdmAligned(const void* p) {
  return reinterpret_cast<uintptr_t>(p) % kDdaA2ATdmAlign == 0;
}

template <typename T>
static ncclResult_t ncclAllToAllDdaFabricTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                               cudaStream_t stream) {
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaFabricBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const int nRanks = comm->nRanks;
  const size_t totalBytes = count * nRanks;
  if (totalBytes > comm->ddaScratchBytes) {
    WARN("DDA fabric alltoall: total %zu bytes exceeds comm scratch %zu bytes", totalBytes,
         comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  if (rcclParamDdaA2APush()) {
    // count is already the byte count (kernel instantiated for int8_t).
    if (ddaAllToAllFabricTdmEligible(comm, count) && ddaA2ATdmAligned(sendbuff) && ddaA2ATdmAligned(recvbuff)) {
      bool launched = false;
      NCCLCHECK(ncclAllToAllDdaFabricTdmDispatch<true>(sendbuff, recvbuff, count, comm, barrierHost, stream, &launched));
      if (launched) {
        CUDACHECK(cudaGetLastError());
        return ncclSuccess;
      }
    }

    dim3 peerGrid, peerBlock;
    if (rcclParamDdaA2APushPeer() && sendbuff != recvbuff &&
        ddaAllToAllFabricPushPeerGeom(comm, count, &peerGrid, &peerBlock)) {
      const size_t partLines = ddaAllToAllFabricPushPeerPartUnits(count / sizeof(uint4), peerGrid.x / nRanks,
                                                                  sizeof(uint4));
      INFO(NCCL_COLL, "DDA fabric AllToAll: launching peer-major push kernel: nRanks=%d count=%zu grid=%u (%u per peer) "
           "block=%u part=%zu B", nRanks, count, peerGrid.x, peerGrid.x / nRanks, peerBlock.x,
           partLines * sizeof(uint4));
      dda::common::ddaAllToAllFabricPushPeer<T><<<peerGrid, peerBlock, 0, stream>>>(
        d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
        partLines);
      CUDACHECK(cudaGetLastError());
      return ncclSuccess;
    }

    auto gridBlock = ddaAllToAllFabricVectorGeom(comm, count);
    const auto& grid = gridBlock.first;
    const auto& block = gridBlock.second;

    INFO(NCCL_COLL, "DDA fabric AllToAll: launching push kernel: nRanks=%d count=%zu grid=%u block=%u%s", nRanks,
         count, grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

    switch (nRanks) {
    case 4:
      dda::common::ddaAllToAllFabricPush<T, 4><<<grid, block, 0, stream>>>(
        d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
      break;
    case 8:
      dda::common::ddaAllToAllFabricPush<T, 8><<<grid, block, 0, stream>>>(
        d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
      break;
    default:
      dda::common::ddaAllToAllFabricPush<T, 0><<<grid, block, 0, stream>>>(
        d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
      break;
    }
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }

  // count is already the byte count (kernel instantiated for int8_t). The pull kernels stage sendbuff into
  // this rank's scratch themselves.
  if (ddaAllToAllFabricTdmEligible(comm, count) && ddaA2ATdmAligned(sendbuff) && ddaA2ATdmAligned(recvbuff)) {
    bool launched = false;
    NCCLCHECK(ncclAllToAllDdaFabricTdmDispatch<false>(sendbuff, recvbuff, count, comm, barrierHost, stream, &launched));
    if (launched) {
      CUDACHECK(cudaGetLastError());
      return ncclSuccess;
    }
  }

  // Use the block cap chosen at init (barrier flag buffer is sized for it).
  auto gridBlock = ddaAllToAllFabricVectorGeom(comm, count);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  INFO(NCCL_COLL, "DDA fabric AllToAll: launching kernel: nRanks=%d count=%zu grid=%u block=%u%s", nRanks, count,
       grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  switch (nRanks) {
  case 4:
    dda::common::ddaAllToAllFabric<T, 4><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
    break;
  case 8:
    dda::common::ddaAllToAllFabric<T, 8><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
    break;
  default:
    dda::common::ddaAllToAllFabric<T, 0><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllToAllDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // Fabric path: requires its own handler + barrier state. Fabric handle
  // exchange works across nodes within an MNNVL clique.
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaFabricBarrierState == nullptr) {
    return false;
  }
  if (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  size_t totalCount = count * comm->nRanks;
  size_t need = totalCount * ncclTypeSize(datatype);
  if (need > comm->ddaScratchBytes) {
    return false;
  }

  // Check for data size divisible by 16 (kernel does 16-byte vectorized loads)
  if ((count * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return true;
}

uint32_t ncclAllToAllDdaFabricBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const size_t bytes = count * ncclTypeSize(datatype);
  dim3 grid, block;
  const bool peerMajor = !ddaAllToAllFabricTdmEligible(comm, bytes) && rcclParamDdaA2APush() &&
                         rcclParamDdaA2APushPeer() && ddaAllToAllFabricPushPeerGeom(comm, bytes, &grid, &block);
  if (ddaAllToAllFabricTdmEligible(comm, bytes)) {
    grid = ddaAllToAllFabricTdmGeom(comm, bytes, ddaAllToAllFabricTdmTileBytes()).first;
  } else if (!peerMajor) {
    grid = ddaAllToAllFabricVectorGeom(comm, bytes).first;
  }
  return grid.x * grid.y;
}

ncclResult_t ncclAllToAllDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  int typeSize = ncclTypeSize(datatype);
  return ncclAllToAllDdaFabricTyped<int8_t>(sendbuff, recvbuff, count * typeSize, comm, stream);
}
