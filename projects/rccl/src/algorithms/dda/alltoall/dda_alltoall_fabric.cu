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
#include <hip/hip_ext.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

// Push model: each rank writes its chunks into peers' scratch instead of reading from them. Env: RCCL_DDA_A2A_PUSH.
RCCL_PARAM(DdaA2APush, "DDA_A2A_PUSH", 0);
// Peer-major push: each block serves one peer (out of place only). Env: RCCL_DDA_A2A_PUSH_PEER.
RCCL_PARAM(DdaA2APushPeer, "DDA_A2A_PUSH_PEER", 0);
// Peer-major push threads per block (one warp .. 512, multiple of 64).
RCCL_PARAM(DdaA2APushPeerThreads, "DDA_A2A_PUSH_PEER_THREADS", 256);
// Peer-major push through LDS with the gfx1250 Tensor Data Mover, with RCCL_DDA_A2A_PUSH=1 and
// RCCL_DDA_A2A_PUSH_PEER=1. Env: RCCL_DDA_A2A_TDM.
RCCL_PARAM(DdaA2ATdm, "DDA_A2A_TDM", 0);
// TDM tile size in KiB: 8 (x2 tiles per warp batch, default), 16 (x1) or 4 (x4). Other values use 8.
RCCL_PARAM(DdaA2ATdmTileKB, "DDA_A2A_TDM_TILE_KB", 8);

namespace {

using nccl_dda_detail::DdaFabricBarrierState;
using dda::common::kDdaA2ATdmAlign;
using dda::common::kDdaA2ATdmMaxWarpsPerBlock;
using dda::common::kDdaA2ATdmWindowBytes;

// Single source of the launch geometry: grid/block for a byte payload. The
// kernel is instantiated for int8_t, so `bytes` is the per-block element count.
static inline std::pair<dim3, dim3> ddaAllToAllFabricGeom(ncclComm* comm, size_t bytes) {
  return dda::common::getGridAndBlockDims(bytes, 1, comm->ddaFabricMaxBlocks);
}

// Peer-major push grid: nRanks groups of blocks, one group per peer offset, within the barrier's block cap.
// Depends only on bytes, so every rank launches the same grid. Returns false when the cap cannot give every peer
// a block.
static inline bool ddaAllToAllFabricPushPeerGeom(ncclComm* comm, size_t bytes, dim3* grid, dim3* block) {
  constexpr uint32_t kThreadsPerWarp = 64;
  const uint32_t maxPerPeer = static_cast<uint32_t>(comm->ddaFabricMaxBlocks) / comm->nRanks;
  if (maxPerPeer == 0) return false;
  const int64_t t = rcclParamDdaA2APushPeerThreads();
  const uint32_t threadsCap =
    (t >= kThreadsPerWarp && t <= 512) ? static_cast<uint32_t>(t) / kThreadsPerWarp * kThreadsPerWarp : 256;
  const size_t lines = dda::common::divRoundUp(bytes, sizeof(uint4));
  const uint32_t perPeer = std::min<uint32_t>(maxPerPeer, dda::common::divRoundUp(lines, threadsCap));
  const uint32_t warps = dda::common::divRoundUp(dda::common::divRoundUp(lines, perPeer), kThreadsPerWarp);
  *grid = dim3(perPeer * comm->nRanks, 1, 1);
  *block = dim3(std::min<uint32_t>(threadsCap, warps * kThreadsPerWarp), 1, 1);
  return true;
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

static inline bool ddaA2ATdmAligned(const void* p) {
  return reinterpret_cast<uintptr_t>(p) % kDdaA2ATdmAlign == 0;
}

// Everything but sendbuff / recvbuff alignment, which the block-count report cannot see.
static bool ddaAllToAllFabricTdmEligible(ncclComm* comm, size_t bytesPerRank) {
  if (!rcclParamDdaA2ATdm()) return false;
  if (!TDM_TOOLCHAIN_AVAILABLE || comm->archName == nullptr || !IsArchMatch(comm->archName, "gfx1250")) return false;
  // The kernel is bounded to 512 threads, i.e. 16 wave32 warps.
  if (comm->WarpSize * kDdaA2ATdmMaxWarpsPerBlock > 512) return false;
  if (bytesPerRank % kDdaA2ATdmAlign != 0) return false;
  return ddaA2ATdmAligned(comm->ddaScratch);
}

// Tiles per peer-major TDM push part when a peer chunk of `tiles` is split over perPeer blocks: the even split
// rounded up to 64 KiB. The alignment is halved while rounding would leave more than a quarter of the blocks without
// a part (small messages).
static inline size_t ddaAllToAllFabricTdmPushPeerPartTiles(size_t tiles, uint32_t perPeer, int tileBytes) {
  constexpr size_t kAlignBytes = 64 * 1024;
  size_t alignTiles = std::max<size_t>(1, kAlignBytes / tileBytes);
  const size_t even = dda::common::divRoundUp(tiles, perPeer);
  auto partTiles = [&]() { return dda::common::divRoundUp(even, alignTiles) * alignTiles; };
  while (alignTiles > 1 && 4 * dda::common::divRoundUp(tiles, partTiles()) < 3 * static_cast<size_t>(perPeer)) {
    alignTiles /= 2;
  }
  return partTiles();
}

// Peer-major TDM push grid: nRanks groups of blocks within the barrier's block cap, each block with enough warps (up
// to 16) to give every warp one batch of its block's part. Returns false when the cap cannot give every peer a
// block. Depends only on bytesPerRank, so every rank launches the same grid.
static inline bool ddaAllToAllFabricTdmPushPeerGeom(ncclComm* comm, size_t bytesPerRank, int tileBytes, dim3* grid,
                                                    dim3* block, size_t* partTiles) {
  const uint32_t maxPerPeer = static_cast<uint32_t>(comm->ddaFabricMaxBlocks) / comm->nRanks;
  if (maxPerPeer == 0) return false;
  const size_t unrollTiles = kDdaA2ATdmWindowBytes / tileBytes;
  const size_t tilesPerPeer = dda::common::divRoundUp(bytesPerRank, static_cast<size_t>(tileBytes));
  const uint32_t perPeer = std::min<uint32_t>(maxPerPeer, dda::common::divRoundUp(tilesPerPeer, unrollTiles));
  *partTiles = ddaAllToAllFabricTdmPushPeerPartTiles(tilesPerPeer, perPeer, tileBytes);
  const uint32_t warps =
    std::min<uint32_t>(kDdaA2ATdmMaxWarpsPerBlock, dda::common::divRoundUp(*partTiles, unrollTiles));
  *grid = dim3(perPeer * comm->nRanks, 1, 1);
  *block = dim3(warps * comm->WarpSize, 1, 1);
  return true;
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
    WARN("DDA fabric AllToAll: TDM path needs %d B of dynamic LDS (device opt-in %d B); using the vector kernels",
         kMaxDynSmem, optin);
  }
  if (cached) state[cudaDev].store(ok ? 1 : -1, std::memory_order_release);
  return ok;
}

// *launched stays false if the block cap cannot give every peer a block or the device cannot grant the LDS.
template <int TileBytes>
static ncclResult_t ncclAllToAllDdaFabricTdmPushPeerLaunch(const void* sendbuff, void* recvbuff, size_t bytesPerRank,
                                                           ncclComm* comm, dda::common::FabricGpuBarrier barrierHost,
                                                           cudaStream_t stream, bool* launched) {
  constexpr int kUnrollTiles = kDdaA2ATdmWindowBytes / TileBytes;
  *launched = false;
  dim3 grid, block;
  size_t partTiles;
  if (!ddaAllToAllFabricTdmPushPeerGeom(comm, bytesPerRank, TileBytes, &grid, &block, &partTiles)) return ncclSuccess;
  if (!ddaAllToAllFabricTdmPrepare<&dda::common::ddaAllToAllFabricTdmPushPeer<TileBytes, kUnrollTiles>>(comm->cudaDev)) {
    return ncclSuccess;
  }
  const size_t dynSmem = (block.x / comm->WarpSize) * kDdaA2ATdmWindowBytes;

  INFO(NCCL_COLL,
       "DDA fabric AllToAll: launching TDM peer-major push kernel: nRanks=%d count=%zu tile=%d x%d grid=%u (%u per "
       "peer) block=%u part=%zu B lds=%zu",
       comm->nRanks, bytesPerRank, TileBytes, kUnrollTiles, grid.x, grid.x / comm->nRanks, block.x,
       partTiles * TileBytes, dynSmem);

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricTdmPushPeer<TileBytes, kUnrollTiles>), grid, block, dynSmem,
                        stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0,
                        reinterpret_cast<uint8_t* const*>(comm->ddaPeerPtrsDev), static_cast<uint8_t*>(recvbuff),
                        static_cast<const uint8_t*>(sendbuff), bytesPerRank, comm->rank, comm->nRanks, barrierHost,
                        partTiles);
  *launched = true;
  return ncclSuccess;
}

static ncclResult_t ncclAllToAllDdaFabricTdmPushPeerDispatch(const void* sendbuff, void* recvbuff,
                                                             size_t bytesPerRank, ncclComm* comm,
                                                             dda::common::FabricGpuBarrier barrierHost,
                                                             cudaStream_t stream, bool* launched) {
  switch (ddaAllToAllFabricTdmTileBytes()) {
  case 4 << 10:
    return ncclAllToAllDdaFabricTdmPushPeerLaunch<4 << 10>(sendbuff, recvbuff, bytesPerRank, comm, barrierHost, stream,
                                                           launched);
  case 16 << 10:
    return ncclAllToAllDdaFabricTdmPushPeerLaunch<16 << 10>(sendbuff, recvbuff, bytesPerRank, comm, barrierHost,
                                                            stream, launched);
  default:
    return ncclAllToAllDdaFabricTdmPushPeerLaunch<8 << 10>(sendbuff, recvbuff, bytesPerRank, comm, barrierHost, stream,
                                                           launched);
  }
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

  // Use the block cap chosen at init (barrier flag buffer is sized for it);
  // count is already the byte count (kernel instantiated for int8_t).
  auto gridBlock = ddaAllToAllFabricGeom(comm, count);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  // In-place calls take the vector push: a peer-major block drains into recvbuff slots that other blocks may
  // still be reading from sendbuff.
  const bool peerPush = rcclParamDdaA2APush() && rcclParamDdaA2APushPeer() && sendbuff != recvbuff;
  // count is already the byte count (kernel instantiated for int8_t).
  if (peerPush && ddaAllToAllFabricTdmEligible(comm, count) && ddaA2ATdmAligned(sendbuff) &&
      ddaA2ATdmAligned(recvbuff)) {
    bool launched = false;
    NCCLCHECK(ncclAllToAllDdaFabricTdmPushPeerDispatch(sendbuff, recvbuff, count, comm, barrierHost, stream,
                                                       &launched));
    if (launched) {
      CUDACHECK(cudaGetLastError());
      return ncclSuccess;
    }
  }

  dim3 peerGrid, peerBlock;
  if (peerPush && ddaAllToAllFabricPushPeerGeom(comm, count, &peerGrid, &peerBlock)) {
    INFO(NCCL_COLL,
         "DDA fabric AllToAll: launching peer-major push kernel: nRanks=%d count=%zu grid=%u (%u per peer) block=%u",
         nRanks, count, peerGrid.x, peerGrid.x / nRanks, peerBlock.x);

    const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricPushPeer<T>), peerGrid, peerBlock, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), count,
                          static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }

  if (rcclParamDdaA2APush()) {
    INFO(NCCL_COLL, "DDA fabric AllToAll: launching push kernel: nRanks=%d count=%zu grid=%u block=%u%s", nRanks,
         count, grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

    const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
    switch (nRanks) {
    case 4:
      hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricPush<T, 4>), grid, block, 0, stream,
                            /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                            count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
      break;
    case 8:
      hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricPush<T, 8>), grid, block, 0, stream,
                            /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                            count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
      break;
    default:
      hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricPush<T, 0>), grid, block, 0, stream,
                            /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                            count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
      break;
    }
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }

  INFO(NCCL_COLL, "DDA fabric AllToAll: launching kernel: nRanks=%d count=%zu grid=%u block=%u%s", nRanks, count,
       grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  // Stage sendbuff into this rank's scratch before the peer exchange. A single
  // host-launched cudaMemcpyAsync avoids the per-block in-kernel copy race on
  // the fabric path.
  CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalBytes, cudaMemcpyDeviceToDevice, stream));

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabric<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), count, comm->rank, nRanks,
                          barrierHost);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabric<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), count, comm->rank, nRanks,
                          barrierHost);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabric<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), count, comm->rank, nRanks,
                          barrierHost);
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
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
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
  size_t partTiles;
  const bool peerPush = rcclParamDdaA2APush() && rcclParamDdaA2APushPeer();
  const bool tdmPeer = peerPush && ddaAllToAllFabricTdmEligible(comm, bytes) &&
                       ddaAllToAllFabricTdmPushPeerGeom(comm, bytes, ddaAllToAllFabricTdmTileBytes(), &grid, &block,
                                                        &partTiles);
  if (!tdmPeer && (!peerPush || !ddaAllToAllFabricPushPeerGeom(comm, bytes, &grid, &block))) {
    grid = ddaAllToAllFabricGeom(comm, bytes).first;
  }
  return grid.x * grid.y;
}

ncclResult_t ncclAllToAllDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclComm* comm, cudaStream_t stream) {
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return ncclInvalidArgument;
  }
  int typeSize = ncclTypeSize(datatype);
  return ncclAllToAllDdaFabricTyped<int8_t>(sendbuff, recvbuff, count * typeSize, comm, stream);
}
