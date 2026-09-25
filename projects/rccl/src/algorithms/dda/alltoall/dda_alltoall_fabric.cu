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

// Grid covers one warp per UnrollPeers (tile, peer) items, up to 16 warps per block and
// the barrier's block cap. Depends only on count, so every rank launches the same grid.
static inline std::pair<dim3, dim3> ddaAllToAllFabricTdmGeom(ncclComm* comm, size_t bytesPerRank, int tileBytes) {
  const size_t unrollPeers = kDdaA2ATdmWindowBytes / tileBytes;
  const size_t tilesPerPeer = (bytesPerRank + tileBytes - 1) / tileBytes;
  const size_t warpsNeeded = dda::common::divRoundUp(tilesPerPeer * comm->nRanks, unrollPeers);
  const uint32_t blocks =
    dda::common::calcBlockCount(warpsNeeded, kDdaA2ATdmMaxWarpsPerBlock, comm->ddaFabricMaxBlocks);
  const uint32_t warpsPerBlock =
    std::min<uint32_t>(kDdaA2ATdmMaxWarpsPerBlock, dda::common::divRoundUp(warpsNeeded, blocks));
  return std::make_pair(dim3(blocks, 1, 1), dim3(warpsPerBlock * comm->WarpSize, 1, 1));
}

// Raise the kernel's dynamic LDS cap once per device. Returns false if the device cannot grant it.
template <int TileBytes, int UnrollPeers>
static bool ddaAllToAllFabricTdmPrepare(int cudaDev) {
  constexpr int kMaxDevs = 64;
  static std::atomic<int8_t> state[kMaxDevs]; // 0 unknown, 1 ready, -1 unavailable
  const bool cached = cudaDev >= 0 && cudaDev < kMaxDevs;
  if (cached && state[cudaDev].load(std::memory_order_acquire) != 0) return state[cudaDev].load() > 0;

  constexpr int kMaxDynSmem = kDdaA2ATdmMaxWarpsPerBlock * kDdaA2ATdmWindowBytes;
  int optin = 0;
  bool ok = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev) == cudaSuccess &&
            optin >= kMaxDynSmem &&
            cudaFuncSetAttribute(reinterpret_cast<const void*>(&dda::common::ddaAllToAllFabricTdm<TileBytes, UnrollPeers>),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynSmem) == cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    WARN("DDA fabric AllToAll: TDM path needs %d B of dynamic LDS (device opt-in %d B); using the vector kernel",
         kMaxDynSmem, optin);
  }
  if (cached) state[cudaDev].store(ok ? 1 : -1, std::memory_order_release);
  return ok;
}

template <int TileBytes>
static ncclResult_t ncclAllToAllDdaFabricTdmLaunch(void* recvbuff, size_t bytesPerRank, ncclComm* comm,
                                                   dda::common::FabricGpuBarrier barrierHost, cudaStream_t stream,
                                                   bool* launched) {
  constexpr int kUnrollPeers = kDdaA2ATdmWindowBytes / TileBytes;
  *launched = false;
  if (!ddaAllToAllFabricTdmPrepare<TileBytes, kUnrollPeers>(comm->cudaDev)) return ncclSuccess;

  auto gridBlock = ddaAllToAllFabricTdmGeom(comm, bytesPerRank, TileBytes);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;
  const size_t dynSmem = (block.x / comm->WarpSize) * kDdaA2ATdmWindowBytes;

  INFO(NCCL_COLL, "DDA fabric AllToAll: launching TDM kernel: nRanks=%d count=%zu tile=%d x%d grid=%u block=%u lds=%zu",
       comm->nRanks, bytesPerRank, TileBytes, kUnrollPeers, grid.x, block.x, dynSmem);

  dda::common::ddaAllToAllFabricTdm<TileBytes, kUnrollPeers><<<grid, block, dynSmem, stream>>>(
    reinterpret_cast<uint8_t* const*>(comm->ddaPeerPtrsDev), static_cast<uint8_t*>(recvbuff), bytesPerRank,
    comm->rank, comm->nRanks, barrierHost);
  *launched = true;
  return ncclSuccess;
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

  // Stage sendbuff into this rank's scratch before the peer exchange. A single
  // host-launched cudaMemcpyAsync avoids the per-block in-kernel copy race on
  // the fabric path.
  CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalBytes, cudaMemcpyDeviceToDevice, stream));

  // count is already the byte count (kernel instantiated for int8_t).
  if (ddaAllToAllFabricTdmEligible(comm, count) && reinterpret_cast<uintptr_t>(recvbuff) % kDdaA2ATdmAlign == 0) {
    bool launched = false;
    switch (ddaAllToAllFabricTdmTileBytes()) {
    case 4 << 10:
      NCCLCHECK(ncclAllToAllDdaFabricTdmLaunch<4 << 10>(recvbuff, count, comm, barrierHost, stream, &launched));
      break;
    case 16 << 10:
      NCCLCHECK(ncclAllToAllDdaFabricTdmLaunch<16 << 10>(recvbuff, count, comm, barrierHost, stream, &launched));
      break;
    default:
      NCCLCHECK(ncclAllToAllDdaFabricTdmLaunch<8 << 10>(recvbuff, count, comm, barrierHost, stream, &launched));
      break;
    }
    if (launched) {
      CUDACHECK(cudaGetLastError());
      return ncclSuccess;
    }
  }

  // Use the block cap chosen at init (barrier flag buffer is sized for it).
  auto gridBlock = ddaAllToAllFabricGeom(comm, count);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  INFO(NCCL_COLL, "DDA fabric AllToAll: launching kernel: nRanks=%d count=%zu grid=%u block=%u%s", nRanks, count,
       grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  switch (nRanks) {
  case 4:
    dda::common::ddaAllToAllFabric<T, 4>
      <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), count, comm->rank, nRanks, barrierHost);
    break;
  case 8:
    dda::common::ddaAllToAllFabric<T, 8>
      <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), count, comm->rank, nRanks, barrierHost);
    break;
  default:
    dda::common::ddaAllToAllFabric<T, 0>
      <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), count, comm->rank, nRanks, barrierHost);
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
  const auto grid = ddaAllToAllFabricTdmEligible(comm, bytes) ?
                      ddaAllToAllFabricTdmGeom(comm, bytes, ddaAllToAllFabricTdmTileBytes()).first :
                      ddaAllToAllFabricGeom(comm, bytes).first;
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
