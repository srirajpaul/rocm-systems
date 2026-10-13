/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_fabric.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_fabric_tdm.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

// Move the fabric all-reduce through LDS with the gfx1250 Tensor Data Mover. Opt-in. Env: RCCL_DDA_AR_TDM.
RCCL_PARAM(DdaArTdm, "DDA_AR_TDM", 0);
// TDM warps per block before the block cap is reached (1..16); fewer spreads the same warps over more CUs.
RCCL_PARAM(DdaArTdmWarps, "DDA_AR_TDM_WARPS", 16);

namespace {

using nccl_dda_detail::DdaFabricBarrierState;
using dda::common::kDdaArTdmAlign;
using dda::common::kDdaArTdmMaxDynSmem;
using dda::common::kDdaArTdmMaxWarpsPerBlock;
using dda::common::kDdaArTdmWindowBytes;

// Flat all-reduce kernel below this size; tree kernel at or above it.
constexpr size_t kDdaFlatTreeThresholdBytes = 1ULL << 18;

static inline bool ddaAllReduceFabricUseTree(ncclComm* comm, size_t count, int typeSize) {
  return count * typeSize > kDdaFlatTreeThresholdBytes && count % static_cast<size_t>(comm->nRanks) == 0;
}

// Single source of the launch geometry: grid/block for `count` elements of
// `typeSize` bytes, capped at the init-time block count. Each tree-kernel thread
// covers one 16 B line of a shard (count / nRanks elements), so the tree grid is
// sized on the shard; sizing it on count would launch blocks with no lines.
static inline std::pair<dim3, dim3> ddaAllReduceFabricGeom(ncclComm* comm, size_t count, int typeSize) {
  if (!ddaAllReduceFabricUseTree(comm, count, typeSize)) {
    return dda::common::getGridAndBlockDims(count, typeSize, comm->ddaFabricMaxBlocks);
  }
  constexpr uint32_t kThreadsPerBlock = 512;
  const size_t vecs = dda::common::divRoundUp(count / comm->nRanks * typeSize, sizeof(uint4));
  const uint32_t blocks = dda::common::calcBlockCount(vecs, kThreadsPerBlock, comm->ddaFabricMaxBlocks);
  const size_t warps = dda::common::divRoundUp(dda::common::divRoundUp(vecs, blocks), comm->WarpSize);
  const uint32_t threads = std::min<uint32_t>(kThreadsPerBlock, static_cast<uint32_t>(warps * comm->WarpSize));
  return std::make_pair(dim3(blocks, 1, 1), dim3(threads, 1, 1));
}

// A warp keeps one tile per rank in its window, so the tile is the largest power of two with
// tile * nRanks <= window, at least 1 KiB. 0 means the clique is too large for the TDM kernels.
static inline int ddaAllReduceFabricTdmTileBytes(int nRanks) {
  if (nRanks <= 4) return 4 << 10;
  if (nRanks <= 8) return 2 << 10;
  if (nRanks <= 16) return 1 << 10;
  return 0;
}

static inline bool ddaArTdmAligned(const void* p) {
  return reinterpret_cast<uintptr_t>(p) % kDdaArTdmAlign == 0;
}

// Everything but sendbuff/recvbuff alignment, which the block-count report cannot see.
static bool ddaAllReduceFabricTdmEligible(ncclComm* comm, size_t count, int typeSize) {
  if (!rcclParamDdaArTdm()) return false;
  if (!TDM_TOOLCHAIN_AVAILABLE || comm->archName == nullptr || !IsArchMatch(comm->archName, "gfx1250")) return false;
  // The kernels are bounded to 512 threads, i.e. 16 wave32 warps.
  if (comm->WarpSize * kDdaArTdmMaxWarpsPerBlock > 512) return false;
  if (ddaAllReduceFabricTdmTileBytes(comm->nRanks) == 0) return false;
  const size_t regionBytes =
    ddaAllReduceFabricUseTree(comm, count, typeSize) ? count / comm->nRanks * typeSize : count * typeSize;
  if (regionBytes % kDdaArTdmAlign != 0) return false;
  return ddaArTdmAligned(comm->ddaScratch);
}

// One warp per tile (of a shard on the tree path, of the whole buffer on the flat path). Blocks are sized for
// the RCCL_DDA_AR_TDM_WARPS target until the block cap is reached, then filled up to 16 warps. Depends only on
// count, so every rank launches the same grid.
static inline std::pair<dim3, dim3> ddaAllReduceFabricTdmGeom(ncclComm* comm, size_t count, int typeSize) {
  const int tileBytes = ddaAllReduceFabricTdmTileBytes(comm->nRanks);
  const size_t regionBytes =
    ddaAllReduceFabricUseTree(comm, count, typeSize) ? count / comm->nRanks * typeSize : count * typeSize;
  const size_t nTiles = dda::common::divRoundUp(regionBytes, tileBytes);
  const int64_t w = rcclParamDdaArTdmWarps();
  const uint32_t target =
    (w >= 1 && w <= kDdaArTdmMaxWarpsPerBlock) ? static_cast<uint32_t>(w) : kDdaArTdmMaxWarpsPerBlock;
  const uint32_t blocks = dda::common::calcBlockCount(nTiles, target, comm->ddaFabricMaxBlocks);
  const uint32_t warpsPerBlock =
    std::min<uint32_t>(kDdaArTdmMaxWarpsPerBlock, dda::common::divRoundUp(nTiles, blocks));
  return std::make_pair(dim3(blocks, 1, 1), dim3(warpsPerBlock * comm->WarpSize, 1, 1));
}

// Raise the kernel's dynamic LDS cap once per device. Returns false if the device cannot grant it.
template <auto Kernel>
static bool ddaAllReduceFabricTdmPrepare(int cudaDev) {
  constexpr int kMaxDevs = 64;
  static std::atomic<int8_t> state[kMaxDevs]; // 0 unknown, 1 ready, -1 unavailable
  const bool cached = cudaDev >= 0 && cudaDev < kMaxDevs;
  if (cached && state[cudaDev].load(std::memory_order_acquire) != 0) return state[cudaDev].load() > 0;

  int optin = 0;
  bool ok = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev) == cudaSuccess &&
            optin >= kDdaArTdmMaxDynSmem &&
            cudaFuncSetAttribute(reinterpret_cast<const void*>(Kernel), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kDdaArTdmMaxDynSmem) == cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    WARN("DDA fabric AllReduce: TDM path needs %d B of dynamic LDS (device opt-in %d B); using the vector kernel",
         kDdaArTdmMaxDynSmem, optin);
  }
  if (cached) state[cudaDev].store(ok ? 1 : -1, std::memory_order_release);
  return ok;
}

template <typename T, int TileBytes, bool Tree>
static ncclResult_t ncclAllReduceDdaFabricTdmLaunch(const void* sendbuff, void* recvbuff, size_t count,
                                                    ncclComm* comm, dda::common::FabricGpuBarrier barrierHost,
                                                    cudaStream_t stream, bool* launched) {
  *launched = false;
  bool ready;
  if constexpr (Tree) {
    ready = ddaAllReduceFabricTdmPrepare<&dda::common::ddaAllReduceTreeFabricTdm<T, TileBytes>>(comm->cudaDev);
  } else {
    ready = ddaAllReduceFabricTdmPrepare<&dda::common::ddaAllReduceFlatFabricTdm<T, TileBytes>>(comm->cudaDev);
  }
  if (!ready) return ncclSuccess;

  auto gridBlock = ddaAllReduceFabricTdmGeom(comm, count, sizeof(T));
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;
  const size_t dynSmem = (block.x / comm->WarpSize) * kDdaArTdmWindowBytes;
  const size_t bytes = count * sizeof(T);

  INFO(NCCL_COLL, "DDA fabric AllReduce: launching TDM %s kernel: nRanks=%d bytes=%zu tile=%d grid=%u block=%u lds=%zu",
       Tree ? "tree (two-shot)" : "flat (one-shot)", comm->nRanks, bytes, TileBytes, grid.x, block.x, dynSmem);

  auto* peers = reinterpret_cast<uint8_t* const*>(comm->ddaPeerPtrsDev);
  if constexpr (Tree) {
    dda::common::ddaAllReduceTreeFabricTdm<T, TileBytes><<<grid, block, dynSmem, stream>>>(
      peers, static_cast<uint8_t*>(recvbuff), bytes, static_cast<const uint8_t*>(sendbuff), comm->rank, comm->nRanks,
      barrierHost);
  } else {
    dda::common::ddaAllReduceFlatFabricTdm<T, TileBytes><<<grid, block, dynSmem, stream>>>(
      peers, static_cast<uint8_t*>(recvbuff), bytes, static_cast<const uint8_t*>(sendbuff), comm->rank, comm->nRanks,
      barrierHost);
  }
  *launched = true;
  return ncclSuccess;
}

template <typename T, bool Tree>
static ncclResult_t ncclAllReduceDdaFabricTdmDispatch(const void* sendbuff, void* recvbuff, size_t count,
                                                      ncclComm* comm, dda::common::FabricGpuBarrier barrierHost,
                                                      cudaStream_t stream, bool* launched) {
  switch (ddaAllReduceFabricTdmTileBytes(comm->nRanks)) {
  case 4 << 10:
    return ncclAllReduceDdaFabricTdmLaunch<T, 4 << 10, Tree>(sendbuff, recvbuff, count, comm, barrierHost, stream,
                                                             launched);
  case 2 << 10:
    return ncclAllReduceDdaFabricTdmLaunch<T, 2 << 10, Tree>(sendbuff, recvbuff, count, comm, barrierHost, stream,
                                                             launched);
  default:
    return ncclAllReduceDdaFabricTdmLaunch<T, 1 << 10, Tree>(sendbuff, recvbuff, count, comm, barrierHost, stream,
                                                             launched);
  }
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                                cudaStream_t stream) {
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaFabricBarrierState == nullptr) {
    return ncclInvalidUsage;
  }
  if (count * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA fabric allreduce: element count %zu needs %zu bytes; comm scratch is %zu bytes", count, count * sizeof(T),
         comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  const int nRanks = comm->nRanks;
  const size_t sizeBytes = count * sizeof(T);
  const bool wantTree = sizeBytes > kDdaFlatTreeThresholdBytes;
  const bool treeOk = wantTree && (count % static_cast<size_t>(nRanks) == 0);

  if (wantTree && !treeOk) {
    INFO(NCCL_ALL, "DDA fabric: size %zu B > 256KB but count %zu not divisible by %d; using flat kernel", sizeBytes,
         count, nRanks);
  }

  // Block cap chosen at init (barrier flag buffer is sized for it).
  auto gridBlock = ddaAllReduceFabricGeom(comm, count, sizeof(T));
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  if (ddaAllReduceFabricTdmEligible(comm, count, sizeof(T)) && ddaArTdmAligned(sendbuff) && ddaArTdmAligned(recvbuff)) {
    bool launched = false;
    const ncclResult_t res =
      treeOk ? ncclAllReduceDdaFabricTdmDispatch<T, true>(sendbuff, recvbuff, count, comm, barrierHost, stream,
                                                          &launched) :
               ncclAllReduceDdaFabricTdmDispatch<T, false>(sendbuff, recvbuff, count, comm, barrierHost, stream,
                                                           &launched);
    NCCLCHECK(res);
    if (launched) {
      CUDACHECK(cudaGetLastError());
      return ncclSuccess;
    }
  }

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  // Specialize the kernel for common clique sizes (compile-time NRANKS -> the
  // unrolled CollCommon reduce/gather), and fall back to the runtime kernel
  // (NRANKS_CT == 0) for any other size.
  INFO(NCCL_COLL, "DDA fabric AllReduce: launching %s kernel: nRanks=%d count=%zu sizeBytes=%zu grid=%u block=%u%s",
       treeOk ? "tree (two-shot)" : "flat (one-shot)", nRanks, count, sizeBytes, grid.x, block.x,
       (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  if (treeOk) {
    // NRANKS_CT 4/8: unrolled reduce; 0: runtime fallback. The kernel stages sendbuff into scratch itself.
    switch (nRanks) {
    case 4:
      INFO(NCCL_COLL, "DDA fabric AllReduce: tree path, NRANKS_CT=4 (unrolled)");
      dda::common::ddaAllReduceTreeFabric<T, 4, false>
        <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff),
                                     comm->rank, nRanks, barrierHost, nullptr);
      break;
    case 8:
      INFO(NCCL_COLL, "DDA fabric AllReduce: tree path, NRANKS_CT=8 (unrolled)");
      dda::common::ddaAllReduceTreeFabric<T, 8, false>
        <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff),
                                     comm->rank, nRanks, barrierHost, nullptr);
      break;
    default:
      INFO(NCCL_COLL, "DDA fabric AllReduce: tree path, NRANKS_CT=0 (runtime, nRanks=%d)", nRanks);
      dda::common::ddaAllReduceTreeFabric<T, 0, false>
        <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff),
                                     comm->rank, nRanks, barrierHost, nullptr);
      break;
    }
  } else {
    switch (nRanks) {
    case 4:
      INFO(NCCL_COLL, "DDA fabric AllReduce: flat path, NRANKS_CT=4 (unrolled)");
      dda::common::ddaAllReduceFlatFabric<T, 4, false>
        <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff),
                                     comm->rank, nRanks, barrierHost, nullptr);
      break;
    case 8:
      INFO(NCCL_COLL, "DDA fabric AllReduce: flat path, NRANKS_CT=8 (unrolled)");
      dda::common::ddaAllReduceFlatFabric<T, 8, false>
        <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff),
                                     comm->rank, nRanks, barrierHost, nullptr);
      break;
    default:
      INFO(NCCL_COLL, "DDA fabric AllReduce: flat path, NRANKS_CT=0 (runtime, nRanks=%d)", nRanks);
      dda::common::ddaAllReduceFlatFabric<T, 0, false>
        <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff),
                                     comm->rank, nRanks, barrierHost, nullptr);
      break;
    }
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr) {
    return false;
  }
  // Fabric path: requires its own handler + barrier state. Fabric handle
  // exchange works across nodes within an MNNVL clique
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaFabricBarrierState == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  // Checks shared by both DDA all-reduce backends.
  if (comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }
  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes > comm->ddaScratchBytes) {
    return false;
  }
  if (bytes % 16) {
    // 16-byte alignment: the DDA kernels do 16-byte vectorized loads.
    return false;
  }
  if (bytes > kDdaFlatTreeThresholdBytes) {
    if (count % comm->nRanks || ((count / comm->nRanks) * ncclTypeSize(datatype)) % 16) {
      // Two-shot/tree path: each rank reduces count/nRanks elements, so that
      // per-rank slice must also be 16-byte aligned.
      return false;
    }
  }
  return true;
}

uint32_t ncclAllReduceDdaFabricBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const int typeSize = ncclTypeSize(datatype);
  const auto grid = ddaAllReduceFabricTdmEligible(comm, count, typeSize) ?
                      ddaAllReduceFabricTdmGeom(comm, count, typeSize).first :
                      ddaAllReduceFabricGeom(comm, count, typeSize).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllReduceDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaFabricTyped<float>(sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaFabricTyped<half>(sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaFabricTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
