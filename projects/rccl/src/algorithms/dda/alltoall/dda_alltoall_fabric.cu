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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

// Push model: each rank writes its chunks into peers' scratch instead of reading from them.
// Env: RCCL_DDA_A2A_PUSH. RCCL_DDA_A2A_TDM=1 moves that exchange through the tensor data mover
// when the pointers and the per-rank byte count are 256-byte aligned. RCCL_DDA_A2A_TDM_TILE_KB
// is 4, 8, or 16 (default 8).
RCCL_PARAM(DdaA2APush, "DDA_A2A_PUSH", 0);
RCCL_PARAM(DdaA2ATdm, "DDA_A2A_TDM", 0);
RCCL_PARAM(DdaA2ATdmTileKB, "DDA_A2A_TDM_TILE_KB", 8);
// Peer-major TDM push knobs. RCCL_DDA_A2A_TDM_TILE_KB is the tile each wave stages: 4, 8 or 16 KiB (a 16 KiB window of
// several tiles), or 20 or 40 KiB (a single tile that is the window; peer-major kernel only). RCCL_DDA_A2A_TDM_WARPS is the
// number of waves per block (default and maximum 16; also bounded by the device's dynamic LDS limit divided by the window).
// RCCL_DDA_A2A_TDM_BPP forces the blocks per peer (0 = automatic; at most the block cap divided by the rank count).
// Tuning notes (not defaults; re-check on each machine): the best TransferBench TDM all-to-all settings were about 32 thread
// blocks per peer transfer, 256 to 1024 threads per block and the device maximum of dynamic LDS per block, i.e. a large
// staging window per wave. Here that is BPP=32, a smaller WARPS and a larger TILE_KB (for example 40 KiB with 8 waves).
RCCL_PARAM(DdaA2ATdmWarps, "DDA_A2A_TDM_WARPS", 16);
RCCL_PARAM(DdaA2ATdmBpp, "DDA_A2A_TDM_BPP", 0);
// RCCL_DDA_A2A_PUSH_PEER=1 (with RCCL_DDA_A2A_PUSH=1): peer-major vector push. Each block works on a single peer
// and the grid is nRanks * blocksPerPeer. Out of place only.
RCCL_PARAM(DdaA2APushPeer, "DDA_A2A_PUSH_PEER", 0);
// RCCL_DDA_A2A_PUSH_PEER_PIPE: 1 (default) cuts each block's share into pieces (see ddaAllToAllFabricPeerPieces) so
// a drain overlaps the next push; 0 = a single piece.
RCCL_PARAM(DdaA2APushPeerPipe, "DDA_A2A_PUSH_PEER_PIPE", 1);
// RCCL_DDA_A2A_PUSH_PEER_PIECES > 0 forces the piece count of the peer-major pushes; 0 (default) = automatic.
RCCL_PARAM(DdaA2APushPeerPieces, "DDA_A2A_PUSH_PEER_PIECES", 0);
// RCCL_DDA_A2A_PUSH_PEER_KEEP: number of trailing pieces each push block of the peer-major vector kernel drains itself;
// the earlier pieces are drained by the local-copy blocks (group 0). 0 moves every drain to group 0. Default -1 = auto,
// see ddaAllToAllFabricPeerKeep.
RCCL_PARAM(DdaA2APushPeerKeep, "DDA_A2A_PUSH_PEER_KEEP", -1);

namespace {

using nccl_dda_detail::DdaFabricBarrierState;

// Single source of the launch geometry: grid/block for a byte payload. The
// kernel is instantiated for int8_t, so `bytes` is the per-block element count.
static inline std::pair<dim3, dim3> ddaAllToAllFabricGeom(ncclComm* comm, size_t bytes) {
  return dda::common::getGridAndBlockDims(bytes, 1, comm->ddaFabricMaxBlocks);
}

// Pieces of a peer-major push: one per MiB of the chunk per peer, 1..maxPieces (1 if the pipeline is off), or
// RCCL_DDA_A2A_PUSH_PEER_PIECES (<= 64) if set. The vector push stops at 16 pieces, the TDM push at 32.
static inline unsigned ddaAllToAllFabricPeerPieces(size_t chunkBytes, unsigned maxPieces) {
  if (!rcclParamDdaA2APushPeerPipe()) return 1;
  const int64_t forced = rcclParamDdaA2APushPeerPieces();
  if (forced > 0) return static_cast<unsigned>(std::min<int64_t>(forced, 64));
  return static_cast<unsigned>(std::clamp<size_t>(chunkBytes >> 20, 1, maxPieces));
}

// Trailing pieces each push block of a peer-major kernel drains itself (RCCL_DDA_A2A_PUSH_PEER_KEEP); the earlier pieces
// are drained by the local-copy blocks. Auto (-1): none (every drain moves to the local-copy blocks) for a chunk per peer in
// [autoMinBytes, autoMaxBytes), all of them outside it: the move loses below autoMinBytes, where the transfer is
// latency-bound, and above autoMaxBytes the local-copy blocks become the bottleneck.
static inline unsigned ddaAllToAllFabricPeerKeep(size_t chunkBytes, unsigned pieces, size_t autoMinBytes,
                                                  size_t autoMaxBytes) {
  const int64_t keep = rcclParamDdaA2APushPeerKeep();
  if (keep >= 0) return static_cast<unsigned>(std::min<int64_t>(keep, 64));
  return chunkBytes >= autoMinBytes && chunkBytes < autoMaxBytes ? 0u : pieces;
}

struct DdaA2APeerGeom {
  dim3 grid;
  dim3 block;
  int blocksPerPeer;
};

// Peer-major grid: nRanks * blocksPerPeer blocks of up to kMaxWaves 32-thread waves (the vector push uses 8 waves,
// 256 threads: that keeps small messages fast and 512 is slower below ~64 MB). `waves` is the number of waves of work
// for ONE peer's chunk. blocksPerPeer is capped so the grid stays inside the barrier's flag allocation. The result
// depends only on that, nRanks and the init-time cap, so every rank gets the same grid. Returns blocksPerPeer == 0
// when there are more peers than blocks. balanceRounds (TDM): a wave does `rounds` rounds of work, so take the fewest
// blocks that still need only as many rounds as the cap would, instead of a last round that few blocks share.
static inline DdaA2APeerGeom ddaAllToAllFabricPeerGeom(ncclComm* comm, size_t waves, size_t kMaxWaves,
                                                       bool balanceRounds = false) {
  constexpr size_t kWave = 32;
  const size_t cap = static_cast<size_t>(comm->ddaFabricMaxBlocks) / static_cast<size_t>(comm->nRanks);
  if (cap == 0) return {dim3(1), dim3(1), 0};
  waves = std::max<size_t>(1, waves);
  size_t bpp = (waves + kMaxWaves - 1) / kMaxWaves;
  bpp = std::max<size_t>(1, std::min(bpp, cap));
  if (balanceRounds) {
    const size_t rounds = (waves + cap * kMaxWaves - 1) / (cap * kMaxWaves);
    bpp = std::max<size_t>(1, std::min(cap, (waves + rounds * kMaxWaves - 1) / (rounds * kMaxWaves)));
  }
  const size_t wavesPerBlock = std::min(kMaxWaves, (waves + bpp - 1) / bpp);
  return {dim3(static_cast<unsigned>(bpp * comm->nRanks)), dim3(static_cast<unsigned>(wavesPerBlock * kWave)),
          static_cast<int>(bpp)};
}

// One warp per window of UnrollPeers tiles. The block count stays inside the
// barrier's flag allocation.
static inline std::pair<dim3, dim3> ddaAllToAllFabricTdmGeom(ncclComm* comm, size_t bytesPerRank, int tileBytes) {
  const size_t unrollPeers = dda::common::kDdaA2ATdmWindowBytes / tileBytes;
  const size_t tilesPerPeer = (bytesPerRank + tileBytes - 1) / tileBytes;
  const size_t warpsNeeded =
      dda::common::divRoundUp(tilesPerPeer * static_cast<size_t>(comm->nRanks), unrollPeers);
  const uint32_t blocks = dda::common::calcBlockCount(warpsNeeded, dda::common::kDdaA2ATdmMaxWarpsPerBlock,
                                                      comm->ddaFabricMaxBlocks);
  const uint32_t warpsPerBlock =
      std::min<uint32_t>(dda::common::kDdaA2ATdmMaxWarpsPerBlock, dda::common::divRoundUp(warpsNeeded, blocks));
  return {dim3(blocks), dim3(warpsPerBlock * dda::common::kDdaA2ATdmWave)};
}

static bool ddaAllToAllFabricTdmEligible(ncclComm* comm, const void* sendbuff, const void* recvbuff,
                                         size_t bytesPerRank, bool peerKernel = false) {
  if (!rcclParamDdaA2ATdm() || !TDM_TOOLCHAIN_AVAILABLE || comm->archName == nullptr ||
      !IsArchMatch(comm->archName, "gfx1250")) {
    return false;
  }
  const int tileKB = rcclParamDdaA2ATdmTileKB();
  const bool bigTile = tileKB == 20 || tileKB == 40;
  if (tileKB != 4 && tileKB != 8 && tileKB != 16 && !(bigTile && peerKernel)) return false;
  if ((bytesPerRank % dda::common::kDdaA2ATdmAlign) != 0) return false;
  auto aligned = [](const void* p) {
    return (reinterpret_cast<uintptr_t>(p) % dda::common::kDdaA2ATdmAlign) == 0;
  };
  return aligned(sendbuff) && aligned(recvbuff) && aligned(comm->ddaScratch);
}

// Opt this instantiation into the device's full dynamic LDS once; each launch then asks for only what it needs.
// The static is per template instantiation, so each tile shape is set up independently.
template <auto Kernel>
static ncclResult_t ddaAllToAllFabricTdmPrepare() {
  static std::atomic<bool> done{false};
  if (done.load(std::memory_order_acquire)) return ncclSuccess;
  int dev = 0;
  int maxSmem = 0;
  CUDACHECK(cudaGetDevice(&dev));
  CUDACHECK(cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  if (dda::common::kDdaA2ATdmMaxSmemBytes > maxSmem) return ncclInvalidUsage;
  CUDACHECK(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmem));
  done.store(true, std::memory_order_release);
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

  // Use the block cap chosen at init (barrier flag buffer is sized for it);
  // count is already the byte count (kernel instantiated for int8_t).
  auto gridBlock = ddaAllToAllFabricGeom(comm, count);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  const char* unroll = (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)";
  // Peer-major TDM push (RCCL_DDA_A2A_PUSH_PEER=1 with RCCL_DDA_A2A_TDM=1): each block works on exactly one peer.
  // Out of place only, like the vector peer-major push. Falls through to the all-peer TDM push when there are more
  // peers than blocks.
  if (rcclParamDdaA2APush() && rcclParamDdaA2APushPeer() && sendbuff != recvbuff &&
      ddaAllToAllFabricTdmEligible(comm, sendbuff, recvbuff, count, /*peerKernel=*/true)) {
    const int tileBytes = rcclParamDdaA2ATdmTileKB() << 10;
    // The wave window is one tile for the 20 and 40 KiB tiles, and kDdaA2ATdmWindowBytes worth of tiles otherwise.
    const size_t tilesPerWave = std::max<size_t>(1, dda::common::kDdaA2ATdmWindowBytes / tileBytes);
    const size_t windowBytes = tilesPerWave * tileBytes;
    int devMaxSmem = 0;
    {
      int dev = 0;
      CUDACHECK(cudaGetDevice(&dev));
      CUDACHECK(cudaDeviceGetAttribute(&devMaxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    }
    const size_t maxWaves = std::max<size_t>(
        1, std::min<size_t>(static_cast<size_t>(std::clamp<int64_t>(rcclParamDdaA2ATdmWarps(), 1,
                                                                    dda::common::kDdaA2ATdmMaxWarpsPerBlock)),
                            static_cast<size_t>(devMaxSmem) / windowBytes));
    const size_t tiles = (count + tileBytes - 1) / tileBytes;
    DdaA2APeerGeom pg = ddaAllToAllFabricPeerGeom(comm, (tiles + tilesPerWave - 1) / tilesPerWave, maxWaves,
                                                  /*balanceRounds=*/true);
    if (rcclParamDdaA2ATdmBpp() > 0 && pg.blocksPerPeer > 0) {
      // Forced blocks per peer, at most the cap the barrier's flag buffer was sized for.
      const size_t cap = static_cast<size_t>(comm->ddaFabricMaxBlocks) / static_cast<size_t>(nRanks);
      const size_t bpp = std::min<size_t>(static_cast<size_t>(rcclParamDdaA2ATdmBpp()), cap);
      pg.blocksPerPeer = static_cast<int>(bpp);
      pg.grid = dim3(static_cast<unsigned>(bpp * nRanks));
      pg.block = dim3(static_cast<unsigned>(maxWaves * dda::common::kDdaA2ATdmWave));
    }
    if (pg.blocksPerPeer > 0) {
      const int smemBytes = static_cast<int>((pg.block.x / dda::common::kDdaA2ATdmWave) * windowBytes);

      const uint8_t* sendBytes = static_cast<const uint8_t*>(sendbuff);
      uint8_t* recvBytes = static_cast<uint8_t*>(recvbuff);
      uint8_t* const* scratch = reinterpret_cast<uint8_t* const*>(d_ipcbuffs);
      ncclResult_t prep = ncclSuccess;
      switch (tileBytes) {
      case 4 << 10:
        prep = ddaAllToAllFabricTdmPrepare<dda::common::ddaAllToAllFabricTdmPushPeer<4 << 10, 4>>();
        break;
      case 16 << 10:
        prep = ddaAllToAllFabricTdmPrepare<dda::common::ddaAllToAllFabricTdmPushPeer<16 << 10, 1>>();
        break;
      case 20 << 10:
        prep = ddaAllToAllFabricTdmPrepare<dda::common::ddaAllToAllFabricTdmPushPeer<20 << 10, 1>>();
        break;
      case 40 << 10:
        prep = ddaAllToAllFabricTdmPrepare<dda::common::ddaAllToAllFabricTdmPushPeer<40 << 10, 1>>();
        break;
      default:
        prep = ddaAllToAllFabricTdmPrepare<dda::common::ddaAllToAllFabricTdmPushPeer<8 << 10, 2>>();
        break;
      }
      if (prep == ncclSuccess) {
        const unsigned pieces = ddaAllToAllFabricPeerPieces(count, 32);
        const unsigned keep = ddaAllToAllFabricPeerKeep(count, pieces, size_t{64} << 20, size_t{256} << 20);
        INFO(NCCL_COLL,
             "DDA fabric AllToAll: launching peer-major TDM push kernel: nRanks=%d count=%zu tile=%d grid=%u "
             "block=%u blocksPerPeer=%d smem=%d pieces=%u keep=%u",
             nRanks, count, tileBytes, pg.grid.x, pg.block.x, pg.blocksPerPeer, smemBytes, pieces, keep);
        const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
        switch (tileBytes) {
        case 4 << 10:
          hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricTdmPushPeer<4 << 10, 4>), pg.grid, pg.block, smemBytes,
                                stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, scratch, recvBytes,
                                sendBytes, count, comm->rank, nRanks, pg.blocksPerPeer, pieces, keep, barrierHost);
          break;
        case 16 << 10:
          hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricTdmPushPeer<16 << 10, 1>), pg.grid, pg.block,
                                smemBytes, stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, scratch,
                                recvBytes, sendBytes, count, comm->rank, nRanks, pg.blocksPerPeer, pieces, keep,
                                barrierHost);
          break;
        case 20 << 10:
          hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricTdmPushPeer<20 << 10, 1>), pg.grid, pg.block,
                                smemBytes, stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, scratch,
                                recvBytes, sendBytes, count, comm->rank, nRanks, pg.blocksPerPeer, pieces, keep,
                                barrierHost);
          break;
        case 40 << 10:
          hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricTdmPushPeer<40 << 10, 1>), pg.grid, pg.block,
                                smemBytes, stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, scratch,
                                recvBytes, sendBytes, count, comm->rank, nRanks, pg.blocksPerPeer, pieces, keep,
                                barrierHost);
          break;
        default:
          hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricTdmPushPeer<8 << 10, 2>), pg.grid, pg.block, smemBytes,
                                stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, scratch, recvBytes,
                                sendBytes, count, comm->rank, nRanks, pg.blocksPerPeer, pieces, keep, barrierHost);
          break;
        }
        CUDACHECK(cudaGetLastError());
        return ncclSuccess;
      }
    }
  }

  if (rcclParamDdaA2APush() && ddaAllToAllFabricTdmEligible(comm, sendbuff, recvbuff, count)) {
    const int tileBytes = rcclParamDdaA2ATdmTileKB() << 10;
    auto tdmGridBlock = ddaAllToAllFabricTdmGeom(comm, count, tileBytes);
    const auto& tdmGrid = tdmGridBlock.first;
    const auto& tdmBlock = tdmGridBlock.second;
    const int warpsPerBlock = tdmBlock.x / dda::common::kDdaA2ATdmWave;
    const int smemBytes = warpsPerBlock * dda::common::kDdaA2ATdmWindowBytes;
    const uint8_t* sendBytes = static_cast<const uint8_t*>(sendbuff);
    uint8_t* recvBytes = static_cast<uint8_t*>(recvbuff);
    uint8_t* const* scratch = reinterpret_cast<uint8_t* const*>(d_ipcbuffs);

    ncclResult_t prep = ncclSuccess;
    switch (tileBytes) {
    case 4 << 10:
      prep = ddaAllToAllFabricTdmPrepare<dda::common::ddaAllToAllFabricTdmPush<4 << 10, 4>>();
      break;
    case 16 << 10:
      prep = ddaAllToAllFabricTdmPrepare<dda::common::ddaAllToAllFabricTdmPush<16 << 10, 1>>();
      break;
    default:
      prep = ddaAllToAllFabricTdmPrepare<dda::common::ddaAllToAllFabricTdmPush<8 << 10, 2>>();
      break;
    }
    if (prep == ncclSuccess) {
      INFO(NCCL_COLL,
           "DDA fabric AllToAll: launching TDM push kernel: nRanks=%d count=%zu tile=%d grid=%u block=%u smem=%d",
           nRanks, count, tileBytes, tdmGrid.x, tdmBlock.x, smemBytes);
      const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
      switch (tileBytes) {
      case 4 << 10:
        hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricTdmPush<4 << 10, 4>), tdmGrid, tdmBlock, smemBytes,
                              stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, scratch, recvBytes, sendBytes,
                              count, comm->rank, nRanks, barrierHost);
        break;
      case 16 << 10:
        hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricTdmPush<16 << 10, 1>), tdmGrid, tdmBlock, smemBytes,
                              stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, scratch, recvBytes, sendBytes,
                              count, comm->rank, nRanks, barrierHost);
        break;
      default:
        hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricTdmPush<8 << 10, 2>), tdmGrid, tdmBlock, smemBytes,
                              stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, scratch, recvBytes, sendBytes,
                              count, comm->rank, nRanks, barrierHost);
        break;
      }
      CUDACHECK(cudaGetLastError());
      return ncclSuccess;
    }
    INFO(NCCL_COLL, "DDA fabric AllToAll: TDM push unavailable (smem %d), using the vector push", smemBytes);
  }
  // Peer-major push: each block works on exactly one peer. Out of place only: a block drains into recvbuff slots
  // that other blocks of this rank may still be reading from sendbuff, so in-place calls take the all-peer push.
  if (rcclParamDdaA2APush() && rcclParamDdaA2APushPeer() && sendbuff != recvbuff) {
    const size_t vecWaves = (count * sizeof(T) / sizeof(uint4) + 31) / 32;
    const DdaA2APeerGeom pg = ddaAllToAllFabricPeerGeom(comm, vecWaves, 8);
    if (pg.blocksPerPeer > 0) {
      const unsigned pieces = ddaAllToAllFabricPeerPieces(count * sizeof(T), 16);
      const unsigned keep = ddaAllToAllFabricPeerKeep(count * sizeof(T), pieces, size_t{8} << 20, SIZE_MAX);
      INFO(NCCL_COLL,
           "DDA fabric AllToAll: launching peer-major push kernel: nRanks=%d count=%zu grid=%u block=%u "
           "blocksPerPeer=%d pieces=%u keep=%u",
           nRanks, count, pg.grid.x, pg.block.x, pg.blocksPerPeer, pieces, keep);
      const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
      hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricPushPeer<T>), pg.grid, pg.block, 0, stream,
                            /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                            count, static_cast<const T*>(sendbuff), comm->rank, nRanks, pg.blocksPerPeer, pieces,
                            keep, barrierHost);
      CUDACHECK(cudaGetLastError());
      return ncclSuccess;
    }
    INFO(NCCL_COLL, "DDA fabric AllToAll: peer-major push needs %d blocks, only %zu allowed; using the all-peer push",
         nRanks, static_cast<size_t>(comm->ddaFabricMaxBlocks));
  }

  if (rcclParamDdaA2APush()) {
    INFO(NCCL_COLL, "DDA fabric AllToAll: launching push kernel: nRanks=%d count=%zu grid=%u block=%u%s", nRanks,
         count, grid.x, block.x, unroll);

    const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
    const T* sendPtr = static_cast<const T*>(sendbuff);
    T* recvPtr = static_cast<T*>(recvbuff);
    switch (nRanks) {
    case 4:
      hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricPush<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                            stopEvent, /*flags=*/0, d_ipcbuffs, recvPtr, count, sendPtr, comm->rank, nRanks,
                            barrierHost);
      break;
    case 8:
      hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricPush<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                            stopEvent, /*flags=*/0, d_ipcbuffs, recvPtr, count, sendPtr, comm->rank, nRanks,
                            barrierHost);
      break;
    default:
      hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricPush<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                            stopEvent, /*flags=*/0, d_ipcbuffs, recvPtr, count, sendPtr, comm->rank, nRanks,
                            barrierHost);
      break;
    }
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }

  INFO(NCCL_COLL, "DDA fabric AllToAll: launching kernel: nRanks=%d count=%zu grid=%u block=%u%s", nRanks, count,
       grid.x, block.x, unroll);

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
  const auto grid = ddaAllToAllFabricGeom(comm, count * ncclTypeSize(datatype)).first;
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
