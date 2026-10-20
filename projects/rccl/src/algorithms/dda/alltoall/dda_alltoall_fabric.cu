/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/dda_alltoall.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/alltoall/alltoall_dda_fabric.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "param.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>
#include <utility>

// Push model: each rank writes its chunks into peers' scratch instead of reading from them. Env: RCCL_DDA_A2A_PUSH.
RCCL_PARAM(DdaA2APush, "DDA_A2A_PUSH", 0);
// Peer-major push: each block serves one peer (out of place only). Env: RCCL_DDA_A2A_PUSH_PEER.
RCCL_PARAM(DdaA2APushPeer, "DDA_A2A_PUSH_PEER", 0);
// Peer-major push threads per block (one warp .. 512, multiple of 64).
RCCL_PARAM(DdaA2APushPeerThreads, "DDA_A2A_PUSH_PEER_THREADS", 256);

namespace {

using nccl_dda_detail::DdaFabricBarrierState;

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
  dim3 peerGrid, peerBlock;
  if (rcclParamDdaA2APush() && rcclParamDdaA2APushPeer() && sendbuff != recvbuff &&
      ddaAllToAllFabricPushPeerGeom(comm, count, &peerGrid, &peerBlock)) {
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
  if (!rcclParamDdaA2APush() || !rcclParamDdaA2APushPeer() ||
      !ddaAllToAllFabricPushPeerGeom(comm, bytes, &grid, &block)) {
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
