/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "rccl_common.h"
#include "param.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>

// Zero selects 1024 threads for fused staging, or 512 for small sixteen-rank
// messages and the generic path.
// Overrides must agree across ranks; block count cannot exceed allocated flags.
RCCL_PARAM(DdaRsHierThreads, "DDA_RS_HIER_THREADS", 0);
RCCL_PARAM(DdaRsHierBlocks, "DDA_RS_HIER_BLOCKS", 0);
RCCL_PARAM(DdaRsHierPipeline, "DDA_RS_HIER_PIPELINE", 0);
RCCL_PARAM(DdaRsHierChunkBytes, "DDA_RS_HIER_CHUNK_BYTES", 1048576);
// 0: destination-major; 1: rotate from the next rail peer; 2: also interleave waves.
RCCL_PARAM(DdaRsHierSchedule, "DDA_RS_HIER_SCHEDULE", 1);

namespace {

using nccl_dda_detail::DdaFabricBarrierState;

template <typename T>
static ncclResult_t ncclReduceScatterDdaFabricTyped(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                    ncclComm* comm, cudaStream_t stream) {
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaFabricBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const int nRanks = comm->nRanks;
  const size_t totalCount = recvcount * nRanks;
  if (totalCount * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA fabric reduce-scatter: total element count %zu needs %zu bytes; comm scratch is %zu bytes", totalCount,
         totalCount * sizeof(T), comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  // At most the block cap chosen at init (barrier flag buffer is sized for it).
  const int nBlocksMax = rcclDdaFabricVmmBlocks(comm);
  // For reduce-scatter we use recvcount for grid calculation since each rank
  // processes its own shard.
  auto gridBlock = dda::common::getGridAndBlockDims(recvcount, sizeof(T), nBlocksMax);
  auto grid = gridBlock.first;
  auto block = gridBlock.second;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  const bool stageEightRanks = nRanks == 8 && barrierState->rsHierarchy.nScaleUp == 4;
  const bool stageSixteenRanks = nRanks == 16 && barrierState->rsHierarchy.nScaleUp == 8;
  const bool stageInKernel = stageEightRanks || stageSixteenRanks;
  if (!stageInKernel) {
    CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalCount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
  }

  if (barrierState->rsHierarchy.nScaleUp != 0) {
    const int schedule = stageInKernel ? std::clamp<int64_t>(rcclParamDdaRsHierSchedule(), 0, 2) : 0;
    const int tunedThreads = stageSixteenRanks && totalCount * sizeof(T) <= (size_t(8) << 20) ? 512 : 1024;
    const int64_t requestedThreads = rcclParamDdaRsHierThreads() > 0
                                       ? rcclParamDdaRsHierThreads() : (stageInKernel ? tunedThreads : 512);
    const unsigned threads = std::clamp<int64_t>(requestedThreads, 64, 1024) / 64 * 64;
    const size_t vectors = recvcount * sizeof(T) / 16;
    const unsigned cap = rcclParamDdaRsHierBlocks() > 0
                           ? std::min<int64_t>(rcclParamDdaRsHierBlocks(), nBlocksMax) : nBlocksMax;
    grid.x = std::max<size_t>(1, std::min<size_t>(cap, (vectors + threads - 1) / threads));
    if (stageInKernel) {
      const size_t trips = std::max<size_t>(1, (vectors + size_t(cap) * threads - 1) / (size_t(cap) * threads));
      // For short loops, avoid leaving many blocks idle on the last trip.
      // Longer loops benefit more from keeping the full grid's parallelism.
      if (trips <= 3) grid.x = std::max<size_t>(1, (vectors + trips * threads - 1) / (trips * threads));
    }
    block.x = std::min<size_t>(threads, std::max<size_t>(64, (vectors + 63) / 64 * 64));
    if (stageInKernel && rcclParamDdaRsHierPipeline() != 0) {
      const unsigned pipeThreads = rcclParamDdaRsHierThreads() > 0 ? threads : 1024;
      const unsigned producers = std::min<size_t>(cap / 2, vectors / (2 * pipeThreads));
      if (producers != 0) {
        const size_t maxChunk = std::max<size_t>(pipeThreads,
          std::clamp<int64_t>(rcclParamDdaRsHierChunkBytes(), 1024, 1048576) / 16);
        const size_t chunkVectors = std::min<size_t>(maxChunk, vectors / (2 * producers)) / 64 * 64;
        grid.x = 2 * producers;
        block.x = pipeThreads;
        INFO(NCCL_COLL, "DDA fabric ReduceScatter: launching kernel: hierarchical pipeline "
                       "scale-in=2 scaleup=%d nRanks=%d recvcount=%zu grid=%u block=%u chunkBytes=%zu schedule=%d",
             barrierState->rsHierarchy.nScaleUp, nRanks, recvcount, grid.x, block.x, chunkVectors * 16, schedule);
        if (stageEightRanks) {
          hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricHierarchicalPipeline<T, 4>), grid, block, 0, stream,
                             nullptr, rcclTakeAddonStopEvent(comm), 0, d_ipcbuffs, static_cast<T*>(recvbuff),
                             recvcount, comm->rank, barrierState->rsHierarchy, barrierHost,
                             static_cast<const T*>(sendbuff), chunkVectors * 16 / sizeof(T), schedule);
        } else {
          hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricHierarchicalPipeline<T, 8>), grid, block, 0, stream,
                             nullptr, rcclTakeAddonStopEvent(comm), 0, d_ipcbuffs, static_cast<T*>(recvbuff),
                             recvcount, comm->rank, barrierState->rsHierarchy, barrierHost,
                             static_cast<const T*>(sendbuff), chunkVectors * 16 / sizeof(T), schedule);
        }
        CUDACHECK(cudaGetLastError());
        return ncclSuccess;
      }
    }
    INFO(NCCL_COLL, "DDA fabric ReduceScatter: launching kernel: hierarchical scale-in=2 scaleup=%d "
                   "nRanks=%d recvcount=%zu grid=%u block=%u schedule=%d",
         barrierState->rsHierarchy.nScaleUp, nRanks, recvcount, grid.x, block.x, schedule);
    if (stageEightRanks) {
      hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricHierarchical<T, 4, true>), grid, block, 0, stream,
                         /*startEvent=*/nullptr, rcclTakeAddonStopEvent(comm), /*flags=*/0, d_ipcbuffs,
                         static_cast<T*>(recvbuff), recvcount, comm->rank, barrierState->rsHierarchy, barrierHost,
                         static_cast<const T*>(sendbuff), schedule);
    } else if (stageSixteenRanks) {
      hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricHierarchical<T, 8, true>), grid, block, 0, stream,
                         /*startEvent=*/nullptr, rcclTakeAddonStopEvent(comm), /*flags=*/0, d_ipcbuffs,
                         static_cast<T*>(recvbuff), recvcount, comm->rank, barrierState->rsHierarchy, barrierHost,
                         static_cast<const T*>(sendbuff), schedule);
    } else {
      hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricHierarchical<T>), grid, block, 0, stream,
                         /*startEvent=*/nullptr, rcclTakeAddonStopEvent(comm), /*flags=*/0, d_ipcbuffs,
                         static_cast<T*>(recvbuff), recvcount, comm->rank, barrierState->rsHierarchy, barrierHost,
                         static_cast<const T*>(sendbuff), schedule);
    }
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }

  INFO(NCCL_COLL, "DDA fabric ReduceScatter: launching kernel: nRanks=%d recvcount=%zu grid=%u block=%u%s", nRanks,
       recvcount, grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabric<T, 4, false>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                          recvcount, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost, nullptr);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabric<T, 8, false>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                          recvcount, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost, nullptr);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabric<T, 0, false>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                          recvcount, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost, nullptr);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclReduceScatterDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                        ncclDataType_t datatype, ncclRedOp_t op) {
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
  if (recvcount == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  size_t totalCount = recvcount * comm->nRanks;
  size_t need = totalCount * ncclTypeSize(datatype);
  if (need > comm->ddaScratchBytes) {
    return false;
  }

  // Check 16-byte alignment for total data
  if ((totalCount * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  // Check per-rank byte alignment
  if ((recvcount * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return true;
}

ncclResult_t ncclReduceScatterDdaFabric(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                        ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclReduceScatterDdaFabricTyped<float>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat16:
    return ncclReduceScatterDdaFabricTyped<half>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclBfloat16:
    return ncclReduceScatterDdaFabricTyped<bf16>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat8e4m3:
    return ncclReduceScatterDdaFabricTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat8e5m2:
    return ncclReduceScatterDdaFabricTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, recvcount, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
