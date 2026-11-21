/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_hier.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "param.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>

// RCCL_DDA_RS_HIER: 1 (default) runs the hierarchical kernel when the comm splits into packages
// (comm->ddaHierScaleIn, gfx1250 partition mode), 0 always runs the flat kernel. Must be the same on every rank.
RCCL_PARAM(DdaRsHier, "DDA_RS_HIER", 1);
// RCCL_DDA_RS_HIER_THRESHOLD: largest total ReduceScatter (bytes) sent to the hierarchical kernel; the selector lifts
// the RS VMM tier to it on partitioned comms. 0 leaves the arch table's VMM cap.
RCCL_PARAM(DdaRsHierThreshold, "DDA_RS_HIER_THRESHOLD", (int64_t)256 * 1024 * 1024);
// RCCL_DDA_RS_HIER_MIN_BYTES: the hierarchical kernel runs only above this total size; at or below it the flat kernel
// is faster (it reads every peer in one pass, the hierarchical kernel takes two passes and one more barrier).
RCCL_PARAM(DdaRsHierMinBytes, "DDA_RS_HIER_MIN_BYTES", (int64_t)16 * 1024 * 1024);

namespace {

using nccl_dda_detail::DdaFabricBarrierState;

template <typename T>
static void launchReduceScatterDdaFabricHier(T** d_ipcbuffs, void* recvbuff, size_t recvcount, const void* sendbuff,
                                             ncclComm* comm, int scaleIn, dim3 grid, dim3 block,
                                             const dda::common::FabricGpuBarrier& barrierHost, hipEvent_t stopEvent,
                                             cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  auto launch = [&](auto kernel) {
    hipExtLaunchKernelGGL(kernel, grid, block, 0, stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs,
                          static_cast<T*>(recvbuff), recvcount, static_cast<const T*>(sendbuff), comm->rank, nRanks,
                          scaleIn, barrierHost);
  };
  if (scaleIn == 2 && nRanks == 8) {
    launch(dda::common::ddaReduceScatterFabricHier<T, 2, 4>);
  } else if (scaleIn == 2 && nRanks == 16) {
    launch(dda::common::ddaReduceScatterFabricHier<T, 2, 8>);
  } else if (scaleIn == 2) {
    launch(dda::common::ddaReduceScatterFabricHier<T, 2, 0>);
  } else {
    launch(dda::common::ddaReduceScatterFabricHier<T, 0, 0>);
  }
}

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
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  // The hierarchical kernel stages its own input, so it skips the scratch pre-copy.
  const bool hier = rcclParamDdaRsHier() && totalCount * sizeof(T) > static_cast<size_t>(rcclParamDdaRsHierMinBytes());
  const int scaleIn = hier ? comm->ddaHierScaleIn : 0;
  if (scaleIn > 0) {
    INFO(NCCL_COLL,
         "DDA fabric ReduceScatter: launching hierarchical kernel: nRanks=%d scaleIn=%d recvcount=%zu grid=%u block=%u",
         nRanks, scaleIn, recvcount, grid.x, block.x);
    launchReduceScatterDdaFabricHier<T>(d_ipcbuffs, recvbuff, recvcount, sendbuff, comm, scaleIn, grid, block,
                                        barrierHost, rcclTakeAddonStopEvent(comm), stream);
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }

  CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalCount * sizeof(T), cudaMemcpyDeviceToDevice, stream));

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

size_t ncclReduceScatterDdaFabricHierMaxBytes(const ncclComm* comm) {
  if (comm == nullptr || comm->ddaHierScaleIn <= 0 || !rcclParamDdaRsHier()) {
    return 0;
  }
  const int64_t cap = rcclParamDdaRsHierThreshold();
  return cap > 0 ? static_cast<size_t>(cap) : 0;
}

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
