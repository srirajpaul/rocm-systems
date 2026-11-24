/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_gather/dda_all_gather.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/all_gather/all_gather_dda_fabric.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>

namespace {

using nccl_dda_detail::DdaFabricBarrierState;

// Single source of the launch geometry: grid/block for a byte payload. The
// kernel is instantiated for int8_t, so `bytes` is the per-block element count.
static inline std::pair<dim3, dim3> ddaAllGatherFabricGeom(ncclComm* comm, size_t bytes) {
  return dda::common::getGridAndBlockDims(bytes, 1, comm->ddaFabricMaxBlocks);
}

template <typename T>
static ncclResult_t ncclAllGatherDdaFabricTyped(const void* sendbuff, void* recvbuff, size_t sendcount, ncclComm* comm,
                                                cudaStream_t stream) {
  const auto scratch = ncclDdaGetScratch(comm, ncclDdaScratchSimple);
  if (comm->ddaFabricMemHandler == nullptr || scratch.scratch == nullptr || scratch.peerPtrsDev == nullptr ||
      comm->ddaFabricBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const int nRanks = comm->nRanks;
  // Each rank stages only its own contribution; user output holds the gathered result.
  const size_t totalCount = sendcount;
  if (!ncclDdaScratchFits("all_gather Simple bytes", totalCount * sizeof(T), scratch.bytes)) {
    WARN("DDA fabric allgather: send element count %zu needs %zu bytes; comm scratch is %zu bytes", sendcount,
         totalCount * sizeof(T), scratch.bytes);
    return ncclInvalidArgument;
  }

  // Block cap chosen at init (barrier flag buffer is sized for it); sendcount is
  // already the byte count (kernel instantiated for int8_t).
  auto gridBlock = ddaAllGatherFabricGeom(comm, sendcount);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = scratch.peerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  INFO(NCCL_COLL, "DDA fabric AllGather: launching kernel: nRanks=%d sendcount=%zu grid=%u block=%u%s", nRanks,
       sendcount, grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // Specialize the kernel for common clique sizes (compile-time NRANKS -> the
  // unrolled CollCommon allGather), and fall back to the runtime kernel
  // (NRANKS_CT == 0) for any other size.
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabric<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), sendcount,
                          static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabric<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), sendcount,
                          static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabric<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), sendcount,
                          static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllGatherDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                    ncclDataType_t datatype) {
  const auto scratch = ncclDdaGetScratch(comm, ncclDdaScratchSimple);
  size_t checkedBytes;
  if (comm == nullptr || comm->nRanks < 1 ||
      !ncclDdaByteCount(sendcount, (size_t)ncclTypeSize(datatype), (size_t)comm->nRanks, &checkedBytes)) return false;

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
  if (scratch.scratch == nullptr || scratch.peerPtrsDev == nullptr) {
    return false;
  }
  if (sendcount == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  size_t need = sendcount * ncclTypeSize(datatype);

  // Check for data size divisible by 16 (kernel does 16-byte vectorized loads)
  if ((sendcount * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return ncclDdaScratchFits("all_gather Simple bytes", need, scratch.bytes);
}

uint32_t ncclAllGatherDdaFabricBlocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype) {
  const auto grid = ddaAllGatherFabricGeom(comm, sendcount * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllGatherDdaFabric(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                    ncclComm* comm, cudaStream_t stream) {
  size_t checkedBytes;
  if (comm == nullptr || comm->nRanks < 1 ||
      !ncclDdaByteCount(sendcount, (size_t)ncclTypeSize(datatype), (size_t)comm->nRanks, &checkedBytes))
    return ncclInvalidArgument;

  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return ncclInvalidArgument;
  }
  int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricTyped<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
