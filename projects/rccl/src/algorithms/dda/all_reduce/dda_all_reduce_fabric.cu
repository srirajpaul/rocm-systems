/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_fabric.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_simple_opt.h"
#include "archinfo.h"
#include "param.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>
#include <utility>

// Independent controls preserve the original path for comparisons and fallbacks.
RCCL_PARAM(DdaSimpleArOpt, "DDA_SIMPLE_AR_OPT", 0);
RCCL_PARAM(DdaSimpleArFused, "DDA_SIMPLE_AR_FUSED", 1);
RCCL_PARAM(DdaSimpleArBatch, "DDA_SIMPLE_AR_BATCH", 1);
RCCL_PARAM(DdaSimpleArThreads, "DDA_SIMPLE_AR_THREADS", 0);
RCCL_PARAM(DdaSimpleArBlocks, "DDA_SIMPLE_AR_BLOCKS", 0);

namespace {

using nccl_dda_detail::DdaFabricBarrierState;

// Flat all-reduce kernel below this size; tree kernel at or above it.
constexpr size_t kDdaFlatTreeThresholdBytes = 1ULL << 18;

static bool ddaSimpleArOpt(ncclComm* comm, size_t bytes) {
  return rcclParamDdaSimpleArOpt() > 0 && bytes > kDdaFlatTreeThresholdBytes &&
      bytes % (comm->nRanks * sizeof(uint4)) == 0 && comm->archName != nullptr &&
      IsArchMatch(comm->archName, "gfx1250");
}

static int ddaSimpleArBatch() {
  const int64_t value = rcclParamDdaSimpleArBatch();
  return value == 2 || value == 4 ? static_cast<int>(value) : 1;
}


// Single source of the launch geometry: grid/block for `count` elements of
// `typeSize` bytes, capped at rcclDdaFabricVmmBlocks() (at most the init-time block count) and at `cap`.
static inline std::pair<dim3, dim3> ddaAllReduceFabricGeom(ncclComm* comm, size_t count, int typeSize,
                                                           rcclDdaLaunchCap cap) {
  size_t blocks = dda::common::ddaCapBlocks((size_t)rcclDdaFabricVmmBlocks(comm), cap.blocks);
  const bool optimized = ddaSimpleArOpt(comm, count * typeSize);
  // An override may exceed the rank-scaled heuristic, but never the allocated
  // barrier capacity or the caller's scheduler budget.
  if (optimized && rcclParamDdaSimpleArBlocks() > 0)
    blocks = dda::common::ddaCapBlocks(static_cast<size_t>(
        std::min<int64_t>(rcclParamDdaSimpleArBlocks(), comm->ddaFabricMaxBlocks)), cap.blocks);
  auto geom = dda::common::getGridAndBlockDims(count, typeSize, blocks);
  if (optimized) {
    const int64_t threads = rcclParamDdaSimpleArThreads();
    if (threads == 128 || threads == 256 || threads == 512) geom.second.x = static_cast<unsigned>(threads);
  }
  geom.second.x = dda::common::ddaCapThreads(geom.second.x, cap.threads);
  return geom;
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                                cudaStream_t stream, rcclDdaLaunchCap cap) {
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

  // At most the block cap chosen at init (barrier flag buffer is sized for it).
  auto gridBlock = ddaAllReduceFabricGeom(comm, count, sizeof(T), cap);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  // Specialize the kernel for common clique sizes (compile-time NRANKS -> the
  // unrolled CollCommon reduce/gather), and fall back to the runtime kernel
  // (NRANKS_CT == 0) for any other size.
  INFO(NCCL_COLL, "DDA fabric AllReduce: launching %s kernel: nRanks=%d count=%zu sizeBytes=%zu grid=%u block=%u%s",
       treeOk ? "tree (two-shot)" : "flat (one-shot)", nRanks, count, sizeBytes, grid.x, block.x,
       (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (treeOk && ddaSimpleArOpt(comm, sizeBytes)) {
    const bool fused = rcclParamDdaSimpleArFused() != 0;
    const int batch = ddaSimpleArBatch();
    INFO(NCCL_COLL, "DDA Simple optimized: ranks=%d bytes=%zu fused=%d batch=%d grid=%u block=%u",
         nRanks, sizeBytes, static_cast<int>(fused), batch, grid.x, block.x);
    if (!fused)
      CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, sizeBytes, cudaMemcpyDeviceToDevice, stream));
#define LAUNCH_SIMPLE_OPT(N, B, F) \
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceSimpleOpt<T, N, B, F>), grid, block, 0, stream, \
        nullptr, stopEvent, 0, d_ipcbuffs, static_cast<T*>(recvbuff), sendbuff == recvbuff ? nullptr : static_cast<const T*>(sendbuff), \
        count, comm->rank, nRanks, barrierHost)
#define LAUNCH_SIMPLE_OPT_F(N, B) \
    do { if (fused) LAUNCH_SIMPLE_OPT(N, B, true); else LAUNCH_SIMPLE_OPT(N, B, false); } while (0)
#define LAUNCH_SIMPLE_OPT_B(N) \
    do { if (batch == 4) LAUNCH_SIMPLE_OPT_F(N, 4); \
         else if (batch == 2) LAUNCH_SIMPLE_OPT_F(N, 2); \
         else LAUNCH_SIMPLE_OPT_F(N, 1); } while (0)
    switch (nRanks) {
    case 4: LAUNCH_SIMPLE_OPT_B(4); break;
    case 8: LAUNCH_SIMPLE_OPT_B(8); break;
    default: LAUNCH_SIMPLE_OPT_B(0); break;
    }
#undef LAUNCH_SIMPLE_OPT_B
#undef LAUNCH_SIMPLE_OPT_F
#undef LAUNCH_SIMPLE_OPT
  } else if (treeOk) {
    CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    // NRANKS_CT 4/8: unrolled CollCommon reduce; 0: runtime fallback.
    switch (nRanks) {
    case 4:
      INFO(NCCL_COLL, "DDA fabric AllReduce: tree path, NRANKS_CT=4 (unrolled)");
      hipExtLaunchKernelGGL((dda::common::ddaAllReduceTreeFabric<T, 4, false>), grid, block, 0, stream,
                            /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                            count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost, nullptr);
      break;
    case 8:
      INFO(NCCL_COLL, "DDA fabric AllReduce: tree path, NRANKS_CT=8 (unrolled)");
      hipExtLaunchKernelGGL((dda::common::ddaAllReduceTreeFabric<T, 8, false>), grid, block, 0, stream,
                            /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                            count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost, nullptr);
      break;
    default:
      INFO(NCCL_COLL, "DDA fabric AllReduce: tree path, NRANKS_CT=0 (runtime, nRanks=%d)", nRanks);
      hipExtLaunchKernelGGL((dda::common::ddaAllReduceTreeFabric<T, 0, false>), grid, block, 0, stream,
                            /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                            count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost, nullptr);
      break;
    }
  } else {
    switch (nRanks) {
    case 4:
      INFO(NCCL_COLL, "DDA fabric AllReduce: flat path, NRANKS_CT=4 (unrolled)");
      hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatFabric<T, 4, false>), grid, block, 0, stream,
                            /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                            count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost, nullptr);
      break;
    case 8:
      INFO(NCCL_COLL, "DDA fabric AllReduce: flat path, NRANKS_CT=8 (unrolled)");
      hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatFabric<T, 8, false>), grid, block, 0, stream,
                            /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                            count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost, nullptr);
      break;
    default:
      INFO(NCCL_COLL, "DDA fabric AllReduce: flat path, NRANKS_CT=0 (runtime, nRanks=%d)", nRanks);
      hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatFabric<T, 0, false>), grid, block, 0, stream,
                            /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff),
                            count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost, nullptr);
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
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
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

uint32_t ncclAllReduceDdaFabricBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype, rcclDdaLaunchCap cap) {
  const auto grid = ddaAllReduceFabricGeom(comm, count, ncclTypeSize(datatype), cap).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllReduceDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaFabricTyped<float>(sendbuff, recvbuff, count, comm, stream, cap);
  case ncclFloat16:
    return ncclAllReduceDdaFabricTyped<half>(sendbuff, recvbuff, count, comm, stream, cap);
  case ncclBfloat16:
    return ncclAllReduceDdaFabricTyped<bf16>(sendbuff, recvbuff, count, comm, stream, cap);
  case ncclFloat8e4m3:
    return ncclAllReduceDdaFabricTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, count, comm, stream, cap);
  case ncclFloat8e5m2:
    return ncclAllReduceDdaFabricTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, count, comm, stream, cap);
  default:
    return ncclInvalidArgument;
  }
}
