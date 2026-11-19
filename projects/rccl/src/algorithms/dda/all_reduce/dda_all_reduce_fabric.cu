/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_fabric.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_simple_push.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_simple_peer_warps.h"
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

// Four-rank gfx1250 Simple allreduce. Other rank counts and architectures
// retain the existing implementation and automatic one/two-shot threshold.
RCCL_PARAM(DdaSimpleArPeerPush, "DDA_SIMPLE_AR_PEER_PUSH", 1);
// Opt-in warp-per-peer counterpart of LL128 PeerParallel. Keeps the existing
// Simple push kernel available; requires PeerPush and four-rank gfx1250.
RCCL_PARAM(DdaSimpleArPeerParallel, "DDA_SIMPLE_AR_PEER_PARALLEL", 0);
// Rounds of reduced slices buffered per peer group: 1/2/4. A 512-thread
// CTA has four groups, so these buffer 4/8/16 slices using 8/16/32 KiB LDS.
// Four-rank h13 repeats favor two rounds: eight slices with half the LDS of four rounds.
RCCL_PARAM(DdaSimpleArBatchRounds, "DDA_SIMPLE_AR_BATCH_ROUNDS", 2);
// -1: automatic; 0: force two-shot when shape-eligible; positive: one-shot through that byte count.
RCCL_PARAM(DdaSimpleArOneShotThreshold, "DDA_SIMPLE_AR_ONESHOT_THRESHOLD", -1);

namespace {

using nccl_dda_detail::DdaFabricBarrierState;

static inline bool ddaSimpleArPeerPushEnabled(const ncclComm* comm) {
  return rcclParamDdaSimpleArPeerPush() != 0 && comm != nullptr && comm->nRanks == 4 &&
         comm->archName != nullptr && IsArchMatch(comm->archName, "gfx1250") && comm->WarpSize == 32;
}

static inline size_t ddaSimpleArOneShotThreshold(const ncclComm* comm) {
  const int64_t overrideBytes = rcclParamDdaSimpleArOneShotThreshold();
  if (overrideBytes >= 0) return static_cast<size_t>(overrideBytes);
  // h13 four-rank bfloat16 repeats: flat wins through 3 MiB, while push wins
  // from 3.5 MiB. The legacy path keeps its original 256 KiB crossover.
  return ddaSimpleArPeerPushEnabled(comm) ? (3ULL << 20) : (1ULL << 18);
}

// Single source of the launch geometry: grid/block for `count` elements of
// `typeSize` bytes, capped at rcclDdaFabricVmmBlocks() (at most the init-time block count) and at `cap`.
static inline std::pair<dim3, dim3> ddaAllReduceFabricGeom(ncclComm* comm, size_t count, int typeSize,
                                                           rcclDdaLaunchCap cap) {
  const size_t blocks = dda::common::ddaCapBlocks((size_t)rcclDdaFabricVmmBlocks(comm), cap.blocks);
  auto geom = dda::common::getGridAndBlockDims(count, typeSize, blocks);
  geom.second.x = dda::common::ddaCapThreads(geom.second.x, cap.threads);
  // A peer group needs four complete wave32 warps. Small forced two-shot
  // calls may grow to one group, but an explicit smaller thread cap wins.
  const size_t bytes = count * typeSize;
  if (rcclParamDdaSimpleArPeerParallel() != 0 && ddaSimpleArPeerPushEnabled(comm) &&
      bytes > ddaSimpleArOneShotThreshold(comm) && bytes <= comm->ddaScratchBytes / 2 &&
      (cap.threads == 0 || cap.threads >= 128)) {
    geom.second.x = std::max(128u, (geom.second.x / 128) * 128);
  }
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
  const bool wantTree = sizeBytes > ddaSimpleArOneShotThreshold(comm);
  const bool treeOk = wantTree && (count % static_cast<size_t>(nRanks) == 0);

  if (wantTree && !treeOk) {
    INFO(NCCL_ALL, "DDA fabric: size %zu B > one-shot threshold %zu but count %zu not divisible by %d; using flat kernel",
         sizeBytes, ddaSimpleArOneShotThreshold(comm), count, nRanks);
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
  // Transposed input and reduced output occupy separate message-sized regions.
  // Fall back to the existing tree kernel when the second region does not fit.
  const bool peerPush = treeOk && ddaSimpleArPeerPushEnabled(comm) &&
                        sizeBytes <= comm->ddaScratchBytes / 2;
  if (peerPush && rcclParamDdaSimpleArPeerParallel() != 0 && block.x >= 128 && block.x % 128 == 0) {
    const int64_t requestedRounds = rcclParamDdaSimpleArBatchRounds();
    const int batchRounds = requestedRounds == 4 ? 4 : (requestedRounds == 2 ? 2 : 1);
    INFO(NCCL_COLL, "DDA fabric AllReduce: tree peer-warp path, NRANKS_CT=4 "
                   "groups=%u vectorsPerSlice=128 scratchRegions=2 batchRounds=%d "
                   "slicesPerBatch=%u grid=%u block=%u",
         block.x / 128, batchRounds, (block.x / 128) * batchRounds, grid.x, block.x);
    const auto kernel = batchRounds == 4 ? dda::common::ddaAllReduceTwoShotSimplePeerParallel<T, 4>
                      : batchRounds == 2 ? dda::common::ddaAllReduceTwoShotSimplePeerParallel<T, 2>
                                         : dda::common::ddaAllReduceTwoShotSimplePeerParallel<T, 1>;
    hipExtLaunchKernelGGL(kernel, grid, block, 0, stream,
                         /*startEvent=*/nullptr, stopEvent, /*flags=*/0,
                         d_ipcbuffs, static_cast<T*>(recvbuff), count,
                         static_cast<const T*>(sendbuff), comm->rank, barrierHost);
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }
  if (peerPush) {
    INFO(NCCL_COLL, "DDA fabric AllReduce: tree peer-push path, NRANKS_CT=4 "
                   "scratchRegions=2 shardVectors=%zu grid=%u block=%u",
         sizeBytes / 64, grid.x, block.x);
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTreeFabricPush4<T>), grid, block, 0, stream,
                         /*startEvent=*/nullptr, stopEvent, /*flags=*/0,
                         d_ipcbuffs, static_cast<T*>(recvbuff), count,
                         static_cast<const T*>(sendbuff), comm->rank, barrierHost);
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }
  if (treeOk) {
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
  if (bytes > ddaSimpleArOneShotThreshold(comm)) {
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
