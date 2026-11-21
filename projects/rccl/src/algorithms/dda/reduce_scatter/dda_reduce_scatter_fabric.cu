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
#include "bootstrap.h"
#include "param.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <type_traits>

// RCCL_DDA_RS_HIER: 1 (default) runs the hierarchical kernel when the comm splits into packages
// (comm->ddaHierScaleIn, gfx1250 partition mode), 0 always runs the flat kernel. Must be the same on every rank.
RCCL_PARAM(DdaRsHier, "DDA_RS_HIER", 1);
// RCCL_DDA_RS_HIER_THRESHOLD: largest total ReduceScatter (bytes) sent to the hierarchical kernel; the selector lifts
// the RS VMM tier to it on partitioned comms. 0 leaves the arch table's VMM cap.
RCCL_PARAM(DdaRsHierThreshold, "DDA_RS_HIER_THRESHOLD", (int64_t)256 * 1024 * 1024);
// RCCL_DDA_RS_HIER_MIN_BYTES: the hierarchical kernel runs only above this total size; at or below it the flat kernel
// is faster (it reads every peer in one pass, the hierarchical kernel takes two passes and one more barrier).
RCCL_PARAM(DdaRsHierMinBytes, "DDA_RS_HIER_MIN_BYTES", (int64_t)16 * 1024 * 1024);
// RCCL_DDA_RS_HIER_MODE: 0 (default) the pull kernel; 1 the push kernel (stores to peers, local reductions, pipelined
// pieces); 2 loads at scale-in and stores at scale-up; 3 the pull kernel synced with epoch flags; 4 the pull kernel
// pipelined over pieces; 5 the pull kernel without its closing comm sync, on the private scratch region of
// RCCL_DDA_RS_HIER_PRIVATE (mode 0 when the comm has none or the message does not fit a half); 6 mode 2 on the
// private region, with a package-only first sync and no closing sync; 7 mode 6 with the staging copy replaced by
// stores into the package peers; 8 mode 6 with partials for owners on other hosts loaded by the owner instead of
// stored (mode 2 for 6-8 without the region).
// RCCL_DDA_RS_HIER_PIECES: pieces per block of modes 1 and 4.
// RCCL_DDA_RS_HIER_PUSH_SYS_STORE: 1 (default) mode 1 pushes with system-scope stores, 0 with plain ones. All must be
// the same on every rank.
RCCL_PARAM(DdaRsHierMode, "DDA_RS_HIER_MODE", 0);
RCCL_PARAM(DdaRsHierPieces, "DDA_RS_HIER_PIECES", 4);
RCCL_PARAM(DdaRsHierPushSysStore, "DDA_RS_HIER_PUSH_SYS_STORE", 1);

namespace {

using nccl_dda_detail::DdaFabricBarrierState;

constexpr const char* kDdaRsHierModeName[] = {"pull",           "push",          "pushup",
                                              "pull-epoch",     "pull-pipe",     "pull-private",
                                              "pushup-private", "pushin-pushup", "pushup-farpull"};

// Consecutive ranks per host when the comm is a series of equal host blocks that each hold whole packages, else
// nRanks (every rank counts as near). The FAR_PULL push-up kernel loads, rather than stores, across hosts.
int ddaHierHostSize(const ncclComm* comm) {
  const ncclPeerInfo* info = comm->peerInfo;
  const int nRanks = comm->nRanks;
  if (info == nullptr || comm->ddaHierScaleIn <= 0) return nRanks;
  int size = 1;
  while (size < nRanks && info[size].hostHash == info[0].hostHash) size++;
  if (nRanks % size != 0 || size % comm->ddaHierScaleIn != 0) return nRanks;
  for (int r = 0; r < nRanks; ++r) {
    if (info[r].hostHash != info[r - r % size].hostHash) return nRanks;
  }
  return size;
}

template <typename T>
static void launchReduceScatterDdaFabricHier(T** d_ipcbuffs, void* recvbuff, size_t recvcount, const void* sendbuff,
                                             ncclComm* comm, int scaleIn, dim3 grid, dim3 block,
                                             const dda::common::FabricGpuBarrier& barrierHost, hipEvent_t stopEvent,
                                             cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  // kernelFor(scaleIn, nGroups) is the instantiation for those compile-time sizes (0 = runtime).
  auto launchSized = [&](auto kernelFor, auto... extra) {
    auto launch = [&](auto kernel) {
      hipExtLaunchKernelGGL(kernel, grid, block, 0, stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs,
                            static_cast<T*>(recvbuff), recvcount, static_cast<const T*>(sendbuff), comm->rank, nRanks,
                            scaleIn, extra..., barrierHost);
    };
    if (scaleIn == 2 && nRanks == 8) {
      launch(kernelFor(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{}));
    } else if (scaleIn == 2 && nRanks == 16) {
      launch(kernelFor(std::integral_constant<int, 2>{}, std::integral_constant<int, 8>{}));
    } else if (scaleIn == 2) {
      launch(kernelFor(std::integral_constant<int, 2>{}, std::integral_constant<int, 0>{}));
    } else {
      launch(kernelFor(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{}));
    }
  };
  int64_t mode = rcclParamDdaRsHierMode();
  const size_t totalBytes = recvcount * nRanks * sizeof(T);
  if (mode == 5 && (comm->ddaHierRsPrivHalfBytes < totalBytes)) {
    mode = 0;
  }
  if (mode == 5) {
    // Alternate the two halves: launch n + 2 reuses launch n's half only after this rank has passed launch n + 1's
    // syncs with every rank that loads from it (its package peers and its tile's ranks), so they are done with it.
    const size_t half = static_cast<size_t>(comm->ddaHierRsPrivCalls++ & 1);
    const size_t scratchOff = (comm->ddaHierRsPrivOffset + half * comm->ddaHierRsPrivHalfBytes) / sizeof(T);
    launchSized(
      [](auto s, auto g) {
        return dda::common::ddaReduceScatterFabricHier<T, decltype(s)::value, decltype(g)::value, false, false>;
      },
      scratchOff);
    return;
  }
  if (mode == 0) {
    launchSized(
      [](auto s, auto g) {
        return dda::common::ddaReduceScatterFabricHier<T, decltype(s)::value, decltype(g)::value, false, true>;
      },
      size_t{0});
    return;
  }
  if (mode == 3) {
    launchSized(
      [](auto s, auto g) {
        return dda::common::ddaReduceScatterFabricHier<T, decltype(s)::value, decltype(g)::value, true, true>;
      },
      size_t{0});
    return;
  }
  if (mode >= 6 && mode <= 8 && comm->ddaHierRsPrivHalfBytes < totalBytes) {
    mode = 2;
  }
  const int hostSize = ddaHierHostSize(comm);
  if (mode >= 6 && mode <= 8) {
    const size_t half = static_cast<size_t>(comm->ddaHierRsPrivCalls++ & 1);
    const size_t scratchOff = (comm->ddaHierRsPrivOffset + half * comm->ddaHierRsPrivHalfBytes) / sizeof(T);
    if (mode == 6) {
      launchSized(
        [](auto s, auto g) {
          return dda::common::ddaReduceScatterFabricHierPushUp<T, decltype(s)::value, decltype(g)::value, true, false,
                                                               false>;
        },
        scratchOff, hostSize);
    } else if (mode == 7) {
      launchSized(
        [](auto s, auto g) {
          return dda::common::ddaReduceScatterFabricHierPushUp<T, decltype(s)::value, decltype(g)::value, true, true,
                                                               false>;
        },
        scratchOff, hostSize);
    } else {
      launchSized(
        [](auto s, auto g) {
          return dda::common::ddaReduceScatterFabricHierPushUp<T, decltype(s)::value, decltype(g)::value, true, false,
                                                               true>;
        },
        scratchOff, hostSize);
    }
    return;
  }
  if (mode == 2) {
    launchSized(
      [](auto s, auto g) {
        return dda::common::ddaReduceScatterFabricHierPushUp<T, decltype(s)::value, decltype(g)::value, false, false,
                                                             false>;
      },
      size_t{0}, hostSize);
    return;
  }
  const unsigned pieces = static_cast<unsigned>(std::max<int64_t>(rcclParamDdaRsHierPieces(), 1));
  if (mode == 4) {
    launchSized(
      [](auto s, auto g) {
        return dda::common::ddaReduceScatterFabricHierPipe<T, decltype(s)::value, decltype(g)::value>;
      },
      pieces);
    return;
  }
  if (rcclParamDdaRsHierPushSysStore()) {
    launchSized(
      [](auto s, auto g) {
        return dda::common::ddaReduceScatterFabricHierPush<T, decltype(s)::value, decltype(g)::value, true>;
      },
      pieces);
  } else {
    launchSized(
      [](auto s, auto g) {
        return dda::common::ddaReduceScatterFabricHierPush<T, decltype(s)::value, decltype(g)::value, false>;
      },
      pieces);
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

  // The hierarchical kernels read sendbuff themselves, so they skip the scratch pre-copy.
  const bool hier = rcclParamDdaRsHier() && totalCount * sizeof(T) > static_cast<size_t>(rcclParamDdaRsHierMinBytes());
  const int scaleIn = hier ? comm->ddaHierScaleIn : 0;
  if (scaleIn > 0) {
    INFO(NCCL_COLL,
         "DDA fabric ReduceScatter: launching hierarchical kernel: nRanks=%d scaleIn=%d recvcount=%zu grid=%u block=%u "
         "%s pieces=%lld hostSize=%d",
         nRanks, scaleIn, recvcount, grid.x, block.x,
         kDdaRsHierModeName[std::clamp<int64_t>(rcclParamDdaRsHierMode(), 0, 8)], (long long)rcclParamDdaRsHierPieces(),
         ddaHierHostSize(comm));
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

ncclResult_t ncclReduceScatterDdaFabricHierDrain(ncclComm* comm) {
  // Mode 5 skips the closing comm sync, so a peer may still load this rank's private region after this rank's last
  // launch. Every rank finishes its own work, then waits for all ranks to do the same before the region is freed.
  if (comm == nullptr || comm->ddaHierRsPrivCalls == 0 || comm->bootstrap == nullptr ||
      (comm->abortFlag != nullptr && __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0)) {
    return ncclSuccess;
  }
  CUDACHECK(cudaDeviceSynchronize());
  NCCLCHECK(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0x2395));
  comm->ddaHierRsPrivCalls = 0;
  return ncclSuccess;
}

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
