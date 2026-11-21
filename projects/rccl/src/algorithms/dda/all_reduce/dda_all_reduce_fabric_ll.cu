/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric all-reduce.
 * All-reduce analogue of dda_all_gather_fabric_ll.cu; reuses the codepath-
 * agnostic ddaAllReduceFlatLL kernel from all_reduce_dda_ll.h.
 *
 * Carries both LL all-reduce tiers: the one-shot kernel above and the two-shot
 * kernel from all_reduce_dda_ll_twoshot.h, which transports one shard per rank
 * instead of the whole message, plus the LL128 one-shot tier from
 * all_reduce_dda_ll128.h. They share a scratch layout and epoch counter, so
 * keeping the launchers in one translation unit also keeps the invariants that
 * tie them (the static_assert below) next to the code it constrains. LL128
 * LL128 one-shot and two-shot are gated by RCCL_DDA_LL128 (not RCCL_DDA_LL); the two protocols are independently controlled.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/all_reduce/all_reduce_dda_ll.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll_twoshot.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_twoshot.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_peer_parallel.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "param.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

RCCL_PARAM_DECLARE(DdaLL);
RCCL_PARAM_DECLARE(DdaLLOneShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128OneShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128AsyncWire);
RCCL_PARAM(DdaLL128ArPeerParallel, "DDA_LL128_AR_PEER_PARALLEL", 1);
RCCL_PARAM(DdaLL128ArPartialReduce, "DDA_LL128_AR_PARTIAL_REDUCE", 0);
RCCL_PARAM(DdaLL128ArPeersPerWarp, "DDA_LL128_AR_PEERS_PER_WARP", 0);
RCCL_PARAM(DdaLL128ArBlockThreads, "DDA_LL128_AR_BLOCK_THREADS", 0);
RCCL_PARAM(DdaLL128ArSliceBatch, "DDA_LL128_AR_SLICE_BATCH", 0);
// -1: measured automatic policy; 0: legacy one-shot; 1: peer-parallel at all supported sizes/ranks.
RCCL_PARAM(DdaLL128ArOneShotPeerParallel, "DDA_LL128_AR_ONESHOT_PEER_PARALLEL", -1);

namespace {

using dda::common::ddaLLArSlotPkts;
using dda::common::ddaLLArTwoShotSlotPkts;

using dda::common::ddaBankSize;
using dda::common::ddaLL128ArSlotWords;
using dda::common::ddaLL128ArTwoShotSlotWords;
using dda::common::ddaLL128Slices;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;

// Block width of each tier when uncapped.
constexpr unsigned kDdaArLLOneShotThreads = 512;
constexpr unsigned kDdaArLLTwoShotThreads = 1024;
constexpr unsigned kDdaArLL128Threads = 512;

// Single source of the LL and LL128 launch geometry: a 1-D grid over `units` of work, `unitThreads` threads each,
// in blocks of `threads`, capped low (LL serves small messages where latency, not occupancy, dominates). `cap` lowers
// the blocks and the threads (dda::common::ddaCapThreads); {0, 0} keeps the tier's.
static inline std::pair<dim3, dim3> ddaArFabricLLGrid(const ncclComm* comm, size_t units, unsigned threads,
                                                      unsigned unitThreads, rcclDdaLaunchCap cap) {
  threads = dda::common::ddaCapThreads(threads, cap.threads);
  const size_t nBlocksMax = dda::common::ddaCapBlocks((size_t)std::max(comm->ddaFabricMaxBlocks, 1), cap.blocks);
  const size_t unitsPerBlock = threads / unitThreads;
  const size_t blocks = std::min((units + unitsPerBlock - 1) / unitsPerBlock, nBlocksMax);
  return std::make_pair(dim3((unsigned)std::max<size_t>(blocks, 1)), dim3(threads));
}

// Per-tier geometry for `bytes` of message, shared by each launcher and the *Blocks reporters.
// LL one-shot: 8-byte packets of the whole message, one thread each.
static inline std::pair<dim3, dim3> ddaArLLOneShotGeom(const ncclComm* comm, size_t bytes, rcclDdaLaunchCap cap) {
  return ddaArFabricLLGrid(comm, bytes >> 3, kDdaArLLOneShotThreads, 1, cap);
}

// LL two-shot: 8-byte packets of one shard, one thread each.
static inline std::pair<dim3, dim3> ddaArLLTwoShotGeom(const ncclComm* comm, size_t bytes, rcclDdaLaunchCap cap) {
  return ddaArFabricLLGrid(comm, (bytes >> 3) / (size_t)std::max(comm->nRanks, 1), kDdaArLLTwoShotThreads, 1, cap);
}

// One-shot uses one complete peer group per full-message slice. Its smaller
// blocks at 4/8 ranks avoid reserving idle groups for small messages.
static bool ddaArLL128OneShotPeerParallel(const ncclComm* comm, size_t bytes, rcclDdaLaunchCap cap) {
  const int n = comm->nRanks;
  const int64_t mode = rcclParamDdaLL128ArOneShotPeerParallel();
  // At four ranks, the legacy one-shot is faster through 8 KiB on h13.
  // An explicit positive mode still permits the peer-parallel kernel there.
  return rcclParamDdaLL128ArPeerParallel() != 0 && mode != 0 && (mode > 0 || n != 4 || bytes > 8 * 1024) &&
         comm->archName != nullptr && IsArchMatch(comm->archName, "gfx1250") &&
         comm->WarpSize == kDdaLL128Warp && (n == 4 || n == 8 || n == 16 || n == 32) &&
         (cap.threads == 0 || cap.threads >= n * kDdaLL128Warp) && rcclParamDdaLL128AsyncWire() == 0;
}

// LL128 one-shot: one block per slice with peer parallelism; otherwise one warp.
static inline std::pair<dim3, dim3> ddaArLL128OneShotGeom(const ncclComm* comm, size_t bytes, rcclDdaLaunchCap cap) {
  if (ddaArLL128OneShotPeerParallel(comm, bytes, cap)) {
    const unsigned threads = comm->nRanks * kDdaLL128Warp;
    return ddaArFabricLLGrid(comm, ddaLL128Slices(bytes), threads, threads, cap);
  }
  return ddaArFabricLLGrid(comm, ddaLL128Slices(bytes), kDdaArLL128Threads, kDdaLL128Warp, cap);
}

// A peer-parallel CTA has 16 warps (32 for 32 ranks), grouped by nRanks. Keep
// the existing kernel for other architectures/rank counts and smaller thread caps.
static bool ddaArLL128PeerParallel(const ncclComm* comm, rcclDdaLaunchCap cap) {
  const int n = comm->nRanks;
  return rcclParamDdaLL128ArPeerParallel() != 0 && comm->archName != nullptr &&
         IsArchMatch(comm->archName, "gfx1250") && comm->WarpSize == kDdaLL128Warp &&
         (n == 4 || n == 8 || n == 16 || n == 32) &&
         (cap.threads == 0 || cap.threads >= (n == 32 ? 1024 : 512)) &&
         rcclParamDdaLL128AsyncWire() == 0;
}

// Explicit experiments at power-of-two ranks. Count the local input as one
// contribution; eight requested contributions at four ranks clamp to four.
// Zero and unsupported settings preserve the existing selection.
static int ddaArLL128PeersPerWarp(int nRanks) {
  const int64_t value = rcclParamDdaLL128ArPeersPerWarp();
  if ((nRanks != 4 && nRanks != 8 && nRanks != 16) ||
      (value != 1 && value != 2 && value != 4 && value != 8)) return 0;
  return value > nRanks ? nRanks : (int)value;
}

// Block tuning is opt-in together with an explicit peer factor. Invalid
// sizes preserve 512 threads; every valid block holds complete slice groups.
static unsigned ddaArLL128BlockThreads(int nRanks, bool partialReduce) {
  const int factor = ddaArLL128PeersPerWarp(nRanks);
  const int64_t threads = rcclParamDdaLL128ArBlockThreads();
  if (factor > 0 && (threads == 64 || threads == 128 || threads == 256 || threads == 512)) {
    const int groupThreads = (nRanks / factor) * kDdaLL128Warp;
    if (threads >= groupThreads && threads % groupThreads == 0) return (unsigned)threads;
  }
  return !partialReduce && nRanks == 32 ? 1024 : 512;
}

// At four ranks, the legacy two-shot sustains higher bandwidth from 64 MiB.
// Share this choice between the geometry reporter and the actual launcher.
static bool ddaArLL128TwoShotPeerParallel(const ncclComm* comm, size_t bytes, rcclDdaLaunchCap cap) {
  // Twelve ranks are supported only by the opt-in eight-warp partial tree.
  // Keep the shared eligibility helper unchanged: one-shot uses it too.
  const bool partial12 = comm->nRanks == 12 && rcclParamDdaLL128ArPartialReduce() != 0 &&
                         rcclParamDdaLL128ArPeerParallel() != 0 && comm->archName != nullptr &&
                         IsArchMatch(comm->archName, "gfx1250") && comm->WarpSize == kDdaLL128Warp &&
                         (cap.threads == 0 || cap.threads >= 512) && rcclParamDdaLL128AsyncWire() == 0;
  return (ddaArLL128PeerParallel(comm, cap) || partial12) &&
         !(comm->nRanks == 4 && bytes >= 64 * 1024 * 1024 && ddaArLL128PeersPerWarp(comm->nRanks) == 0);
}

// An explicit factor takes precedence over the earlier fixed-eight-warp
// partial mode. With neither option, preserve the existing specialization.
static bool ddaArLL128PartialReduce(int nRanks) {
  const int peersPerWarp = ddaArLL128PeersPerWarp(nRanks);
  if (peersPerWarp != 0) return peersPerWarp > 1;
  return nRanks > dda::common::kDdaLL128WordsPerThread && rcclParamDdaLL128ArPartialReduce() != 0;
}

// LL128 two-shot: complete peer-parallel slice groups per CTA, or the original warp per slice.
static inline std::pair<dim3, dim3> ddaArLL128TwoShotGeom(const ncclComm* comm, size_t bytes, rcclDdaLaunchCap cap) {
  const size_t shardBytes = bytes / (size_t)std::max(comm->nRanks, 1);
  if (ddaArLL128TwoShotPeerParallel(comm, bytes, cap)) {
    const bool partialReduce = ddaArLL128PartialReduce(comm->nRanks);
    const unsigned threads = ddaArLL128BlockThreads(comm->nRanks, partialReduce);
    const unsigned peerWarps = dda::common::ddaLL128ArTwoShotPeerWarps(
        comm->nRanks, partialReduce, ddaArLL128PeersPerWarp(comm->nRanks));
    return ddaArFabricLLGrid(comm, ddaLL128Slices(shardBytes), threads,
                             peerWarps * kDdaLL128Warp, cap);
  }
  return ddaArFabricLLGrid(comm, ddaLL128Slices(shardBytes), kDdaArLL128Threads, kDdaLL128Warp, cap);
}

// Preserve the public resource/shape eligibility and explicit threshold overrides.
// Only the new kernel uses the measured default crossover. In particular, an
// unsupported shard shape or a smaller thread cap retains the legacy selector.
static bool ddaArLL128UseOneShot(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op, rcclDdaLaunchCap cap) {
  if (!ddaLL128ArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) return false;
  if (ncclGetEnv("RCCL_DDA_LL128_ONESHOT_THRESHOLD") != nullptr || !ddaArLL128PeerParallel(comm, cap)) return true;
  const size_t bytes = count * ncclTypeSize(datatype);
  // h13 measurements: one-shot wins through 512/256/128/128 KiB at 4/8/16/32
  // ranks. The four-rank small-message legacy kernel shares this one-shot crossover.
  // Preserve the previous crossover when peer-parallel one-shot is disabled.
  const size_t threshold = rcclParamDdaLL128ArOneShotPeerParallel() != 0
                             ? (comm->nRanks == 4 ? 512 * 1024 : comm->nRanks == 8 ? 256 * 1024 : 128 * 1024)
                             : comm->nRanks == 4 ? 2048 : comm->nRanks == 8 ? 512 : 256;
  return bytes <= threshold ||
         !ddaLL128ArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op);
}

// The LDS-staged writers need gfx1250's async LDS stores (ASYNC_COPY_SUPPORTED) and one LL128 warp per wave.
static bool ddaArLL128StagedWireOk(const ncclComm* comm) {
  return comm->archName != nullptr && IsArchMatch(comm->archName, "gfx1250") && comm->WarpSize == kDdaLL128Warp;
}

// Calls pick(std::integral_constant<int, N>{}): NRANKS_CT 4/8 unroll the peer loops, 0 is the runtime fallback.
template <typename Pick>
static auto ddaArPickRanksCt(int nRanks, Pick pick) {
  switch (nRanks) {
  case 4:
    return pick(std::integral_constant<int, 4>{});
  case 8:
    return pick(std::integral_constant<int, 8>{});
  default:
    return pick(std::integral_constant<int, 0>{});
  }
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLLTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                                  cudaStream_t stream, rcclDdaLaunchCap cap) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet (for logging)

  const auto geom = ddaArLLOneShotGeom(comm, bytes, cap);
  const dim3 grid = geom.first;
  const dim3 block = geom.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Shared epoch counter (same as AG/RS) so bank = flag & 1 is consistent
  // across all LL operation types and cannot alias scratch banks.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  INFO(NCCL_COLL, "DDA fabric AllReduce LL: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u", nRanks, bytes, nPk, grid.x,
       block.x);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback (up to
  // kDdaMaxNranks).
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// Smallest per-rank shard whose LL128 two-shot takes the staged writer: enough slices per warp to pay off.
constexpr size_t kDdaArLL128AsyncMinShardBytes = 2 * 1024 * 1024;

// Writer of an LL128 two-shot launch: staged from kDdaArLL128AsyncMinShardBytes where ddaArLL128StagedWireOk.
// Opt-in with RCCL_DDA_LL128_ASYNC_WIRE=1; at 0 (default) every launch takes the cooperative writer and no dynamic LDS.
static int ddaArLL128Wire(const ncclComm* comm, size_t bytes) {
  if (rcclParamDdaLL128AsyncWire() == 0) {
    return dda::common::kDdaLL128WireCoop;
  }
  const size_t shard = bytes / (size_t)std::max(comm->nRanks, 1);
  return ddaArLL128StagedWireOk(comm) && shard >= kDdaArLL128AsyncMinShardBytes ? dda::common::kDdaLL128WireAsync
                                                                                  : dda::common::kDdaLL128WireCoop;
}

// Dynamic LDS of an LL128 launch: the staged writer's two windows per warp.
static size_t ddaArLL128SharedBytes(int wire, unsigned threads) {
  return wire == dda::common::kDdaLL128WireCoop ? 0 : dda::common::ddaLL128StageBytes(threads);
}

// Signature of the LL128 one-shot and two-shot kernels.
template <typename T>
using DdaArLL128Kernel = void (*)(T* const*, T*, const T*, size_t, int, int, uint32_t*, int, size_t, size_t,
                                  dda::common::DdaKernelParams);

template <typename T>
static DdaArLL128Kernel<T> ddaArLL128TwoShotKernel(int nRanks, int wire) {
  return ddaArPickRanksCt(nRanks, [wire](auto n) -> DdaArLL128Kernel<T> {
    constexpr int kN = decltype(n)::value;
    return wire == dda::common::kDdaLL128WireAsync
             ? &dda::common::ddaAllReduceTwoShotLL128<T, kN, dda::common::kDdaLL128WireAsync>
             : &dda::common::ddaAllReduceTwoShotLL128<T, kN, dda::common::kDdaLL128WireCoop>;
  });
}

template <typename T, int N, int P>
static DdaArLL128Kernel<T> ddaArLL128BlockKernel(unsigned threads) {
  constexpr int factor = P == 1 ? 0 : P;
  constexpr int groupThreads = (N / P) * kDdaLL128Warp;
  if constexpr (groupThreads <= 64) {
    if (threads == 64) return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, N, (P > 1), factor, 64>;
  }
  if constexpr (groupThreads <= 128) {
    if (threads == 128) return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, N, (P > 1), factor, 128>;
  }
  if constexpr (groupThreads <= 256) {
    if (threads == 256) return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, N, (P > 1), factor, 256>;
  }
  return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, N, (P > 1), factor>;
}

template <typename T, int N>
static DdaArLL128Kernel<T> ddaArLL128PeersPerWarpKernel(int peersPerWarp, unsigned threads) {
  switch (peersPerWarp) {
  case 2: return ddaArLL128BlockKernel<T, N, 2>(threads);
  case 4: return ddaArLL128BlockKernel<T, N, 4>(threads);
  case 8:
    if constexpr (N >= 8) return ddaArLL128BlockKernel<T, N, 8>(threads);
    else return ddaArLL128BlockKernel<T, N, 4>(threads);
  default: return ddaArLL128BlockKernel<T, N, 1>(threads);
  }
}

// Pair two iterations without changing the grid. Require the explicit
// one-peer, 512-thread control so that other experiments retain their shape.
static bool ddaArLL128SlicePair(int nRanks, bool partialReduce, int peersPerWarp) {
  return rcclParamDdaLL128ArSliceBatch() == 2 && !partialReduce && peersPerWarp == 1 &&
         (nRanks == 4 || nRanks == 8 || nRanks == 16) && ddaArLL128BlockThreads(nRanks, false) == 512;
}

template <typename T>
static DdaArLL128Kernel<T> ddaArLL128PeerParallelKernel(int nRanks, bool partialReduce, int peersPerWarp) {
  if (ddaArLL128SlicePair(nRanks, partialReduce, peersPerWarp)) {
    if (nRanks == 4) return &dda::common::ddaAllReduceTwoShotLL128PeerParallelSlicePair<T, 4>;
    if (nRanks == 8) return &dda::common::ddaAllReduceTwoShotLL128PeerParallelSlicePair<T, 8>;
    return &dda::common::ddaAllReduceTwoShotLL128PeerParallelSlicePair<T, 16>;
  }
  if (peersPerWarp != 0) {
    const unsigned threads = ddaArLL128BlockThreads(nRanks, partialReduce);
    if (nRanks == 4) return ddaArLL128PeersPerWarpKernel<T, 4>(peersPerWarp, threads);
    if (nRanks == 8) return ddaArLL128PeersPerWarpKernel<T, 8>(peersPerWarp, threads);
    return ddaArLL128PeersPerWarpKernel<T, 16>(peersPerWarp, threads);
  }
  if (partialReduce) {
    if (nRanks == 12) return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, 12, true>;
    if (nRanks == 32) return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, 32, true>;
    return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, 16, true>;
  }
  switch (nRanks) {
  case 4: return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, 4>;
  case 8: return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, 8>;
  case 32: return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, 32>;
  default: return &dda::common::ddaAllReduceTwoShotLL128PeerParallel<T, 16>;
  }
}

template <typename T>
static DdaArLL128Kernel<T> ddaArLL128OneShotPeerParallelKernel(int nRanks) {
  switch (nRanks) {
  case 4: return &dda::common::ddaAllReduceOneShotLL128PeerParallel<T, 4>;
  case 8: return &dda::common::ddaAllReduceOneShotLL128PeerParallel<T, 8>;
  case 32: return &dda::common::ddaAllReduceOneShotLL128PeerParallel<T, 32>;
  default: return &dda::common::ddaAllReduceOneShotLL128PeerParallel<T, 16>;
  }
}

// Both one-shot kernels retain the slice-indexed scratch and epoch layout.
template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLL128OneShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                            ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t slices = ddaLL128Slices(bytes);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  const auto geom = ddaArLL128OneShotGeom(comm, bytes, cap);
  const dim3 grid = geom.first;
  const dim3 block = geom.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter the LL tiers use: they share a scratch layout, so one
  // monotonic flag is what keeps either from accepting a line the other left.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  const bool peerParallel = ddaArLL128OneShotPeerParallel(comm, bytes, cap);
  INFO(NCCL_COLL,
       "DDA fabric AllReduce LL128 one-shot: nRanks=%d bytes=%zu slices=%zu grid=%u block=%u "
       "peerParallel=%d (bankSize=%zu)",
       nRanks, bytes, slices, grid.x, block.x, (int)peerParallel, bankSize);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (peerParallel) {
    hipExtLaunchKernelGGL(ddaArLL128OneShotPeerParallelKernel<T>(nRanks), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), bytes, comm->rank, nRanks, epochDev, epochLen,
                          slices, bankSize, params);
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }
  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL128<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          bytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL128<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          bytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL128<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          bytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// The peer-parallel two-shot grid covers 16/nRanks slices per CTA (one at 32 ranks),
// or two with partial reductions at 12/16/32 ranks. The legacy kernel covers 16.
// All retain the scratch layout indexed by shard and slice.
template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLL128TwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                            ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t shardBytes = bytes / (size_t)nRanks;
  const size_t slices = ddaLL128Slices(shardBytes);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  const auto geom = ddaArLL128TwoShotGeom(comm, bytes, cap);
  const dim3 grid = geom.first;
  const dim3 block = geom.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter every other LL/LL128 tier uses.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  const int wire = ddaArLL128Wire(comm, bytes);
  const size_t lds = ddaArLL128SharedBytes(wire, block.x);
  const bool peerParallel = ddaArLL128TwoShotPeerParallel(comm, bytes, cap);
  const bool partialReduce = peerParallel && ddaArLL128PartialReduce(nRanks);
  const int peerFactor = ddaArLL128PeersPerWarp(nRanks);
  const int peerWarps = peerParallel ? dda::common::ddaLL128ArTwoShotPeerWarps(nRanks, partialReduce, peerFactor) : 1;
  const int peersPerWarp = peerParallel ? (nRanks + peerWarps - 1) / peerWarps : 0;

  INFO(NCCL_COLL,
       "DDA fabric AllReduce LL128 two-shot: nRanks=%d bytes=%zu shardBytes=%zu slices=%zu grid=%u block=%u wire=%d "
       "lds=%zu peerParallel=%d partialReduce=%d peersPerWarp=%d peerWarps=%d sliceBatch=%d (bankSize=%zu)",
       nRanks, bytes, shardBytes, slices, grid.x, block.x, wire, lds, (int)peerParallel, (int)partialReduce,
       peersPerWarp, peerWarps, peerParallel && ddaArLL128SlicePair(nRanks, partialReduce, peerFactor) ? 2 : 1, bankSize);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  const auto kernel = peerParallel ? ddaArLL128PeerParallelKernel<T>(nRanks, partialReduce, peerFactor)
                                   : ddaArLL128TwoShotKernel<T>(nRanks, wire);
  hipExtLaunchKernelGGL(kernel, grid, block, lds, stream, /*startEvent=*/nullptr,
                        stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                        shardBytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// Every phase of the two-shot kernel loops over one shard, with the peer fan-out
// inside it, so the grid covers count/nRanks packets rather than the whole
// message; sizing it on the message would only add blocks whose gtid starts past
// the loop bound.
template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLLTwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                         ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = (bytes >> 3 ) / (size_t)nRanks; // 8 payload bytes per packet

  const auto geom = ddaArLLTwoShotGeom(comm, bytes, cap);
  const dim3 grid = geom.first;
  const dim3 block = geom.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter the one-shot tier uses: the two share a scratch layout, so
  // one monotonic flag is what keeps either from accepting a line the other left.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  INFO(NCCL_COLL, "DDA fabric AllReduce LL two-shot: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u", nRanks, bytes, nPk,
       grid.x, block.x);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback (up to
  // kDdaMaxNranks).
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

// Shape/resource eligibility for the one-shot variant, independent of whether
// the tier is switched on for this size.
bool ddaLLArOneShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (count * ncclTypeSize(datatype) > (size_t)rcclParamDdaLLOneShotThreshold()) {
      return false;
  }

  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // Fabric path: requires the fabric handler + scratch + peer table.
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
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

  if (bytes % 16 != 0) {
    return false;
  }
  // One packet carries 8B of payload, so the whole message has to fit the slot
  // the bank gives each rank. A bank too small for one packet per rank takes the
  // tier out entirely.
  if ((bytes >> 3) > ddaLLArSlotPkts(ddaBankSize(comm->ddaScratchBytes), comm->nRanks)) {
    return false;
  }

  return true;
}

// Shape/resource eligibility for the two-shot variant.
bool ddaLLArTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (count * ncclTypeSize(datatype) > rcclDdaLLTwoShotThreshold(comm)) {
      return false;
  }

  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // Fabric path: requires the fabric handler + scratch + peer table.
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
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

  if (bytes % (size_t)comm->nRanks != 0) {
    return false;
  }

  const size_t bytesPerRank = bytes / (size_t)comm->nRanks;

  if (bytesPerRank % 16 != 0) {
    return false;
  }

  // A two-shot slot is half the one-shot's because a bank carries two staging
  // areas, so only the shard has to fit that halved slot.
  if ((bytesPerRank >> 3) > ddaLLArTwoShotSlotPkts(ddaBankSize(comm->ddaScratchBytes), comm->nRanks)) {
    return false;
  }

  return true;
}

// Shape/resource eligibility for the LL128 one-shot variant.
bool ddaLL128ArOneShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op) {
  if (count * ncclTypeSize(datatype) > (size_t)rcclParamDdaLL128OneShotThreshold()) {
    return false;
  }

  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
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

  // The LL128 line format packs 16B-aligned data with no chunk straddling a
  // line, so a partial 16B chunk has nowhere to go, and the 16B wire accesses
  // need both user buffers aligned to match.
  if (bytes % 16 != 0) {
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(sendbuff) % 16) != 0 || (reinterpret_cast<uintptr_t>(recvbuff) % 16) != 0) {
    return false;
  }
  // can the bank hold the data in the slot
  const size_t slotWords = ddaLL128ArSlotWords(ddaBankSize(comm->ddaScratchBytes), comm->nRanks);
  if (ddaLL128Slices(bytes) * (size_t)kDdaLL128WireWordsPerSlice > slotWords) {
    return false;
  }

  return true;
}

// Shape/resource eligibility for the LL128 two-shot variant.
bool ddaLL128ArTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op) {
  if (count * ncclTypeSize(datatype) > rcclDdaLL128TwoShotThreshold(comm)) {
    return false;
  }

  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
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

  // Every rank owns exactly one shard, and the LL128 line format packs 16B-aligned
  // data with no chunk straddling a line, so the shard itself must be a whole
  // number of 16B chunks.
  if (bytes % (size_t)comm->nRanks != 0) {
    return false;
  }

  const size_t shardBytes = bytes / (size_t)comm->nRanks;

  if (shardBytes % 16 != 0) {
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(sendbuff) % 16) != 0 || (reinterpret_cast<uintptr_t>(recvbuff) % 16) != 0) {
    return false;
  }

  // A two-shot slot is half the one-shot's because a bank carries two staging
  // areas, so the shard has to fit the slices that halved stride holds.
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(ddaBankSize(comm->ddaScratchBytes), comm->nRanks);
  if (ddaLL128Slices(shardBytes) * (size_t)kDdaLL128WireWordsPerSlice > slotWords) {
    return false;
  }

  return true;
}

// Tier selection: enabled, within this tier's threshold, and shape-eligible.
// Tested cheapest-first, so a message that qualifies for several takes the
// earliest; a run hands sizes over to a later tier by lowering the earlier
// tier's threshold and raising the later one's.
bool ncclAllReduceDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op) {
  return ddaLLArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op) ||
         ddaLLArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op);
}

bool ncclAllReduceDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype, ncclRedOp_t op) {
  return ddaLL128ArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op) ||
         ddaLL128ArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op);
}

ncclResult_t ncclAllReduceDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  const size_t bytes = count * ncclTypeSize(datatype);

  if (ddaLLArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL, "AllReduce: taking DDA fabric LL one-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    (void)op;
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLLTyped<float>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLLTyped<half>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLLTyped<bf16>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e4m3:
      return ncclAllReduceDdaFabricLLTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e5m2:
      return ncclAllReduceDdaFabricLLTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, count, comm, stream, cap);
    default:
      return ncclInvalidArgument;
    }
  }

  if (ddaLLArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL, "AllReduce: taking DDA fabric LL two-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLLTwoShotTyped<float>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLLTwoShotTyped<half>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLLTwoShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e4m3:
      return ncclAllReduceDdaFabricLLTwoShotTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e5m2:
      return ncclAllReduceDdaFabricLLTwoShotTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, count, comm, stream, cap);
    default:
      return ncclInvalidArgument;
    }
  }

  // Callers gate on ncclAllReduceDdaFabricLLEligible, which is the disjunction of
  // the two selectors above, so neither matching means the caller skipped it.
  WARN("ncclAllReduceDdaFabricLL called for a message no LL tier claims: count=%zu datatype=%d bytes=%zu", count,
       (int)datatype, bytes);
  return ncclInternalError;
}

ncclResult_t ncclAllReduceDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  const size_t bytes = count * ncclTypeSize(datatype);

  if (ddaArLL128UseOneShot(comm, sendbuff, recvbuff, count, datatype, op, cap)) {
    INFO(NCCL_COLL,
         "AllReduce: taking DDA fabric LL128 one-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLL128OneShotTyped<float>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLL128OneShotTyped<half>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLL128OneShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e4m3:
      return ncclAllReduceDdaFabricLL128OneShotTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, count, comm, stream,
                                                                           cap);
    case ncclFloat8e5m2:
      return ncclAllReduceDdaFabricLL128OneShotTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, count, comm, stream,
                                                                           cap);
    default:
      return ncclInvalidArgument;
    }
  }

  if (ddaLL128ArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL,
         "AllReduce: taking DDA fabric LL128 two-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLL128TwoShotTyped<float>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLL128TwoShotTyped<half>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLL128TwoShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e4m3:
      return ncclAllReduceDdaFabricLL128TwoShotTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, count, comm, stream,
                                                                           cap);
    case ncclFloat8e5m2:
      return ncclAllReduceDdaFabricLL128TwoShotTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, count, comm, stream,
                                                                           cap);
    default:
      return ncclInvalidArgument;
    }
  }

  WARN("ncclAllReduceDdaFabricLL128 called for a message no LL128 tier claims: count=%zu datatype=%d bytes=%zu", count,
       (int)datatype, bytes);
  return ncclInternalError;
}

// Reports the grid of the tier ncclAllReduceDdaFabricLL launches: one-shot when eligible, else two-shot.
uint32_t ncclAllReduceDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype, rcclDdaLaunchCap cap) {
  const size_t bytes = count * ncclTypeSize(datatype);
  const bool oneShot = ddaLLArOneShotEligible(comm, nullptr, nullptr, count, datatype, ncclSum);
  const dim3 grid = (oneShot ? ddaArLLOneShotGeom(comm, bytes, cap) : ddaArLLTwoShotGeom(comm, bytes, cap)).first;
  return grid.x * grid.y;
}

// Reports the grid of the tier ncclAllReduceDdaFabricLL128 launches: one-shot when eligible, else two-shot.
uint32_t ncclAllReduceDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype,
                                           rcclDdaLaunchCap cap) {
  const size_t bytes = count * ncclTypeSize(datatype);
  const bool oneShot = ddaArLL128UseOneShot(comm, nullptr, nullptr, count, datatype, ncclSum, cap);
  const dim3 grid =
      (oneShot ? ddaArLL128OneShotGeom(comm, bytes, cap) : ddaArLL128TwoShotGeom(comm, bytes, cap)).first;
  return grid.x * grid.y;
}
