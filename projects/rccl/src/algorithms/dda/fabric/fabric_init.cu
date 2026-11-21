/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/fabric/fabric_init.h"

#include "alloc.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "algorithms/dda/fabric/fabric_mem_handler.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "bootstrap.h"
#include "rccl_common.h"
#include "param.h"

#include <cuda_runtime.h>

#include <utility>
#include <vector>

using dda::common::kDdaMaxNranks;
using nccl_dda_detail::ddaFabricMaxNBlocksForScratch;
using nccl_dda_detail::ddaFabricScratchSizing;
using nccl_dda_detail::ddaLLEpochCount;
using nccl_dda_detail::DdaFabricBarrierState;
using nccl_dda_detail::DdaFabricMaxBlocksOverride;

RCCL_PARAM(DdaFabricBufferSizeForScratch, "DDA_FABRIC_BUFFER_SIZE", -1);
// RCCL_DDA_RS_HIER_PRIVATE: 1 appends a private region of two RCCL_DDA_RS_HIER_THRESHOLD halves to the scratch of a
// partitioned comm, for the hierarchical ReduceScatter without a closing barrier (RCCL_DDA_RS_HIER_MODE=5). 0 (default)
// allocates nothing extra. Must be the same on every rank.
RCCL_PARAM(DdaRsHierPrivate, "DDA_RS_HIER_PRIVATE", 0);
RCCL_PARAM_DECLARE(DdaRsHierThreshold);
// RCCL_DDA_A2A_PUSH_PEER_EPOCH: 1 (default) uses epoch flags for the peer-major vector and TDM alltoall pushes, 0 the
// one-token CAS flags (see FabricGpuBarrier::signalWaitPeer / postPeer / waitPeer). Must be the same on every rank.
// With 0 the pushes run without drain blocks (RCCL_DDA_A2A_PUSH_PEER_DRAIN is ignored).
RCCL_PARAM(DdaA2APushPeerEpoch, "DDA_A2A_PUSH_PEER_EPOCH", 1);

bool ncclDdaUseFabricPath(ncclComm* comm) {
  if (comm == nullptr) {
    return false;
  }
  return comm->MNNVL == 1 && IsArchMatch(comm->archName, "gfx1250");
}

// Partitions of one card share every busId bit but the function number.
static bool ddaSameCard(const ncclPeerInfo& a, const ncclPeerInfo& b) {
  return a.hostHash == b.hostHash && (a.busId & ~0xfLL) == (b.busId & ~0xfLL);
}

// Consecutive ranks per physical device in partition mode, or 0 unless every package holds the same
// number of consecutive ranks and there are at least two packages to cross. Same rule as the
// symmetric hierarchical AllGather's scale-in size.
static int ddaHierScaleInSize(ncclComm* comm) {
  const ncclPeerInfo* peerInfo = comm->peerInfo;
  if (!comm->hasMloPart || peerInfo == nullptr || comm->nRanks < 4) return 0;

  int size = 1;
  while (size < comm->nRanks && ddaSameCard(peerInfo[size], peerInfo[0])) size++;
  if (size < 2 || comm->nRanks % size != 0 || comm->nRanks / size < 2) return 0;

  for (int r = 1; r < comm->nRanks; r++) {
    if (ddaSameCard(peerInfo[r], peerInfo[r - 1]) != (r % size != 0)) return 0;
  }
  return size;
}

ncclResult_t ncclDdaFabricCommInit(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }

  if (comm->nRanks < 2 || comm->nRanks > kDdaMaxNranks || comm->bootstrap == nullptr) {
    return ncclSuccess;
  }

  // Fabric DDA assumes every rank shares one UALink clique; skip (fall back to
  // normal RCCL) if this comm spans multiple cliques -- e.g. multiple racks.
  if (comm->clique.size != comm->nRanks) {
    INFO(
      NCCL_INIT,
      "ncclDdaFabricCommInit: comm spans multiple fabric cliques (nRanks %d, clique.size %d); skipping fabric DDA path",
      comm->nRanks, comm->clique.size);
    return ncclSuccess;
  }

  const int nRanks = comm->nRanks;
  const int64_t llEnabled = rcclParamDdaLL();
  const int64_t ll128Enabled = rcclParamDdaLL128();
  // Size scratch to the largest DDA/CE-scratch table cap (all collectives,
  // including graph VMM), not AllReduce VMM alone. Otherwise AG CE-Scratch
  // or AR LL128 can win the selector and then fail the ddaScratchBytes check.
  // ddaVmmMaxGraph is in that max even if this comm never captures: scratch
  // is allocated once and exported across the clique, so a later graph AR
  // must still fit. On gfx1250 that is a silent 256 MiB tax vs eager VMM.
  // RCCL_DDA_FABRIC_BUFFER_SIZE overrides.
  const size_t llThresh = rcclDdaLLThreshold(comm, ncclFuncAllReduce);
  const size_t ll128Thresh = rcclDdaLL128Threshold(comm, ncclFuncAllReduce);
  const size_t simpleThresh = rcclDdaScratchPayloadCap(comm);
  const int64_t fabricScratchOverride = rcclParamDdaFabricBufferSizeForScratch();

  // Right-sized from the DDA thresholds and nRanks (env-overridable) instead of
  // a fixed 10 GiB. RCCL_DDA_FABRIC_BUFFER_SIZE=0 disables the fabric DDA path.
  size_t bytes = ddaFabricScratchSizing(nRanks, fabricScratchOverride, rcclParamDdaEnable(), simpleThresh, llEnabled,
                                        ll128Enabled, ll128Thresh);
  if (bytes == 0) {
    return ncclSuccess;
  }
  // The private region sits past ddaScratchBytes, which every other collective stays below.
  const int hierScaleIn = ddaHierScaleInSize(comm);
  constexpr size_t kPrivAlign = 4096;
  const size_t privOffset = (bytes + kPrivAlign - 1) / kPrivAlign * kPrivAlign;
  size_t privHalf = 0;
  if (rcclParamDdaRsHierPrivate() && hierScaleIn > 0 && rcclParamDdaRsHierThreshold() > 0) {
    privHalf = (static_cast<size_t>(rcclParamDdaRsHierThreshold()) + kPrivAlign - 1) / kPrivAlign * kPrivAlign;
  }
  const size_t allocBytes = privHalf > 0 ? privOffset + 2 * privHalf : bytes;

  // Scratch (temp) buffer via VMM: the fabric path requires a fabric-capable
  // (cuMem) allocation so the handle can be exported across the clique. If VMM
  // is unavailable the fabric path is skipped (DDA disabled, normal RCCL path
  // used). ncclCuMemAlloc rounds size up to the allocation granularity.
  if (!ncclCuMemEnable()) {
    INFO(NCCL_INIT, "ncclDdaFabricCommInit: VMM unavailable; skipping fabric DDA path");
    return ncclSuccess;
  }

  const char* fabricMaxBlocksOverride = getenv("RCCL_DDA_FABRIC_MAXBLOCKS");
  DdaFabricMaxBlocksOverride parsedOverride;
  const int localBlocksMax =
    ddaFabricMaxNBlocksForScratch(comm->cuCount, fabricMaxBlocksOverride, &parsedOverride);

  if (parsedOverride.specified && !parsedOverride.valid) {
    WARN("ncclDdaFabricCommInit: Ignoring invalid RCCL_DDA_FABRIC_MAXBLOCKS='%s'; using CU-derived cap %d.",
         fabricMaxBlocksOverride, localBlocksMax);
  } else if (parsedOverride.valid && parsedOverride.requested < 1) {
    WARN("ncclDdaFabricCommInit: RCCL_DDA_FABRIC_MAXBLOCKS=%ld is below the minimum; using 1.",
         parsedOverride.requested);
  } else if (parsedOverride.valid && parsedOverride.requested > localBlocksMax) {
    WARN("ncclDdaFabricCommInit: RCCL_DDA_FABRIC_MAXBLOCKS=%ld exceeds CU-derived cap (%d); using %d. "
         "The override can only lower the block count, not raise it.",
         parsedOverride.requested, localBlocksMax, localBlocksMax);
  }

  std::vector<int> blockCaps(nRanks, 0);
  blockCaps[comm->rank] = localBlocksMax;
  // Fabric cliques are normally homogeneous, but harvested or disabled CUs
  // can produce different local caps. Every rank must launch the same number
  // of blocks because the barrier pairs participants by blockIdx.
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, blockCaps.data(), sizeof(int)));

  int nBlocksMax = localBlocksMax;
  for (int rank = 0; rank < nRanks; ++rank) {
    if (blockCaps[rank] < nBlocksMax) {
      nBlocksMax = blockCaps[rank];
    }
  }

  const size_t eagerVmmAr = rcclDdaVmmThreshold(comm, ncclFuncAllReduce);
  INFO(NCCL_INIT,
       "ncclDdaFabricCommInit: allocating %zu-byte scratch; payload cap %lld includes graph VMM "
       "(ddaVmmMaxGraph) even for eager-only comms; AR eager VMM=%zu. "
       "Override with RCCL_DDA_FABRIC_BUFFER_SIZE.",
       bytes, (long long)simpleThresh, eagerVmmAr);

  // Owned resources: handed to comm on success, freed at `fail` otherwise.
  // All declared before any goto so the cleanup label is reachable without
  // jumping over an initialization.
  void* scratch = nullptr;
  CUmemGenericAllocationHandle scratchHandle{};
  ncclFabricMemHandler* handler = nullptr;
  void* peerDev = nullptr;
  void** peerHost = nullptr;
  DdaFabricBarrierState* barrierState = nullptr;
  uint32_t* epochDev = nullptr;
  std::vector<void*> h_ptrs(nRanks, nullptr);
  const size_t epochLen = ddaLLEpochCount(nRanks, nBlocksMax);
  ncclResult_t res = ncclSuccess;

  res = ncclCuMemAlloc(&scratch, &scratchHandle, ncclCuMemHandleType, allocBytes, comm->memManager);
  if (res != ncclSuccess || scratch == nullptr) {
    INFO(NCCL_INIT, "ncclDdaFabricCommInit: VMM scratch alloc failed; skipping fabric DDA path");
    scratch = nullptr;
    goto fail;
  }

  handler = new (std::nothrow) ncclFabricMemHandler(comm->bootstrap, comm->rank, nRanks, comm->memManager);
  if (handler == nullptr) {
    WARN("ncclDdaFabricCommInit: OOM allocating ncclFabricMemHandler");
    goto fail;
  }

  NCCLCHECKGOTO(handler->addSelfDeviceMem(scratch, scratchHandle, allocBytes), res, fail);
  NCCLCHECKGOTO(handler->exchangeMemPtrs(), res, fail);

  CUDACHECKGOTO(cudaMalloc(&peerDev, nRanks * sizeof(void*)), res, fail);

  for (int i = 0; i < nRanks; ++i) {
    NCCLCHECKGOTO(handler->getPeerDeviceMemPtr(i, &h_ptrs[i]), res, fail);
  }

  CUDACHECKGOTO(cudaMemcpy(peerDev, h_ptrs.data(), nRanks * sizeof(void*), cudaMemcpyHostToDevice), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&peerHost, nRanks), res, fail);
  CUDACHECKGOTO(cudaMemcpy(peerHost, h_ptrs.data(), nRanks * sizeof(void*), cudaMemcpyHostToHost), res, fail);

  {
    auto cudaOk = [](cudaError_t err, const char* what) {
      if (CUDACLEARERROR(err) != cudaSuccess) {
        WARN("ncclDdaFabricCommInit: %s failed: %s", what, cudaGetErrorString(err));
        return false;
      }
      return true;
    };
    // Past the barrier's pointer exchange, failures set !ready so every rank still reaches the allgather.
    bool ready = true;
    auto barrierPair =
      dda::common::FabricGpuBarrier::mallocAndInit(nRanks, nBlocksMax, comm->rank, comm->bootstrap, comm->memManager,
                                                   dda::common::DdaKernelParams{comm->abortFlagDev});
    if (!barrierPair.first) {
      WARN("ncclDdaFabricCommInit: FabricGpuBarrier malloc/init failed");
      ready = false;
    } else if ((barrierState = new (std::nothrow) DdaFabricBarrierState()) == nullptr) {
      WARN("ncclDdaFabricCommInit: OOM allocating DdaFabricBarrierState");
      ready = false;
    } else {
      barrierState->resources = std::move(barrierPair.first);
      barrierState->barrierHost = barrierPair.second;
      barrierState->barrierHost.setEpochFlags(rcclParamDdaA2APushPeerEpoch() != 0);
    }

    // Own stream: syncing the null stream would also consume async errors from the app's blocking streams.
    cudaStream_t zeroStream = nullptr;
    ready = ready && cudaOk(cudaStreamCreateWithFlags(&zeroStream, cudaStreamNonBlocking), "zero-fill stream create");
    // Zeroed once; later LL calls rely on monotonic epochs (>= 1) and two banks, not re-zeroing.
    ready = ready && cudaOk(cudaMemsetAsync(scratch, 0, allocBytes, zeroStream), "scratch memset");
    // Zeroed epoch cells make the first device-derived LL flag 1.
    ready = ready && cudaOk(cudaMalloc(&epochDev, epochLen * sizeof(uint32_t)), "epoch alloc") &&
            cudaOk(cudaMemsetAsync(epochDev, 0, epochLen * sizeof(uint32_t), zeroStream), "epoch memset");
    // Fills must land before any rank's first LL kernel (and before `fail` frees scratch); the allgather orders ranks.
    if (zeroStream != nullptr) {
      ready = cudaOk(cudaStreamSynchronize(zeroStream), "zero-fill sync") && ready;
      CUDACHECKIGNORE(cudaStreamDestroy(zeroStream));
    }

    std::vector<int> allReady(nRanks, 0);
    allReady[comm->rank] = ready ? 1 : 0;
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allReady.data(), sizeof(int)), res, fail);
    for (int r = 0; r < nRanks; ++r) {
      if (!allReady[r]) {
        INFO(NCCL_INIT, "ncclDdaFabricCommInit: rank %d failed to set up its LL buffers; skipping fabric DDA path", r);
        goto fail;
      }
    }
  }

  // Success: hand ownership of every resource to comm.
  comm->ddaFabricMemHandler = handler;
  comm->ddaScratch = scratch;
  comm->ddaScratchBytes = bytes;
  comm->ddaScratchIsVmm = true;
  comm->ddaPeerPtrsDev = peerDev;
  comm->ddaPeerPtrsHost = peerHost;
  comm->ddaFabricBarrierState = barrierState;
  comm->ddaFabricMaxBlocks = nBlocksMax;
  comm->ddaLLEpochDev = epochDev;
  comm->ddaLLEpochLen = (int)epochLen;
  comm->ddaHierScaleIn = hierScaleIn;
  comm->ddaHierRsPrivOffset = privHalf > 0 ? privOffset : 0;
  comm->ddaHierRsPrivHalfBytes = privHalf;
  comm->ddaHierRsPrivCalls = 0;
  comm->ddaHierRsArriveDev = nullptr;
  if (privHalf > 0) {
    // Mode 9's 2 * kDdaHierMaxPieces arrival counters (zero, and left zero by every launch), then one private-region
    // half cell per block (DdaHierHalf), all zero.
    const size_t kArriveCells = 32 + static_cast<size_t>(nBlocksMax);
    if (cudaMalloc(&comm->ddaHierRsArriveDev, kArriveCells * sizeof(uint32_t)) != cudaSuccess ||
        cudaMemset(comm->ddaHierRsArriveDev, 0, kArriveCells * sizeof(uint32_t)) != cudaSuccess) {
      (void)cudaGetLastError();
      if (comm->ddaHierRsArriveDev != nullptr) CUDACHECKIGNORE(cudaFree(comm->ddaHierRsArriveDev));
      comm->ddaHierRsArriveDev = nullptr;
    }
    INFO(NCCL_INIT, "ncclDdaFabricCommInit: hierarchical RS private scratch: 2 x %zu bytes at offset %zu", privHalf,
         privOffset);
  }
  if (comm->ddaHierScaleIn > 0) {
    INFO(NCCL_INIT, "ncclDdaFabricCommInit: hierarchical scale-in size %d (%d packages of nRanks %d)",
         comm->ddaHierScaleIn, nRanks / comm->ddaHierScaleIn, nRanks);
  }
  INFO(NCCL_INIT,
       "ncclDdaFabricCommInit: nRanks %d, scratch %zu bytes (vmm, gfx1250 fabric path; derived from RCCL DDA params; "
       "RCCL_DDA_FABRIC_BUFFER_SIZE=%lld), LL enabled=%lld threshold=%lld, "
       "LL128 enabled=%lld threshold=%lld, Simple threshold=%lld, FabricGpuBarrier nBlocks=%d, peer table on device",
       nRanks, bytes, (long long)fabricScratchOverride, (long long)llEnabled, (long long)llThresh,
       (long long)ll128Enabled, (long long)ll128Thresh, (long long)simpleThresh, nBlocksMax);
  INFO(NCCL_INIT,
       "ncclDdaFabricCommInit: CU count=%d, local max blocks=%d, communicator max blocks=%d",
       comm->cuCount, localBlocksMax, nBlocksMax);
  return ncclSuccess;

fail:
  if (barrierState != nullptr) {
    delete barrierState;
  }
  if (epochDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(epochDev));
  }
  if (peerHost != nullptr) {
    free(peerHost);
  }
  if (peerDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(peerDev));
  }
  if (handler != nullptr) {
    delete handler;
  }
  if (scratch != nullptr) {
    (void)ncclCuMemFree(scratch, comm->memManager);
  }
  return ncclSuccess;
}

ncclResult_t ncclDdaFabricCommFini(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }
  NCCLCHECK(ncclReduceScatterDdaFabricHierDrain(comm));
  if (comm->ddaFabricBarrierState != nullptr) {
    delete static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
    comm->ddaFabricBarrierState = nullptr;
  }
  CUDACHECKIGNORE(cudaFree(comm->ddaPeerPtrsDev));
  comm->ddaPeerPtrsDev = nullptr;
  CUDACHECKIGNORE(cudaFree(comm->ddaLLEpochDev));
  comm->ddaLLEpochDev = nullptr;
  comm->ddaLLEpochLen = 0;
  comm->ddaHierScaleIn = 0;
  comm->ddaHierRsPrivOffset = 0;
  comm->ddaHierRsPrivHalfBytes = 0;
  CUDACHECKIGNORE(cudaFree(comm->ddaHierRsArriveDev));
  comm->ddaHierRsArriveDev = nullptr;
  free(comm->ddaPeerPtrsHost);
  comm->ddaPeerPtrsHost = nullptr;
  // Destroying the fabric handler unmaps/frees the imported peer scratch buffers.
  if (comm->ddaFabricMemHandler != nullptr) {
    delete static_cast<ncclFabricMemHandler*>(comm->ddaFabricMemHandler);
    comm->ddaFabricMemHandler = nullptr;
  }
  // Free this rank's scratch buffer with the allocator that produced it.
  if (comm->ddaScratch != nullptr) {
    if (comm->ddaScratchIsVmm) {
      (void)ncclCuMemFree(comm->ddaScratch, comm->memManager);
    } else {
      CUDACHECKIGNORE(cudaFree(comm->ddaScratch));
    }
  }
  comm->ddaScratch = nullptr;
  comm->ddaScratchBytes = 0;
  comm->ddaScratchIsVmm = false;
  return ncclSuccess;
}
