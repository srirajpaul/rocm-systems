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
#include "bootstrap.h"
#include "rccl_common.h"
#include "param.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

using dda::common::kDdaMaxNranks;
using nccl_dda_detail::ddaFabricMaxNBlocksForScratch;
using nccl_dda_detail::ddaLLEpochCount;
using nccl_dda_detail::DdaFabricBarrierState;
using nccl_dda_detail::DdaFabricMaxBlocksOverride;

RCCL_PARAM(CeScratchSize, "CE_SCRATCH_SIZE", 256LL * 1024 * 1024);
RCCL_PARAM(DdaSimpleScratchSize, "DDA_SIMPLE_SCRATCH_SIZE", 256LL * 1024 * 1024);
RCCL_PARAM(DdaLL128ScratchSize, "DDA_LL128_SCRATCH_SIZE", 64LL * 1024 * 1024);
RCCL_PARAM(DdaLLScratchSize, "DDA_LL_SCRATCH_SIZE", 8LL * 1024 * 1024);

static void freeFabricScratch(ncclComm* comm, ncclDdaFabricScratch& state) {
  for (auto& region : state.regions) {
    CUDACHECKIGNORE(cudaFree(region.epochDev));
    CUDACHECKIGNORE(cudaFree(region.peerPtrsDev));
    free(region.peerPtrsHost);
  }
  CUDACHECKIGNORE(cudaFree(state.tokenPeerPtrsDev));
  if (state.allocation != nullptr) (void)ncclCuMemFree(state.allocation, comm->memManager);
  state = {};
}

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

  if (!rcclParamDdaEnable()) return ncclSuccess;

  const int nRanks = comm->nRanks;
  const char* names[ncclDdaScratchCount] = {"RCCL_CE_SCRATCH_SIZE", "RCCL_DDA_SIMPLE_SCRATCH_SIZE",
                                           "RCCL_DDA_LL128_SCRATCH_SIZE", "RCCL_DDA_LL_SCRATCH_SIZE"};
  struct Config {
    int64_t capacities[ncclDdaScratchCount];
    int valid;
  } local = {{rcclParamCeScratchSize(), rcclParamDdaSimpleScratchSize(),
              rcclParamDdaLL128ScratchSize(), rcclParamDdaLLScratchSize()}, 1};
  // RCCL parameters cache values. Also reject malformed/overflowing input, which
  // the generic parameter reader would otherwise replace with its default.
  for (int i = 0; i < ncclDdaScratchCount; ++i) {
    const char* value = ncclGetEnv(names[i]);
    if (value != nullptr) {
      errno = 0;
      char* end = nullptr;
      const long long parsed = strtoll(value, &end, 0);
      if (errno == ERANGE || end == value || *end != '\0' || parsed < 0) local.valid = 0;
    }
  }
  ncclDdaFabricScratch state{};
  local.valid = local.valid && ncclDdaScratchLayout(local.capacities, kDdaMaxNranks, &state);
  std::vector<Config> configs(nRanks);
  configs[comm->rank] = local;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, configs.data(), sizeof(Config)));
  for (int r = 0; r < nRanks; ++r) {
    if (!configs[r].valid || memcmp(configs[r].capacities, local.capacities, sizeof(local.capacities)) != 0) {
      WARN("Invalid or mismatched fixed DDA scratch capacities at rank %d", r);
      return ncclInvalidArgument;
    }
  }
  if (ncclGetEnv("RCCL_DDA_FABRIC_BUFFER_SIZE") != nullptr) {
    WARN("RCCL_DDA_FABRIC_BUFFER_SIZE is deprecated and ignored; use RCCL_CE_SCRATCH_SIZE, "
         "RCCL_DDA_SIMPLE_SCRATCH_SIZE, RCCL_DDA_LL128_SCRATCH_SIZE and RCCL_DDA_LL_SCRATCH_SIZE (bytes)");
  }
  const size_t bytes = state.allocationBytes;
  if (bytes == 0) return ncclSuccess;

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

  for (int i = 0; i < ncclDdaScratchCount; ++i)
    INFO(NCCL_INIT, "DDA fixed scratch: %s=%zu offset=%zu dataBytes=%zu", names[i], state.capacities[i],
         state.offsets[i], state.regions[i].bytes);

  CUmemGenericAllocationHandle scratchHandle{};
  ncclFabricMemHandler* handler = nullptr;
  DdaFabricBarrierState* barrierState = nullptr;
  const size_t epochLen = ddaLLEpochCount(nRanks, nBlocksMax);
  std::vector<void*> h_ptrs(nRanks, nullptr);
  std::vector<int> allReady(nRanks, 0);
  ncclResult_t res = ncclSuccess;
  auto vote = [&](bool ready) -> ncclResult_t {
    allReady[comm->rank] = ready ? 1 : 0;
    NCCLCHECK(bootstrapAllGather(comm->bootstrap, allReady.data(), sizeof(int)));
    for (int r = 0; r < nRanks; ++r)
      if (!allReady[r]) return ncclSystemError;
    return ncclSuccess;
  };
  auto cudaOk = [](cudaError_t err, const char* what) {
    if (CUDACLEARERROR(err) != cudaSuccess) {
      WARN("ncclDdaFabricCommInit: %s failed: %s", what, cudaGetErrorString(err));
      return false;
    }
    return true;
  };

  res = ncclCuMemAlloc(&state.allocation, &scratchHandle, ncclCuMemHandleType, bytes, comm->memManager);
  handler = new (std::nothrow) ncclFabricMemHandler(comm->bootstrap, comm->rank, nRanks, comm->memManager);
  // All ranks must enter handle exchange together, including allocation failures.
  if (vote(res == ncclSuccess && state.allocation != nullptr && handler != nullptr) != ncclSuccess) goto fail;
  NCCLCHECKGOTO(handler->addSelfDeviceMem(state.allocation, scratchHandle, bytes), res, fail);
  NCCLCHECKGOTO(handler->exchangeMemPtrs(), res, fail);

  {
    bool ready = true;
    auto barrierPair = dda::common::FabricGpuBarrier::mallocAndInit(
      nRanks, nBlocksMax, comm->rank, comm->bootstrap, comm->memManager,
      dda::common::DdaKernelParams{comm->abortFlagDev});
    if (!barrierPair.first) {
      ready = false;
    } else if ((barrierState = new (std::nothrow) DdaFabricBarrierState()) == nullptr) {
      ready = false;
    } else {
      barrierState->resources = std::move(barrierPair.first);
      barrierState->barrierHost = barrierPair.second;
      barrierState->barrierHost.setEpochFlags(rcclParamDdaA2APushPeerEpoch() != 0);
    }

    cudaStream_t zeroStream = nullptr;
    ready = ready && cudaOk(cudaStreamCreateWithFlags(&zeroStream, cudaStreamNonBlocking), "zero stream create");
    ready = ready && cudaOk(cudaMemsetAsync(state.allocation, 0, bytes, zeroStream), "scratch memset");
    for (int i = 0; ready && i < ncclDdaScratchCount; ++i) {
      auto& region = state.regions[i];
      if (region.bytes == 0) continue;
      const size_t tokenBytes = i == ncclDdaScratchLL128 ? 2 * state.tokenBankBytes : 0;
      const size_t offset = state.offsets[i] + tokenBytes;
      region.scratch = static_cast<char*>(state.allocation) + offset;
      for (int r = 0; r < nRanks; ++r) {
        void* base = nullptr;
        ready = ready && handler->getPeerDeviceMemPtr(r, &base) == ncclSuccess;
        h_ptrs[r] = base != nullptr ? static_cast<char*>(base) + offset : nullptr;
      }
      ready = ready && ncclCalloc(&region.peerPtrsHost, nRanks) == ncclSuccess;
      ready = ready && cudaOk(cudaMemcpy(region.peerPtrsHost, h_ptrs.data(), nRanks * sizeof(void*),
                                         cudaMemcpyHostToHost), "host peer table copy");
      ready = ready && cudaOk(cudaMalloc(&region.peerPtrsDev, nRanks * sizeof(void*)), "peer table alloc");
      ready = ready && cudaOk(cudaMemcpy(region.peerPtrsDev, h_ptrs.data(), nRanks * sizeof(void*),
                                         cudaMemcpyHostToDevice), "peer table copy");
      if (i == ncclDdaScratchLL || i == ncclDdaScratchLL128) {
        region.epochLen = (int)epochLen;
        ready = ready && cudaOk(cudaMalloc(&region.epochDev, epochLen * sizeof(uint32_t)), "epoch alloc");
        ready = ready && cudaOk(cudaMemsetAsync(region.epochDev, 0, epochLen * sizeof(uint32_t), zeroStream),
                                 "epoch memset");
      }
      if (ready && i == ncclDdaScratchLL128) {
        for (auto& ptr : h_ptrs) ptr = static_cast<char*>(ptr) - tokenBytes;
        ready = ready && cudaOk(cudaMalloc(&state.tokenPeerPtrsDev, nRanks * sizeof(void*)), "token table alloc");
        ready = ready && cudaOk(cudaMemcpy(state.tokenPeerPtrsDev, h_ptrs.data(), nRanks * sizeof(void*),
                                           cudaMemcpyHostToDevice), "token table copy");
      }
    }
    if (zeroStream != nullptr) {
      ready = cudaOk(cudaStreamSynchronize(zeroStream), "zero-fill sync") && ready;
      CUDACHECKIGNORE(cudaStreamDestroy(zeroStream));
    }
    // Initialization must finish on every rank before any remote flag reader runs.
    if (vote(ready) != ncclSuccess) goto fail;
  }

  comm->ddaFabricScratch = state;
  comm->ddaFabricMemHandler = handler;
  comm->ddaFabricBarrierState = barrierState;
  comm->ddaFabricMaxBlocks = nBlocksMax;
  INFO(NCCL_INIT, "DDA fabric: fixed allocation=%zu bytes, LL128 token bank=%zu bytes, max blocks=%d, epochs=%zu",
       bytes, state.tokenBankBytes, nBlocksMax, epochLen);
  return ncclSuccess;

fail:
  INFO(NCCL_INIT, "ncclDdaFabricCommInit: resource initialization failed; skipping fabric DDA path");
  delete barrierState;
  delete handler;
  freeFabricScratch(comm, state);
  return ncclSuccess;
}

ncclResult_t ncclDdaFabricCommFini(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }
  if (comm->ddaFabricBarrierState != nullptr) {
    delete static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
    comm->ddaFabricBarrierState = nullptr;
  }
  // Imported mappings are released before freeing the sole owning allocation.
  delete comm->ddaFabricMemHandler;
  comm->ddaFabricMemHandler = nullptr;
  freeFabricScratch(comm, comm->ddaFabricScratch);
  return ncclSuccess;
}
