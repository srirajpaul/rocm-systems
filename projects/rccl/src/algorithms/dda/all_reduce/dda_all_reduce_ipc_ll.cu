/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA IPC all-reduce.
 * IPC counterpart of dda_all_reduce_fabric_ll.cu; reuses the same
 * codepath-agnostic kernels from all_reduce_dda_ll.h,
 * all_reduce_dda_ll_twoshot.h, all_reduce_dda_ll128.h, and
 * all_reduce_dda_ll128_twoshot.h.
 *
 * Carries the LL all-reduce tiers (one-shot and two-shot) and the LL128
 * one-shot and two-shot tiers. The variants share a scratch layout and epoch
 * counter; the static_asserts below keep the halved two-shot slots aligned
 * with the one-shot banks.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/all_reduce/all_reduce_dda_ll.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_twoshot.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll_twoshot.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

RCCL_PARAM_DECLARE(DdaLL);
RCCL_PARAM_DECLARE(DdaLLOneShotThreshold);
RCCL_PARAM_DECLARE(DdaLLTwoShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128OneShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128TwoShotThreshold);

namespace {

using dda::common::kDdaLLArSlotStridePkts;
using dda::common::kDdaLLArTwoShotSlotStridePkts;
using dda::common::kDdaLL128ArSlotWords;
using dda::common::kDdaLL128ArTwoShotSlotWords;
using dda::common::kDdaLLMaxBytes;
using dda::common::LLPacket16;

using dda::common::ddaLL128ArDataBytesPerSlice;
using dda::common::ddaLL128ArMaxSlices;
using dda::common::ddaLL128ArSlices;
using dda::common::ddaLL128ArTwoShotMaxSlices;
using dda::common::ddaLL128ArWireWordPerSlice;

using nccl_dda_detail::ddaMaxNBlocksForScratch;

// LL scratch footprint: 2 banks * nRanks slots * slotStride packets * 16B.
static inline size_t ddaLLArScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLArSlotStridePkts * sizeof(LLPacket16);
}

// LL128 one-shot footprint: 2 banks * nRanks slots * slotWords * 8B.
static inline size_t ddaLL128ArOneShotScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLL128ArSlotWords * sizeof(uint64_t);
}

// Two-shot footprint, 2 banks * nRanks slots * slotStride packets * 16B * 2 phases.
static inline size_t ddaLLArTwoShotScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLArTwoShotSlotStridePkts * sizeof(LLPacket16) * 2;
}

// LL128 two-shot footprint: 2 banks * nRanks slots * slotWords * 8B * 2 stages.
static inline size_t ddaLL128ArTwoShotScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLL128ArTwoShotSlotWords * sizeof(uint64_t) * 2;
}

static_assert(kDdaLL128ArTwoShotSlotWords == kDdaLL128ArSlotWords / 2,
              "LL128 all-reduce tiers share one scratch and epoch; \
              keep LL128-ts slot half of LL128 because the scratch is used in two stages");

static_assert(kDdaLLArTwoShotSlotStridePkts == kDdaLLArSlotStridePkts / 2,
              "LL all-reduce tiers share one scratch and epoch; \
              keep LL-ts slot half of LL because the scratch is used in two phases");

static inline int ddaIpcLlMaxBlocks() {
  int n = ddaMaxNBlocksForScratch();
  return n < 1 ? 1 : n;
}

// IPC path: single node, IPC handler, shared scratch/peer table, and the LL
// epoch array allocated at IPC comm init.
static inline bool ddaIpcLlResourcesOk(ncclComm* comm) {
  return comm != nullptr && comm->bootstrap != nullptr && comm->nNodes == 1 && comm->ddaIpcMemHandler != nullptr &&
         comm->ddaScratch != nullptr && comm->ddaPeerPtrsDev != nullptr && comm->ddaLLEpochDev != nullptr;
}

static inline std::pair<dim3, dim3> ddaAllReduceIpcLLGeom(size_t count, int typeSize) {
  const size_t nPk = ((size_t)count * (size_t)typeSize) >> 3; // 8 payload bytes per packet
  const unsigned threads = 256;
  const int nBlocksMax = ddaIpcLlMaxBlocks();
  unsigned blocks = (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  return std::make_pair(dim3(blocks), dim3(threads));
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcLLTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                               cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3;

  const unsigned threads = 512;
  const int nBlocksMax = ddaIpcLlMaxBlocks();
  unsigned blocks = (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL, "DDA IPC AllReduce LL: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u", nRanks, bytes, nPk, grid.x,
       block.x);

  switch (nRanks) {
  case 4:
    dda::common::ddaAllReduceFlatLL<T, 4><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    dda::common::ddaAllReduceFlatLL<T, 8><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    dda::common::ddaAllReduceFlatLL<T, 0><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcLL128OneShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                         ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const int wireWordPerSlice = ddaLL128ArWireWordPerSlice(comm->WarpSize);
  const int dataBytesPerSlice = ddaLL128ArDataBytesPerSlice(comm->WarpSize, comm->ll128LineElems);
  const size_t slices = ddaLL128ArSlices(bytes, comm->WarpSize, comm->ll128LineElems);
  const size_t slotWords = dda::common::kDdaLL128ArSlotWords;

  const unsigned threads = 512;

  using word_type = uint64_t;
  constexpr size_t kWordsPerThread = 4;
  constexpr size_t kBytesPerThread = kWordsPerThread * sizeof(word_type);

  const int kLineWords = comm->ll128LineElems;
  const size_t total_bytes = std::ceil((double)bytes * kLineWords / (kLineWords - 1));
  const size_t nPk = std::ceil((double)total_bytes / kBytesPerThread);

  const int nBlocksMax = ddaIpcLlMaxBlocks();
  unsigned blocks = (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL,
       "DDA IPC AllReduce LL128 one-shot: nRanks=%d bytes=%zu slices=%zu grid=%u block=%u "
       "wave=%d lineElems=%d wire=%dB data=%dB slotWords=%zu",
       nRanks, bytes, slices, grid.x, block.x, comm->WarpSize, comm->ll128LineElems, wireWordPerSlice * 8,
       dataBytesPerSlice, slotWords);

  switch (nRanks) {
  case 4:
    dda::common::ddaAllReduceFlatLL128<T, 4, kWordsPerThread><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), bytes, comm->rank, nRanks, epochDev, epochLen,
      slices, slotWords, wireWordPerSlice, dataBytesPerSlice);
    break;
  case 8:
    dda::common::ddaAllReduceFlatLL128<T, 8, kWordsPerThread><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), bytes, comm->rank, nRanks, epochDev, epochLen,
      slices, slotWords, wireWordPerSlice, dataBytesPerSlice);
    break;
  default:
    dda::common::ddaAllReduceFlatLL128<T, 0, kWordsPerThread><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), bytes, comm->rank, nRanks, epochDev, epochLen,
      slices, slotWords, wireWordPerSlice, dataBytesPerSlice);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcLLTwoShotTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                                      cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = (bytes >> 3) / (size_t)nRanks;

  const unsigned threads = 1024;
  const int nBlocksMax = ddaIpcLlMaxBlocks();
  unsigned blocks = (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL, "DDA IPC AllReduce LL two-shot: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u", nRanks, bytes, nPk,
       grid.x, block.x);

  switch (nRanks) {
  case 4:
    dda::common::ddaAllReduceTwoShotLL<T, 4><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    dda::common::ddaAllReduceTwoShotLL<T, 8><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    dda::common::ddaAllReduceTwoShotLL<T, 0><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcLL128TwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                         ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t shardBytes = bytes / (size_t)nRanks;
  const int wireWordPerSlice = ddaLL128ArWireWordPerSlice(comm->WarpSize);
  const int dataBytesPerSlice = ddaLL128ArDataBytesPerSlice(comm->WarpSize, comm->ll128LineElems);
  const size_t slices = ddaLL128ArSlices(shardBytes, comm->WarpSize, comm->ll128LineElems);
  const size_t slotWords = dda::common::kDdaLL128ArTwoShotSlotWords;

  const unsigned threads = 512;

  using word_type = uint64_t;
  constexpr size_t kWordsPerThread = 4;
  constexpr size_t kBytesPerThread = kWordsPerThread * sizeof(word_type);

  const int kLineWords = comm->ll128LineElems;
  const size_t total_bytes = std::ceil((double)shardBytes * kLineWords / (kLineWords - 1));
  const size_t nPk = std::ceil((double)total_bytes / kBytesPerThread);

  const int nBlocksMax = ddaIpcLlMaxBlocks();
  unsigned blocks = (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL,
       "DDA IPC AllReduce LL128 two-shot: nRanks=%d bytes=%zu shardBytes=%zu slices=%zu grid=%u block=%u "
       "wave=%d lineElems=%d wire=%dB data=%dB slotWords=%zu",
       nRanks, bytes, shardBytes, slices, grid.x, block.x, comm->WarpSize, comm->ll128LineElems, wireWordPerSlice * 8,
       dataBytesPerSlice, slotWords);

  switch (nRanks) {
  case 4:
    dda::common::ddaAllReduceTwoShotLL128<T, 4, kWordsPerThread><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev,
      epochLen, slices, slotWords, wireWordPerSlice, dataBytesPerSlice);
    break;
  case 8:
    dda::common::ddaAllReduceTwoShotLL128<T, 8, kWordsPerThread><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev,
      epochLen, slices, slotWords, wireWordPerSlice, dataBytesPerSlice);
    break;
  default:
    dda::common::ddaAllReduceTwoShotLL128<T, 0, kWordsPerThread><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev,
      epochLen, slices, slotWords, wireWordPerSlice, dataBytesPerSlice);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ddaLLArOneShotIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (rcclParamDdaLL() == 0) {
    return false;
  }

  if (count * ncclTypeSize(datatype) > (size_t)rcclParamDdaLLOneShotThreshold()) {
    return false;
  }

  if (!ddaIpcLlResourcesOk(comm)) {
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
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);

  if (bytes % 16 != 0) {
    return false;
  }
  if (bytes * 2 > kDdaLLMaxBytes) {
    return false;
  }
  if (ddaLLArScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

bool ddaLLArTwoShotIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (rcclParamDdaLL() == 0) {
    return false;
  }

  if (count * ncclTypeSize(datatype) > (size_t)rcclParamDdaLLTwoShotThreshold()) {
    return false;
  }

  if (!ddaIpcLlResourcesOk(comm)) {
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
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
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

  if (bytesPerRank * 2 * 2 > kDdaLLMaxBytes) {
    return false;
  }

  if (ddaLLArTwoShotScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

bool ddaLL128ArOneShotIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (rcclParamDdaLL() == 0) {
    return false;
  }

  if (count * ncclTypeSize(datatype) > (size_t)rcclParamDdaLL128OneShotThreshold()) {
    return false;
  }

  if (!ddaIpcLlResourcesOk(comm)) {
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
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);

  if (bytes % 16 != 0) {
    return false;
  }
  if (ddaLL128ArOneShotScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  if (ddaLL128ArSlices(bytes, comm->WarpSize, comm->ll128LineElems) > ddaLL128ArMaxSlices(comm->WarpSize)) {
    return false;
  }

  return true;
}

bool ddaLL128ArTwoShotIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (rcclParamDdaLL() == 0) {
    return false;
  }

  if (count * ncclTypeSize(datatype) > (size_t)rcclParamDdaLL128TwoShotThreshold()) {
    return false;
  }

  if (!ddaIpcLlResourcesOk(comm)) {
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
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);

  if (bytes % (size_t)comm->nRanks != 0) {
    return false;
  }

  const size_t shardBytes = bytes / (size_t)comm->nRanks;

  if (shardBytes % 16 != 0) {
    return false;
  }
  if (ddaLL128ArTwoShotScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  if (ddaLL128ArSlices(shardBytes, comm->WarpSize, comm->ll128LineElems) >
      ddaLL128ArTwoShotMaxSlices(comm->WarpSize)) {
    return false;
  }

  return true;
}

bool ncclAllReduceDdaIpcLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op) {
  return ddaLLArOneShotIpcEligible(comm, sendbuff, recvbuff, count, datatype, op) ||
         ddaLLArTwoShotIpcEligible(comm, sendbuff, recvbuff, count, datatype, op) ||
         ddaLL128ArOneShotIpcEligible(comm, sendbuff, recvbuff, count, datatype, op) ||
         ddaLL128ArTwoShotIpcEligible(comm, sendbuff, recvbuff, count, datatype, op);
}

ncclResult_t ncclAllReduceDdaIpcLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  const size_t bytes = count * ncclTypeSize(datatype);

  if (ddaLLArOneShotIpcEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL, "AllReduce: taking DDA IPC LL one-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    (void)op;
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaIpcLLTyped<float>(sendbuff, recvbuff, count, comm, stream);
    case ncclFloat16:
      return ncclAllReduceDdaIpcLLTyped<half>(sendbuff, recvbuff, count, comm, stream);
    case ncclBfloat16:
      return ncclAllReduceDdaIpcLLTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
    default:
      return ncclInvalidArgument;
    }
  }

  if (ddaLLArTwoShotIpcEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL, "AllReduce: taking DDA IPC LL two-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaIpcLLTwoShotTyped<float>(sendbuff, recvbuff, count, comm, stream);
    case ncclFloat16:
      return ncclAllReduceDdaIpcLLTwoShotTyped<half>(sendbuff, recvbuff, count, comm, stream);
    case ncclBfloat16:
      return ncclAllReduceDdaIpcLLTwoShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
    default:
      return ncclInvalidArgument;
    }
  }

  if (ddaLL128ArOneShotIpcEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL,
         "AllReduce: taking DDA IPC LL128 one-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    (void)op;
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaIpcLL128OneShotTyped<float>(sendbuff, recvbuff, count, comm, stream);
    case ncclFloat16:
      return ncclAllReduceDdaIpcLL128OneShotTyped<half>(sendbuff, recvbuff, count, comm, stream);
    case ncclBfloat16:
      return ncclAllReduceDdaIpcLL128OneShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
    default:
      return ncclInvalidArgument;
    }
  }

  if (ddaLL128ArTwoShotIpcEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL,
         "AllReduce: taking DDA IPC LL128 two-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaIpcLL128TwoShotTyped<float>(sendbuff, recvbuff, count, comm, stream);
    case ncclFloat16:
      return ncclAllReduceDdaIpcLL128TwoShotTyped<half>(sendbuff, recvbuff, count, comm, stream);
    case ncclBfloat16:
      return ncclAllReduceDdaIpcLL128TwoShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
    default:
      return ncclInvalidArgument;
    }
  }

  WARN("ncclAllReduceDdaIpcLL called for a message no LL tier claims: count=%zu datatype=%d bytes=%zu", count,
       (int)datatype, bytes);
  return ncclInternalError;
}

uint32_t ncclAllReduceDdaIpcLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  (void)comm;
  const auto grid = ddaAllReduceIpcLLGeom(count, ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}
