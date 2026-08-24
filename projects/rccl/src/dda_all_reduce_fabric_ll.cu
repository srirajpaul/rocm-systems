/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric all-reduce.
 * All-reduce analogue of dda_all_gather_fabric_ll.cu; reuses the codepath-
 * agnostic ddaAllReduceFlatLL kernel from all_reduce_dda_ll.h.
 *
 * Carries the LL all-reduce tiers (one-shot and two-shot) and the LL128 one-shot
 * tier. The two LL variants share a scratch layout and epoch counter, so keeping
 * both launchers in one translation unit also keeps the invariant that ties them
 * (the static_assert below) next to the code it constrains. LL128 one-shot uses
 * the same DDA_LL enable and its own threshold.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_dda_ll.h"
#include "algorithms/all_reduce/all_reduce_dda_ll128.h"
#include "algorithms/all_reduce/all_reduce_dda_ll_twoshot.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "fabric_gpu_barrier.h" // meta::comms::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

RCCL_PARAM_DECLARE(DdaLL);
RCCL_PARAM_DECLARE(DdaLLOneShotThreshold);
RCCL_PARAM_DECLARE(DdaLLTwoShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128OneShotThreshold);

namespace {

using meta::comms::kDdaLLArSlotStridePkts;
using meta::comms::kDdaLLArTwoShotSlotStridePkts;
using meta::comms::kDdaLL128ArSlotWords;
using meta::comms::kDdaLLMaxBytes;
using meta::comms::LLPacket16;

using meta::comms::ddaLL128ArDataBytesPerSlice;
using meta::comms::ddaLL128ArMaxSlices;
using meta::comms::ddaLL128ArSlices;
using meta::comms::ddaLL128ArWireWordPerSlice;

// LL scratch footprint: 2 banks * nRanks slots * slotStride packets * 16B.
static inline size_t ddaLLArScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLArSlotStridePkts * sizeof(LLPacket16);
}

// LL128 one-shot footprint: 2 banks * nRanks slots * slotWords * 8B. Sized off
// kDdaLLMaxBytes so it matches the LL tiers exactly; see kDdaLL128ArSlotWords.
static inline size_t ddaLL128ArOneShotScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLL128ArSlotWords * sizeof(uint64_t);
}

// Two-shot footprint, 2 banks * nRanks slots * slotStride packets * 16B  * 2 phases.
static inline size_t ddaLLArTwoShotScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLArTwoShotSlotStridePkts * sizeof(LLPacket16) * 2;
}

// The two tiers stage through one scratch under one epoch, so bank 1 has to
// start at the same offset for both; otherwise a launch of one could write over
// lines the other is still polling an epoch later.
static_assert(kDdaLLArTwoShotSlotStridePkts == kDdaLLArSlotStridePkts / 2,
              "LL all-reduce tiers share one scratch and epoch; \
              keep LL-ts slot half of LL because the scratch is used in two phases");

// Single source of the launch geometry: 1-D grid over 8-byte LL packets, capped
// low (LL serves tiny messages where latency, not occupancy, dominates).
static inline std::pair<dim3, dim3> ddaAllReduceFabricLLGeom(ncclComm* comm, size_t count, int typeSize) {
  const size_t nPk = ((size_t)count * (size_t)typeSize) >> 3; // 8 payload bytes per packet
  const unsigned threads = 256;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  return std::make_pair(dim3(blocks), dim3(threads));
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLLTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                                  cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet (for logging)

  const unsigned threads = 512;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Shared epoch counter (same as AG/RS) so bank = flag & 1 is consistent
  // across all LL operation types and cannot alias scratch banks.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL, "DDA fabric AllReduce LL: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u", nRanks, bytes, nPk, grid.x,
       block.x);

  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback (up to
  // kDdaMaxNranks).
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllReduceFlatLL<T, 4><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    meta::comms::ddaAllReduceFlatLL<T, 8><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    meta::comms::ddaAllReduceFlatLL<T, 0><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLL128OneShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                            ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const int wireWordPerSlice = ddaLL128ArWireWordPerSlice(comm->WarpSize);
  const int dataBytesPerSlice = ddaLL128ArDataBytesPerSlice(comm->WarpSize, comm->ll128LineElems);
  const size_t slices = ddaLL128ArSlices(bytes, comm->WarpSize, comm->ll128LineElems);
  const size_t slotWords = meta::comms::kDdaLL128ArSlotWords;

  const unsigned threads = 512;

  // word is uint64_t (8 bytes)
  using word_type = uint64_t;
  constexpr size_t kWordsPerThread = 4;
  constexpr size_t kBytesPerThread = kWordsPerThread * sizeof(word_type);

  // kLineWords is 16 for LL128, i.e. 16 uint64_t together form 128B
  const int kLineWords = comm->ll128LineElems;
  // total bytes with flags. we use 1 word from kLineWords as flag
  const size_t total_bytes = std::ceil((double)bytes * kLineWords / (kLineWords - 1));

  const size_t nPk = std::ceil((double)total_bytes / kBytesPerThread);

  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks =
    (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL,
       "DDA fabric AllReduce LL128 one-shot: nRanks=%d bytes=%zu slices=%zu grid=%u block=%u "
       "wave=%d lineElems=%d wire=%dB data=%dB slotWords=%zu",
       nRanks, bytes, slices, grid.x, block.x, comm->WarpSize, comm->ll128LineElems,
       wireWordPerSlice * 8, dataBytesPerSlice, slotWords);

  switch (nRanks) {
  case 4:
    meta::comms::ddaAllReduceFlatLL128<T, 4, kWordsPerThread><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), bytes, comm->rank, nRanks, epochDev, epochLen,
      slices, slotWords, wireWordPerSlice, dataBytesPerSlice);
    break;
  case 8:
    meta::comms::ddaAllReduceFlatLL128<T, 8, kWordsPerThread><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), bytes, comm->rank, nRanks, epochDev, epochLen,
      slices, slotWords, wireWordPerSlice, dataBytesPerSlice);
    break;
  default:
    meta::comms::ddaAllReduceFlatLL128<T, 0, kWordsPerThread><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), bytes, comm->rank, nRanks, epochDev, epochLen,
      slices, slotWords, wireWordPerSlice, dataBytesPerSlice);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// Every phase of the two-shot kernel loops over one shard, with the peer fan-out
// inside it, so the grid covers count/nRanks packets rather than the whole
// message; sizing it on the message would only add blocks whose gtid starts past
// the loop bound.
template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLLTwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                         ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = (bytes >> 3 ) / (size_t)nRanks; // 8 payload bytes per packet

  const unsigned threads = 1024;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter the one-shot tier uses: the two share a scratch layout, so
  // one monotonic flag is what keeps either from accepting a line the other left.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL, "DDA fabric AllReduce LL two-shot: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u", nRanks, bytes, nPk,
       grid.x, block.x);

  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback (up to
  // kDdaMaxNranks).
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllReduceTwoShotLL<T, 4><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    meta::comms::ddaAllReduceTwoShotLL<T, 8><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    meta::comms::ddaAllReduceTwoShotLL<T, 0><<<grid, block, 0, stream>>>(
      peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, nRanks, epochDev, epochLen);
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
  if (rcclParamDdaLL() == 0) {
      return false;
  }

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
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
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
  // expand from 8B to 16B
  if (bytes * 2 > kDdaLLMaxBytes) {
    return false;
  }
  if (ddaLLArScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

// Shape/resource eligibility for the two-shot variant.
bool ddaLLArTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (rcclParamDdaLL() == 0) {
      return false;
  }

  if (count * ncclTypeSize(datatype) > (size_t)rcclParamDdaLLTwoShotThreshold()) {
      return false;
  }

  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // Fabric path: requires the fabric handler + scratch + peer table.
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
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

  // expand from 8B to 16B and two copy phases
  if (bytesPerRank * 2 * 2 > kDdaLLMaxBytes) {
    return false;
  }

  if (ddaLLArTwoShotScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

// Shape/resource eligibility for the LL128 one-shot variant.
bool ddaLL128ArOneShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (rcclParamDdaLL() == 0) {
    return false;
  }

  if (count * ncclTypeSize(datatype) > (size_t)rcclParamDdaLL128OneShotThreshold()) {
    return false;
  }

  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
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

  // The LL128 line format packs 16B-aligned data with no chunk straddling a
  // line, so a partial 16B chunk has nowhere to go.
  if (bytes % 16 != 0) {
    return false;
  }
  if (ddaLL128ArOneShotScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  // A slot is a fixed kDdaLL128ArSlotWords, so the message has to fit the slices
  // that stride holds.
  if (ddaLL128ArSlices(bytes, comm->WarpSize, comm->ll128LineElems) >
      ddaLL128ArMaxSlices(comm->WarpSize)) {
    return false;
  }

  return true;
}

// Tier selection: enabled, within this tier's threshold, and shape-eligible.
// LL one-shot is tested first, then LL two-shot, then LL128 one-shot.
bool ncclAllReduceDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op) {
  return ddaLLArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op) ||
         ddaLLArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op) ||
         ddaLL128ArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op);
}

ncclResult_t ncclAllReduceDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  const size_t bytes = count * ncclTypeSize(datatype);

  if (ddaLLArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL, "AllReduce: taking DDA fabric LL one-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    (void)op;
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLLTyped<float>(sendbuff, recvbuff, count, comm, stream);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLLTyped<half>(sendbuff, recvbuff, count, comm, stream);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLLTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
    default:
      return ncclInvalidArgument;
    }
  }

  if (ddaLLArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL, "AllReduce: taking DDA fabric LL two-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLLTwoShotTyped<float>(sendbuff, recvbuff, count, comm, stream);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLLTwoShotTyped<half>(sendbuff, recvbuff, count, comm, stream);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLLTwoShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
    default:
      return ncclInvalidArgument;
    }
  }

  if (ddaLL128ArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL,
         "AllReduce: taking DDA fabric LL128 one-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    (void)op;
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLL128OneShotTyped<float>(sendbuff, recvbuff, count, comm, stream);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLL128OneShotTyped<half>(sendbuff, recvbuff, count, comm, stream);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLL128OneShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
    default:
      return ncclInvalidArgument;
    }
  }

  // Callers gate on ncclAllReduceDdaFabricLLEligible, which is the disjunction of
  // the tier selectors above, so none matching means the caller skipped it.
  WARN("ncclAllReduceDdaFabricLL called for a message no LL tier claims: count=%zu datatype=%d bytes=%zu", count,
       (int)datatype, bytes);
  return ncclInternalError;
}

uint32_t ncclAllReduceDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const auto grid = ddaAllReduceFabricLLGeom(comm, count, ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}
