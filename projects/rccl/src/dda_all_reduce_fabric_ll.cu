/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launchers + eligibility for the LL-protocol DDA fabric all-reduce, both
 * the one-shot flat variant (ddaAllReduceFlatLL, all_reduce_dda_ll.h) and the
 * two-shot reduce-scatter + all-gather variant (ddaAllReduceTwoShotLL,
 * all_reduce_dda_ll_twoshot.h). Both live here because they form one dispatch
 * tier: they stage in the same scratch bytes and must share the tier's epoch
 * counter (comm->ddaLLArEpochDev) so a stale line from either variant can never
 * false-match the other's flag.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_dda_ll.h"
#include "algorithms/all_reduce/all_reduce_dda_ll_twoshot.h"
#include "checks.h"
#include "comm.h"
#include "dda_init_detail.h" // nccl_dda_detail::kDdaFabricLLArMaxBlocks
#include "debug.h"
#include "fabric_gpu_barrier.h" // meta::comms::kDdaMaxNranks

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

using meta::comms::kDdaLLArMaxBytes;
using meta::comms::kDdaLLArSlotStridePkts;
using meta::comms::kDdaLLArTwoShotMaxBytes;
using meta::comms::kDdaLLArTwoShotMinRanks;
using meta::comms::kDdaLLArTwoShotStages;
using meta::comms::LLPacket16;

// LL scratch footprint: 2 banks * nRanks slots * slotStride packets * 16B.
static inline size_t ddaLLArScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLArSlotStridePkts * sizeof(LLPacket16);
}

// Two-shot LL scratch footprint: 2 banks * 2 stages * nRanks slots * slotStride
// packets * 16B. The slot stride matches the shard exactly (compact per-call
// layout, as in the LL128 all-reduce) so the whole staging area stays small and
// L2-friendly: 8 * message bytes in total.
static inline size_t ddaLLArTwoShotScratchSize(int nRanks, size_t shardPkts) {
  return (size_t)2 * (size_t)kDdaLLArTwoShotStages * (size_t)nRanks * shardPkts * sizeof(LLPacket16);
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLLTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                                  cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet

  const unsigned threads = 256;
  // LL only serves tiny messages (<= DDA_LL_THRESHOLD, 32 KiB) where latency,
  // not occupancy, dominates; cap the grid low so we avoid the launch/sync
  // overhead of a wide grid (LL128/Simple use the full comm->ddaFabricMaxBlocks).
  int nBlocksMax = std::min(comm->ddaFabricMaxBlocks, nccl_dda_detail::kDdaFabricLLArMaxBlocks);
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
  // Dedicated small, high-namespace epoch array for the LL AR tier (see
  // kDdaFabricLLArMaxBlocks / kDdaLLArEpochSeed) so the per-launch epoch reset
  // stays cheap and cannot false-match LL128 flags on the shared scratch.
  uint32_t* epochDev = comm->ddaLLArEpochDev;
  const int epochLen = comm->ddaLLArEpochLen;

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
static ncclResult_t ncclAllReduceDdaFabricLLTwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                         ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3;                    // 8 payload bytes per packet
  const size_t shardPkts = nPk / (size_t)nRanks;    // whole by eligibility
  // Sweeps 1 and 3 walk (peer, packet) pairs, so that, not the shard, sets how
  // wide the grid can usefully be.
  const size_t pairs = (size_t)(nRanks - 1) * shardPkts;

  const unsigned threads = 256;
  // Same cap as the one-shot variant: this tier is latency-bound, and a grid
  // wider than what stays co-resident could stall the flag polling.
  int nBlocksMax = std::min(comm->ddaFabricMaxBlocks, nccl_dda_detail::kDdaFabricLLArMaxBlocks);
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>((pairs + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  // flatBlockId (blockIdx.x) must stay within the device epoch array.
  if ((int)blocks > comm->ddaLLArEpochLen) {
    blocks = (unsigned)comm->ddaLLArEpochLen;
    if (blocks == 0) {
      blocks = 1;
    }
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Shares the LL AR tier's epoch array with the one-shot variant: both stage in
  // the same scratch bytes, so only a common monotonic counter keeps a leftover
  // line from one variant from false-matching the other's flag.
  uint32_t* epochDev = comm->ddaLLArEpochDev;
  const int epochLen = comm->ddaLLArEpochLen;

  INFO(NCCL_COLL, "DDA fabric AllReduce LL two-shot: nRanks=%d bytes=%zu shardPkts=%zu grid=%u block=%u", nRanks, bytes,
       shardPkts, grid.x, block.x);

  // NRANKS_CT 4/8: unrolled publish loops; 0: runtime fallback (up to
  // kDdaMaxNranks).
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllReduceTwoShotLL<T, 4><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                        static_cast<const T*>(sendbuff), count,
                                                                        comm->rank, nRanks, epochDev, epochLen,
                                                                        shardPkts);
    break;
  case 8:
    meta::comms::ddaAllReduceTwoShotLL<T, 8><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                        static_cast<const T*>(sendbuff), count,
                                                                        comm->rank, nRanks, epochDev, epochLen,
                                                                        shardPkts);
    break;
  default:
    meta::comms::ddaAllReduceTwoShotLL<T, 0><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                        static_cast<const T*>(sendbuff), count,
                                                                        comm->rank, nRanks, epochDev, epochLen,
                                                                        shardPkts);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
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
  // Payload is staged as 8-byte LL packets, so it must be a whole number of
  // packets.
  if (bytes % 8 != 0) {
    return false;
  }
  if (bytes > kDdaLLArMaxBytes) {
    return false;
  }
  if (ddaLLArScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllReduceDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
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

bool ncclAllReduceDdaFabricLLTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                             ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // Fabric path: requires the fabric handler + scratch + peer table.
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  // The tier's epoch array carries the flag/bank derivation; without it the
  // kernel cannot sync.
  if (comm->ddaLLArEpochDev == nullptr || comm->ddaLLArEpochLen < 1) {
    return false;
  }
  if (comm->nRanks < kDdaLLArTwoShotMinRanks || comm->nRanks > meta::comms::kDdaMaxNranks) {
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
  // Each rank reduces one shard staged as 8-byte LL packets, so the message must
  // split into a whole number of packets per rank. Anything else falls through
  // to the one-shot LL variant.
  if (bytes % (8 * (size_t)comm->nRanks) != 0) {
    return false;
  }
  if (bytes > kDdaLLArTwoShotMaxBytes) {
    return false;
  }
  const size_t shardPkts = (bytes >> 3) / (size_t)comm->nRanks;
  if (ddaLLArTwoShotScratchSize(comm->nRanks, shardPkts) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllReduceDdaFabricLLTwoShot(const void* sendbuff, void* recvbuff, size_t count,
                                             ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                                             cudaStream_t stream) {
  (void)op;
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
