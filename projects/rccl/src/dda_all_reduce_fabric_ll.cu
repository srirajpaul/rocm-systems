/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric all-reduce.
 * All-reduce analogue of dda_all_gather_fabric_ll.cu; reuses the codepath-
 * agnostic ddaAllReduceFlatLL kernel from all_reduce_dda_ll.h.
 *
 * Two LL variants dispatch from here, both codepath-agnostic kernels shared with
 * the IPC launcher: ddaAllReduceFlatLL (one-shot, whole message to every peer)
 * and ddaAllReduceTwoShotLL (reduce-scatter then all-gather over shards, which
 * moves nRanks/2 fewer bytes per rank for a second round trip). They are separate
 * dispatch tiers -- collectives.cc sends the small messages to the one-shot entry
 * point and the larger ones to the two-shot one -- and both advance the same LL
 * epoch counter, which is what keeps a leftover line from either scratch layout
 * from matching the other's flag.
 *
 * The two-shot kernel works one shard at a time -- count / nRanks elements -- with
 * the peer fan-out inside each phase, so its launcher sizes the grid on the shard
 * rather than on the whole message the way the one-shot launcher does.
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
using meta::comms::kDdaLLArTwoShotSlotStridePkts;
using meta::comms::LLPacket16;

// Both tiers stage through one scratch under one epoch counter, so they have to
// agree on where bank 1 starts or one can overwrite lines the other is still
// polling. The two-shot layout spans two stages of half-size slots per bank, so
// the strides line up exactly when its slot is half the one-shot slot.
static_assert(2 * kDdaLLArTwoShotSlotStridePkts == kDdaLLArSlotStridePkts,
              "LL all-reduce tiers must bank at the same stride; keep kDdaLLArMaxBytes == kDdaLLArTwoShotMaxBytes");

// LL scratch footprint: 2 banks * nRanks slots * slotStride packets * 16B.
static inline size_t ddaLLArScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLArSlotStridePkts * sizeof(LLPacket16);
}

// Two-shot tier scratch footprint. Its slot stride is fixed like the one-shot
// layout above, so this does not depend on the message either -- a scratch too
// small for it rules the tier out at any size.
static inline size_t ddaLLArTwoShotScratchSize(int nRanks) {
  // 2 banks * 2 stages (reduce-scatter, then all-gather) * nRanks slots.
  const size_t regions = (size_t)2 * 2 * (size_t)nRanks;
  return regions * kDdaLLArTwoShotSlotStridePkts * sizeof(LLPacket16);
}

// Requirements both LL variants share on the fabric path: the resources they
// stage through, the clique shape, and a payload that is a whole number of
// 8-byte LL packets. Each variant adds its own size cap and scratch-fit check.
static bool ddaLLArFabricCommon(const ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op) {
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
  // Payload is staged as 8-byte LL packets, so it must be a whole number of
  // packets.
  return (count * ncclTypeSize(datatype)) % 8 == 0;
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
static ncclResult_t ncclAllReduceDdaFabricLLTwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                         ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet
  if (nPk % (size_t)nRanks != 0) {
    WARN("DDA fabric AllReduce LL two-shot: %zu bytes does not divide evenly across %d ranks", bytes, nRanks);
    return ncclInvalidArgument;
  }
  const size_t nPkRank = nPk / (size_t)nRanks;

  const unsigned threads = 256;
  // Every phase loops over one shard with the peer fan-out inside it, so the grid
  // covers nPkRank rather than the whole message: sizing it on nPk would only add
  // blocks whose gtid starts past the loop bound.
  int nBlocksMax = std::min(comm->ddaFabricMaxBlocks, nccl_dda_detail::kDdaFabricLLArMaxBlocks);
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>((nPkRank + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter every LL collective on this path uses, so the monotonic
  // flag stops this kernel from ever accepting a line another layout left behind.
  // The flag alone would not stop a foreign layout from writing over a line this
  // kernel is still polling; the static_assert above is what rules that out
  // against the one-shot all-reduce, by keeping the two bank strides equal.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL, "DDA fabric AllReduce LL two-shot: nRanks=%d bytes=%zu nPk=%zu nPkRank=%zu grid=%u block=%u", nRanks,
       bytes, nPk, nPkRank, grid.x, block.x);

  // The slot stride is a constant both sides read, so nothing about the layout is
  // passed in; ddaLLArTwoShotScratchSize sizes the fit check from that same
  // constant. NRANKS_CT 4/8: unrolled peer loops; 0: runtime fallback.
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

bool ncclAllReduceDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (!ddaLLArFabricCommon(comm, count, datatype, op)) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes > kDdaLLArMaxBytes) {
    return false;
  }
  // The one-shot slot stride is fixed at the cap, so the footprint does not depend
  // on this message: a scratch too small for it rules the variant out at any size.
  return ddaLLArScratchSize(comm->nRanks) <= comm->ddaScratchBytes;
}

// Same shared requirements as the one-shot variant, but its own size band: it
// reaches further (kDdaLLArTwoShotMaxBytes) and in exchange needs the payload to
// split evenly across the ranks.
bool ncclAllReduceDdaFabricLLTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                             ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (!ddaLLArFabricCommon(comm, count, datatype, op)) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);
  // The kernel shards the message across the ranks and works in whole 8-byte
  // packets, so the shard has to come out even on both counts.
  if (bytes % (8 * (size_t)comm->nRanks) != 0) {
    return false;
  }
  if (bytes > kDdaLLArTwoShotMaxBytes) {
    return false;
  }
  // A shard also has to fit the slot it is published into. The cap above already
  // implies this at every supported clique size, but the kernel asserts it, so
  // state it here rather than leave the invariant to a check release builds
  // compile out.
  if ((bytes >> 3) / (size_t)comm->nRanks > kDdaLLArTwoShotSlotStridePkts) {
    return false;
  }
  return ddaLLArTwoShotScratchSize(comm->nRanks) <= comm->ddaScratchBytes;
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
