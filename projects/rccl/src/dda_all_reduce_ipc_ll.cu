/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launchers + eligibility for the LL-protocol DDA IPC all-reduce. The
 * ddaAllReduceFlatLL (one-shot) and ddaAllReduceTwoShotLL kernels are
 * codepath-agnostic -- they reach peers only through the scratch pointer table --
 * so this file is the IPC counterpart of dda_all_reduce_fabric_ll.cu and shares
 * those kernels. The only difference is where the table comes from:
 * hipIpcOpenMemHandle here, imported fabric handles there.
 *
 * The two variants are separate dispatch tiers -- collectives.cc sends the small
 * messages to the one-shot entry point and the larger ones to the two-shot one --
 * but they share the tier's epoch counter, and they must: they lay their lines out
 * differently over the same scratch bytes, and it is that one shared monotonic flag
 * that keeps a leftover line from either layout from matching the other's flag.
 *
 * The two-shot kernel works a shard at a time -- count / nRanks elements -- with the
 * peer fan-out inside each phase, so its launcher sizes the grid on the shard, not
 * on the whole message the way the one-shot launcher does.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_dda_ll.h"
#include "algorithms/all_reduce/all_reduce_dda_ll_twoshot.h"
#include "checks.h"
#include "comm.h"
#include "dda_init_detail.h" // nccl_dda_detail::ddaMaxNBlocksForScratch, kDdaNranks
#include "debug.h"

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
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::kDdaNranks;

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

// Requirements both variants share: the resources they sync and stage through, the
// clique shape, and a payload that is a whole number of 8-byte LL packets. Each
// variant adds its own size cap and scratch-fit check on top.
static bool ddaLLArIpcCommon(const ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // IPC path: requires the IPC handler + scratch + peer table. The barrier state
  // the Simple IPC kernels need is not checked -- LL syncs through its flags.
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  // The epoch array carries the flag/bank derivation for both variants; without it
  // neither kernel can sync.
  if (comm->ddaLLArEpochDev == nullptr || comm->ddaLLArEpochLen < 1) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != kDdaNranks) {
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
  // packets. Note this is looser than the Simple IPC tier's 16-byte requirement.
  return (count * ncclTypeSize(datatype)) % 8 == 0;
}

// Grid for either LL variant: one block per `threads` work items, capped by the
// IPC block limit and by the epoch array (blockIdx.x indexes it).
static dim3 ddaLLArGrid(const ncclComm* comm, size_t workItems, unsigned threads) {
  int nBlocksMax = std::min(ddaMaxNBlocksForScratch(), comm->ddaLLArEpochLen);
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>((workItems + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  return dim3(blocks);
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcLLTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                               cudaStream_t stream) {
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet

  // The IPC path caps every DDA kernel at DDA_IPC_MAXBLOCKS, which is already the
  // narrow grid this latency-bound tier wants.
  const unsigned threads = 256;
  dim3 block(threads);
  dim3 grid = ddaLLArGrid(comm, nPk, threads);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLArEpochDev;
  const int epochLen = comm->ddaLLArEpochLen;

  INFO(NCCL_COLL, "DDA IPC AllReduce LL one-shot: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u", comm->nRanks, bytes,
       nPk, grid.x, block.x);

  // Unlike the fabric launcher there is no runtime-nRanks instantiation: the IPC
  // path only ever runs at kDdaNranks, so the reduce loop is always unrolled.
  meta::comms::ddaAllReduceFlatLL<T, kDdaNranks><<<grid, block, 0, stream>>>(
    peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, comm->nRanks, epochDev,
    epochLen);

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcLLTwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                      ncclComm* comm, cudaStream_t stream) {
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet
  if (nPk % comm->nRanks != 0) {
    ERROR("DDA IPC AllReduce LL two-shot: %zu bytes does not divide evenly across %d ranks", bytes, comm->nRanks);
    return ncclInvalidArgument;
  }
  const size_t nPk_rank = nPk / comm->nRanks;

  // Every phase loops over one shard with the peer fan-out inside it, so the grid
  // covers nPk_rank rather than the whole message: sizing it on nPk would only add
  // blocks whose gtid starts past the loop bound.
  const unsigned threads = 256;
  dim3 block(threads);
  dim3 grid = ddaLLArGrid(comm, nPk_rank, threads);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLArEpochDev;
  const int epochLen = comm->ddaLLArEpochLen;

  INFO(NCCL_COLL, "DDA IPC AllReduce LL two-shot: nRanks=%d bytes=%zu nPk=%zu nPk_rank=%zu grid=%u block=%u", comm->nRanks, bytes,
       nPk, nPk_rank, grid.x, block.x);

  // The slot stride is a constant both sides read, so nothing about the layout is
  // passed in; ddaLLArTwoShotScratchSize sizes the fit check from that same constant.
  meta::comms::ddaAllReduceTwoShotLL<T, kDdaNranks><<<grid, block, 0, stream>>>(
    peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank, comm->nRanks, epochDev,
    epochLen);

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaIpcLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (!ddaLLArIpcCommon(comm, count, datatype, op)) {
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
bool ncclAllReduceDdaIpcLLTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                          ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (!ddaLLArIpcCommon(comm, count, datatype, op)) {
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
  // implies this at kDdaNranks, but the kernel asserts it, so state it here rather
  // than leave the invariant to a check that release builds compile out.
  if ((bytes >> 3) / (size_t)comm->nRanks > kDdaLLArTwoShotSlotStridePkts) {
    return false;
  }
  return ddaLLArTwoShotScratchSize(comm->nRanks) <= comm->ddaScratchBytes;
}

ncclResult_t ncclAllReduceDdaIpcLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
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

ncclResult_t ncclAllReduceDdaIpcLLTwoShot(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                          ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
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
