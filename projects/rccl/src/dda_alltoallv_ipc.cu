/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_alltoallv_ipc.h"

#include "algorithms/CollCommon.h"
#include "algorithms/alltoallv/alltoallv_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ipc_gpu_barrier.h"
#include "ipc_init_detail.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

using nccl_dda_ipc_detail::DdaIpcBarrierState;
using nccl_dda_ipc_detail::ddaMaxNBlocksForScratch;
using nccl_dda_ipc_detail::kDdaNranks;

constexpr int kDdaAllToAllvThreadsPerBlock = 512;

static bool ddaAllToAllvResourcesReady(const ncclComm* comm) {
  return comm != nullptr && comm->ddaIpcMemHandler != nullptr &&
      comm->ddaIpcScratch != nullptr && comm->ddaIpcPeerPtrsDev != nullptr &&
      comm->ddaIpcBarrierState != nullptr;
}

} // namespace

bool ncclAllToAllvDdaIpcEligible(
    ncclComm* comm,
    const void* /*sendbuff*/,
    const size_t /*sendcounts*/[],
    const size_t /*sdispls*/[],
    void* /*recvbuff*/,
    const size_t /*recvcounts*/[],
    const size_t /*rdispls*/[],
    ncclDataType_t datatype) {
  // IMPORTANT: only branch on state that is identical on every rank of the
  // communicator. The per-rank counts/displacements deliberately do not
  // influence the decision, otherwise ranks could disagree and deadlock on the
  // in-kernel barrier. Per-chunk capacity is validated at launch time.
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (!ddaAllToAllvResourcesReady(comm)) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != kDdaNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }
  return true;
}

ncclResult_t ncclAllToAllvDdaIpc(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm* comm,
    cudaStream_t stream) {
  if (!ddaAllToAllvResourcesReady(comm)) {
    return ncclInvalidUsage;
  }
  if (comm->nRanks != kDdaNranks) {
    return ncclInvalidUsage;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }

  const size_t typeSize = static_cast<size_t>(ncclTypeSize(datatype));
  // Each rank's scratch buffer is split into kDdaNranks equal slots; a chunk to
  // a given peer must fit in one slot.
  const size_t slotStride = comm->ddaIpcScratchBytes / kDdaNranks;

  meta::comms::DdaAllToAllvArgs<kDdaNranks> args{};
  for (int i = 0; i < kDdaNranks; ++i) {
    const size_t sBytes = sendcounts[i] * typeSize;
    const size_t rBytes = recvcounts[i] * typeSize;
    if (sBytes > slotStride) {
      WARN(
          "DDA IPC alltoallv: send chunk to peer %d needs %zu bytes; per-peer "
          "scratch slot is %zu bytes",
          i,
          sBytes,
          slotStride);
      return ncclInvalidArgument;
    }
    args.sendcounts[i] = sBytes;
    args.sdispls[i] = sdispls[i] * typeSize;
    args.recvcounts[i] = rBytes;
    args.rdispls[i] = rdispls[i] * typeSize;
  }

  auto* barrierState =
      static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  int8_t** d_ipcbuffs = reinterpret_cast<int8_t**>(comm->ddaIpcPeerPtrsDev);

  // The in-kernel barrier synchronizes matching block indices across ranks, so
  // every rank must launch an identical grid regardless of its local counts.
  // Use a fixed grid (the barrier is provisioned for ddaMaxNBlocksForScratch
  // blocks) and rely on grid-stride loops to cover arbitrary chunk sizes.
  dim3 grid(static_cast<unsigned>(ddaMaxNBlocksForScratch()), 1, 1);
  dim3 block(kDdaAllToAllvThreadsPerBlock, 1, 1);

  meta::comms::ddaAllToAllvIpc<int8_t, kDdaNranks><<<grid, block, 0, stream>>>(
      d_ipcbuffs,
      static_cast<int8_t*>(recvbuff),
      static_cast<const int8_t*>(sendbuff),
      comm->rank,
      slotStride,
      args,
      barrierHost);
  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}
