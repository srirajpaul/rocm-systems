/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_gather.h"

#include "algorithms/CollCommon.h"
#include "algorithms/all_gather/all_gather_dda.h"
#include "algorithms/all_gather/all_gather_symm_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ipc_gpu_barrier.h"
#include "dda_init_detail.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

namespace {

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::kDdaNranks;

// The symmetric kernel reaches peers through their windows, so it needs neither the
// staging scratch nor the host-resolved peer pointer array. It does require both user
// buffers to live in symmetric windows, which comm->symmetricSupport alone does not
// imply -- that flag only says the comm *can* host windows.
static bool ddaAllGatherUseSymm(ncclComm* comm, const void* sendbuff, void* recvbuff, ncclSymPtr<char>& recvSymPtr,
                                ncclSymPtr<char>& sendSymPtr) {
  if (!comm->symmetricSupport) return false;
  if (meta::comms::ncclPtrToSymPtr(comm, recvbuff, recvSymPtr) != ncclSuccess) return false;
  if (meta::comms::ncclPtrToSymPtr(comm, const_cast<void*>(sendbuff), sendSymPtr) != ncclSuccess) return false;
  return true;
}

template <typename T>
static ncclResult_t ncclAllGatherDdaIpcTyped(const void* sendbuff, void* recvbuff, size_t sendcount, ncclComm* comm,
                                             cudaStream_t stream) {
  ncclSymPtr<char> recvSymPtr, sendSymPtr;
  const bool useSymm = ddaAllGatherUseSymm(comm, sendbuff, recvbuff, recvSymPtr, sendSymPtr);

  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }
  if (!useSymm && (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr)) {
    return ncclInvalidUsage;
  }

  const size_t totalCount = sendcount * comm->nRanks;
  if (!useSymm && totalCount * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA IPC allgather: send element count %zu needs %zu bytes; comm scratch is %zu bytes", sendcount,
         totalCount * sizeof(T), comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  // For allgather, we use sendcount for grid calculation
  auto gridBlock = meta::comms::getGridAndBlockDims(sendcount, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  // Must be useSymm, not comm->symmetricSupport: the latter only says the comm can host
  // windows, so gating on it launches the symmetric kernel with null-window symmetric
  // pointers whenever the user buffers are not registered.
  if (useSymm) {
    INFO(NCCL_COLL, "ddaAllGatherIpcSymm count %zu", sendcount);
    meta::comms::ddaAllGatherIpcSymm<T, kDdaNranks, false><<<grid, block, 0, stream>>>(
      recvSymPtr, sendcount, sendSymPtr, comm->rank, barrierHost);
  } else {
    INFO(NCCL_COLL, "ddaAllGatherIpc count %zu", sendcount);
    T** d_ipcbuffs = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);

    meta::comms::ddaAllGatherIpc<T, kDdaNranks, false><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), sendcount, static_cast<const T*>(sendbuff), comm->rank, barrierHost);
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllGatherDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                 ncclDataType_t datatype) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  ncclSymPtr<char> recvSymPtr, sendSymPtr;
  const bool useSymm = ddaAllGatherUseSymm(comm, sendbuff, recvbuff, recvSymPtr, sendSymPtr);
  if (!useSymm && (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr)) {
    return false;
  }
  if (sendcount == 0) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != nccl_dda_detail::kDdaNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  size_t need = sendcount * ncclTypeSize(datatype);
  if (!useSymm && need > comm->ddaScratchBytes) {
    return false;
  }

  // Check for data size divisible by 16
  if ((sendcount * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllGatherDdaIpc(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                 ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaIpcTyped<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
