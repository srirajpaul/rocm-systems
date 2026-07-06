/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host path: launch meta::comms::ddaAllToAllvIpc from ncclAlltoAllv.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALLTOALLV_IPC_H_
#define DDA_ALLTOALLV_IPC_H_

#include "nccl.h"

#include <cstddef>

struct ncclComm;

/**
 * Check whether the DDA alltoallv fast path can be used for this comm.
 *
 * The decision only depends on globally-identical comm configuration (topology,
 * rank count, IPC resources, datatype) so that every rank in the communicator
 * reaches the same conclusion without exchanging the per-rank counts. This is
 * required because a divergent decision would deadlock the in-kernel barrier.
 */
bool ncclAllToAllvDdaIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype);

/**
 * Execute a DDA alltoallv operation using IPC.
 *
 * Each per-peer chunk must fit within one scratch slot
 * (comm->ddaIpcScratchBytes / nRanks bytes); larger chunks return
 * ncclInvalidArgument.
 */
ncclResult_t ncclAllToAllvDdaIpc(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm* comm,
    cudaStream_t stream);

#endif
