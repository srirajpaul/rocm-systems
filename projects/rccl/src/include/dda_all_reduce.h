/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA all-reduce paths launched from ncclAllReduce
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALL_REDUCE_H_
#define DDA_ALL_REDUCE_H_

#include "nccl.h"

struct ncclComm;

// IPC path (single node, fixed kDdaNranks ranks).
bool ncclAllReduceDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// Fabric path (runtime nRanks up to kDdaMaxNranks, single- or multi-node).
bool ncclAllReduceDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// LL-protocol fabric path (small-message fast lane, flag-based sync, no barrier).
bool ncclAllReduceDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// Two-shot (reduce-scatter + all-gather) variant of the LL fabric path. Same
// tier as the one-shot LL above, but moves nRanks/2 fewer bytes per rank at the
// cost of a second publish/poll round trip, so it serves the upper part of the
// LL size range (see DDA_LL_TWOSHOT_MIN_BYTES in collectives.cc).
bool ncclAllReduceDdaFabricLLTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                             ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaFabricLLTwoShot(const void* sendbuff, void* recvbuff, size_t count,
                                             ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                                             cudaStream_t stream);

// LL128-protocol fabric path (mid-message fast lane, 128B lines, no barrier).
bool ncclAllReduceDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

#endif
