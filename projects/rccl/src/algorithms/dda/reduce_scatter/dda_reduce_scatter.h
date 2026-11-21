/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host entry points for the DDA reduce-scatter paths launched from ncclReduceScatter.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_REDUCE_SCATTER_H_
#define DDA_REDUCE_SCATTER_H_

#include "nccl.h"

struct ncclComm;

/**
 * Check if DDA reduce-scatter is eligible for the given parameters
 */
bool ncclReduceScatterDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                     ncclDataType_t datatype, ncclRedOp_t op);

/**
 * Execute DDA reduce-scatter operation using IPC
 */
ncclResult_t ncclReduceScatterDdaIpc(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                     ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

/**
 * Check if DDA reduce-scatter is eligible for the fabric/VMM path (runtime
 * nRanks up to kDdaMaxNranks, single- or multi-node within an MNNVL clique).
 */
bool ncclReduceScatterDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                        ncclDataType_t datatype, ncclRedOp_t op);

/**
 * Execute DDA reduce-scatter operation using the fabric/VMM path
 */
ncclResult_t ncclReduceScatterDdaFabric(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                        ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// Largest total message (bytes) the fabric path runs with the hierarchical kernel on this comm, or 0 when the comm
// does not split into packages or RCCL_DDA_RS_HIER=0. The selector lifts the RS VMM tier to it.
size_t ncclReduceScatterDdaFabricHierMaxBytes(const ncclComm* comm);

// Before the private scratch region of RCCL_DDA_RS_HIER_MODE=5 is freed: waits until every rank's launches are done,
// since that mode has no closing barrier. A no-op unless the comm launched mode 5.
ncclResult_t ncclReduceScatterDdaFabricHierDrain(ncclComm* comm);

// LL-protocol fabric path (small-shard fast lane, 16B lines, no barrier).
bool ncclReduceScatterDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                          ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclReduceScatterDdaFabricLL(const void* sendbuff, void* recvbuff, size_t recvcount,
                                          ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// LL128-protocol fabric path (mid-shard fast lane, 128B lines, no barrier).
bool ncclReduceScatterDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                             ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclReduceScatterDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t recvcount,
                                             ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                                             cudaStream_t stream);

#endif
