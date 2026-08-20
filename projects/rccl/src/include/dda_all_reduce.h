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
//
// The tier has three variants -- LL one-shot, LL two-shot (transports only the
// shard each rank owns), and LL128 one-shot. Picking between them, including the
// DDA_LL enable and per-variant thresholds, is internal to dda_all_reduce_fabric_ll.cu:
// Eligible reports whether any variant claims the message, and the entry point
// launches whichever one did.
bool ncclAllReduceDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op);

// The per-variant predicates the gate above is the disjunction of. Declared here
// so unit tests can exercise one tier without the other masking it; hidden in
// Release by -fvisibility=hidden, as the rest of the internal surface is.
bool ddaLLArOneShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op);

bool ddaLLArTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op);

bool ddaLL128ArOneShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

// LL128-protocol fabric path (mid-message fast lane, 128B lines, no barrier).
bool ncclAllReduceDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclAllReduceDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);

#endif
