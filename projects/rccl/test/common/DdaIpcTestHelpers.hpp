/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstdint>
#include <cstring>

#include "comm.h"
#include "dda_init_detail.h"

namespace RcclUnitTesting
{

// Minimal ncclComm stand-in for DDA IPC eligibility unit tests.
struct DdaIpcMockComm
{
    ncclComm comm{};
    char     bootstrapPlaceholder{0};

    DdaIpcMockComm() { reset(); }

    void reset()
    {
        std::memset(&comm, 0, sizeof(comm));
        comm.bootstrap           = &bootstrapPlaceholder;
        comm.nNodes              = 1;
        comm.nRanks              = nccl_dda_detail::kDdaNranks;
        comm.ddaScratchBytes  = DDA_IPC_BUFFER_SIZE;
        // The LL all-reduce tier derives its flag and scratch bank from these
        // cells, so its eligibility check requires them.
        comm.ddaLLArEpochDev  = reinterpret_cast<uint32_t*>(0x5);
        comm.ddaLLArEpochLen  = DDA_IPC_MAXBLOCKS;
        setIpcResourcesPresent(true);
    }

    void setIpcResourcesPresent(bool present)
    {
        if (present)
        {
            comm.ddaIpcMemHandler =
                reinterpret_cast<ncclIpcMemHandler*>(0x1);
            comm.ddaScratch       = reinterpret_cast<void*>(0x2);
            comm.ddaPeerPtrsDev   = reinterpret_cast<void*>(0x3);
            comm.ddaIpcBarrierState  =
                reinterpret_cast<nccl_dda_detail::DdaIpcBarrierState*>(0x4);
        }
        else
        {
            comm.ddaIpcMemHandler   = nullptr;
            comm.ddaScratch      = nullptr;
            comm.ddaPeerPtrsDev  = nullptr;
            comm.ddaIpcBarrierState = nullptr;
        }
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting
