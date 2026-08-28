/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstdint>
#include <cstring>

#include "comm.h"
#include "algorithms/dda/dda_init_detail.h"

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
        // Same sizing the real comm gets, so the LL all-reduce tiers see the
        // capacity they actually run with rather than the pre-LL base size.
        comm.ddaScratchBytes  = nccl_dda_detail::ddaIpcScratchSizing(
            nccl_dda_detail::kDdaNranks, /*overrideBytes=*/-1, /*llEnabled=*/1);
        comm.WarpSize            = 64;
        comm.ll128LineElems      = 8;
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
            comm.ddaLLEpochDev   = reinterpret_cast<uint32_t*>(0x5);
            comm.ddaLLEpochLen   = 24;
        }
        else
        {
            comm.ddaIpcMemHandler   = nullptr;
            comm.ddaScratch      = nullptr;
            comm.ddaPeerPtrsDev  = nullptr;
            comm.ddaIpcBarrierState = nullptr;
            comm.ddaLLEpochDev      = nullptr;
            comm.ddaLLEpochLen      = 0;
        }
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting
