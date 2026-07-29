/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstring>

#include "comm.h"
#include "dda_init_detail.h"
#include "fabric_gpu_barrier.h"

namespace RcclUnitTesting
{

// Minimal ncclComm stand-in for DDA fabric (VMM) eligibility unit tests.
struct DdaFabricMockComm
{
    ncclComm comm{};
    char     bootstrapPlaceholder{0};

    DdaFabricMockComm() { reset(); }

    void reset()
    {
        std::memset(&comm, 0, sizeof(comm));
        comm.bootstrap          = &bootstrapPlaceholder;
        comm.nNodes             = 1;
        comm.nRanks             = 8; // any value in [2, kDdaMaxNranks]
        comm.ddaScratchBytes    = DDA_FABRIC_BUFFER_SIZE;
        comm.ddaFabricMaxBlocks = DDA_FABRIC_MAXBLOCKS;
        // Epoch cells for the LL tiers; a device pointer is never dereferenced
        // by the host-side eligibility checks, so a placeholder suffices.
        comm.ddaLLEpochDev = reinterpret_cast<uint32_t*>(0x10);
        comm.ddaLLEpochLen = static_cast<int>(nccl_dda_detail::ddaLLEpochCount(
            comm.nRanks, nccl_dda_detail::kDdaFabricLLArMaxBlocks));
        comm.ddaLLArEpochDev = reinterpret_cast<uint32_t*>(0x11);
        comm.ddaLLArEpochLen = nccl_dda_detail::kDdaFabricLLArMaxBlocks;
        setFabricResourcesPresent(true);
    }

    void setFabricResourcesPresent(bool present)
    {
        if (present)
        {
            comm.ddaFabricMemHandler =
                reinterpret_cast<ncclFabricMemHandler*>(0x1);
            comm.ddaScratch     = reinterpret_cast<void*>(0x2);
            comm.ddaPeerPtrsDev = reinterpret_cast<void*>(0x3);
            comm.ddaFabricBarrierState =
                reinterpret_cast<nccl_dda_detail::DdaFabricBarrierState*>(0x4);
        }
        else
        {
            comm.ddaFabricMemHandler   = nullptr;
            comm.ddaScratch            = nullptr;
            comm.ddaPeerPtrsDev        = nullptr;
            comm.ddaFabricBarrierState = nullptr;
        }
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting
