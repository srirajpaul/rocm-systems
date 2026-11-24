/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

#include "comm.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

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
        // Capacity is not under test by default. Individual tests override this
        // with an exact small value when exercising scratch-size rejection.
        for (auto& region : comm.ddaFabricScratch.regions)
            region.bytes = std::numeric_limits<size_t>::max();
        comm.ddaFabricMaxBlocks = DDA_FABRIC_MAXBLOCKS;
        comm.archName            = const_cast<char*>("gfx1250");
        // Non-null so launcher tests can tell the abort flag was forwarded.
        comm.abortFlagDev       = reinterpret_cast<uint32_t*>(0x6);
        setFabricResourcesPresent(true);
    }

    void setFabricResourcesPresent(bool present)
    {
        if (present)
        {
            comm.ddaFabricMemHandler =
                reinterpret_cast<ncclFabricMemHandler*>(0x1);
            comm.ddaFabricBarrierState =
                reinterpret_cast<nccl_dda_detail::DdaFabricBarrierState*>(0x4);
            for (int i = 0; i < ncclDdaScratchCount; ++i) {
                auto& region = comm.ddaFabricScratch.regions[i];
                region.scratch = reinterpret_cast<void*>(0x1000 + 0x100 * i);
                region.peerPtrsDev = reinterpret_cast<void*>(0x2000 + 0x100 * i);
                region.peerPtrsHost = reinterpret_cast<void**>(0x3000 + 0x100 * i);
                region.epochDev = reinterpret_cast<uint32_t*>(0x4000 + 0x100 * i);
                region.epochLen = DDA_FABRIC_MAXBLOCKS;
            }
            comm.ddaFabricScratch.tokenPeerPtrsDev = reinterpret_cast<void*>(0x5000);
            comm.ddaFabricScratch.tokenBankBytes = 2304;

        }
        else
        {
            comm.ddaFabricMemHandler   = nullptr;
            comm.ddaFabricBarrierState = nullptr;
            for (auto& region : comm.ddaFabricScratch.regions) {
                const size_t bytes = region.bytes;
                region = {};
                region.bytes = bytes;
            }
            comm.ddaFabricScratch.tokenPeerPtrsDev = nullptr;

        }
    }

    ncclComm* get() { return &comm; }
};

} // namespace RcclUnitTesting
