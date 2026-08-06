/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/DdaFabricTestHelpers.hpp"

#include "dda_all_gather.h"
#include "dda_all_reduce.h"
#include "dda_alltoall.h"
#include "dda_reduce_scatter.h"
#include "fabric_gpu_barrier.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

class DdaFabricEligibilityTest : public ::testing::Test
{
protected:
    DdaFabricMockComm mockComm_;
    void*             sendbuff_{reinterpret_cast<void*>(0x1000)};
    void*             recvbuff_{reinterpret_cast<void*>(0x2000)};
};

// ---------------------------------------------------------------------------
// AllGather
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, AllGather_NullComm)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        nullptr, sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MissingFabricResources)
{
    mockComm_.setFabricResourcesPresent(false);
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MissingBarrierState)
{
    mockComm_.comm.ddaFabricBarrierState = nullptr;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MissingScratch)
{
    mockComm_.comm.ddaScratch = nullptr;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_ZeroCount)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_TooFewRanks)
{
    mockComm_.comm.nRanks = 1;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_TooManyRanks)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks + 1;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_MaxRanksEligible)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks;
    EXPECT_TRUE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_UnalignedCount)
{
    EXPECT_FALSE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_EligibleFloat16)
{
    EXPECT_TRUE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclFloat16));
}

TEST_F(DdaFabricEligibilityTest, AllGather_EligibleBfloat16)
{
    EXPECT_TRUE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclBfloat16));
}

// Unlike the IPC path, the fabric path spans an MNNVL clique, so multi-node
// comms stay eligible (nNodes is not part of the eligibility check).
TEST_F(DdaFabricEligibilityTest, AllGather_MultiNodeStillEligible)
{
    mockComm_.comm.nNodes = 2;
    EXPECT_TRUE(ncclAllGatherDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllGather_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllGatherDdaFabric(
                  sendbuff_, recvbuff_, 4, ncclInt32, mockComm_.get(), nullptr),
              ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// AllReduce
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, AllReduce_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_EligibleBfloat16)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclBfloat16, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_UnsupportedOp)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclProd));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_ZeroCount)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_UnalignedCount)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_MinRanksEligible)
{
    mockComm_.comm.nRanks = 2;
    EXPECT_TRUE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_TooManyRanks)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks + 1;
    EXPECT_FALSE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

// Above the flat/tree threshold (256KB) the two-shot path requires count to be
// divisible by nRanks with a 16-byte-aligned per-rank slice. count=131072
// (512KB f32, nRanks=8) satisfies both.
TEST_F(DdaFabricEligibilityTest, AllReduce_TreePathEligible)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 131072, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduce_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllReduceDdaFabric(sendbuff_,
                                     recvbuff_,
                                     4,
                                     ncclInt32,
                                     ncclSum,
                                     mockComm_.get(),
                                     nullptr),
              ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// AllReduce LL
//
// One gate fronts both LL tiers: ncclAllReduceDdaFabricLLEligible reports
// whether the one-shot or the two-shot variant claims the message, and each
// applies its own enable and threshold internally.
//
// These run under the default parameter values -- DDA_LL=1 with a 32 KiB
// DDA_LL_THRESHOLD, and DDA_LL_TWOSHOT_THRESHOLD=0, which matches no message and
// so leaves the two-shot tier inert. They deliberately do not setenv: RCCL_PARAM
// caches on first read, and the NCCL_NO_CACHE opt-out is resolved once per
// process, so a test that changed a threshold would depend on being the first in
// the binary to touch it. What is asserted here is therefore the shared shape and
// resource guards plus the one-shot threshold boundary; which tier serves a size
// once the two-shot threshold is raised belongs to a multi-rank run.
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, AllReduceLL_NullComm)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        nullptr, sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_MissingFabricResources)
{
    mockComm_.setFabricResourcesPresent(false);
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_MissingScratch)
{
    mockComm_.comm.ddaScratch = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_ZeroCount)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_UnsupportedOp)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclProd));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32, ncclSum));
}

// Payload is staged as 8-byte LL packets, so a message that is not a whole
// number of packets is rejected. One float is 4 bytes.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_UnalignedBytes)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_TooFewRanks)
{
    mockComm_.comm.nRanks = 1;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_TooManyRanks)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks + 1;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_MinRanksEligible)
{
    mockComm_.comm.nRanks = 2;
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

// The staging footprint scales with nRanks, so the widest comm is also the one
// that has to still fit the scratch.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_MaxRanksEligible)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks;
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_EligibleFloat16)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclFloat16, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLL_EligibleBfloat16)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclBfloat16, ncclSum));
}

// 8192 floats is exactly the default 32 KiB DDA_LL_THRESHOLD, which the tier
// takes with <=.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_AtThresholdEligible)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8192, ncclFloat32, ncclSum));
}

// 8208 floats is 32832 bytes: past the one-shot threshold, but a whole number of
// packets and divisible by 8 * nRanks, so it satisfies the two-shot shape rules
// and would be claimed if that tier's threshold were raised. Rejecting it is what
// "inert by default" means -- if this starts passing, the two-shot tier has begun
// claiming traffic without being asked to.
TEST_F(DdaFabricEligibilityTest, AllReduceLL_TwoShotInertByDefault)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8208, ncclFloat32, ncclSum));
}

// ---------------------------------------------------------------------------
// AllToAll
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, AllToAll_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllToAll_ZeroCount)
{
    EXPECT_FALSE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllToAll_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32));
}

// Scratch must hold count*nRanks elements (the full exchange), not just count.
TEST_F(DdaFabricEligibilityTest, AllToAll_ScratchTooSmallForTotal)
{
    mockComm_.comm.ddaScratchBytes = 64;
    EXPECT_FALSE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllToAll_UnalignedCount)
{
    EXPECT_FALSE(ncclAllToAllDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32));
}

TEST_F(DdaFabricEligibilityTest, AllToAll_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllToAllDdaFabric(
                  sendbuff_, recvbuff_, 4, ncclInt32, mockComm_.get(), nullptr),
              ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// ReduceScatter
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, ReduceScatter_EligibleFloat32)
{
    EXPECT_TRUE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_UnsupportedOp)
{
    EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclMax));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_ZeroCount)
{
    EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

// recvcount=3 keeps the total (recvcount*nRanks) 16-byte aligned but leaves the
// per-rank slice unaligned, exercising the per-rank alignment guard.
TEST_F(DdaFabricEligibilityTest, ReduceScatter_PerRankUnaligned)
{
    EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 64;
    EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_MaxRanksEligible)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks;
    EXPECT_TRUE(ncclReduceScatterDdaFabricEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, ReduceScatter_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclReduceScatterDdaFabric(sendbuff_,
                                         recvbuff_,
                                         4,
                                         ncclInt32,
                                         ncclSum,
                                         mockComm_.get(),
                                         nullptr),
              ncclInvalidArgument);
}

} // namespace RcclUnitTesting
