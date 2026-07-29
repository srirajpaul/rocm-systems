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
// AllReduce, LL two-shot tier
//
// The default mock comm has nRanks=8, so an eligible float32 message needs a
// byte count divisible by 8*nRanks = 64: count=1024 (4 KiB) is used throughout.
// ---------------------------------------------------------------------------

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_EligibleFloat16)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat16, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_EligibleBfloat16)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclBfloat16, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_NullComm)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        nullptr, sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_MissingFabricResources)
{
    mockComm_.setFabricResourcesPresent(false);
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_MissingScratch)
{
    mockComm_.comm.ddaScratch = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

// The kernel reaches peer scratch only through this table, so it is checked
// separately from the other fabric resources.
TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_MissingPeerPtrs)
{
    mockComm_.comm.ddaPeerPtrsDev = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

// The kernel derives its flag and scratch bank from the LL AllReduce tier's
// epoch cells; without them it has no way to sync.
TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_MissingEpochArray)
{
    mockComm_.comm.ddaLLArEpochDev = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_EmptyEpochArray)
{
    mockComm_.comm.ddaLLArEpochLen = 0;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_ZeroCount)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_UnsupportedOp)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclProd));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclInt32, ncclSum));
}

// Fewer than kDdaLLArTwoShotMinRanks (4) ranks would move the same bytes as
// one-shot while paying two round trips, so the tier declines them.
TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_TooFewRanks)
{
    mockComm_.comm.nRanks = 3;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_MinRanksEligible)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_TRUE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_TooManyRanks)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks + 1;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

// count=144 float32 is 576 bytes == 8 * kDdaMaxNranks, i.e. exactly one packet
// per rank at the widest supported clique.
TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_MaxRanksEligible)
{
    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks;
    EXPECT_TRUE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 144, ncclFloat32, ncclSum));
}

// count=1000 float32 is 4000 bytes: a whole number of 8-byte packets overall,
// but not a whole number per rank at nRanks=8. Such sizes fall through to the
// one-shot LL variant.
TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_ShardNotWholePackets)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1000, ncclFloat32, ncclSum));
}

// kDdaLLArTwoShotMaxBytes is 128 KiB: 32768 float32 elements is exactly the cap,
// 65536 is twice it.
TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_MaxBytesEligible)
{
    EXPECT_TRUE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 32768, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_OverMaxBytes)
{
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 65536, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 1024;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLTwoShotEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricEligibilityTest, AllReduceLLTwoShot_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllReduceDdaFabricLLTwoShot(sendbuff_,
                                              recvbuff_,
                                              1024,
                                              ncclInt32,
                                              ncclSum,
                                              mockComm_.get(),
                                              nullptr),
              ncclInvalidArgument);
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
