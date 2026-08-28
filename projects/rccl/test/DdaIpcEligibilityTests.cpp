/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/DdaAlltoAllTestHelpers.hpp"
#include "common/DdaIpcTestHelpers.hpp"

#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "gtest/gtest.h"
#include "algorithms/dda/dda_init_detail.h"

namespace RcclUnitTesting
{

class DdaIpcEligibilityTest : public ::testing::Test
{
protected:
    DdaIpcMockComm mockComm_;
    void*          sendbuff_{reinterpret_cast<void*>(0x10)};
    void*          recvbuff_{reinterpret_cast<void*>(0x20)};
};

TEST_F(DdaIpcEligibilityTest, AllGather_NullComm)
{
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        nullptr, sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_MissingIpcResources)
{
    mockComm_.setIpcResourcesPresent(false);
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_ZeroCount)
{
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_MultiNode)
{
    mockComm_.comm.nNodes = 2;
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_WrongRankCount)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_UnalignedCount)
{
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_EligibleFloat16)
{
    EXPECT_TRUE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclFloat16));
}

TEST_F(DdaIpcEligibilityTest, AllGather_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllGatherDdaIpc(
                  sendbuff_, recvbuff_, 4, ncclInt32, mockComm_.get(), nullptr),
              ncclInvalidArgument);
}

TEST_F(DdaIpcEligibilityTest, AllToAll_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllToAllDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllToAll_ScratchTooSmallForTotal)
{
    mockComm_.comm.ddaScratchBytes = 64;
    EXPECT_FALSE(ncclAllToAllDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllToAll_UnalignedCount)
{
    EXPECT_FALSE(ncclAllToAllDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllToAll_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllToAllDdaIpc(
                  sendbuff_, recvbuff_, 4, ncclInt32, mockComm_.get(), nullptr),
              ncclInvalidArgument);
}

TEST_F(DdaIpcEligibilityTest, AllToAll_CountAt4MbTotal_Eligible)
{
    EXPECT_TRUE(ncclAllToAllDdaIpcEligible(
        mockComm_.get(),
        sendbuff_,
        recvbuff_,
        kAlltoAllFloat32CountAt4MbThreshold,
        ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllToAll_StagingBytesAtThresholdFitsScratch)
{
    const size_t stagingBytes = testAlltoAllDdaIpcStagingBytes(
        kAlltoAllFloat32CountAt4MbThreshold,
        mockComm_.comm.nRanks,
        sizeof(float));
    EXPECT_EQ(stagingBytes, kDdaAlltoAllGfx950ThresholdBytes);
    EXPECT_LE(stagingBytes, mockComm_.comm.ddaScratchBytes);
}

TEST_F(DdaIpcEligibilityTest, AllToAll_StagingBytesOneCountOverThresholdStillEligible)
{
    // Eligibility is independent of the 4 MiB dispatch cap enforced in collectives.cc.
    const size_t count = kAlltoAllFloat32CountAt4MbThreshold + 4;
    const size_t stagingBytes = testAlltoAllDdaIpcStagingBytes(
        count, mockComm_.comm.nRanks, sizeof(float));
    EXPECT_GT(stagingBytes, kDdaAlltoAllGfx950ThresholdBytes);
    EXPECT_TRUE(ncclAllToAllDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, count, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllToAll_MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(ncclAllToAllDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllToAll_ZeroCount)
{
    EXPECT_FALSE(ncclAllToAllDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, ReduceScatter_EligibleFloat32)
{
    EXPECT_TRUE(ncclReduceScatterDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, ReduceScatter_UnsupportedOp)
{
    EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclMax));
}

TEST_F(DdaIpcEligibilityTest, ReduceScatter_NonAlignedRecvcount)
{
    EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, ReduceScatter_PerRankUnaligned)
{
    EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 2, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, ReduceScatter_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclReduceScatterDdaIpc(sendbuff_,
                                     recvbuff_,
                                     4,
                                     ncclInt32,
                                     ncclSum,
                                     mockComm_.get(),
                                     nullptr),
              ncclInvalidArgument);
}

// ---------------------------------------------------------------------------
// AllReduce LL (IPC): same four-tier gate as fabric, with IPC resources.
// Default thresholds: one-shot 1 MiB, two-shot 16 MiB, LL128 one-shot 32 MiB.
// ---------------------------------------------------------------------------

TEST_F(DdaIpcEligibilityTest, AllReduceLL_NullComm)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        nullptr, sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_MissingIpcResources)
{
    mockComm_.setIpcResourcesPresent(false);
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_MissingEpoch)
{
    mockComm_.comm.ddaLLEpochDev = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_MultiNode)
{
    mockComm_.comm.nNodes = 2;
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_ZeroCount)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_UnsupportedOp)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclProd));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_UnalignedBytes)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_OneShotEligible)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_TRUE(ddaLLArOneShotIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
    EXPECT_FALSE(ddaLLArTwoShotIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_TwoShotClaimsPastOneShotThreshold)
{
    EXPECT_TRUE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 262176, ncclFloat32, ncclSum));
    EXPECT_FALSE(ddaLLArOneShotIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 262176, ncclFloat32, ncclSum));
    EXPECT_TRUE(ddaLLArTwoShotIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 262176, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL128_OneShotPastTwoShotThreshold)
{
    // 16 MiB + 128 B is past the LL two-shot cap and inside the LL128 one-shot cap.
    EXPECT_TRUE(ddaLL128ArOneShotIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4194336, ncclFloat32, ncclSum));
    EXPECT_FALSE(ddaLLArOneShotIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4194336, ncclFloat32, ncclSum));
    EXPECT_FALSE(ddaLLArTwoShotIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4194336, ncclFloat32, ncclSum));
}

// ---------------------------------------------------------------------------
// IPC scratch sizing
// ---------------------------------------------------------------------------

TEST(DdaIpcScratchSizingTest, ExplicitOverrideTakesPrecedence)
{
    const size_t forced = nccl_dda_detail::ddaIpcScratchSizing(8, 4096, 1);
    EXPECT_EQ(forced, 4096u);

    const size_t disabled = nccl_dda_detail::ddaIpcScratchSizing(8, 0, 1);
    EXPECT_EQ(disabled, 0u);
}

TEST(DdaIpcScratchSizingTest, DisabledLLUsesBaseBufferSize)
{
    const size_t sizing = nccl_dda_detail::ddaIpcScratchSizing(8, -1, 0);
    EXPECT_EQ(sizing, (size_t)DDA_IPC_BUFFER_SIZE);
}

// The four LL all-reduce tiers share one bank layout of
// 2 * nRanks * kDdaLLMaxBytes, which is far larger than DDA_IPC_BUFFER_SIZE.
// Without this floor every tier fails its scratch-capacity check and the LL
// path can never be selected on the IPC codepath.
TEST(DdaIpcScratchSizingTest, LLFloorDominatesBaseBufferSize)
{
    constexpr size_t llFloor = 2 * 8 * nccl_dda_detail::kDdaLLMaxBytes;
    static_assert(llFloor > (size_t)DDA_IPC_BUFFER_SIZE, "LL floor should exceed the base IPC buffer");

    const size_t sizing = nccl_dda_detail::ddaIpcScratchSizing(8, -1, 1);
    EXPECT_EQ(sizing, llFloor);
}

TEST(DdaIpcScratchSizingTest, LLFloorScalesWithRankCount)
{
    EXPECT_EQ(nccl_dda_detail::ddaIpcScratchSizing(4, -1, 1),
              (size_t)2 * 4 * nccl_dda_detail::kDdaLLMaxBytes);
    EXPECT_EQ(nccl_dda_detail::ddaIpcScratchSizing(8, -1, 1),
              (size_t)2 * 8 * nccl_dda_detail::kDdaLLMaxBytes);
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllReduceDdaIpcLL(sendbuff_,
                                   recvbuff_,
                                   4,
                                   ncclInt32,
                                   ncclSum,
                                   mockComm_.get(),
                                   nullptr),
              ncclInternalError);
}

} // namespace RcclUnitTesting
