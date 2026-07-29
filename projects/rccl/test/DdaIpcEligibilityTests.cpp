/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/DdaAlltoAllTestHelpers.hpp"
#include "common/DdaIpcTestHelpers.hpp"

#include "dda_all_gather.h"
#include "dda_all_reduce.h"
#include "dda_alltoall.h"
#include "dda_reduce_scatter.h"
#include "gtest/gtest.h"
#include "dda_init_detail.h"

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
    const size_t count = kAlltoAllFloat32CountAt4MbThreshold + 1;
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
// AllReduce, LL protocol
// ---------------------------------------------------------------------------

TEST_F(DdaIpcEligibilityTest, AllReduceLL_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_EligibleFloat16)
{
    EXPECT_TRUE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat16, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_EligibleBfloat16)
{
    EXPECT_TRUE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclBfloat16, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_NullComm)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        nullptr, sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_MissingIpcResources)
{
    mockComm_.setIpcResourcesPresent(false);
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

// The kernel reaches peer scratch only through this table, so it is checked
// separately from the other IPC resources.
TEST_F(DdaIpcEligibilityTest, AllReduceLL_MissingPeerPtrs)
{
    mockComm_.comm.ddaPeerPtrsDev = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

// LL flags replace the GPU barrier, so the tier runs without the barrier state
// the Simple IPC kernels require.
TEST_F(DdaIpcEligibilityTest, AllReduceLL_EligibleWithoutBarrierState)
{
    mockComm_.comm.ddaIpcBarrierState = nullptr;
    EXPECT_TRUE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

// The kernel derives its flag and scratch bank from the LL AllReduce tier's
// epoch cells; without them it has no way to sync.
TEST_F(DdaIpcEligibilityTest, AllReduceLL_MissingEpochArray)
{
    mockComm_.comm.ddaLLArEpochDev = nullptr;
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_EmptyEpochArray)
{
    mockComm_.comm.ddaLLArEpochLen = 0;
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_MultiNode)
{
    mockComm_.comm.nNodes = 2;
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

// The IPC kernels fix the clique size at compile time.
TEST_F(DdaIpcEligibilityTest, AllReduceLL_WrongRankCount)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_ZeroCount)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_UnsupportedOp)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclProd));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclInt32, ncclSum));
}

// count=1 float32 is 4 bytes, half an 8-byte LL packet.
TEST_F(DdaIpcEligibilityTest, AllReduceLL_NotWholePackets)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1, ncclFloat32, ncclSum));
}

// count=6 float32 is 24 bytes: whole LL packets, but not the 16-byte multiple the
// Simple IPC tier needs, so LL claims sizes that tier would decline.
TEST_F(DdaIpcEligibilityTest, AllReduceLL_EligibleWhenSimpleTierUnaligned)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 6, ncclFloat32, ncclSum));
    EXPECT_TRUE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 6, ncclFloat32, ncclSum));
}

// kDdaLLArMaxBytes is 128 KiB: 32768 float32 elements is exactly the cap. 32770 is
// the next count past it that still lands on a whole 8-byte packet, so the cap is
// what rejects it.
TEST_F(DdaIpcEligibilityTest, AllReduceLL_MaxBytesEligible)
{
    EXPECT_TRUE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 32768, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_OverMaxBytes)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 32770, ncclFloat32, ncclSum));
}

// The staging area is a fixed 4 MiB -- 2 banks * 8 slots * the 128 KiB cap doubled
// by the 8B->16B line expansion -- regardless of the message, so a small scratch
// rules the tier out at any size.
TEST_F(DdaIpcEligibilityTest, AllReduceLL_ScratchTooSmall)
{
    mockComm_.comm.ddaScratchBytes = 1024;
    EXPECT_FALSE(ncclAllReduceDdaIpcLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 1024, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, AllReduceLL_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllReduceDdaIpcLL(sendbuff_,
                                    recvbuff_,
                                    1024,
                                    ncclInt32,
                                    ncclSum,
                                    mockComm_.get(),
                                    nullptr),
              ncclInvalidArgument);
}

} // namespace RcclUnitTesting
