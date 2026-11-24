/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE.txt for license information.
 ************************************************************************/
#include <gtest/gtest.h>
#include "dda_scratch.h"

TEST(DdaScratchLayout, FixedDefaultsAndReservedTokensDoNotOverlap) {
  const int64_t capacities[] = {256LL << 20, 256LL << 20, 64LL << 20, 8LL << 20};
  ncclDdaFabricScratch layout{};
  ASSERT_TRUE(ncclDdaScratchLayout(capacities, 144, &layout));
  EXPECT_EQ(layout.allocationBytes, 584ULL << 20);
  for (int i = 0; i < ncclDdaScratchCount; ++i) {
    EXPECT_EQ(layout.capacities[i], capacities[i]);
    EXPECT_EQ(layout.offsets[i] % 128, 0u);
    if (i) EXPECT_GE(layout.offsets[i], layout.offsets[i-1] + layout.capacities[i-1]);
  }
  EXPECT_GE(layout.tokenBankBytes, 144u * 16);
  EXPECT_EQ(layout.tokenBankBytes % 128, 0u);
  EXPECT_EQ(layout.regions[ncclDdaScratchLL128].bytes + 2 * layout.tokenBankBytes, 64ULL << 20);
}

TEST(DdaScratchLayout, PaddingCannotIncreaseCapacity) {
  const int64_t capacities[] = {1, 129, 8193, 257};
  ncclDdaFabricScratch layout{};
  ASSERT_TRUE(ncclDdaScratchLayout(capacities, 144, &layout));
  EXPECT_EQ(layout.offsets[ncclDdaScratchSimple], 128u);
  EXPECT_EQ(layout.offsets[ncclDdaScratchLL128], 384u);
  EXPECT_EQ(layout.regions[ncclDdaScratchSimple].bytes, 129u);
  EXPECT_EQ(layout.regions[ncclDdaScratchLL].bytes, 257u);
  EXPECT_EQ(layout.regions[ncclDdaScratchLL128].bytes, 8193u - 2 * layout.tokenBankBytes);
}

TEST(DdaScratchLayout, DisabledAndTinyRegionsCannotUnderflow) {
  for (int64_t tiny : {0, 1, 4607, 4608}) {
    const int64_t capacities[] = {0, 128, tiny, 0};
    ncclDdaFabricScratch layout{};
    ASSERT_TRUE(ncclDdaScratchLayout(capacities, 144, &layout));
    EXPECT_EQ(layout.regions[ncclDdaScratchLL128].bytes, 0u);
    EXPECT_EQ(layout.regions[ncclDdaScratchSimple].bytes, 128u);
    EXPECT_EQ(layout.regions[ncclDdaScratchLL].bytes, 0u);
  }
  const int64_t zero[] = {0, 0, 0, 0};
  ncclDdaFabricScratch layout{};
  ASSERT_TRUE(ncclDdaScratchLayout(zero, 144, &layout));
  EXPECT_EQ(layout.allocationBytes, 0u);
}

TEST(DdaScratchLayout, InvalidAndOverflowingTotalsAreRejected) {
  int64_t capacities[] = {256, 256, -1, 8};
  ncclDdaFabricScratch layout{};
  EXPECT_FALSE(ncclDdaScratchLayout(capacities, 144, &layout));
  capacities[2] = INT64_MAX;
  EXPECT_FALSE(ncclDdaScratchLayout(capacities, 144, &layout));
}

TEST(DdaScratchLayout, RankScaledFootprintArithmeticCannotWrap) {
  size_t bytes;
  EXPECT_TRUE(ncclDdaByteCount(1024, 4, 8, &bytes));
  EXPECT_EQ(bytes, 32768u);
  EXPECT_FALSE(ncclDdaByteCount(SIZE_MAX / 4 + 1, 4, 1, &bytes));
  EXPECT_FALSE(ncclDdaByteCount(SIZE_MAX / 8, 4, 8, &bytes));
  EXPECT_FALSE(ncclDdaByteCount(1, 0, 8, &bytes));
}
