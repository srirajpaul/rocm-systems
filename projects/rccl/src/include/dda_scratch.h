/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE.txt for license information.
 ************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

// Payload capacities are configuration, never derived from tuning or rank count.
enum ncclDdaScratchKind { ncclDdaScratchCE, ncclDdaScratchSimple, ncclDdaScratchLL128,
                          ncclDdaScratchLL, ncclDdaScratchCount };

struct ncclDdaScratchRegion {
  void* scratch;
  size_t bytes;  // Addressable data bytes, excluding the LL128 token reservation.
  void* peerPtrsDev;
  void** peerPtrsHost;
  uint32_t* epochDev;
  int epochLen;
};

struct ncclDdaFabricScratch {
  void* allocation;
  size_t allocationBytes;
  size_t capacities[ncclDdaScratchCount];
  size_t offsets[ncclDdaScratchCount];
  ncclDdaScratchRegion regions[ncclDdaScratchCount];
  void* tokenPeerPtrsDev;  // Two reserved LL-format Broadcast token banks, LL128 epochs.
  size_t tokenBankBytes;
};

// maxRanks is supplied from the compile-time fabric limit, never comm->nRanks.
inline bool ncclDdaScratchLayout(const int64_t* capacities, size_t maxRanks, ncclDdaFabricScratch* out) {
  *out = {};
  constexpr size_t align = 128;
  const size_t max = std::numeric_limits<size_t>::max();
  if (maxRanks > (max - align + 1) / 16) return false;
  out->tokenBankBytes = (maxRanks * 16 + align - 1) / align * align;
  if (out->tokenBankBytes > max / 2) return false;
  for (int i = 0; i < ncclDdaScratchCount; ++i) {
    if (capacities[i] < 0 || (uint64_t)capacities[i] > max) return false;
    const size_t bytes = (size_t)capacities[i];
    if (out->allocationBytes > max - (align - 1)) return false;
    const size_t offset = (out->allocationBytes + align - 1) / align * align;
    if (bytes > max - offset) return false;
    out->offsets[i] = offset;
    out->capacities[i] = bytes;
    out->regions[i].bytes = bytes;
    out->allocationBytes = offset + bytes;
  }
  // Leave room for allocator granularity rounding and reject unrepresentable totals.
  if (out->allocationBytes > (size_t)std::numeric_limits<int64_t>::max()) return false;
  const size_t tokens = 2 * out->tokenBankBytes;
  auto& ll128 = out->regions[ncclDdaScratchLL128];
  ll128.bytes = ll128.bytes > tokens ? ll128.bytes - tokens : 0;
  return true;
}

inline bool ncclDdaByteCount(size_t count, size_t typeSize, size_t ranks, size_t* bytes) {
  const size_t max = std::numeric_limits<size_t>::max();
  if (typeSize == 0 || ranks == 0 || count > max / typeSize) return false;
  const size_t perRank = count * typeSize;
  if (perRank > max / ranks) return false;
  *bytes = perRank * ranks;
  return true;
}
