/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

namespace nccl_dda_detail {

// Passed by value to the kernel. Rank numbers need not follow PCI order.
template <int MaxRanks>
struct DdaRsHierarchy {
  int nScaleUp = 0; // Zero means the communicator cannot use the hierarchy.
  int sibling = -1;
  int scaleUpRanks[MaxRanks / 2]{};
};

// The caller checks that every peer reports DPX. Verify complete physical-device
// pairs, not just an even rank count: subcommunicators may contain one tile per
// device, duplicate tiles, or reordered ranks. PCI IDs are only unique per host.
template <int MaxRanks, typename Peer>
DdaRsHierarchy<MaxRanks> ddaRsHierarchy(const Peer* peers, int nRanks, int self) {
  DdaRsHierarchy<MaxRanks> result;
  if (peers == nullptr || nRanks < 4 || nRanks > MaxRanks || nRanks % 2 != 0 || self < 0 || self >= nRanks)
    return result;

  const auto sameCard = [&](int a, int b) {
    return peers[a].hostHash == peers[b].hostHash && (peers[a].busId & ~0xfLL) == (peers[b].busId & ~0xfLL);
  };
  for (int r = 0; r < nRanks; ++r) {
    const int tile = peers[r].busId & 0xf;
    if (tile > 1) return {};
    int sibling = -1;
    for (int p = 0; p < nRanks; ++p) {
      if (p == r || !sameCard(r, p)) continue;
      if (sibling != -1 || (peers[p].busId & 0xf) == tile) return {};
      sibling = p;
    }
    if (sibling == -1) return {};
    if (r == self) result.sibling = sibling;
  }
  const int tile = peers[self].busId & 0xf;
  for (int r = 0; r < nRanks; ++r) {
    if ((peers[r].busId & 0xf) == tile) result.scaleUpRanks[result.nScaleUp++] = r;
  }
  return result;
}

} // namespace nccl_dda_detail
