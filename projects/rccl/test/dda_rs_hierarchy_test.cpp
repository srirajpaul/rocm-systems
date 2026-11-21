/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE.txt for license information.
 ************************************************************************/

// CPU-only topology regression test; no HIP runtime, GPU or gtest required.
#include "algorithms/dda/reduce_scatter/dda_rs_hierarchy.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); std::exit(1); \
} } while (0)

struct Peer {
  uint64_t hostHash;
  int64_t busId;
};

int main() {
  constexpr int maxRanks = 144;
  std::array<Peer, maxRanks> peers{};
  std::array<int, maxRanks> order{};
  std::mt19937 rng(2395);
  for (int n : {4, 6, 8, 16, 32, 64, 144}) {
    std::iota(order.begin(), order.end(), 0);
    for (int trial = 0; trial < 20; ++trial) {
      if (trial) std::shuffle(order.begin(), order.begin() + n, rng);
      for (int r = 0; r < n; ++r) {
        // Repeated PCI IDs on different hosts must still form distinct pairs.
        peers[r] = {uint64_t(order[r] / 8), 0x1000 + (order[r] % 8 / 2) * 0x100 + order[r] % 2};
      }
      for (int self = 0; self < n; ++self) {
        auto h = nccl_dda_detail::ddaRsHierarchy<maxRanks>(peers.data(), n, self);
        CHECK(h.nScaleUp == n / 2);
        CHECK(h.sibling >= 0 && h.sibling < n);
        CHECK(order[h.sibling] == (order[self] ^ 1));
        bool visited[maxRanks]{};
        for (int p = 0; p < h.nScaleUp; ++p) {
          CHECK(h.scaleUpRanks[p] >= 0 && h.scaleUpRanks[p] < n);
          const int tile = order[h.scaleUpRanks[p]];
          CHECK(tile % 2 == order[self] % 2);
          CHECK(!visited[tile]);
          visited[tile] = visited[tile ^ 1] = true;
        }
        // Every source contributes exactly once through its paired partial sum.
        for (int r = 0; r < n; ++r) CHECK(visited[r]);
      }
    }
  }
  const auto rejected = [&](int n, int rank) {
    return nccl_dda_detail::ddaRsHierarchy<maxRanks>(peers.data(), n, rank).nScaleUp == 0;
  };
  for (int r = 0; r < 8; ++r) peers[r] = {1, 0x1000 + (r / 2) * 0x100 + r % 2};
  CHECK(rejected(0, 0) && rejected(2, 0) && rejected(7, 0) && rejected(146, 0));
  CHECK(rejected(8, -1) && rejected(8, 8));
  CHECK(nccl_dda_detail::ddaRsHierarchy<maxRanks>(static_cast<const Peer*>(nullptr), 8, 0).nScaleUp == 0);
  peers[1].hostHash = 2;
  CHECK(rejected(8, 0)); // Incomplete physical pair across hosts.
  peers[1] = peers[0];
  CHECK(rejected(8, 0)); // Duplicate logical tile.
  for (int r = 0; r < 8; ++r) peers[r] = {1, 0x1000 + r};
  CHECK(rejected(8, 0)); // CPX, not two-tile DPX.
  for (int r = 0; r < 8; ++r) peers[r] = {1, 0x1000 + r * 0x100};
  CHECK(rejected(8, 0)); // One tile per card.
  std::puts("PASS: DPX hierarchy topology, shuffled ranks, repeated PCI IDs and invalid communicators");
}
