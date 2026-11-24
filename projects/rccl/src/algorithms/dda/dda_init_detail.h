/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/device/CollCommon_ll128.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

#define DDA_IPC_MAXBLOCKS 24
#define DDA_IPC_BUFFER_SIZE 268435456

#define DDA_FABRIC_MAXBLOCKS 256

namespace nccl_dda_detail {

using dda::common::ddaLL128Slices;
using dda::common::kDdaLL128WireBytesPerSlice;
using dda::common::kDdaLLMaxBytes;
constexpr int kDdaNranks = dda::common::NRANKS;

// Per-comm IPC barrier state stored in ncclComm::ddaIpcBarrierState.
struct DdaIpcBarrierState {
  std::unique_ptr<dda::common::IpcGpuBarrierResources> resources;
  dda::common::IpcGpuBarrier barrierHost;
};

// Per-comm fabric barrier state stored in ncclComm::ddaFabricBarrierState.
struct DdaFabricBarrierState {
  std::unique_ptr<dda::common::FabricGpuBarrierResources> resources;
  dda::common::FabricGpuBarrier barrierHost;
};

inline int ddaMaxNBlocksForScratch() {
  unsigned maxBlocks = DDA_IPC_MAXBLOCKS;
  return static_cast<int>(maxBlocks);
}

struct DdaFabricMaxBlocksOverride {
  bool specified = false;
  bool valid = false;
  long requested = 0;
};

inline int ddaFabricMaxNBlocksForScratch(int cuCount, const char* overrideValue,
                                         DdaFabricMaxBlocksOverride* parsedOverride = nullptr) {
  int maxBlocks = cuCount;
  if (maxBlocks < 1) {
    maxBlocks = 1;
  }
  if (maxBlocks > DDA_FABRIC_MAXBLOCKS) {
    maxBlocks = DDA_FABRIC_MAXBLOCKS;
  }

  DdaFabricMaxBlocksOverride parsed;
  if (overrideValue != nullptr && overrideValue[0] != '\0') {
    parsed.specified = true;
    char* endptr = nullptr;
    parsed.requested = strtol(overrideValue, &endptr, 10);
    // Only apply if the entire string was a valid integer
    if (endptr != overrideValue && *endptr == '\0') {
      parsed.valid = true;
      // Clamp to [1, maxBlocks] while still a long to avoid int overflow
      if (parsed.requested < 1) {
        maxBlocks = 1;
      } else if (parsed.requested < maxBlocks) {
        maxBlocks = static_cast<int>(parsed.requested);
      }
      // requested >= maxBlocks: keep maxBlocks unchanged (override cannot raise)
    }
  }

  if (parsedOverride != nullptr) {
    *parsedOverride = parsed;
  }

  return maxBlocks;
}

constexpr int kDdaLLAgMaxBlocksPerPeer = 8;

// Number of device epoch cells for the LL collectives. it is sized for the larger of the two
// max(AG total blocks, AR total blocks).
inline size_t ddaLLEpochCount(int nRanks, int arMaxBlocks) {
  const size_t ag = (size_t)nRanks * (size_t)kDdaLLAgMaxBlocksPerPeer;
  const size_t ar = (size_t)arMaxBlocks;
  return ag > ar ? ag : ar;
}

} // namespace nccl_dda_detail
