/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Runtime (nRanks-flexible) GPU barrier for the DDA fabric/VMM path.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cuda.h>
#include <cstdint>
#include <memory>

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/device/device_buffer.h"
#include "algorithms/dda/fabric/fabric_mem_handler.h"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"

struct ncclMemManager;

namespace dda::common {

// Upper bound on the number of ranks the fabric DDA path supports (DPX).
// Note: LL scratch scales as 2 * nRanks * 16 MiB; at 144 ranks this is ~4.5 GiB per GPU.
constexpr int kDdaMaxNranks = 144;

class FabricGpuBarrier;

struct FabricGpuBarrierResources {
  std::unique_ptr<ncclFabricMemHandler> fabricMemHandler;
  // This rank's own flag buffer (VMM-backed when available).
  std::unique_ptr<DeviceBuffer> selfFlagBuf;
  // Device-resident array of nRanks flag-buffer pointers (FlagType*[nRanks]).
  std::unique_ptr<DeviceBuffer> peerFlagsDev;
};

class FabricGpuBarrier {
public:
  using FlagType = uint32_t;

  __host__ FabricGpuBarrier() = default;

  // Allocates this rank's flag buffer, exchanges peer flag-buffer pointers via
  // fabric shareable handles and builds a device-resident pointer table.
  static __host__ std::pair<std::unique_ptr<FabricGpuBarrierResources>, FabricGpuBarrier> mallocAndInit(
    int nRanks, int nBlocks, int selfRank, void* bootstrap, struct ncclMemManager* manager);

  // Attaches the comm's LL epoch cells (comm->ddaLLEpochDev), which the barrier
  // shares with the LL/LL128 kernels so every fabric collective advances one epoch.
  __host__ void setEpochs(uint32_t* epochDev, int epochLen) {
    epochDev_ = epochDev;
    epochLen_ = epochLen;
  }

  // Barrier k (1-based) of a kernel posts epoch beginEpoch() + k, and the kernel
  // ends with endEpoch(beginEpoch() + nBarriers). Both must be called by every
  // thread of the block.
  __device__ __forceinline__ FlagType beginEpoch() const {
    return ddaGetLLEpochInc(epochDev_, blockIdx.x, 1);
  }
  __device__ __forceinline__ void endEpoch(FlagType epoch) const {
    ddaSetLLEpoch(epochDev_, epochLen_, blockIdx.x, gridDim.x, epoch);
  }

  // Flags carry the epoch and only grow, so a slot is never reset and a waiter only
  // needs "flag >= epoch". A peer can be at most one barrier ahead (it cannot pass
  // epoch e before we post e), so the wrap-safe signed-difference compare is exact.
  template <bool hasPreviousMemAccess, bool hasSubsequentMemAccess>
  __device__ __forceinline__ void syncOnSameBlockIdx(FlagType epoch) {
    static_assert(hasPreviousMemAccess || hasSubsequentMemAccess);

    if constexpr (hasPreviousMemAccess) {
      __syncthreads();
    }

    // Strided so the barrier stays correct when blockDim.x < nRanks_.
    if (static_cast<int>(threadIdx.x) < nRanks_) {
      if constexpr (hasPreviousMemAccess) {
        __atomic_thread_fence(__ATOMIC_RELEASE);
      }
      for (int peerRank = threadIdx.x; peerRank < nRanks_; peerRank += blockDim.x) {
        __scoped_atomic_store_n(peerFlags_[peerRank] + getFlagIdx(selfRank_, blockIdx.x), epoch, __ATOMIC_RELAXED,
                                __MEMORY_SCOPE_SYSTEM);
      }
      FlagType* selfBuf = peerFlags_[selfRank_];
      for (int peerRank = threadIdx.x; peerRank < nRanks_; peerRank += blockDim.x) {
        FlagType* flag = selfBuf + getFlagIdx(peerRank, blockIdx.x);
        while (static_cast<int32_t>(__scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) - epoch) <
               0) {
        }
      }
      if constexpr (hasSubsequentMemAccess) {
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
      }
    }

    if constexpr (hasSubsequentMemAccess) {
      __syncthreads();
    }
  }

private:
  int nBlocks_{-1};
  int selfRank_{-1};
  int nRanks_{-1};
  FlagType** peerFlags_{nullptr};
  uint32_t* epochDev_{nullptr};
  int epochLen_{0};

  __host__ FabricGpuBarrier(int nBlocks, int selfRank, int nRanks, FlagType** peerFlags)
    : nBlocks_(nBlocks), selfRank_(selfRank), nRanks_(nRanks), peerFlags_(peerFlags) {}

  __device__ inline int getFlagIdx(int rank, int block) {
    return block * nRanks_ + rank;
  }
};

} // namespace dda::common
