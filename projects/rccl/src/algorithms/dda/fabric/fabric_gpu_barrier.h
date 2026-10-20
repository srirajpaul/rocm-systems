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

// Flags carry an epoch and only grow, so a slot is never reset and a waiter only needs "flag >= epoch". The epoch
// comes from the comm's LL epoch cells, so every fabric collective advances the same per-block counter: a kernel reads
// flag = beginEpoch(), passes flag + k to its k-th sync and stores the last value it used with endEpoch(). A peer can
// be at most one sync ahead on a slot (it cannot pass epoch e before we post e), so the wrap-safe signed-difference
// compare is exact.
class FabricGpuBarrier {
public:
  using FlagType = uint32_t;

  __host__ FabricGpuBarrier() = default;

  // Allocates this rank's flag buffer, exchanges peer flag-buffer pointers via
  // fabric shareable handles and builds a device-resident pointer table.
  static __host__ std::pair<std::unique_ptr<FabricGpuBarrierResources>, FabricGpuBarrier> mallocAndInit(
    int nRanks, int nBlocks, int selfRank, void* bootstrap, struct ncclMemManager* manager);

  // Attaches the comm's LL epoch cells (comm->ddaLLEpochDev), which the barrier shares with the LL/LL128 kernels so
  // every fabric collective advances one epoch.
  __host__ void setEpochs(uint32_t* epochDev, int epochLen) {
    epochDev_ = epochDev;
    epochLen_ = epochLen;
  }

  // Both must be called by every thread of the block.
  __device__ __forceinline__ FlagType beginEpoch() const {
    return ddaGetLLEpochInc(epochDev_, blockIdx.x, 1);
  }
  __device__ __forceinline__ void endEpoch(FlagType epoch) const {
    ddaSetLLEpoch(epochDev_, epochLen_, blockIdx.x, gridDim.x, epoch);
  }

  // Per-block-slot records {epoch, tag} (nBlocks of them, initialised to all ones) written by endEpochRecord. Every
  // fabric kernel moves the epoch cells, so a record whose epoch equals the epoch beginEpoch() just read means the
  // last fabric kernel on this communicator was one that ended with endEpochRecord, and the record holds its tag.
  __host__ void setRecords(uint32_t* records) {
    records_ = records;
  }
  __device__ __forceinline__ bool lastEndedWithRecord(FlagType flag, uint32_t* tag) const {
    const uint32_t* r = records_ + 2 * blockIdx.x;
    if (r[0] != flag - 1) return false;
    *tag = r[1];
    return true;
  }
  // endEpoch plus a record in every slot (slot e is written by block e % gridDim.x), so a later launch with any grid
  // reads the same record in every block. Must be called by every thread of the block.
  __device__ __forceinline__ void endEpochRecord(FlagType epoch, uint32_t tag) const {
    __syncthreads(); // every thread has read this block's record (lastEndedWithRecord) before it is rewritten
    endEpoch(epoch);
    for (int e = blockIdx.x + threadIdx.x * gridDim.x; e < nBlocks_; e += blockDim.x * gridDim.x) {
      records_[2 * e] = epoch;
      records_[2 * e + 1] = tag;
    }
  }

// Waits until every store this wave has issued has completed. A system-scope store has then reached its destination.
__device__ __forceinline__ void orderDestWrites() {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if defined(__gfx1250__) || defined(__GFX12__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#else
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#else
  __threadfence_system();
#endif
}

  template <bool hasPreviousMemAccess, bool hasSubsequentMemAccess>
  __device__ __forceinline__ void syncOnSameBlockIdx(FlagType epoch) {
    static_assert(hasPreviousMemAccess || hasSubsequentMemAccess);

    if constexpr (hasPreviousMemAccess) {
      __syncthreads();
    }
    // Each thread handles one or more peers in a strided loop so the barrier
    // stays correct when blockDim.x < nRanks_. A small count can launch as few
    // as 64 threads while a clique may have up to kDdaMaxNranks ranks; with a
    // single thread per peer, peers with rank >= blockDim.x would never be
    // signaled or waited on, hanging the barrier.
    if (static_cast<int>(threadIdx.x) < nRanks_) {
      if constexpr (hasPreviousMemAccess) {
        __atomic_thread_fence(__ATOMIC_RELEASE);
      }
      for (int peerRank = threadIdx.x; peerRank < nRanks_; peerRank += blockDim.x) {
        storeFlag(peerFlags_[peerRank] + getFlagIdx(selfRank_, blockIdx.x), epoch);
      }
      FlagType* selfBuf = peerFlags_[selfRank_];
      for (int peerRank = threadIdx.x; peerRank < nRanks_; peerRank += blockDim.x) {
        waitFlagAtLeast(selfBuf + getFlagIdx(peerRank, blockIdx.x), epoch);
      }
      if constexpr (hasSubsequentMemAccess) {
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
      }
    }
    if constexpr (hasSubsequentMemAccess) {
      __syncthreads();
    }
  }

  // Two-rank form of syncOnSameBlockIdx<true, true>: posts epoch into dstRank's slot for block dstBlock and waits for
  // srcRank's slot for this block in this rank's buffer to reach it.
  __device__ __forceinline__ void signalWaitPeer(int dstRank, int srcRank, int dstBlock, FlagType epoch) {
    __syncthreads();
    orderDestWrites();
    if (threadIdx.x == 0) {
      __atomic_thread_fence(__ATOMIC_RELEASE);
      storeFlag(peerFlags_[dstRank] + getFlagIdx(selfRank_, dstBlock), epoch);
      waitFlagAtLeast(peerFlags_[selfRank_] + getFlagIdx(srcRank, blockIdx.x), epoch);
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
    }
    __syncthreads();
  }

  // Single-thread halves of signalWaitPeer with relaxed ordering, for callers that order their data accesses
  // themselves (system-scope accesses and an explicit wait for their own stores). A post does not wait for the
  // previous value to be seen, so successive posts to one slot can carry increasing values.
  __device__ __forceinline__ void postPeerRelaxed(int dstRank, int dstBlock, FlagType value) {
    orderDestWrites();
    storeFlag(peerFlags_[dstRank] + getFlagIdx(selfRank_, dstBlock), value);
  }
  __device__ __forceinline__ void waitPeerRelaxed(int srcRank, FlagType value) {
    waitFlagAtLeast(peerFlags_[selfRank_] + getFlagIdx(srcRank, blockIdx.x), value);
  }

private:
  int nBlocks_{-1};
  int selfRank_{-1};
  int nRanks_{-1};
  FlagType** peerFlags_{nullptr};
  uint32_t* epochDev_{nullptr};
  int epochLen_{0};
  uint32_t* records_{nullptr};

  __host__ FabricGpuBarrier(int nBlocks, int selfRank, int nRanks, FlagType** peerFlags)
    : nBlocks_(nBlocks), selfRank_(selfRank), nRanks_(nRanks), peerFlags_(peerFlags) {}

  __device__ inline int getFlagIdx(int rank, int block) {
    return block * nRanks_ + rank;
  }

  static __device__ __forceinline__ void storeFlag(FlagType* flag, FlagType value) {
    __scoped_atomic_store_n(flag, value, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
  static __device__ __forceinline__ void waitFlagAtLeast(FlagType* flag, FlagType value) {
    while (static_cast<int32_t>(__scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) - value) < 0) {
    }
  }
};

} // namespace dda::common
