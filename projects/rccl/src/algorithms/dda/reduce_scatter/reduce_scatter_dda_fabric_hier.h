/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Hierarchical DDA reduce-scatter kernel for the fabric/VMM path in partition
 * mode (gfx1250 DPX), using FabricGpuBarrier group syncs.
 *
 * Each physical device (package) hosts scaleIn consecutive ranks, so rank
 * r = pkg * scaleIn + tile. The links between the tiles of a package are the
 * scale-in level; the links between packages are the scale-up level. The flat
 * kernel reads nRanks - scaleIn of its nRanks shards across packages. Reducing
 * within the package first leaves nRanks / scaleIn - 1 cross-package shards
 * per rank, half of the flat count on DPX.
 *
 *   Stage:    copy the shards the other tiles of this package reduce
 *             (shard % scaleIn != tile) from sendbuff into this rank's scratch.
 *             Sync the package.
 *   Scale-in: for every shard of this tile (shard % scaleIn == tile), add the
 *             package peers' copies to sendbuff and keep the partial in this
 *             rank's scratch. A package peer only reads shards of its own tile,
 *             so the partial can overwrite in place. Sync the ranks of this tile
 *             on every package.
 *   Scale-up: sum this rank's shard of every package's partial into recvbuff.
 *   End:      sync the comm, so no rank reuses scratch a peer is still reading.
 *
 * Every phase strides a shard with the same grid, so block b only reads what
 * block b of a peer wrote and every sync pairs same-index blocks. recvbuff may
 * alias this rank's shard of sendbuff: a block reads that range only in its
 * scale-in step, before it writes it in scale-up.
 *
 * Templated on compile-time SCALE_IN_CT / NGROUPS_CT (0 = runtime) like the
 * flat fabric kernel's NRANKS_CT, so the DPX 8-rank case fully unrolls.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

template <typename T, int SCALE_IN_CT, int NGROUPS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterFabricHier(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                             const T* __restrict__ sendbuff, int selfRank, int nRanks, int scaleInRt,
                                             FabricGpuBarrier barrier) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  const int scaleIn = (SCALE_IN_CT > 0) ? SCALE_IN_CT : scaleInRt;
  const int nGroups = (NGROUPS_CT > 0) ? NGROUPS_CT : nRanks / scaleIn;
  constexpr int kScaleInUnroll = (SCALE_IN_CT > 0) ? SCALE_IN_CT : 2;
  constexpr int kGroupUnroll = (NGROUPS_CT > 0) ? NGROUPS_CT : 8;
  const int tile = selfRank % scaleIn;
  const int pkgFirst = selfRank - tile;

  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const size_t idxStart = (static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x) * countPerThread;
  const size_t idxStride = static_cast<size_t>(gridDim.x) * blockDim.x * countPerThread;

  T* const self = ipcbuffs[selfRank];

  // Stage: only the package peers read this rank's raw input.
  for (int g = 0; g < nGroups; ++g) {
#pragma unroll kScaleInUnroll
    for (int t = 0; t < scaleIn; ++t) {
      if (t != tile) {
        const size_t off = static_cast<size_t>(g * scaleIn + t) * count;
        copyFromSrcToDest<T>(sendbuff + off, self + off, idxStart, count, idxStride);
      }
    }
  }
  barrier.syncGroupOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(pkgFirst, 1,
                                                                                                      scaleIn);

  // Scale-in: partial[shard] = sum over the package of shard, for the shards of this tile.
  for (size_t idx = idxStart; idx < count; idx += idxStride) {
#pragma unroll kGroupUnroll
    for (int g = 0; g < nGroups; ++g) {
      const size_t off = static_cast<size_t>(g * scaleIn + tile) * count + idx;
      uint4 sum = reinterpret_cast<const uint4*>(&sendbuff[off])[0];
#pragma unroll kScaleInUnroll
      for (int t = 0; t < scaleIn; ++t) {
        if (t != tile) {
          sum = vecElementAdd<T>(sum, reinterpret_cast<const uint4*>(&ipcbuffs[pkgFirst + t][off])[0]);
        }
      }
      *reinterpret_cast<uint4*>(&self[off]) = sum;
    }
  }
  barrier.syncGroupOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(tile, scaleIn,
                                                                                                      nGroups);

  // Scale-up: this rank's shard summed over the partials of every package, own package first.
  const int pkg = selfRank / scaleIn;
  const size_t selfOff = static_cast<size_t>(selfRank) * count;
  for (size_t idx = idxStart; idx < count; idx += idxStride) {
    uint4 sum = reinterpret_cast<const uint4*>(&self[selfOff + idx])[0];
#pragma unroll kGroupUnroll
    for (int k = 1; k < nGroups; ++k) {
      // Start each rank on a different package so the tile's ranks do not all load from one peer at once.
      int g = pkg + k;
      if (g >= nGroups) g -= nGroups;
      sum = vecElementAdd<T>(sum, reinterpret_cast<const uint4*>(&ipcbuffs[g * scaleIn + tile][selfOff + idx])[0]);
    }
    *reinterpret_cast<uint4*>(&recvbuff[idx]) = sum;
  }

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace dda::common
