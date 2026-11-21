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
 * flat fabric kernel's NRANKS_CT, so the DPX 8- and 16-rank cases fully unroll.
 *
 * ddaReduceScatterFabricHierPush runs the same two levels with stores instead
 * of loads: every cross-rank transfer pushes into the receiver's scratch and
 * every reduction reads local memory, so no transfer waits a remote round trip
 * and the input needs no staging copy. Each block cuts its rows into pieces and
 * pipelines them with per-peer tokens, so a piece's scale-up overlaps the next
 * pieces' scale-in and pushes instead of every rank waiting at group barriers.
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

// Loads 16 bytes another rank pushed into this rank's scratch, with system scope so no cache holds a stale copy.
template <typename T>
__device__ __forceinline__ uint4 ddaHierLoadPushed(const T* src) {
  uint4 v;
  ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(src), v.x, v.y, v.z, v.w);
  return v;
}

// Stores 16 bytes into a peer's scratch, system scope or plain; the post that follows drains the stores either way.
template <bool SYS_STORE, typename T>
__device__ __forceinline__ void ddaHierStorePeer(T* dst, const uint4& v) {
  if constexpr (SYS_STORE) {
    ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(dst), v.x, v.y, v.z, v.w);
  } else {
    *reinterpret_cast<uint4*>(dst) = v;
  }
}

// Push form of ddaReduceScatterFabricHier. Rank r = pkg * S + tile (S = scaleIn) uses its scratch slots as:
//   slot g * S + t, t != tile: package peer t's shard g * S + tile, the scale-in input of this rank's shard.
//   slot g * S + tile, g != pkg: package g's partial of shard r, the scale-up input.
// Slot r is unused: this rank keeps its own package's partial of shard r in recvbuff.
//
// A block's rows (idx = idxStart + row * idxStride) are cut into nPieces pieces of perPiece rows. Step k:
//   scale-up piece k - 2: consume the same-tile ranks' tokens, add their partials to recvbuff.
//   scale-in piece k - 1: consume the package peers' tokens, reduce every shard of this tile, keep the own one in
//                         recvbuff and push the others to their owners; post the same-tile ranks.
//   push piece k:         store the shards of the other tiles into their package peers; post the package peers.
// This order cannot deadlock on the one-token flags: a post at step k only waits for the receiver to consume this
// rank's token from step k - 1, which the receiver does at step k before any post of its own.
//
// The entry handshake keeps a push off scratch its receiver's previous collective still reads; the closing comm
// barrier keeps the next collective of any peer off this rank's scratch until it has been read.
template <typename T, int SCALE_IN_CT, int NGROUPS_CT, bool SYS_STORE>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterFabricHierPush(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff,
                                                 size_t count, const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                                 int scaleInRt, unsigned pieces, FabricGpuBarrier barrier) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  const int scaleIn = (SCALE_IN_CT > 0) ? SCALE_IN_CT : scaleInRt;
  const int nGroups = (NGROUPS_CT > 0) ? NGROUPS_CT : nRanks / scaleIn;
  constexpr int kScaleInUnroll = (SCALE_IN_CT > 0) ? SCALE_IN_CT : 2;
  constexpr int kGroupUnroll = (NGROUPS_CT > 0) ? NGROUPS_CT : 8;
  const int tile = selfRank % scaleIn;
  const int pkg = selfRank / scaleIn;
  const int pkgFirst = selfRank - tile;

  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const size_t idxStart = (static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x) * countPerThread;
  const size_t idxStride = static_cast<size_t>(gridDim.x) * blockDim.x * countPerThread;

  // Every rank derives the same piece split: it depends only on count and the launch shape.
  const size_t nRows = (count + idxStride - 1) / idxStride;
  const size_t nPiecesWanted = pieces > 0 ? pieces : 1;
  const size_t perPiece = (nRows + nPiecesWanted - 1) / nPiecesWanted;
  const size_t nPieces = (nRows + perPiece - 1) / perPiece;
  auto rowsOf = [&](size_t piece, auto&& body) {
    const size_t rowEnd = (piece + 1) * perPiece;
    size_t idx = idxStart + piece * perPiece * idxStride;
    for (size_t row = piece * perPiece; row < rowEnd && idx < count; ++row, idx += idxStride) {
      body(idx);
    }
  };

  T* const self = ipcbuffs[selfRank];
  auto pkgPeer = [&](int i) { return pkgFirst + (tile + 1 + i) % scaleIn; };
  auto tilePeer = [&](int i) { return ((pkg + 1 + i) % nGroups) * scaleIn + tile; };

  barrier.syncGroupOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(pkgFirst, 1,
                                                                                                       scaleIn);
  barrier.syncGroupOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(tile, scaleIn,
                                                                                                       nGroups);

  for (size_t k = 0; k < nPieces + 2; ++k) {
    if (k >= 2) {
      barrier.waitPeers(tilePeer, nGroups - 1);
      rowsOf(k - 2, [&](size_t idx) {
        uint4 sum = *reinterpret_cast<const uint4*>(&recvbuff[idx]);
#pragma unroll kGroupUnroll
        for (int g = 0; g < nGroups; ++g) {
          if (g != pkg) {
            sum =
              vecElementAdd<T>(sum, ddaHierLoadPushed(&self[static_cast<size_t>(g * scaleIn + tile) * count + idx]));
          }
        }
        *reinterpret_cast<uint4*>(&recvbuff[idx]) = sum;
      });
    }
    if (k >= 1 && k <= nPieces) {
      barrier.waitPeers(pkgPeer, scaleIn - 1);
      rowsOf(k - 1, [&](size_t idx) {
#pragma unroll kGroupUnroll
        for (int g = 0; g < nGroups; ++g) {
          uint4 sum =
            reinterpret_cast<const uint4*>(&sendbuff[static_cast<size_t>(g * scaleIn + tile) * count + idx])[0];
#pragma unroll kScaleInUnroll
          for (int t = 0; t < scaleIn; ++t) {
            if (t != tile) {
              sum = vecElementAdd<T>(sum, ddaHierLoadPushed(&self[static_cast<size_t>(g * scaleIn + t) * count + idx]));
            }
          }
          if (g == pkg) {
            // recvbuff may alias shard selfRank of sendbuff, which this thread has just read at idx.
            *reinterpret_cast<uint4*>(&recvbuff[idx]) = sum;
          } else {
            ddaHierStorePeer<SYS_STORE>(&ipcbuffs[g * scaleIn + tile][static_cast<size_t>(selfRank) * count + idx],
                                        sum);
          }
        }
      });
      barrier.postPeers(tilePeer, nGroups - 1);
    }
    if (k < nPieces) {
      rowsOf(k, [&](size_t idx) {
#pragma unroll kGroupUnroll
        for (int g = 0; g < nGroups; ++g) {
#pragma unroll kScaleInUnroll
          for (int t = 0; t < scaleIn; ++t) {
            if (t != tile) {
              const uint4 v =
                reinterpret_cast<const uint4*>(&sendbuff[static_cast<size_t>(g * scaleIn + t) * count + idx])[0];
              ddaHierStorePeer<SYS_STORE>(
                &ipcbuffs[pkgFirst + t][static_cast<size_t>(g * scaleIn + tile) * count + idx], v);
            }
          }
        }
      });
      barrier.postPeers(pkgPeer, scaleIn - 1);
    }
  }

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace dda::common
