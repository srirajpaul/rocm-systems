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
 * EPOCH_FLAGS syncs with the barrier's epoch flags (one-way stores) instead of
 * its toggle flags (a remote CAS round trip per peer); the three syncs post
 * epoch, epoch + 1 and epoch + 2.
 * Scratch is used from element scratchOff of every rank's buffer. Without
 * END_BARRIER the kernel skips the closing comm sync; the launcher then gives it
 * a private scratch region no other collective uses and alternates its two
 * halves per launch, so the next launch never overwrites what a slow peer of
 * this one may still load (see ncclReduceScatterDdaFabricTyped).
 *
 * ddaReduceScatterFabricHierPush runs the same two levels with stores instead
 * of loads: every cross-rank transfer pushes into the receiver's scratch and
 * every reduction reads local memory, so no transfer waits a remote round trip
 * and the input needs no staging copy. Each block cuts its rows into pieces and
 * pipelines them with per-peer tokens, so a piece's scale-up overlaps the next
 * pieces' scale-in and pushes instead of every rank waiting at group barriers.
 *
 * ddaReduceScatterFabricHierPipe is the pull kernel cut into pieces and run as
 * a software pipeline, so the staging copy, the scale-in loads and the
 * scale-up loads of different pieces are in flight together.
 *
 * ddaReduceScatterFabricHierPushUp loads at the scale-in level, like the pull
 * kernel, and stores at the scale-up level: each rank pushes every partial to
 * the rank that owns its shard, which then reduces from its own scratch.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

template <typename T, int SCALE_IN_CT, int NGROUPS_CT, bool EPOCH_FLAGS, bool END_BARRIER>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterFabricHier(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                             const T* __restrict__ sendbuff, int selfRank, int nRanks, int scaleInRt,
                                             size_t scratchOff, FabricGpuBarrier barrier) {
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

  T* const self = ipcbuffs[selfRank] + scratchOff;

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
  const FabricGpuBarrier::FlagType epoch = EPOCH_FLAGS ? barrier.epochBegin() : 0;
  auto syncGroup = [&](int firstRank, int rankStride, int nGroupRanks, int step) {
    if constexpr (EPOCH_FLAGS) {
      barrier.epochSyncGroup(firstRank, rankStride, nGroupRanks, epoch + step);
    } else {
      barrier.syncGroupOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(
        firstRank, rankStride, nGroupRanks);
    }
  };
  syncGroup(pkgFirst, 1, scaleIn, 0);

  // Scale-in: partial[shard] = sum over the package of shard, for the shards of this tile.
  if constexpr (SCALE_IN_CT > 0 && NGROUPS_CT > 0) {
    // A row loads every input before it stores any partial. The stores go through the ipcbuffs table, so the compiler
    // cannot move a later load above an earlier store, and an interleaved loop would wait out each load in turn.
    for (size_t idx = idxStart; idx < count; idx += idxStride) {
      uint4 sum[NGROUPS_CT];
      uint4 peer[NGROUPS_CT][SCALE_IN_CT];
#pragma unroll
      for (int g = 0; g < NGROUPS_CT; ++g) {
        const size_t off = static_cast<size_t>(g * SCALE_IN_CT + tile) * count + idx;
        sum[g] = reinterpret_cast<const uint4*>(&sendbuff[off])[0];
#pragma unroll
        for (int t = 0; t < SCALE_IN_CT; ++t) {
          if (t != tile) {
            peer[g][t] = reinterpret_cast<const uint4*>(&ipcbuffs[pkgFirst + t][scratchOff + off])[0];
          }
        }
      }
#pragma unroll
      for (int g = 0; g < NGROUPS_CT; ++g) {
#pragma unroll
        for (int t = 0; t < SCALE_IN_CT; ++t) {
          if (t != tile) {
            sum[g] = vecElementAdd<T>(sum[g], peer[g][t]);
          }
        }
        *reinterpret_cast<uint4*>(&self[static_cast<size_t>(g * SCALE_IN_CT + tile) * count + idx]) = sum[g];
      }
    }
  } else {
    for (size_t idx = idxStart; idx < count; idx += idxStride) {
#pragma unroll kGroupUnroll
      for (int g = 0; g < nGroups; ++g) {
        const size_t off = static_cast<size_t>(g * scaleIn + tile) * count + idx;
        uint4 sum = reinterpret_cast<const uint4*>(&sendbuff[off])[0];
#pragma unroll kScaleInUnroll
        for (int t = 0; t < scaleIn; ++t) {
          if (t != tile) {
            sum = vecElementAdd<T>(sum, reinterpret_cast<const uint4*>(&ipcbuffs[pkgFirst + t][scratchOff + off])[0]);
          }
        }
        *reinterpret_cast<uint4*>(&self[off]) = sum;
      }
    }
  }
  syncGroup(tile, scaleIn, nGroups, 1);

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
      sum = vecElementAdd<T>(
        sum, reinterpret_cast<const uint4*>(&ipcbuffs[g * scaleIn + tile][scratchOff + selfOff + idx])[0]);
    }
    *reinterpret_cast<uint4*>(&recvbuff[idx]) = sum;
  }

  if constexpr (EPOCH_FLAGS) {
    barrier.epochSyncGroup(0, 1, nRanks, epoch + 2);
    barrier.epochEnd(epoch, 1);
  } else if constexpr (END_BARRIER) {
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
  }
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

// Scale-in by loads (or, with PUSH_IN, stores), scale-up by stores. Rank r = pkg * S + tile (S = scaleIn); its
// scratch slots, from element scratchOff of every rank's buffer, are:
//   slot g * S + t, t != tile: shard g * S + t of this rank, staged for package peer t to load; with PUSH_IN, shard
//                              g * S + tile of package peer t, stored there by that peer.
//   slot g * S + tile, g != pkg: package g's partial of shard r, stored by rank g * S + tile.
// The slot sets are disjoint, so partials can land while the scale-in slots are still in use.
//
//   Stage:    copy the shards of the other tiles from sendbuff into scratch (PUSH_IN: store them into the package
//             peers' scratch instead). Sync the package; without PRIVATE also this tile's ranks in the same round, so
//             every rank this one stores into is past its previous collective (the LL tiers have no closing barrier
//             and share the scratch).
//   Scale-in: for every shard of this tile, add the package peers' copies to sendbuff. Keep the own shard's partial
//             in recvbuff; store the others into their owners' scratch. Sync this tile's ranks.
//   Scale-up: add the partials of the other packages from this rank's scratch to recvbuff.
//   End:      without PRIVATE, sync the comm, so no rank reuses scratch a peer still loads or stores into.
//
// PRIVATE runs on the private region of RCCL_DDA_RS_HIER_PRIVATE, whose halves alternate per launch and which no
// other collective uses: launch n + 2 touches launch n's half only after the writer has passed launch n + 1's syncs
// with the receiver, so the receiver is done with it, and no closing sync or wider first sync is needed.
template <typename T, int SCALE_IN_CT, int NGROUPS_CT, bool PRIVATE, bool PUSH_IN>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__
  void ddaReduceScatterFabricHierPushUp(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                        const T* __restrict__ sendbuff, int selfRank, int nRanks, int scaleInRt,
                                        size_t scratchOff, FabricGpuBarrier barrier) {
  static_assert(is_supported_type_v<T>, "dda: unsupported element type");
  static_assert(PRIVATE || !PUSH_IN, "push-in stores into the package peers before any sync: needs the private region");
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

  T* const self = ipcbuffs[selfRank] + scratchOff;
  auto slot = [&](int g, int t, size_t idx) { return static_cast<size_t>(g * scaleIn + t) * count + idx; };

  if constexpr (PUSH_IN) {
    if constexpr (SCALE_IN_CT > 0 && NGROUPS_CT > 0) {
      for (size_t idx = idxStart; idx < count; idx += idxStride) {
        uint4 v[NGROUPS_CT][SCALE_IN_CT];
#pragma unroll
        for (int g = 0; g < NGROUPS_CT; ++g) {
#pragma unroll
          for (int t = 0; t < SCALE_IN_CT; ++t) {
            if (t != tile) {
              v[g][t] = reinterpret_cast<const uint4*>(&sendbuff[slot(g, t, idx)])[0];
            }
          }
        }
#pragma unroll
        for (int g = 0; g < NGROUPS_CT; ++g) {
#pragma unroll
          for (int t = 0; t < SCALE_IN_CT; ++t) {
            if (t != tile) {
              *reinterpret_cast<uint4*>(&ipcbuffs[pkgFirst + t][scratchOff + slot(g, tile, idx)]) = v[g][t];
            }
          }
        }
      }
    } else {
      for (size_t idx = idxStart; idx < count; idx += idxStride) {
        for (int g = 0; g < nGroups; ++g) {
          for (int t = 0; t < scaleIn; ++t) {
            if (t != tile) {
              *reinterpret_cast<uint4*>(&ipcbuffs[pkgFirst + t][scratchOff + slot(g, tile, idx)]) =
                reinterpret_cast<const uint4*>(&sendbuff[slot(g, t, idx)])[0];
            }
          }
        }
      }
    }
  } else {
    for (int g = 0; g < nGroups; ++g) {
#pragma unroll kScaleInUnroll
      for (int t = 0; t < scaleIn; ++t) {
        if (t != tile) {
          const size_t off = static_cast<size_t>(g * scaleIn + t) * count;
          copyFromSrcToDest<T>(sendbuff + off, self + off, idxStart, count, idxStride);
        }
      }
    }
  }
  if constexpr (PRIVATE) {
    barrier.syncGroupOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(pkgFirst, 1,
                                                                                                        scaleIn);
  } else {
    barrier.syncPeerSetOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>([&](int r) {
      return r / scaleIn == pkg || r % scaleIn == tile;
    });
  }

  // Package peer t's copy of this tile's shard g * S + tile: in its scratch (staged) or in this rank's (pushed).
  auto loadPeerCopy = [&](int g, int t, size_t idx) {
    if constexpr (PUSH_IN) {
      return ddaHierLoadPushed(&self[slot(g, t, idx)]);
    } else {
      return reinterpret_cast<const uint4*>(&ipcbuffs[pkgFirst + t][scratchOff + slot(g, tile, idx)])[0];
    }
  };
  auto ownerOf = [&](int k) {
    // Start each rank on a different owner, so this tile's ranks do not all store into one package at once.
    const int g = pkg + k;
    return g >= nGroups ? g - nGroups : g;
  };
  auto storePartial = [&](int g, size_t idx, const uint4& sum) {
    if (g == pkg) {
      // recvbuff may alias shard selfRank of sendbuff, which this thread has already read at idx.
      *reinterpret_cast<uint4*>(&recvbuff[idx]) = sum;
    } else {
      *reinterpret_cast<uint4*>(
        &ipcbuffs[g * scaleIn + tile][scratchOff + static_cast<size_t>(selfRank) * count + idx]) = sum;
    }
  };
  if constexpr (SCALE_IN_CT > 0 && NGROUPS_CT > 0) {
    // A row loads every input before its first store, as in ddaReduceScatterFabricHier's scale-in.
    for (size_t idx = idxStart; idx < count; idx += idxStride) {
      uint4 sum[NGROUPS_CT];
      uint4 peer[NGROUPS_CT][SCALE_IN_CT];
#pragma unroll
      for (int k = 0; k < NGROUPS_CT; ++k) {
        const int g = ownerOf(k);
        sum[k] = reinterpret_cast<const uint4*>(&sendbuff[slot(g, tile, idx)])[0];
#pragma unroll
        for (int t = 0; t < SCALE_IN_CT; ++t) {
          if (t != tile) {
            peer[k][t] = loadPeerCopy(g, t, idx);
          }
        }
      }
#pragma unroll
      for (int k = 0; k < NGROUPS_CT; ++k) {
#pragma unroll
        for (int t = 0; t < SCALE_IN_CT; ++t) {
          if (t != tile) {
            sum[k] = vecElementAdd<T>(sum[k], peer[k][t]);
          }
        }
        storePartial(ownerOf(k), idx, sum[k]);
      }
    }
  } else {
    for (size_t idx = idxStart; idx < count; idx += idxStride) {
#pragma unroll kGroupUnroll
      for (int k = 0; k < nGroups; ++k) {
        const int g = ownerOf(k);
        uint4 sum = reinterpret_cast<const uint4*>(&sendbuff[slot(g, tile, idx)])[0];
#pragma unroll kScaleInUnroll
        for (int t = 0; t < scaleIn; ++t) {
          if (t != tile) {
            sum = vecElementAdd<T>(sum, loadPeerCopy(g, t, idx));
          }
        }
        storePartial(g, idx, sum);
      }
    }
  }
  barrier.syncGroupOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(tile, scaleIn,
                                                                                                      nGroups);

  for (size_t idx = idxStart; idx < count; idx += idxStride) {
    uint4 sum = *reinterpret_cast<const uint4*>(&recvbuff[idx]);
#pragma unroll kGroupUnroll
    for (int k = 1; k < nGroups; ++k) {
      sum = vecElementAdd<T>(sum, ddaHierLoadPushed(&self[slot(ownerOf(k), tile, idx)]));
    }
    *reinterpret_cast<uint4*>(&recvbuff[idx]) = sum;
  }

  if constexpr (!PRIVATE) {
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
  }
}

// Pipelined form of ddaReduceScatterFabricHier, with the same scratch layout. A block's rows
// (idx = idxStart + row * idxStride) are cut into nPieces pieces of perPiece rows, and step k = 0 .. nPieces + 1 is one
// fused pass over perPiece rows:
//   stage piece k:        copy the shards of the other tiles from sendbuff into scratch,
//   scale-in piece k - 1: add the package peers' staged copies to sendbuff, keep the partial in scratch,
//   scale-up piece k - 2: add this rank's shard of every package's partial, write recvbuff,
// then syncs the package and this tile's ranks in one round, which publishes piece k's staged copies and piece k - 1's
// partials. The three parts load independent data, so the local copy, the package-link loads and the cross-package
// loads of a step are in flight together, and they touch disjoint rows, so a store never hits what a peer loads in
// the same step. recvbuff may alias shard selfRank of sendbuff: scale-in reads that shard's rows of piece k - 1 while
// scale-up writes rows of piece k - 2.
template <typename T, int SCALE_IN_CT, int NGROUPS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterFabricHierPipe(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff,
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

  T* const self = ipcbuffs[selfRank];
  const size_t selfOff = static_cast<size_t>(selfRank) * count;

  for (size_t k = 0; k < nPieces + 2; ++k) {
    const bool doStage = k < nPieces;
    const bool doScaleIn = k >= 1 && k <= nPieces;
    const bool doScaleUp = k >= 2;
    for (size_t row = 0; row < perPiece; ++row) {
      if (doStage) {
        const size_t idx = idxStart + (k * perPiece + row) * idxStride;
        if (idx < count) {
#pragma unroll kGroupUnroll
          for (int g = 0; g < nGroups; ++g) {
#pragma unroll kScaleInUnroll
            for (int t = 0; t < scaleIn; ++t) {
              if (t != tile) {
                const size_t off = static_cast<size_t>(g * scaleIn + t) * count + idx;
                *reinterpret_cast<uint4*>(&self[off]) = reinterpret_cast<const uint4*>(&sendbuff[off])[0];
              }
            }
          }
        }
      }
      if (doScaleIn) {
        const size_t idx = idxStart + ((k - 1) * perPiece + row) * idxStride;
        if (idx < count) {
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
      }
      if (doScaleUp) {
        const size_t idx = idxStart + ((k - 2) * perPiece + row) * idxStride;
        if (idx < count) {
          uint4 sum = reinterpret_cast<const uint4*>(&self[selfOff + idx])[0];
#pragma unroll kGroupUnroll
          for (int j = 1; j < nGroups; ++j) {
            // Start each rank on a different package so the tile's ranks do not all load from one peer at once.
            int g = pkg + j;
            if (g >= nGroups) g -= nGroups;
            sum =
              vecElementAdd<T>(sum, reinterpret_cast<const uint4*>(&ipcbuffs[g * scaleIn + tile][selfOff + idx])[0]);
          }
          *reinterpret_cast<uint4*>(&recvbuff[idx]) = sum;
        }
      }
    }
    if (k < nPieces + 1) {
      barrier.syncPeerSetOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>([&](int r) {
        return r / scaleIn == pkg || r % scaleIn == tile;
      });
    }
  }

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace dda::common
