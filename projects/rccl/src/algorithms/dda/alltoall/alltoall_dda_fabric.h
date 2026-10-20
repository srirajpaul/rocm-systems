/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA alltoall kernel for the fabric/VMM path, using FabricGpuBarrier.
 *
 * Templated on a compile-time rank count NRANKS_CT (matching the all-reduce
 * fabric design):
 *   - NRANKS_CT > 0  : specialized for that clique size (unrolled peer loop).
 *   - NRANKS_CT == 0 : runtime fallback driven by the nRanks argument, so a
 *                      single instantiation covers any other clique size up to
 *                      kDdaMaxNranks.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                    int selfRank, int nRanks, FabricGpuBarrier barrier) {
  // Release-acquire barrier ensures the stream-ordered scratch write is visible to peers.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  // use uint4 to do 16-byte loads to maximize memory efficiency. We assume
  // that count % countPerThread == 0, enforced before kernel launch.
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  // Fully unroll when the clique size is a compile-time constant; otherwise
  // partially unroll 8-wide for the runtime fallback (Option A).
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll kUnroll
    for (int r = 0; r < nRanksEff; ++r) {
      int srcRank = r;
      size_t srcIdx = idx + selfRank * idxEnd;
      size_t destIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[destIdx]) = reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

// Push model: each rank writes chunk r of its sendbuff into peer r's scratch at slot selfRank, then drains its
// own scratch into recvbuff after the closing barrier.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricPush(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                        const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                        FabricGpuBarrier barrier) {
  // A peer's scratch is free to overwrite once that peer has reached this op, i.e. its previous
  // stream-ordered scratch -> recvbuff copy has drained.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // Load a whole group before storing it, so kUnroll loads are in flight per thread. Full groups carry no
  // per-rank bounds check, which keeps the staging array in registers; leftover ranks are copied one at a time.
  const size_t destIdx = selfRank * idxEnd;
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    int r0 = 0;
    for (; r0 + kUnroll <= nRanksEff; r0 += kUnroll) {
      uint4 v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) v[u] = reinterpret_cast<const uint4*>(&sendbuff[idx + (r0 + u) * idxEnd])[0];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) *reinterpret_cast<uint4*>(&ipcbuffs[r0 + u][destIdx + idx]) = v[u];
    }
    for (int r = r0; r < nRanksEff; ++r) {
      *reinterpret_cast<uint4*>(&ipcbuffs[r][destIdx + idx]) =
        reinterpret_cast<const uint4*>(&sendbuff[idx + r * idxEnd])[0];
    }
  }

  // Release our remote writes and acquire every peer's writes into our scratch.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  // Drain our scratch into recvbuff. Peer r's thread gtIdx wrote exactly the positions this thread reads
  // from slot r, since every rank launches the same grid.
  const T* selfScratch = ipcbuffs[selfRank];
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll kUnroll
    for (int r = 0; r < nRanksEff; ++r) {
      size_t i = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[i]) = reinterpret_cast<const uint4*>(&selfScratch[i])[0];
    }
  }
}

constexpr int kDdaA2APushPeerUnroll = 8;

template <bool sysScope>
__device__ __forceinline__ void ddaA2APushPeerStore(uint4* p, const uint4& v) {
  if constexpr (sysScope) {
    ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(p), v.x, v.y, v.z, v.w);
  } else {
    *p = v;
  }
}

template <bool sysScope>
__device__ __forceinline__ uint4 ddaA2APushPeerLoad(const uint4* p) {
  uint4 v;
  if constexpr (sysScope) {
    ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(p), v.x, v.y, v.z, v.w);
  } else {
    v = *p;
  }
  return v;
}

// Copies 16 B lines i = begin, begin + stride, ... < end of src to dst, kDdaA2APushPeerUnroll loads in flight per
// thread. sysScopeStore / sysScopeLoad select system-scope accesses, whose helpers are compiler memory barriers, so
// every load of a group is issued before its stores. Full groups carry no per-line bounds check; leftover lines are
// copied one at a time.
template <bool sysScopeStore, typename T, bool sysScopeLoad = false>
__device__ __forceinline__ void ddaA2APushPeerCopy(T* dst, const T* src, size_t begin, size_t end, size_t stride) {
  constexpr int U = kDdaA2APushPeerUnroll;
  const uint4* s = reinterpret_cast<const uint4*>(src);
  uint4* d = reinterpret_cast<uint4*>(dst);
  size_t i = begin;
  for (; i + (U - 1) * stride < end; i += U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = ddaA2APushPeerLoad<sysScopeLoad>(&s[i + u * stride]);
#pragma unroll
    for (int u = 0; u < U; ++u) ddaA2APushPeerStore<sysScopeStore>(&d[i + u * stride], v[u]);
  }
  for (; i < end; i += stride) {
    ddaA2APushPeerStore<sysScopeStore>(&d[i], ddaA2APushPeerLoad<sysScopeLoad>(&s[i]));
  }
}

// Peer-major push: every block serves a single peer, so its stores stream to one destination instead of one line to
// each rank. With P = gridDim.x / nRanks blocks per peer, block b serves peer offset k = b / P, and the P blocks of a
// group interleave over the chunk one block-sized run of 16 B lines at a time. The block pushes chunk (self + k) of
// sendbuff into that peer's scratch at slot selfRank with system-scope stores, then syncs with just those two peers
// and drains the slot of peer (self - k), which that peer's block b wrote. k == 0 copies sendbuff straight to
// recvbuff and drains nothing.
// With pieceLines > 0 (a multiple of the group's stride) the chunk is cut into pieces of pieceLines lines and the
// sync becomes a pipeline: step p pushes piece p, waits for the source's piece p - 1 and drains it while piece p's
// remote stores are in flight, then waits for its own stores and posts piece p. The pieces reuse the block's flag
// slot, so a post first waits for the destination to consume the previous piece; the drain reads with system-scope
// loads, so no fence is needed per piece.
// gridDim.x must be a multiple of nRanks. Out of place only: a block drains into recvbuff slots that other blocks of
// this rank may still be reading from sendbuff.
template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricPushPeer(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                            const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                            FabricGpuBarrier barrier, size_t pieceLines) {
  // A peer's scratch is free to overwrite once that peer has reached this op.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  constexpr size_t countPerThread = sizeof(uint4) / sizeof(T);
  const unsigned perPeer = gridDim.x / nRanks;
  const int k = static_cast<int>(blockIdx.x / perPeer);
  const size_t nLines = count / countPerThread;
  const size_t begin = static_cast<size_t>(blockIdx.x % perPeer) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(perPeer) * blockDim.x;

  const int peer = (selfRank + k) % nRanks;
  const T* src = sendbuff + static_cast<size_t>(peer) * count;
  const size_t selfSlot = static_cast<size_t>(selfRank) * count;
  if (k == 0) {
    ddaA2APushPeerCopy<false>(recvbuff + selfSlot, src, begin, nLines, stride);
    return;
  }

  T* dst = ipcbuffs[peer] + selfSlot;
  const int srcRank = (selfRank - k + nRanks) % nRanks;
  const size_t slot = static_cast<size_t>(srcRank) * count;
  T* drainDst = recvbuff + slot;
  const T* drainSrc = ipcbuffs[selfRank] + slot;

  if (pieceLines == 0) {
    ddaA2APushPeerCopy<true>(dst, src, begin, nLines, stride);
    // Point-to-point closing sync: release our writes to the peer we pushed to and acquire the writes of the peer
    // whose slot we drain, block b on both sides. Blocks of this rank's own chunk neither wrote remotely nor drain,
    // so they skip it.
    barrier.signalWaitPeer(peer, srcRank, blockIdx.x);
    ddaA2APushPeerCopy<false>(drainDst, drainSrc, begin, nLines, stride);
    return;
  }

  const size_t nPieces = (nLines + pieceLines - 1) / pieceLines;
  for (size_t p = 0; p <= nPieces; ++p) {
    if (p < nPieces) {
      const size_t end = (p + 1) * pieceLines < nLines ? (p + 1) * pieceLines : nLines;
      ddaA2APushPeerCopy<true>(dst, src, p * pieceLines + begin, end, stride);
    }
    if (p > 0) {
      if (threadIdx.x == 0) barrier.waitPeerRelaxed(srcRank);
      __syncthreads();
      const size_t end = p * pieceLines < nLines ? p * pieceLines : nLines;
      ddaA2APushPeerCopy<false, T, true /* sysScopeLoad */>(drainDst, drainSrc, (p - 1) * pieceLines + begin, end,
                                                            stride);
    }
    if (p < nPieces) {
      __syncthreads();
      if (threadIdx.x == 0) barrier.postPeerRelaxed(peer, blockIdx.x);
    }
  }
}

} // namespace dda::common
