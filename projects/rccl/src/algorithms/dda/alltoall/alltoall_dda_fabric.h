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

constexpr int kDdaA2APushThreadsPerBlock = 1024;

// Pull alltoall: each rank stages its sendbuff into its own scratch, then reads its slot out of every
// peer's scratch into recvbuff. Every access is system-scope, so neither barrier needs a cache fence.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(kDdaA2APushThreadsPerBlock)
#endif
  __global__ void ddaAllToAllFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                    const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                    FabricGpuBarrier barrier) {
  const uint32_t flag = barrier.beginEpoch();

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

  // Stage every slot of sendbuff into our scratch. Thread gtIdx on each peer later reads exactly the lines this
  // thread writes (same grid on every rank), so the per-block barrier covers them. The previous op's closing
  // barrier guarantees no peer is still reading our scratch.
  T* selfScratch = ipcbuffs[selfRank];
  ddaSysCopyLines<kUnroll>(
    nRanksEff, idxStart, idxEnd, idxStride, [&](int r) { return sendbuff + r * idxEnd; },
    [&](int r) { return selfScratch + r * idxEnd; });

  ddaFabricWaitMemOps();
  // Publish the staged scratch; the reads below are system-scope, so no acquire is needed.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

  const size_t selfSlot = selfRank * idxEnd;
  ddaSysCopyLines<kUnroll>(
    nRanksEff, idxStart, idxEnd, idxStride, [&](int r) { return ipcbuffs[r] + selfSlot; },
    [&](int r) { return recvbuff + r * idxEnd; });

  // Peers must not overwrite their scratch until every rank has read its slot; nothing is published here.
  ddaFabricWaitMemOps();
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */>(flag + 2);
  barrier.endEpoch(flag + 2);
}

template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(kDdaA2APushThreadsPerBlock)
#endif
  __global__ void ddaAllToAllFabricPush(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                        const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                        FabricGpuBarrier barrier) {
  const uint32_t flag = barrier.beginEpoch();
  // Only local sendbuff is read before the next barrier, so no acquire is needed here.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // Every access is system-scope so none of it goes through L2. The line helpers are compiler memory
  // barriers, so issue every load of the group before its stores.
  const size_t destIdx = selfRank * idxEnd;
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    for (int r0 = 0; r0 < nRanksEff; r0 += kUnroll) {
      uint4 v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (r0 + u < nRanksEff) {
          uint32_t t0, t1, t2, t3;
          ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(&sendbuff[idx + (r0 + u) * idxEnd]), t0, t1, t2, t3);
          v[u] = make_uint4(t0, t1, t2, t3);
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (r0 + u < nRanksEff) {
          ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&ipcbuffs[r0 + u][destIdx + idx]), v[u].x, v[u].y, v[u].z,
                             v[u].w);
        }
      }
    }
  }

  // TODO : move these inside barrier
  __syncthreads();

#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if defined(__gfx1250__)
  asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0");
#else
  asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
#endif
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#else
  __threadfence_system();
#endif

  // Publish our remote writes. The drain below reads only with system-scope loads, so no acquire is needed.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 2);

  // Drain our scratch into recvbuff. Peer r's thread gtIdx wrote exactly the positions this thread reads
  // from slot r, since every rank launches the same grid. As above, every access is system-scope and the
  // loads of a group are issued before its stores.
  const T* selfScratch = ipcbuffs[selfRank];
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    for (int r0 = 0; r0 < nRanksEff; r0 += kUnroll) {
      uint32_t w[kUnroll][4];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (r0 + u < nRanksEff) {
          ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(&selfScratch[(r0 + u) * idxEnd + idx]), w[u][0], w[u][1],
                            w[u][2], w[u][3]);
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (r0 + u < nRanksEff) {
          ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&recvbuff[(r0 + u) * idxEnd + idx]), w[u][0], w[u][1], w[u][2],
                             w[u][3]);
        }
      }
    }
  }
  barrier.endEpoch(flag + 2);
}

constexpr int kDdaA2APushPeerUnroll = 8;

// Copies 16 B lines [begin, end) of src to dst with system-scope accesses, U loads in flight per thread. The line
// helpers are compiler memory barriers, so every load of a group is issued before its stores.
template <int U, typename T>
__device__ __forceinline__ void ddaA2APushPeerCopy(T* dst, const T* src, size_t begin, size_t end, size_t stride) {
  constexpr size_t countPerThread = sizeof(uint4) / sizeof(T);
  size_t i = begin;
  for (; i + (U - 1) * stride < end; i += U * stride) {
    uint32_t w[U][4];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(src + (i + u * stride) * countPerThread), w[u][0], w[u][1],
                        w[u][2], w[u][3]);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(dst + (i + u * stride) * countPerThread), w[u][0], w[u][1],
                         w[u][2], w[u][3]);
    }
  }
  for (; i < end; i += stride) {
    uint32_t w0, w1, w2, w3;
    ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(src + i * countPerThread), w0, w1, w2, w3);
    ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(dst + i * countPerThread), w0, w1, w2, w3);
  }
}

// Peer-major push: every block serves a single peer, so its stores stream to one destination instead of one line
// to each rank. With P = gridDim.x / nRanks blocks per peer, block b serves peer offset k = b / P and contiguous
// part j = b % P of that peer's chunk, 16 B lines [j * partLines, (j + 1) * partLines). It pushes to peer (self + k)
// and later drains the slot of peer (self - k), which that peer's block b wrote, so the per-block barrier pairs the
// write with the read. k == 0 copies sendbuff straight to recvbuff and drains nothing.
// With p2p the block instead drains the slot of the peer it pushed to, (self + k), which that peer's block with offset
// nRanks - k (same part) wrote, and waits on that block's flag. The only block that writes recvbuff slot (self + k)
// is then the one that read sendbuff slot (self + k), each thread on the lines it read, so p2p also runs in place.
// gridDim.x must be a multiple of nRanks. Without p2p out of place only: a block drains into recvbuff slots that
// other blocks of this rank may still be reading from sendbuff.
template <typename T, int U = kDdaA2APushPeerUnroll>
#if defined(USE_ROCM)
__launch_bounds__(kDdaA2APushThreadsPerBlock)
#endif
  __global__ void ddaAllToAllFabricPushPeer(T* const* __restrict__ ipcbuffs, T* recvbuff, size_t count,
                                            const T* sendbuff, int selfRank, int nRanks, FabricGpuBarrier barrier,
                                            size_t partLines, bool p2p) {
  const uint32_t flag = barrier.beginEpoch();
  // Only local sendbuff is read before the next barrier, so no acquire is needed here.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

  constexpr size_t countPerThread = sizeof(uint4) / sizeof(T);
  const size_t nParts = gridDim.x / nRanks;
  const int k = static_cast<int>(blockIdx.x / nParts);
  const size_t part = blockIdx.x % nParts;
  const size_t nLines = count / countPerThread;
  const size_t lineBegin = part * partLines;
  const size_t lineEnd = lineBegin + partLines < nLines ? lineBegin + partLines : nLines;
  const size_t begin = lineBegin + threadIdx.x;

  const int peer = (selfRank + k) % nRanks;
  T* dst = (k == 0 ? recvbuff : ipcbuffs[peer]) + static_cast<size_t>(selfRank) * count;
  const T* src = sendbuff + static_cast<size_t>(peer) * count;
  // In place, this rank's own slot is already where it belongs.
  if (dst != src) ddaA2APushPeerCopy<U>(dst, src, begin, lineEnd, blockDim.x);

  ddaFabricWaitMemOps();
  // Publish our remote writes. The drain below reads only with system-scope loads, so no acquire is needed. With p2p
  // this block only tells the peer it wrote and waits for the block whose write it drains; blocks of this rank's own
  // chunk (k == 0) neither wrote remotely nor drain, so they skip it.
  const int srcRank = p2p ? peer : (selfRank - k + nRanks) % nRanks;
  if (!p2p) {
    barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                               true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 2);
  } else if (k != 0) {
    barrier.signalWaitPeer(peer, srcRank, static_cast<int>((nRanks - k) * nParts + part), flag + 2);
  }

  if (k != 0) {
    const size_t src = static_cast<size_t>(srcRank) * count;
    ddaA2APushPeerCopy<U>(recvbuff + src, ipcbuffs[selfRank] + src, begin, lineEnd, blockDim.x);
  }
  barrier.endEpoch(flag + 2);
}

// 16 B line accesses: system scope (Sys) for fabric memory, i.e. peer scratch and this rank's scratch, which peers
// write; plain for this GPU's own sendbuff and recvbuff, which only this GPU touches during the kernel.
typedef unsigned ddaA2AV4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) ddaA2AV4* ddaA2AGV4;
template <bool Sys>
__device__ __forceinline__ void ddaA2ALoadLine(const uint32_t* p, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  if constexpr (Sys) {
    ddaLLLoadLineB128(p, a, b, c, d);
  } else {
    const ddaA2AV4 v = *(ddaA2AGV4)p;
    a = v.x, b = v.y, c = v.z, d = v.w;
  }
}
template <bool Sys>
__device__ __forceinline__ void ddaA2AStoreLine(uint32_t* p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  if constexpr (Sys) {
    ddaLLStoreLineB128(p, a, b, c, d);
  } else {
    const ddaA2AV4 v = {a, b, c, d};
    *(ddaA2AGV4)p = v;
  }
}
// ddaA2APushPeerCopy with the load and store scope chosen per side.
template <int U, bool SysLoad, bool SysStore, typename T>
__device__ __forceinline__ void ddaA2APushPeerCopyM(T* dst, const T* src, size_t begin, size_t end, size_t stride) {
  constexpr size_t countPerThread = sizeof(uint4) / sizeof(T);
  size_t i = begin;
  for (; i + (U - 1) * stride < end; i += U * stride) {
    uint32_t w[U][4];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      ddaA2ALoadLine<SysLoad>(reinterpret_cast<const uint32_t*>(src + (i + u * stride) * countPerThread), w[u][0],
                              w[u][1], w[u][2], w[u][3]);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      ddaA2AStoreLine<SysStore>(reinterpret_cast<uint32_t*>(dst + (i + u * stride) * countPerThread), w[u][0],
                                w[u][1], w[u][2], w[u][3]);
    }
  }
  for (; i < end; i += stride) {
    uint32_t w0, w1, w2, w3;
    ddaA2ALoadLine<SysLoad>(reinterpret_cast<const uint32_t*>(src + i * countPerThread), w0, w1, w2, w3);
    ddaA2AStoreLine<SysStore>(reinterpret_cast<uint32_t*>(dst + i * countPerThread), w0, w1, w2, w3);
  }
}

// Waits for every store this wave has issued to complete.
__device__ __forceinline__ void ddaA2AWaitWaveStores() {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if defined(__gfx1250__)
  asm volatile("s_wait_storecnt 0x0");
#else
  asm volatile("s_waitcnt vmcnt(0)");
#endif
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#else
  __threadfence_system();
#endif
}

// Pipelined peer-major push: the same block roles and parts as ddaAllToAllFabricPushPeer, with each part cut into
// `pieces` pieces of whole 4 KiB. Step i issues the push of piece i, then, while those stores are in flight, waits for
// the source's piece i - 1 and drains it, and only then waits for its own stores and tells the peer piece i is ready;
// a final step drains the last piece. The source is the peer this block pushes to, (self + k): the block drains the
// slot that peer's block with offset nRanks - k (same part) wrote, so the only block writing recvbuff slot (self + k)
// is the one that read sendbuff slot (self + k), each thread on the lines it pushed, and the kernel runs in place.
// Piece i's ready flag is epoch flag + 2 + i in the barrier slot this block owns at the peer, so values only grow and
// the next barrier posts a higher one. Every wave polls the flag itself, so draining does not wait for the block's
// other waves. Scratch accesses are system scope; this GPU's own sendbuff and recvbuff use plain accesses. The entry
// barrier stays a full barrier (the previous call may be a kernel that posts nothing after its drain). gridDim.x must
// be a multiple of nRanks.
template <typename T, int U = kDdaA2APushPeerUnroll>
#if defined(USE_ROCM)
__launch_bounds__(kDdaA2APushThreadsPerBlock)
#endif
  __global__ void ddaAllToAllFabricPushPeerPipe(T* const* __restrict__ ipcbuffs, T* recvbuff, size_t count,
                                                const T* sendbuff, int selfRank, int nRanks, FabricGpuBarrier barrier,
                                                size_t partLines, int pieces) {
  const uint32_t flag = barrier.beginEpoch();
  // Only local sendbuff is read before the next barrier, so no acquire is needed here.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */,
                             true /* sysScopeWrites */, true /* sysScopeReads */>(flag + 1);

  constexpr size_t countPerThread = sizeof(uint4) / sizeof(T);
  constexpr size_t kPieceAlignLines = 4096 / sizeof(uint4);
  const size_t nParts = gridDim.x / nRanks;
  const int k = static_cast<int>(blockIdx.x / nParts);
  const size_t part = blockIdx.x % nParts;
  const size_t nLines = count / countPerThread;
  const size_t lineBegin = part * partLines;
  const size_t lineEnd = lineBegin + partLines < nLines ? lineBegin + partLines : nLines;
  const int peer = (selfRank + k) % nRanks;
  T* dst = (k == 0 ? recvbuff : ipcbuffs[peer]) + static_cast<size_t>(selfRank) * count;
  const T* src = sendbuff + static_cast<size_t>(peer) * count;

  if (k == 0 || lineBegin >= lineEnd) {
    // Own chunk (no remote write, no drain; in place it is already in recvbuff) or an empty part: nothing to signal
    // or wait for.
    if (lineBegin < lineEnd && dst != src) {
      ddaA2APushPeerCopyM<U, false, false>(dst, src, lineBegin + threadIdx.x, lineEnd, blockDim.x);
    }
    barrier.endEpoch(flag + 1 + pieces);
    return;
  }

  const size_t pieceLines =
    (((lineEnd - lineBegin + pieces - 1) / pieces + kPieceAlignLines - 1) / kPieceAlignLines) * kPieceAlignLines;
  auto pieceBegin = [&](int i) {
    const size_t b = lineBegin + static_cast<size_t>(i) * pieceLines;
    return b < lineEnd ? b : lineEnd;
  };
  const size_t srcOff = static_cast<size_t>(peer) * count;
  const T* drainSrc = ipcbuffs[selfRank] + srcOff;
  T* drainDst = recvbuff + srcOff;
  const FabricGpuBarrier::FlagType* ready = barrier.peerFlag(peer, static_cast<int>((nRanks - k) * nParts + part));

  for (int i = 0; i <= pieces; ++i) {
    if (i < pieces) {
      ddaA2APushPeerCopyM<U, false, true>(dst, src, pieceBegin(i) + threadIdx.x, pieceBegin(i + 1), blockDim.x);
    }
    if (i > 0) {
      const FabricGpuBarrier::FlagType want = flag + 1 + i;
      while (static_cast<int32_t>(__scoped_atomic_load_n(ready, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) - want) < 0) {
      }
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      ddaA2APushPeerCopyM<U, true, false>(drainDst, drainSrc, pieceBegin(i - 1) + threadIdx.x, pieceBegin(i),
                                          blockDim.x);
    }
    if (i < pieces) {
      ddaA2AWaitWaveStores();
      __syncthreads();
      if (threadIdx.x == 0) barrier.postPeer(peer, flag + 2 + i);
    }
  }
  barrier.endEpoch(flag + 1 + pieces);
}

} // namespace dda::common
