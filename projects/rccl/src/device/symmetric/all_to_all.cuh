#include "sym_kernels.h"
#include "symmetric/kernel.h"
#include "symmetric/primitives.h"

template<typename T>
static __device__ void scatter(
    ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks,
    ncclLsaBarrierSession<ncclCoopCta>& bar,
    ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {

  bool inPlace = (input == output);
  static_assert(sizeof(T) == 1);

  int const& rank   = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  const size_t nEltsPack = nElts / sizeof(uint4);

  for (int i = t; i < nEltsPack; i += tn) {
    #pragma unroll
    for (int r = 0; r < nRanks; r++) {
      int peer = (rank + r) % nRanks;
      if (peer == rank && inPlace) continue;

      ncclSymPtr<char> srcSlice = input  + (size_t)peer * nElts;
      ncclSymPtr<char> dstSlice = output;

      char*       dst = dstSlice.lsaPtr(peer);
      char const* src = srcSlice.localPtr();

      (reinterpret_cast<uint4*>(dst)) [i] =
      (reinterpret_cast<uint4*>(const_cast<char*>(src))) [i];
    }
  }
}

__device__ __forceinline__ void ncclSymkRun_AlltoAll_ST(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x
  };
  int const& rank   = handler.comm.rank;

  bar.sync(ncclCoopCta(), NCCL_MEM_ORDER_RELAXED);

  handler.forEachWork<char>(
    [&] __device__ (int block, int nBlocks, size_t nElts, size_t nAllElts,
      ncclSymPtr<char> input, ncclSymPtr<char> output) {
        int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                          block, nBlocks,
                          threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int tn = nBlocks*blockDim.x;
        scatter(handler, tn, t, nBlocks, bar, input, output + rank*nElts, nElts);
      }
    );

  bar.sync(ncclCoopCta(), NCCL_MEM_ORDER_RELEASE);
}
