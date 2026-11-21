# DPX hierarchical DDA ReduceScatter

`RCCL_DDA_RS_HIERARCHICAL=1` enables an experimental hierarchy inside DDA's
fabric/VMM (SIMPLE) ReduceScatter path. The default is `0`. Eight- and sixteen-rank
gfx1250-strict DPX communicators use specialized kernels with fused staging.
Small messages can still favor flat DDA. This switch does not
change the existing DDA size thresholds or the LL/LL128 selection priority.
Set it consistently on all communicator ranks before communicator creation.

For a VMM comparison, use the same settings for both runs and change only the
hierarchy switch:

```sh
export RCCL_FORCE_DDA=1
export RCCL_DDA_LL=0
export RCCL_DDA_LL128=0
export RCCL_DDA_THRESHOLD=268435456  # total input bytes, 256 MiB
export RCCL_DDA_FABRIC_BUFFER_SIZE=268435456
export RCCL_DDA_RS_HIERARCHICAL=1  # 0 selects the flat baseline
```

The communicator must report DPX on every rank and contain at least two complete
physical GPU pairs. Pairing uses host identity and PCI device identity; PCI
functions 0 and 1 identify the two tiles. Rank order may be arbitrary. Incomplete
pairs, one-tile-per-GPU subcommunicators, and other partition modes fall back to
flat DDA. Supported reductions and types are those of the existing fabric VMM
path: sum on float32, float16, bfloat16, FP8 E4M3 and FP8 E5M2. Each output shard
must satisfy the existing 16-byte size alignment requirement.

For the usual eight-rank order, stage 1 reduces the scale-in pairs `(0,1)`,
`(2,3)`, `(4,5)` and `(6,7)`. Each even tile computes pair sums for even destination
shards; each odd tile computes pair sums for odd destination shards. Stage 2
reduces those partial sums within `(0,2,4,6)` or `(1,3,5,7)` and writes each rank's
output shard. Reordered communicators use the actual destination rank offsets.

For eight or sixteen ranks, staging is fused into the hierarchical kernel: each
tile copies only the four or eight input shards needed by its sibling. It reads its own contributions
directly from the send buffer and overwrites only its own rail's scratch shards
with pair sums. This removes the separate full-input copy and retains the existing
scratch footprint. Other eligible rank counts retain the stream-ordered copy.
System-scope vector loads/stores provide peer visibility, and every wave drains
its scratch stores before the existing release/acquire barrier publishes them.
All stages assign the same offsets within every shard to the same block, so the
intermediate per-block barrier orders all relevant producers. This ownership also
preserves in-place operation: a block consumes its input before writing its output.
The final barrier protects scratch reuse by subsequent collectives. Graph replay
uses the existing device barrier protocol. Floating-point reduction order differs
from flat DDA, so bitwise equality with the flat algorithm is not guaranteed.

The eight- and sixteen-rank kernels specialize the four- and eight-peer scaleup
loops, issuing all peer loads before their dependent arithmetic. Both default to
1024 threads per block, except sixteen-rank messages with total input at most
8 MiB per rank use 512 threads. Both balance short grid-stride loops to avoid idle blocks.
Large messages retain the full configured block cap. Small shards reduce the
thread count automatically. Optional tuning controls, identical on every rank:

- `RCCL_DDA_RS_HIER_THREADS=0` (default): automatic width; a positive override is
  clamped to 64–1024 and rounded down to a multiple of 64. Generic hierarchies
  default to 512 threads.
- `RCCL_DDA_RS_HIER_BLOCKS=0` (default): use the existing fabric block cap; a
  positive override caps the hierarchical grid without enlarging barrier storage.

These settings do not alter the flat kernel or the global DDA thresholds. The
eight-rank tuning was measured on node 02; the sixteen-rank extension was measured
on nodes 13 and 14, with eight ranks per node. It retains the existing scale-in
reduction followed by scaleup reads, scratch layout, and release/acquire barriers.
Build RCCL with
`bash install.sh --amdgpu_targets gfx1250-strict`. On this machine, `-l` detects
`gfx1250` and produces a library that fails to launch with "device kernel image
is invalid". Build rccl-tests for `gfx1250-strict` as well so its verification
kernels can run.

Initialization logs `DDA ReduceScatter hierarchy: rank=... sibling=...`.
With `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL`, hierarchical launches log
`DDA fabric ReduceScatter: launching kernel: hierarchical scale-in=2 scaleup=...`.

The CPU topology regression test runs without HIP or gtest:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -Isrc test/dda_rs_hierarchy_test.cpp -o /tmp/dda-rs-topology-test
/tmp/dda-rs-topology-test
```

It is also registered with CTest as `rccl-UnitTestsDdaRsHierarchy` when
`BUILD_TESTS=ON`. The MPI regression case is
`DdaFabricSimple.HierarchicalReduceScatterDpx`; launch the MPI test binary with
the VMM environment settings above so a faster LL tier cannot mask the test.
