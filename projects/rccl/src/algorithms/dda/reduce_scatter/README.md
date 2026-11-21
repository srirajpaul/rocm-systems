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
tile copies only the four or eight input shards needed by its sibling. It reads
its own contributions directly from the send buffer. Each pair sum is written
to the destination tile's scratch slot indexed by the producer rank, so the final
reduction reads only local scratch. Incoming sums occupy the destination's own
rail slots, disjoint from the staged input its sibling still needs. This retains
the existing scratch footprint and removes the separate full-input copy.
Other eligible rank counts retain the stream-ordered copy and generic remote-read
final reduction; their scratch still holds input during the pair reduction.
System-scope vector loads/stores provide peer visibility, and every wave drains
its scratch stores before the existing release/acquire barrier publishes them.
All stages assign the same offsets within every shard to the same block, so the
intermediate per-block barrier orders all relevant producers. This ownership also
preserves in-place operation: a block consumes its input before writing its output.
The final barrier protects scratch reuse by subsequent collectives. Graph replay
uses the existing device barrier protocol. Floating-point reduction order differs
from flat DDA, so bitwise equality with the flat algorithm is not guaranteed.

The eight- and sixteen-rank kernels specialize the four- and eight-peer scaleup
loops, issuing all partial-sum loads before their dependent arithmetic. Both default to
1024 threads per block, except sixteen-rank messages with total input at most
8 MiB per rank use 512 threads. Both balance short grid-stride loops to avoid idle blocks.
Large messages retain the full configured block cap. Small shards reduce the
thread count automatically. Optional tuning controls, identical on every rank:

- `RCCL_DDA_RS_HIER_THREADS=0` (default): automatic width; a positive override is
  clamped to 64–1024 and rounded down to a multiple of 64. Generic hierarchies
  default to 512 threads.
- `RCCL_DDA_RS_HIER_BLOCKS=0` (default): use the existing fabric block cap; a
  positive override caps the hierarchical grid without enlarging barrier storage.
- `RCCL_DDA_RS_HIER_SCHEDULE=1` (default): begin sending at the next peer after
  this rank in its scaleup rail and wrap around, sending to self last. This
  distributes simultaneous writes across destinations. `0` restores the
  original destination order; `2` additionally interleaves wave-sized work
  across destinations within each block's existing offset ownership. Values
  are clamped to 0–2. These modes apply to both the sequential and pipelined
  eight-/sixteen-rank push kernels; other rank counts retain their existing path.
  Rotation alone improved large-message timing in the 8-/16-rank experiments.
  Wave interleaving remains an experimental option and was slower than rotation
  alone. Staging, reduction order, launch geometry, memory operations, and
  synchronization are unchanged by the schedule control.
- `RCCL_DDA_RS_HIER_PIPELINE=0` (default): set to `1` to enable the experimental
  double-buffered remote-write kernel on eight or sixteen ranks. It defaults to
  1024 threads; the thread override above also applies to this kernel.
- `RCCL_DDA_RS_HIER_CHUNK_BYTES=1048576`: maximum bytes per producer chunk in
  pipeline mode, clamped to 1 KiB–1 MiB. The actual chunk is reduced to fit two
  banks per producer inside the existing scratch slots and rounded to 64 vectors.

Pipeline mode uses equal numbers of producer and consumer blocks, with their
combined grid bounded by the existing fabric block cap. Producer block `b`
stages and processes offsets `b * chunk + round * producers * chunk` in every
shard. It pushes pair sums into bank `round % 2` of each destination's local
scratch; consumer block `b + producers` reduces those slots into its output.
The opposite rail's staged input and own rail's partial-sum banks stay disjoint.
Very small shards and a block cap of one use the sequential kernel.

The pipeline reuses AllToAll's epoch flag allocation and `epochBegin`/`epochEnd`
protocol. Batched peer post/wait helpers signal readiness from producer rows
and consumption from consumer rows. A producer waits for round `r-2` credits
before overwriting that bank. All blocks advance their epoch by the same number
of rounds, including empty tail chunks; this supports changing grid sizes,
graph replay, and alternating with AllToAll on the same communicator.
The intermediate all-rank partial-sum barrier is replaced by these rail-local
ready/credit exchanges. The staging and closing barriers remain. No extra
scratch or flag allocation is required. This implementation is opt-in because
the measured scheduling and signaling overhead outweighs the overlap benefit
on the tested eight- and sixteen-rank configurations.

These settings do not alter the flat kernel or the global DDA thresholds. The
eight-rank tuning was measured on node 02; the sixteen-rank extension was measured
on nodes 13 and 14, with eight ranks per node. The remote-write variant retains
the same staging, block/thread selection, reduction order, and release/acquire
barriers when pipeline mode is disabled. Pair reduction pushes sums to scaleup
destinations; the final reduction reads their local slots.
Build RCCL with
`bash install.sh --amdgpu_targets gfx1250-strict`. On this machine, `-l` detects
`gfx1250` and produces a library that fails to launch with "device kernel image
is invalid". Build rccl-tests for `gfx1250-strict` as well so its verification
kernels can run.

Use a clean build when comparing revisions. The device-linker DDA fabric
objects now request header dependency tracking on CMake 3.20 or newer; before
this fix an incremental header-only edit could silently retain old GPU code.
`install.sh` cleans by default unless `--no_clean` is supplied. For an existing
separate build, use `cmake --build <build-dir> --clean-first -j 16` once to
establish an unambiguous baseline.

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
