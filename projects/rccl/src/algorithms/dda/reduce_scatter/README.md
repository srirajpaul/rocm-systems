# DPX hierarchical DDA ReduceScatter

`RCCL_DDA_RS_HIERARCHICAL=1` enables an experimental hierarchy inside DDA's
fabric/VMM (SIMPLE) ReduceScatter path. The default is `0`: initial measurements
on eight gfx1250 DPX ranks showed the flat path was faster. This switch does not
change the existing DDA size thresholds or the LL/LL128 selection priority.
Set it consistently on all communicator ranks before communicator creation.

For a VMM comparison, use the same settings for both runs and change only the
hierarchy switch:

```sh
export RCCL_FORCE_DDA=1
export RCCL_DDA_LL=0
export RCCL_DDA_LL128=0
export RCCL_DDA_THRESHOLD=67108864  # total input bytes, 64 MiB
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

The input is first staged using the existing stream-ordered copy. Pair sums
overwrite only the scratch shards the sibling is not reading, keeping the
existing scratch footprint. System-scope vector loads/stores provide peer
visibility. Both stages assign the same offsets within every shard to the same
block, so the intermediate per-block barrier orders all relevant producers.
The final barrier protects scratch reuse by subsequent collectives. Graph replay
uses the existing device barrier protocol. Floating-point reduction order differs
from flat DDA, so bitwise equality with the flat algorithm is not guaranteed.

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
