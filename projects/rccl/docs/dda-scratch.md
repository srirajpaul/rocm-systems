# Fabric DDA scratch capacity

Fabric DDA exports one VMM allocation per GPU/communicator with four isolated
regions. CE staging, Simple, LL128 and LL never share payload addresses. LL and
LL128 have separate per-block epoch arrays; all collectives within one protocol
share its two-bank layout and epoch domain.

| Environment variable | Default bytes | Default MiB |
| --- | ---: | ---: |
| `RCCL_CE_SCRATCH_SIZE` | 268435456 | 256 |
| `RCCL_DDA_SIMPLE_SCRATCH_SIZE` | 268435456 | 256 |
| `RCCL_DDA_LL128_SCRATCH_SIZE` | 67108864 | 64 |
| `RCCL_DDA_LL_SCRATCH_SIZE` | 8388608 | 8 |

These are total region capacities, including both LL/LL128 banks and wire
flags. Set them before communicator initialization. They must agree across all
ranks in the fabric communicator. Zero disables the candidates using that
region. Negative, malformed, overflowing, or mismatched capacities fail
initialization collectively. Hardware, fabric-clique, VMM and DDA enable gates
still apply.

Capacities do not grow with rank count, message thresholds, tuning tables or
graph capture. The allocator may pad the owning VMM allocation; kernels cannot
use that padding to exceed a region's capacity. An operation whose existing
layout cannot fit follows the existing selection/fallback order. Thresholds,
launch geometry and backend priorities are unchanged.

LL128 reserves two Broadcast token banks within its budget. Each holds one
16-byte LL-format token per compile-time supported rank, aligned to 128 bytes
(currently 2304 bytes per token bank at the 144-rank limit). Every LL128 payload
view begins after that prefix. This prevents retained LL128 payload from being
accepted as a Broadcast start token. These tokens use the LL128 epoch domain.

`RCCL_DDA_FABRIC_BUFFER_SIZE` is deprecated and ignored, with a warning listing
the replacements. It no longer overrides the allocation or disables all four
regions. To disable the fabric DDA facility, use `RCCL_DDA_ENABLE=0`; to disable
individual regions, set their new size variables to zero.

At the defaults, configured regions total 584 MiB per GPU/communicator, plus
allocation padding and separate synchronization metadata. At eight ranks the
8 MiB LL region fits AlltoAll payloads through 256 KiB per peer, subject to the
existing dispatch limits. LL ReduceScatter retains its fixed slots and requires
32 MiB at eight ranks, so it falls back with the default LL capacity. It needs
16 MiB at four ranks. A larger fixed LL value can be used for coverage; no slots
or tuning parameters are adjusted to recover coverage automatically.

Initialization logs the capacities and offsets under `NCCL_DEBUG=INFO` with the
`INIT` subsystem. Capacity diagnostics use `TUNING` and identify the collective,
protocol, required amount, available amount, and layout unit. Keep `COLL`
enabled to record the selected backend. `NCCL_DEBUG=VERSION` does not enable
these diagnostics.

The IPC data path retains its existing scratch ownership. CE registration,
`ceARTmpBuf`, CE barriers and Simple barriers are unchanged. Simple double
buffering and initial-barrier removal remain future work.
