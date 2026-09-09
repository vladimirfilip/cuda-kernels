# Fused RMSNorm + residual add (CUDA)

The pre-norm transformer block does this between every sublayer:

```
h   = residual + x                                   # h is the NEW residual stream, written out
out = h * rsqrt(mean(h^2, axis=-1) + eps) * weight   # RMSNorm over the hidden dim
```

PyTorch runs it as two kernels with a full round trip to global memory in
between. Fusing them into one pass removes that round trip. Both `h` and `out`
still have to be written — `h` *is* the residual stream the next block reads — so
the fused op's traffic is `4 * N * H * sizeof(T)`: read `x` and `residual`, write
`h` and `out`.

That makes this purely memory bound, and gives a hard target: HBM bandwidth,
504 GB/s on the reference card. There is no arithmetic worth optimizing here,
only the memory access pattern.

| | |
|---|---|
| Files | [`kernel.cuh`](kernel.cuh) device code · [`main.cu`](main.cu) standalone driver · [`binding.cu`](binding.cu) + [`op.py`](op.py) PyTorch op |
| Correctness | [`test_rmsnorm_fused.py`](test_rmsnorm_fused.py) — 27 cases vs `F.rms_norm`, fp32 + bf16 |
| Benchmark | [`bench.py`](bench.py) → `results/rmsnorm_*.csv` |

```bash
make run   KERNEL=rmsnorm_fused      # ladder + correctness, one command
make test                            # pytest vs the PyTorch reference
make bench KERNEL=rmsnorm_fused      # sweep (N, H, dtype)
make sanitize KERNEL=rmsnorm_fused   # memcheck + racecheck
```

## The optimization ladder

`make run KERNEL=rmsnorm_fused` runs all three rungs back to back over the same
inputs, so the table below reproduces in one command rather than being copied out
of three separate builds. N=4096, H=2048, 200 iterations, 134 MB of traffic —
comfortably past the 48 MB L2, so `%peak` refers to HBM.

| rung | strategy | fp32 ms | fp32 %peak | bf16 ms | bf16 %peak |
|------|----------|---------|-----------|---------|-----------|
| v0 | one thread per row, scalar loop | 1.1347 | 23.5% | 1.4130 | 9.4% |
| v1 | one warp per row, `__shfl_down_sync` reduction | 0.3272 | 81.4% | **0.1420** | **93.7%** |
| v2 | one block per row, shared-mem cross-warp reduction | **0.3139** | **84.8%** | 0.1553 | 85.7% |

**v0 → v1 is the whole story.** A 3.5× fp32 speedup comes from changing *which
thread reads which address*, not from doing less work. In v0, thread `i` owns row
`i`, so the 32 lanes of a warp are reading addresses `H` floats apart — every lane
needs its own memory transaction. In v1 the warp cooperates on one row and walks
it together, so the same 32 loads coalesce into a handful of transactions. The
arithmetic is identical.

The bf16 numbers make the same point more sharply. In v0, bf16 is *slower* than
fp32 (1.41 ms vs 1.13 ms) despite moving half the bytes: when every access is a
separate transaction, per-element format conversion is what dominates, and
halving the payload buys nothing. Once the accesses coalesce in v1, bf16 behaves
as it should and lands at 93.7% of peak.

**v2 is not a clean win, and the table says so.** Assigning a whole block per row
adds a `__syncthreads`-mediated reduction across warps. That pays off in fp32
(84.8% vs 81.4%) where a row is wide enough to keep several warps busy, and
*costs* in bf16 (85.7% vs 93.7%) where the extra synchronisation outweighs the
extra parallelism. At H=2048 the crossover sits between the two dtypes. The
launcher defaults to v2; `launch_rmsnorm_fused(..., RmsNormVariant::kV1Warp)`
selects otherwise.

## Against PyTorch

From `make bench KERNEL=rmsnorm_fused`, at the large end of the sweep
(N=32768, H=4096) where launch overhead is negligible:

| variant | fp32 ms | bf16 ms | what it is |
|---------|---------|---------|------------|
| `unfused` | 6.3253 | 3.2339 | `x + residual` then `F.rms_norm` — the baseline to beat |
| **`fused` (ours)** | **5.0239** | **2.5252** | one kernel |
| `norm` | 2.6782 | 1.3412 | `F.rms_norm` alone on a precomputed `h` — a lower bound, not a competitor |

The fused kernel is **1.26× (fp32) / 1.28× (bf16) faster than the unfused PyTorch
pair** and sustains ~85% of HBM bandwidth. `norm` is listed because it bounds what any fused
implementation could reach: it skips the add and never writes `h`, so it moves
half the bytes. It is not something this kernel can beat while still producing
the residual stream.

Each variant is charged its own traffic in the GB/s column. An earlier version
charged all three the fused op's byte count, which made `norm` appear to exceed
the card's bandwidth — see [`docs/profiling.md`](../../docs/profiling.md).

## Correctness

The sum of squares always accumulates in fp32 regardless of the IO dtype, which
is what keeps bf16 within tolerance. Tests cover fp32 and bf16 across six shapes
including `H = 4097` (not a multiple of any vector width) and a single-token
`N = 1`, at two eps values. The standalone driver additionally checks every rung
against an fp64 CPU reference on each run and exits non-zero on mismatch, so
`make run` doubles as a smoke test.

## Where the remaining 15% is

The kernel does two passes over each row: one to accumulate the sum of squares,
one to scale and write. The second pass re-reads `h`, which it just wrote, and
that read is what the roofline model does not charge for — it is served by L2 at
these sizes but still costs latency. The next rungs would be 128-bit vectorized
loads (`float4` / `__nv_bfloat162`), and caching the row in registers across both
passes to eliminate the re-read entirely. Neither is implemented yet.

Attributing that gap precisely needs Nsight Compute's memory-throughput section,
which is blocked on this box by a driver setting — see
[`docs/profiling.md`](../../docs/profiling.md) for the details and the workaround.
