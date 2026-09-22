# Fused RMSNorm + residual add (CUDA)

The pre-norm transformer block does this between every sublayer:

```
h   = residual + x                                   # h is the new residual stream, written out
out = h * rsqrt(mean(h^2, axis=-1) + eps) * weight   # RMSNorm over the hidden dim
```

PyTorch runs it as two kernels with a full round trip to global memory in
between. Fusing them into one pass removes that round trip. Both `h` and `out`
are written (`h` is the residual stream the next block reads), so the fused op's
traffic is `4 * N * H * sizeof(T)`: read `x` and `residual`, write `h` and `out`.

That makes this memory bound, with a hard target of DRAM bandwidth, 672 GB/s on
the reference card. There is no arithmetic to optimize, only the memory access
pattern.

| | |
|---|---|
| Files | [`kernel.cuh`](kernel.cuh) device code, [`main.cu`](main.cu) standalone driver, [`binding.cu`](binding.cu) + [`op.py`](op.py) PyTorch op |
| Correctness | [`test_rmsnorm_fused.py`](test_rmsnorm_fused.py), 27 cases vs `F.rms_norm`, fp32 + bf16 |
| Benchmark | [`bench.py`](bench.py) -> `results/rmsnorm_*.csv` |

```bash
make run   KERNEL=rmsnorm_fused      # ladder + correctness, one command
make test                            # pytest vs the PyTorch reference
make bench KERNEL=rmsnorm_fused      # sweep (N, H, dtype)
make sanitize KERNEL=rmsnorm_fused   # memcheck + racecheck
```

## The optimization ladder

`make run KERNEL=rmsnorm_fused` runs all three rungs back to back over the same
inputs. N=4096, H=2048, 200 iterations. Traffic is 134 MB in fp32 and 67 MB in
bf16, both past the 48 MB L2, so `%peak` refers to DRAM.

| rung | strategy | fp32 ms | fp32 %peak | bf16 ms | bf16 %peak |
|------|----------|---------|-----------|---------|-----------|
| v0 | one thread per row, scalar loop | 1.6101 | 12.4% | 2.4480 | 4.1% |
| v1 | one warp per row, `__shfl_down_sync` reduction | 0.2451 | 81.5% | 0.1033 | 96.7% |
| v2 | one block per row, shared-mem cross-warp reduction | 0.2417 | 82.6% | 0.1905 | 52.4% |

**v0 to v1 is the main result.** A 6.6x fp32 speedup comes from changing which
thread reads which address, not from doing less work. In v0, thread `i` owns row
`i`, so the 32 lanes of a warp read addresses `H` floats apart and every lane
needs its own memory transaction. In v1 the warp cooperates on one row and walks
it together, so the same 32 loads coalesce into a handful of transactions. The
arithmetic is identical.

The bf16 numbers show the same effect. In v0, bf16 is slower than fp32 (2.45 ms
vs 1.61 ms) despite moving half the bytes: halving the payload buys nothing when
every access is already a separate transaction. Once the accesses coalesce in v1,
bf16 lands at 96.7% of peak.

**v2 is not a clean win.** Assigning a whole block per row adds a
`__syncthreads`-mediated reduction across warps. In fp32 it is a wash (82.6% vs
81.5%). In bf16 it costs nearly half the bandwidth (52.4% vs 96.7%): a bf16 row at
H=2048 is 4 KB, too little work to amortise the extra synchronisation. The launcher
defaults to v2, so the PyTorch op inherits that bf16 penalty;
`launch_rmsnorm_fused(..., RmsNormVariant::kV1Warp)` selects the faster rung.

## Against PyTorch

From `make bench KERNEL=rmsnorm_fused`, at the large end of the sweep
(N=32768, H=4096) where launch overhead is negligible:

| variant | fp32 ms | bf16 ms | what it is |
|---------|---------|---------|------------|
| `unfused` | 4.6265 | 2.3105 | `x + residual` then `F.rms_norm`, the baseline |
| `fused` | 3.7829 | 2.5255 | one kernel |
| `norm` | 1.9084 | 0.9678 | `F.rms_norm` alone on a precomputed `h`, a lower bound |

The fused kernel is 1.22x faster than the unfused PyTorch pair in fp32 and sustains
84.5% of DRAM bandwidth. In bf16 it is 0.91x, i.e. slower (63.3% of peak), across
the whole sweep (0.55-0.91x): the default v2 launch shape is the cause, see the
ladder above. `norm` sets the floor for any kernel
that also writes the residual stream: it skips the add and never writes `h`, so
it moves half the bytes.

Each variant is charged its own traffic in the GB/s column.

## Correctness

The sum of squares always accumulates in fp32 regardless of the IO dtype, which
keeps bf16 within tolerance. Tests cover fp32 and bf16 across six shapes
including `H = 4097` (not a multiple of any vector width) and a single-token
`N = 1`, at two eps values. The standalone driver checks every rung against an
fp64 CPU reference on each run and exits non-zero on mismatch.

## The remaining gap (fp32)

The kernel does two passes over each row: one to accumulate the sum of squares,
one to scale and write. The second pass re-reads `h`, which it just wrote. The
roofline model does not charge for that read on the assumption that it is served
by L2 rather than DRAM.

Nsight Compute on the default v2 kernel (N=4096, H=2048, fp32) says that
assumption does not hold here: `lts__t_sector_hit_rate.pct` (L2 hit rate) is
0.39%, so the re-read is landing in DRAM almost every time, not L2. Achieved
occupancy is 65.4% against a 66.7% ceiling set by warps per block -- occupancy
is not the bottleneck. `dram__bytes.sum.per_second` reads 540 GB/s, consistent
with the re-read actually costing full DRAM bandwidth rather than being free.
The next rungs would be 128-bit vectorized loads (`float4` / `__nv_bfloat162`)
and caching the row in registers across both passes to eliminate the re-read
outright rather than hope L2 absorbs it. Neither is implemented yet.

Commands: `make ncu KERNEL=rmsnorm_fused` runs the full section set; see
[`docs/profiling.md`](../../docs/profiling.md) for the two setup steps this box
needed (`ncu` is a separate apt package from the rest of the toolkit, and isn't
on `PATH` by default) and for how the overhead this adds compares to a normal
run.
