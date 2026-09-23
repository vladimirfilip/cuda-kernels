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
| Correctness | [`test_rmsnorm_fused.py`](test_rmsnorm_fused.py), 33 cases vs `F.rms_norm`, fp32 + bf16 |
| Benchmark | [`bench.py`](bench.py) -> `results/rmsnorm_*.csv` |

```bash
make run   KERNEL=rmsnorm_fused      # ladder + correctness, one command
make test                            # pytest vs the PyTorch reference
make bench KERNEL=rmsnorm_fused      # sweep (N, H, dtype)
make sanitize KERNEL=rmsnorm_fused   # memcheck + racecheck
```

## The optimization ladder

`make run KERNEL=rmsnorm_fused` runs all four rungs back to back over the same
inputs. N=4096, H=2048, 200 iterations. Traffic is 134 MB in fp32 and 67 MB in
bf16, both past the 48 MB L2, so `%peak` refers to DRAM.

| rung | strategy | fp32 ms | fp32 %peak | bf16 ms | bf16 %peak |
|------|----------|---------|-----------|---------|-----------|
| v0 | one thread per row, scalar loop | 1.6100 | 12.4% | 2.4472 | 4.1% |
| v1 | one warp per row, `__shfl_down_sync` reduction | 0.2446 | 81.7% | 0.1033 | 96.7% |
| v2 | one block per row, shared-mem cross-warp reduction | 0.2401 | 83.2% | 0.1906 | 52.4% |
| v3 | v2 + 128-bit vectorized loads + register-cached `h` | 0.2300 | 86.8% | 0.1035 | 96.5% |

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
`__syncthreads`-mediated reduction across warps. In fp32 it is a wash (83.2% vs
81.7%). In bf16 it costs nearly half the bandwidth (52.4% vs 96.7%): a bf16 row at
H=2048 is 4 KB, too little work to amortise the extra synchronisation.

**v3 fixes both of v2's problems at once, without giving up v2's cross-warp
reduction.** It keeps v2's one-block-per-row shape but reads and writes 128 bits
per instruction instead of `sizeof(T)`, and it caches each thread's `h` values
in a per-thread array (`cache` in `rmsnorm_fused_v3`) between passes instead of
re-reading them from global memory (the actual fix for ["the remaining
gap"](#the-remaining-gap-closed-in-v3) below). `-Xptxas -v` says that array is
true registers in fp32 (0 bytes local memory) but spills to local memory in
bf16 (128 bytes, exactly `cache`'s size) -- bf16's extra register pressure from
packing/unpacking `__nv_bfloat162` pairs is enough that the compiler gives up
promoting it. Local memory is still private per-thread scratch backed by L1,
not the shared, DRAM-backed `h` tensor v2 re-reads, so the "no re-read" result
holds in both dtypes; only "register array" is fp32-specific.

In fp32 this is a clean win over every earlier rung (86.8%, the best in the
ladder). In bf16 it recovers essentially all of v1's coalescing advantage
(96.5% vs v1's 96.7%) while keeping v2's cross-warp reduction, and it is *more*
accurate than v2 in bf16 too (`out` tolerance 0.17 vs 0.33): v2's second pass
re-reads `h` after it has been rounded to bf16 and back, while v3's cache still
holds the fp32 value pass 1 computed, so computing `out` no longer costs a
second rounding step. That does mean `out` and the bf16 `h` v3 writes are no
longer derived from the same rounded value -- `out` is now slightly closer to
an unfused fp64 reference than `F.rms_norm(h, ...)` on the bf16 `h` actually
returned would be. Both stay well inside the existing bf16 tolerance. The
launcher defaults to v3.

v3's 128-bit loads need each row's start address 16-byte aligned, which needs
two things: `H` a multiple of the vector width (4 for fp32, 8 for bf16), and
the input/output buffers themselves 16-byte aligned -- the second does not
follow from `.contiguous()`, since a contiguous PyTorch tensor can still start
at a non-16-aligned storage offset. See the comment above `rmsnorm_fused_v3` in
[`kernel.cuh`](kernel.cuh) for the full argument. `launch_rmsnorm_fused()`
checks both at launch time and falls back to v2 otherwise; the suite's
`H = 4097` case and its misaligned-offset test exercise the two fallback
paths separately.

## Against PyTorch

From `make bench KERNEL=rmsnorm_fused`, at the large end of the sweep
(N=32768, H=4096) where launch overhead is negligible:

| variant | fp32 ms | bf16 ms | what it is |
|---------|---------|---------|------------|
| `unfused` | 4.6080 | 2.3142 | `x + residual` then `F.rms_norm`, the baseline |
| `fused` | 3.7427 | 1.8980 | one kernel (v3) |
| `norm` | 1.9152 | 0.9685 | `F.rms_norm` alone on a precomputed `h`, a lower bound |

The fused kernel is 1.23x faster than the unfused PyTorch pair in fp32 (85.4% of
peak) and 1.22x faster in bf16 (84.2% of peak) at this shape. Across the whole
sweep it is 0.98-1.24x in fp32 and 0.77-1.24x in bf16 -- v3 turns bf16 from a
consistent loss against `unfused` (0.55-0.91x under v2) into a near-even-to-clear
win everywhere but the smallest one or two shapes in the sweep (N<=8192,
H<=3072), where per-launch overhead still dominates enough that the extra
vectorized-load setup costs more than it saves. `norm` sets the floor for any
kernel that also writes the residual stream: it skips the add and never writes
`h`, so it moves half the bytes.

Each variant is charged its own traffic in the GB/s column.

## Correctness

The sum of squares always accumulates in fp32 regardless of the IO dtype, which
keeps bf16 within tolerance. Tests cover fp32 and bf16 across seven shapes
including `H = 4097` (a multiple of no vector width), `H = 2052` (a multiple of
fp32's but not bf16's, so the two dtypes take different v3-vs-v2 dispatch
paths at the same shape) and a single-token `N = 1`, at two eps values. A
separate pair of tests builds a contiguous tensor at a non-16-byte-aligned
storage offset per dtype (`.contiguous()` doesn't fix that up), which is the
other way v3's dispatch check can and must fall back to v2. The standalone
driver checks every rung against an fp64 CPU reference on each run and exits
non-zero on mismatch.

## The remaining gap, closed in v3

Every rung before v3 does two passes over each row: one to accumulate the sum
of squares, one to scale and write. That second pass re-reads `h`, which it
just wrote. The roofline model above never charged for that read, on the
assumption that it is served by L2 rather than DRAM.

Nsight Compute on v2 (N=4096, H=2048, fp32) said that assumption does not hold:
`lts__t_sector_hit_rate.pct` (L2 hit rate) was 0.39%, so the re-read was landing
in DRAM almost every time, not L2. Achieved occupancy was 65.4% against a 66.7%
ceiling set by warps per block -- occupancy was not the bottleneck.

v3 removes the re-read outright rather than hoping L2 absorbs it: pass 1 keeps
each thread's `h` values in a per-thread `cache` array (registers in fp32,
L1-backed local memory in bf16 -- see the ladder section above), and pass 2
reads that instead of issuing any load for `h` at all -- there is no second
`hr[i]` access left for the profiler to measure a hit rate on. Under
`ncu --set full`'s replay (same one-launch capture as above, fp32, at the same
shape), the kernel's own `gpu__time_duration.sum` drops from 215.9us (v2) to
175.3us (v3), an 18.8% reduction consistent with one whole pass's worth of
global traffic disappearing. This is also visible without `ncu`: v3 is the
fastest fp32 rung in the ladder above (86.8% of peak) and, in bf16, the fastest
rung that still gets v2's cross-warp reduction (96.5%, against v1's 96.7% with
no cross-warp reduction at all).

`make ncu KERNEL=rmsnorm_fused` runs the full section set; see
[`docs/profiling.md`](../../docs/profiling.md) for what that costs against a
normal run.
