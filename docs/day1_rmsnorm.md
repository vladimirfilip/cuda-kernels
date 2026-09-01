# Day 1 — Fused RMSNorm + residual add: optimization log

Fused op (one kernel instead of add-kernel + norm-kernel + an extra global round trip):

```
h   = residual + x                                   # h is the NEW residual stream, written out
out = h * rsqrt(mean(h^2, axis=-1) + eps) * weight   # RMSNorm over the hidden dim
```

Memory bound. Traffic lower bound = `4 * N * H * sizeof(T)` (read x + residual, write h + out).
Roofline target ≈ HBM bandwidth, ~1790 GB/s on the RTX 5090 (sm_120).

## How to measure

```
make run      KERNEL=rmsnorm_fused GPU=0     # CPU-ref correctness + ms + GB/s, fp32 & bf16
make sanitize KERNEL=rmsnorm_fused GPU=0     # memcheck + racecheck
make torch-test                              # pytest vs F.rms_norm reference
make bench                                   # sweep (N,H,dtype); CSV in bench/results/
make ncu      KERNEL=rmsnorm_fused NCU_SET=full   # per-kernel HW profile  (see caveat)
```

### ncu caveat on this box

`ncu` is installed but `RmProfilingAdminOnly: 1` on the host and there is no passwordless
sudo, so hardware performance counters are blocked (`ERR_NVGPUCTRPERM`) — every `ncu`
section/metric run fails at connect. `nsys` is not installed.

Until that is resolved, the profiling rungs below have to come from one of:
- a box where the host set `NVreg_RestrictProfilingToAdminUsers=0` (or run ncu as root) —
  e.g. the university GPU used for the headline run;
- asking the Vast host to flip the module param;
- falling back to CUDA-events timing + achieved-bandwidth math (what `make run` / `make bench`
  already give) plus reasoning from the access pattern, for the intermediate rungs.

The `ncu evidence` column below stays as the deliverable target; fill it on a profiling-capable box.

## Optimization ladder

| ver | change | dtype | ms/launch | GB/s | % peak | ncu evidence | notes |
|-----|--------|-------|-----------|------|--------|--------------|-------|
| v0  | one thread per row, scalar loop, no smem, no vectorization | fp32 | 1.08 | 124 | 7.0% | _(blocked — see caveat)_ | baseline. Un-coalesced: adjacent threads stride H floats apart. ~50x slower than torch unfused. |
| v0  | " | bf16 | 1.63 | 41 | 2.3% | _(blocked)_ | slower than fp32 here — per-element bf16↔f32 conversion dominates when the access pattern is already terrible. |
| v1  | warp per row, `__shfl_down_sync` tree reduction, grid-stride over rows | | | | | | expect big jump: loads within a row become coalesced across the 32 lanes. |
| v2  | block per row (multiple warps), shared-mem partial sums across warps | | | | | | helps large H / raises occupancy; tune warps-per-row. |
| v3  | vectorized 128-bit loads (`float4` / `__nv_bfloat162`×4), `__ldg`, single reread-free pass (cache h in regs/smem) | | | | | | should approach HBM roofline. |
| v4  | tune rows/block, block size, `__launch_bounds__`; check L2 vs occupancy tradeoff | | | | | | last few %. |

## Reference numbers (context)

From `make bench` on the RTX 5090, N=2048 H=2048:

| variant | fp32 ms | bf16 ms |
|---------|---------|---------|
| torch unfused (`x+residual` then `F.rms_norm`) | 0.025 | 0.016 |
| torch `F.rms_norm` alone | 0.016 | 0.012 |
| **v0 fused (ours)** | **1.10** | **1.62** |

Beating the torch unfused number (fused, single pass, one h write) is the Day 1 goal.

_Note: the GB/s column in `bench_rmsnorm.py` uses the fused op's traffic model for every
row, so it is only meaningful for the `fused` variant; for the torch variants compare `ms`._
