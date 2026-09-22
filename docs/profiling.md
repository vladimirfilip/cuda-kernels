# Profiling setup

What works on the development box and what doesn't.

## Reference machine

Every measurement under `results/` was taken here unless the CSV says otherwise.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 (GB205, `sm_120`), 48 SMs, 12 GB GDDR7 |
| Peak memory bandwidth | 672 GB/s (28 Gbps x 192-bit) |
| L2 cache | 48 MB |
| fp32 vector peak | ~30.9 TFLOP/s (6144 cores x 2 FLOP/clk x ~2.51 GHz) |
| Driver / CUDA | 580.178.04 / 13.0.88 |
| PyTorch / Triton | 2.14.0+cu130 / 3.8.0 |

Peak bandwidth is the divisor for the `%peak` columns. Pass `--peak-gbps`
(Python benchmarks) or the peak argument of the standalone drivers to retarget it.
The defaults in the source are still the 4070 Ti's, so the results under
`results/` were produced with the 5070 values passed explicitly:

```
python kernels/rmsnorm_fused/bench.py --peak-gbps 672
python kernels/matmul/bench.py --peak-gflops 30900
./bin/vector_add 672 48
./bin/rmsnorm_fused 4096 2048 200 672
./bin/matmul 1024 768 512 200 30900
```

## Nsight Systems

`make nsys KERNEL=<name>` produces a timeline and per-kernel duration summary. It
needs no elevated privileges.

```
make nsys KERNEL=vector_add
```

The `cuda_gpu_kern_sum` table gives mean/median kernel duration directly, enough
to fill in an optimization ladder without counters.

## Nsight Compute

`make ncu KERNEL=<name>` needs a driver permission it does not have by default:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

The block is a driver setting:

```console
$ cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
RmProfilingAdminOnly: 1
```

With `RmProfilingAdminOnly: 1`, reading hardware performance counters requires
root. This box's shell runs as root already, so it works with no `sudo` and no
password. Two things trip it up regardless:

- `cuda-nsight-compute-13-0` is a separate apt package from the base CUDA
  toolkit; installing `cuda-nvcc-13-0` etc. does not pull `ncu` in. It puts the
  binary at `/usr/local/cuda/bin/ncu`.
- that directory is not on `PATH` by default (the `Makefile` calls `nvcc` by
  absolute path via `CUDA_HOME`, so this only bites `ncu`). Either add it to
  `PATH` or `make ncu KERNEL=... NCU=/usr/local/cuda/bin/ncu`.

On a box without root, `sudo -E make ncu KERNEL=...` works if passwordless sudo
is set up; otherwise:

- set `NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/nvidia.conf`
  and reload the driver (needs host access, not available inside a container);
- profile on a machine where the host has already done so.

`ncu` serialises and replays kernels to collect counters, so timings taken under
`ncu` are not comparable to normal runs, and the size of the effect depends on
the section set. On this box, with `NCU_SET=full` (the default), `vector_add`
reports 25.73 ms/launch under `ncu` for the row it captures and 0.0047 ms/launch
without it, ~5500x instrumentation overhead -- much worse than a quick sanity
check would suggest, because `full` replays every pass the section set needs.
`NCU_SET=basic` is far cheaper (9 passes vs. 39 here) at the cost of fewer
metrics. Use `ncu` for counters and stall reasons, CUDA events or `nsys` for
time.

As a concrete example, `make ncu KERNEL=rmsnorm_fused` on the default (v2,
block-per-row) fp32 kernel at N=4096 H=2048 shows why the [fused RMSNorm
write-up](../kernels/rmsnorm_fused/README.md#the-remaining-gap-fp32)'s "L2 serves
the re-read" claim does not hold here: `lts__t_sector_hit_rate.pct` (L2 hit rate)
is 0.39%, not the near-100% a cached re-read would need. Achieved occupancy is
65.4% against a 66.7% theoretical ceiling set by warps per block, and
`dram__bytes.sum.per_second` reads 540 GB/s -- essentially all of this kernel's
traffic is going to DRAM, including the re-read.

## compute-sanitizer

```
make sanitize KERNEL=rmsnorm_fused    # memcheck, then racecheck
```

Run it on anything with shared memory and cross-thread reductions.

## The L2 trap

This card has 48 MB of L2. A benchmark whose working set fits inside it does not
measure DRAM bandwidth and will report a figure several times the card's physical
limit. `make run KERNEL=vector_add` shows it:

```
           n   MB moved  ms/launch      GB/s    %peak  working set
      262144        3.1     0.0048     660.3    98.3%  L2-resident (not a DRAM measurement)
     1048576       12.6     0.0083    1517.6   225.8%  L2-resident (not a DRAM measurement)
     4194304       50.3     0.0438    1150.2   171.2%  L2/DRAM transition
    16777216      201.3     0.3362     598.9    89.1%  DRAM-bound
    67108864      805.3     1.3472     597.7    88.9%  DRAM-bound
```

226% of peak is a cache hit, not a fast kernel. The rmsnorm CSV carries
`footprint_mb` and `l2_resident` columns so each row's regime is explicit.
