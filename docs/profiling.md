# Profiling setup

What works on the development box, what doesn't, and why — so the numbers in the
kernel write-ups can be traced back to a tool that actually ran.

## Reference machine

Every measurement committed under `results/` was taken here unless the CSV says
otherwise:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti (AD104, `sm_89`), 60 SMs, 12 GB GDDR6X |
| Peak memory bandwidth | **504 GB/s** (21 Gbps × 192-bit) |
| L2 cache | 48 MB |
| fp32 vector peak | ~40.1 TFLOP/s (7680 cores × 2 FLOP/clk × ~2.61 GHz) |
| Driver / CUDA | 580.178.04 / 13.0.88 |
| PyTorch / Triton | 2.14.0+cu130 / 3.8.0 |

Peak bandwidth is the number the `%peak` columns divide by. Pass `--peak-gbps`
(Python benchmarks) or `argv[4]` (the standalone drivers) to retarget it.

## Nsight Systems — works

`make nsys KERNEL=<name>` produces a timeline and per-kernel duration summary. It
needs no elevated privileges, so it is the tool to reach for first:

```
make nsys KERNEL=vector_add
```

The `cuda_gpu_kern_sum` table it prints gives mean/median kernel duration
directly, which is enough to fill in an optimization ladder without counters.

## Nsight Compute — blocked on this box

`make ncu KERNEL=<name>` fails here:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

The cause is a driver-level setting, not a missing tool — `ncu` is installed at
`/usr/local/cuda/bin/ncu` and connects to the process fine:

```console
$ cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
RmProfilingAdminOnly: 1
```

With `RmProfilingAdminOnly: 1`, reading hardware performance counters requires
root. There is no passwordless sudo on this box, so `make ncu` cannot run
unattended. Any of these fixes it:

- run the target as root (`sudo -E make ncu KERNEL=...`), if you have the password;
- set `NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/nvidia.conf`
  and reload the driver (needs host access, not available inside a container);
- profile on a machine where the host has already done so.

**One thing to be careful about:** `ncu` serialises and replays kernels to
collect counters, so timings taken *under* `ncu` are not comparable to normal
runs. On this box `vector_add` reports 0.51 ms/launch under `ncu` and 0.0061
ms/launch without it — an ~84× difference that is pure instrumentation overhead.
Use `ncu` for counters and stall reasons; use CUDA events or `nsys` for time.

## compute-sanitizer — works

```
make sanitize KERNEL=rmsnorm_fused    # memcheck, then racecheck
```

Worth running on anything with shared memory and cross-thread reductions.

## The L2 trap

This card has 48 MB of L2. A benchmark whose working set fits inside it does not
measure HBM bandwidth, and will happily report a figure several times the card's
physical limit. `make run KERNEL=vector_add` demonstrates it:

```
           n   MB moved  ms/launch      GB/s    %peak  working set
      262144        3.1     0.0060     521.1   103.4%  L2-resident (not an HBM measurement)
     1048576       12.6     0.0057    2217.5   440.0%  L2-resident (not an HBM measurement)
     4194304       50.3     0.0233    2164.3   429.4%  L2/HBM transition
    16777216      201.3     0.4429     454.5    90.2%  HBM-bound
    67108864      805.3     1.7590     457.8    90.8%  HBM-bound
```

440% of peak is not a fast kernel, it is a cache hit. The benchmark sweeps in
`kernels/*/bench.py` start above the L2 footprint for this reason, and the
rmsnorm CSV carries `footprint_mb` and `l2_resident` columns so a reader can
check rather than trust.
