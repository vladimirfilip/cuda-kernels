# Profiling setup

What works on the development box and what doesn't.

## Reference machine

Every measurement under `results/` was taken here unless the CSV says otherwise.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti (AD104, `sm_89`), 60 SMs, 12 GB GDDR6X |
| Peak memory bandwidth | 504 GB/s (21 Gbps x 192-bit) |
| L2 cache | 48 MB |
| fp32 vector peak | ~40.1 TFLOP/s (7680 cores x 2 FLOP/clk x ~2.61 GHz) |
| Driver / CUDA | 580.178.04 / 13.0.88 |
| PyTorch / Triton | 2.14.0+cu130 / 3.8.0 |

Peak bandwidth is the divisor for the `%peak` columns. Pass `--peak-gbps`
(Python benchmarks) or `argv[4]` (standalone drivers) to retarget it.

## Nsight Systems

`make nsys KERNEL=<name>` produces a timeline and per-kernel duration summary. It
needs no elevated privileges.

```
make nsys KERNEL=vector_add
```

The `cuda_gpu_kern_sum` table gives mean/median kernel duration directly, enough
to fill in an optimization ladder without counters.

## Nsight Compute: blocked on this box

`make ncu KERNEL=<name>` fails here:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

`ncu` is installed at `/usr/local/cuda/bin/ncu` and connects to the process. The
block is a driver setting:

```console
$ cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
RmProfilingAdminOnly: 1
```

With `RmProfilingAdminOnly: 1`, reading hardware performance counters requires
root. There is no passwordless sudo on this box. Any of these fixes it:

- run the target as root (`sudo -E make ncu KERNEL=...`), with the password;
- set `NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/nvidia.conf`
  and reload the driver (needs host access, not available inside a container);
- profile on a machine where the host has already done so.

`ncu` serialises and replays kernels to collect counters, so timings taken under
`ncu` are not comparable to normal runs. On this box `vector_add` reports 0.51
ms/launch under `ncu` and 0.0060 ms/launch without it, ~84x instrumentation
overhead. Use `ncu` for counters and stall reasons, CUDA events or `nsys` for
time.

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
      262144        3.1     0.0060     521.1   103.4%  L2-resident (not a DRAM measurement)
     1048576       12.6     0.0057    2217.5   440.0%  L2-resident (not a DRAM measurement)
     4194304       50.3     0.0233    2164.3   429.4%  L2/DRAM transition
    16777216      201.3     0.4429     454.5    90.2%  DRAM-bound
    67108864      805.3     1.7590     457.8    90.8%  DRAM-bound
```

440% of peak is a cache hit, not a fast kernel. The rmsnorm CSV carries
`footprint_mb` and `l2_resident` columns so each row's regime is explicit.
