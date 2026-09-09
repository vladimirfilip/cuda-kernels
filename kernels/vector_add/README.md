# Vector add (CUDA)

`c = a + b`. Nothing to optimize in the kernel itself. It is here to demonstrate
the measurement setup the other slices depend on, and one trap that setup avoids.

| | |
|---|---|
| Files | [`kernel.cuh`](kernel.cuh), [`main.cu`](main.cu) driver |

```bash
make run  KERNEL=vector_add      # sweep working-set sizes across the L2 boundary
make nsys KERNEL=vector_add      # timeline + per-kernel durations
```

## Bandwidth numbers can exceed the card's bandwidth

The driver sweeps `n` from 3 MB of traffic to 800 MB. The reference card has
504 GB/s of DRAM bandwidth and 48 MB of L2:

```
           n   MB moved  ms/launch      GB/s    %peak  working set
      262144        3.1     0.0060     521.1   103.4%  L2-resident (not a DRAM measurement)
     1048576       12.6     0.0057    2217.5   440.0%  L2-resident (not a DRAM measurement)
     4194304       50.3     0.0233    2164.3   429.4%  L2/DRAM transition
    16777216      201.3     0.4429     454.5    90.2%  DRAM-bound
    67108864      805.3     1.7590     457.8    90.8%  DRAM-bound
```

At 3 MB the kernel is latency bound and the effective-bandwidth figure is low. At
12.6 MB it runs well but the working set still never leaves L2, so almost no
traffic reaches DRAM and the 440% figure is measuring cache. Only the last two
rows, where the footprint is several times L2, measure what a roofline model
assumes.

This is why the rmsnorm CSV carries `footprint_mb` and `l2_resident` columns: an
L2-resident row is labelled, not hidden. A memory-bound kernel benchmarked inside
L2 will look excellent and tell you nothing.

The crossover is gradual rather than a cliff: at 50 MB the footprint just exceeds
L2 but most accesses still hit, so that row still reports 429%.

## Timing methodology

Every driver in this repo follows the same shape:

- one warm-up launch before timing, to pay context setup and JIT cost;
- CUDA events around a loop of launches, not wall-clock around one: a single
  launch at these sizes is microseconds and launch overhead dominates;
- divide by iteration count, so the reported figure is per-launch;
- check correctness in the same run and exit non-zero on mismatch.

Timings taken under `ncu` are not comparable to normal runs, because it
serialises and replays kernels. This kernel reports about 0.006 ms/launch
normally and 0.51 ms/launch under `ncu`. See [`docs/profiling.md`](../../docs/profiling.md).
