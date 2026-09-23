# Matmul: naive to register-tiled (CUDA)

`C = A @ B` in fp32, row-major, `A: [M,K] . B: [K,N] -> C: [M,N]`. Three rungs,
one shared header, a comparison against cuBLAS.

| | |
|---|---|
| Files | [`kernel.cuh`](kernel.cuh) all variants, [`main.cu`](main.cu) driver, [`binding.cu`](binding.cu) + [`op.py`](op.py) PyTorch ops |
| Correctness | [`test_matmul.py`](test_matmul.py), 20 cases (both `tiled` and `v2`); the numeric ones use an fp64 reference |
| Benchmark | [`bench.py`](bench.py) -> `results/matmul_*.csv` |

```bash
make run   KERNEL=matmul     # all three variants, correctness + GFLOP/s
make test                    # pytest vs fp64 reference
make bench KERNEL=matmul     # against cuBLAS
```

Tests and the driver use non-square shapes. The driver defaults to M=1024, K=768,
N=512; the suite includes `(17, 33, 65)`, `(100, 1, 100)` and `(129, 257, 63)`,
several of them smaller than one tile in some dimension and not multiples of
`TILE_WIDTH=16` or v2's `BM=BN=64`. `A` and `B` have different strides, so
square-only coverage would not exercise the indexing.

## The three rungs

**v0, naive.** One thread per output element, walking the full `K` inner product
from global memory. Every thread in a block re-reads the same rows of `A` and
columns of `B` its neighbours are reading. The kernel is limited by redundant
global traffic, not arithmetic.

**v1, tiled.** Each block stages a `16 x 16` tile of `A` and of `B` into shared
memory, then every thread computes its partial products out of that tile. Each
element is fetched from global memory once per tile instead of once per thread,
roughly a `TILE_WIDTH`-fold reduction in global traffic. But every thread still
computes one output element, so each value pulled from shared memory feeds
exactly one FMA -- the kernel is now bound by shared-memory bandwidth instead of
global traffic.

**v2, register-tiled.** Same `BK=8`-deep shared tiles (now `BM=BN=64`), but each
of the block's 256 threads owns a `TM x TN = 4 x 4` patch of the output instead
of one element. Per `k` step, a thread stages 4 values from `As` and 4 from `Bs`
into registers (8 shared-memory reads) and does `TM*TN = 16` FMAs against them --
the read:FMA ratio goes from v1's 1:0.5 to 1:2, so the same shared-memory
traffic now feeds 4x the arithmetic. See the comment above `matmul_v2_kernel` in
[`kernel.cuh`](kernel.cuh) for the full derivation.

From `make run KERNEL=matmul` (M=1024, K=768, N=512):

| variant | ms/launch | GFLOP/s | vs naive |
|---------|-----------|---------|----------|
| naive | 0.5887 | 1368 | 1.00x |
| tiled | 0.4344 | 1854 | 1.35x |
| v2 (register) | 0.1226 | 6567 | 4.80x |

v2's win over v1 is smallest at small or narrow problems: register tiling adds
fixed overhead (staging into registers, a bigger 64x64 output tile the last
block partially wastes) that only pays for itself once there is enough
arithmetic per tile to amortise it. At `M=1, K=512, N=1` v2 is *slower* than
naive (see the correctness sweep below); at the 1024x768x512 default it is
already 4.8x naive and 3.5x v1.

## Against cuBLAS

`make bench KERNEL=matmul`, TF32 disabled on both sides:

| M=K=N | cuBLAS | tiled | v2 | v2 %peak | v2/cuBLAS |
|-------|--------|-------|-----|----------|-----------|
| 512 | 2852 GFLOP/s | 1481 GFLOP/s | 3080 GFLOP/s | 10.0% | 1.08x |
| 1024 | 9995 | 1823 | 6276 | 20.3% | 0.63x |
| 2048 | 15210 | 1889 | 7726 | 25.0% | 0.51x |
| 4096 | 15540 | 1904 | 7997 | 25.9% | 0.52x |

512^3 is the noisiest cell in this sweep -- cuBLAS itself varies by ~15% run to
run there, small enough that launch overhead and clock/thermal state move it
more than the algorithm does. v2 lands ahead of cuBLAS there in every run so
far, but treat the margin, not the direction, as the noisy part. Past that, the
comparison is stable: v2 plateaus at ~8 TFLOP/s, 26% of the card's 31 TFLOP/s
fp32 peak -- up from tiled's 6% -- while cuBLAS keeps scaling (50.3% of peak by
4096^3).

The remaining gap is the same shape as the one v2 just closed, one level up:
v2's `4x4` register tile gives each thread `16` FMAs per `8` shared-memory
reads, better than v1's `1` FMA per `2` reads but still well short of what the
FMA units can sustain, and the tile is loaded fresh every phase with no
overlap between staging the next tile and computing the current one. Closing
it further needs a larger per-thread tile (`8x8`), vectorized shared-memory
loads (`float4`), double-buffering across `k0` phases so the next tile's global
loads issue while the current tile's FMAs run, and eventually tensor cores.
That ladder is not implemented here.
