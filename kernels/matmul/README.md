# Matmul: naive to shared-memory tiled (CUDA)

`C = A @ B` in fp32, row-major, `A: [M,K] . B: [K,N] -> C: [M,N]`. Two rungs, one
shared header, a comparison against cuBLAS.

| | |
|---|---|
| Files | [`kernel.cuh`](kernel.cuh) both variants, [`main.cu`](main.cu) driver, [`binding.cu`](binding.cu) + [`op.py`](op.py) PyTorch op |
| Correctness | [`test_matmul.py`](test_matmul.py), 10 cases; the numeric ones use an fp64 reference |
| Benchmark | [`bench.py`](bench.py) -> `results/matmul_*.csv` |

```bash
make run   KERNEL=matmul     # both variants, correctness + GFLOP/s
make test                    # pytest vs fp64 reference
make bench KERNEL=matmul     # against cuBLAS
```

Tests and the driver use non-square shapes. The driver defaults to M=1024, K=768,
N=512; the suite includes `(17, 33, 65)`, `(100, 1, 100)` and `(129, 257, 63)`,
several of them not multiples of `TILE_WIDTH=16`. `A` and `B` have different
strides, so square-only coverage would not exercise the indexing.

## The two rungs

**v0, naive.** One thread per output element, walking the full `K` inner product
from global memory. Every thread in a block re-reads the same rows of `A` and
columns of `B` its neighbours are reading. The kernel is limited by redundant
global traffic, not arithmetic.

**v1, tiled.** Each block stages a `16 x 16` tile of `A` and of `B` into shared
memory, then every thread computes its partial products out of that tile. Each
element is fetched from global memory once per tile instead of once per thread,
roughly a `TILE_WIDTH`-fold reduction in global traffic.

From `make run KERNEL=matmul` (M=1024, K=768, N=512):

| variant | ms/launch | GFLOP/s | vs naive |
|---------|-----------|---------|----------|
| naive | 0.3263 | 2468 | 1.00x |
| tiled | 0.2518 | 3198 | 1.30x |

## Against cuBLAS

`make bench KERNEL=matmul`, TF32 disabled on both sides:

| M=K=N | cuBLAS | tiled | tiled %peak | ratio |
|-------|--------|-------|-------------|-------|
| 512 | 2078 GFLOP/s | 2114 GFLOP/s | 5.3% | 1.02x |
| 1024 | 13190 | 3061 | 7.6% | 0.23x |
| 2048 | 24538 | 3303 | 8.2% | 0.14x |
| 4096 | 27120 | 3283 | 8.2% | 0.12x |

The tiled kernel is flat at ~3.3 TFLOP/s, 8% of the card's 40 TFLOP/s fp32 peak.
cuBLAS scales to 27 TFLOP/s (68%). At 512^3 the two are level; cuBLAS pulls away
as size grows.

The gap is structural. A 16x16 tile with one output per thread gives each thread
one fused multiply-add per two shared-memory loads, so the kernel is bound by
shared-memory bandwidth well before the FMA units. Closing it needs register
tiling (each thread computing a small patch of `C`, say 4x4, so operands loaded
once from shared memory feed many FMAs), vectorized loads, double-buffering
across tiles, and tensor cores. That ladder is not implemented here. This slice
stops at the rung that demonstrates the shared-memory idea.
