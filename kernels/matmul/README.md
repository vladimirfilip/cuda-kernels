# Matmul: naive → shared-memory tiled (CUDA)

`C = A @ B` in fp32, row-major, `A: [M,K] · B: [K,N] → C: [M,N]`. Two rungs, one
shared header, and an honest comparison against cuBLAS.

| | |
|---|---|
| Files | [`kernel.cuh`](kernel.cuh) both variants · [`main.cu`](main.cu) driver · [`binding.cu`](binding.cu) + [`op.py`](op.py) PyTorch op |
| Correctness | [`test_matmul.py`](test_matmul.py) — 10 cases vs an fp64 reference |
| Benchmark | [`bench.py`](bench.py) → `results/matmul_*.csv` |

```bash
make run   KERNEL=matmul     # both variants, correctness + GFLOP/s
make test                    # pytest vs fp64 reference
make bench KERNEL=matmul     # against cuBLAS
```

## The two rungs

**v0, naive.** One thread per output element, walking the full `K` inner
product from global memory. Every thread in a block re-reads the same rows of `A`
and columns of `B` that its neighbours are reading. The kernel is limited by
redundant global traffic, not arithmetic.

**v1, tiled.** Each block stages a `16 × 16` tile of `A` and of `B` into shared
memory, then every thread in the block computes its partial products out of that
tile. Each element is fetched from global memory once per tile instead of once
per thread — roughly a `TILE_WIDTH`-fold reduction in global traffic.

From `make run KERNEL=matmul` (M=1024, K=768, N=512):

| variant | ms/launch | GFLOP/s | vs naive |
|---------|-----------|---------|----------|
| naive | 0.3263 | 2468 | 1.00× |
| tiled | 0.2518 | 3198 | **1.30×** |

## Against cuBLAS

`make bench KERNEL=matmul`, TF32 disabled on both sides so the comparison is
arithmetic-for-arithmetic:

| M=K=N | cuBLAS | tiled | tiled %peak | ratio |
|-------|--------|-------|-------------|-------|
| 512 | 2078 GFLOP/s | 2114 GFLOP/s | 5.3% | **1.02×** |
| 1024 | 13190 | 3061 | 7.6% | 0.23× |
| 2048 | 24538 | 3303 | 8.2% | 0.14× |
| 4096 | 27120 | 3283 | 8.2% | 0.12× |

This is the interesting part, and it is not a flattering result. **The tiled
kernel is flat at ~3.3 TFLOP/s — 8% of the card's 40 TFLOP/s fp32 peak — while
cuBLAS scales to 27 TFLOP/s (68%).** At 512³ the two are level only because both
are launch-overhead bound there; cuBLAS pulls away the moment the problem is big
enough to matter.

The gap is structural, not a tuning detail. A 16×16 tile with one output per
thread gives each thread one fused multiply-add per two shared-memory loads,
so the kernel is bound by shared-memory bandwidth long before it approaches the
FMA units. Closing it needs **register tiling** — each thread computing a small
patch of `C` (say 4×4) so that operands loaded once from shared memory feed many
FMAs — plus vectorized loads, double-buffering across tiles, and ultimately
tensor cores. That ladder is not implemented here; this slice stops at the rung
that demonstrates the shared-memory idea.

Reporting the 8% is the point. A matmul write-up that shows only "1.3× faster
than naive" is hiding the number that matters.

## A note on the indexing bug this slice used to have

The naive kernel indexed `B` as `b[k * K + x]` when `B` is `[K, N]` — the column
stride is `N`, not `K`. Every benchmark and check used square matrices, where
`K == N`, so it was invisible. The driver now defaults to **M=1024, K=768, N=512**
and the tests use mostly non-square, non-tile-aligned shapes, including
`(17, 33, 65)`, `(100, 1, 100)` and `(129, 257, 63)`.

Square-only test coverage on a kernel with two different strides is not coverage.
