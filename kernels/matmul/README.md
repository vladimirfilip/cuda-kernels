# Matmul: naive to register-tiled (CUDA)

`C = A @ B` in fp32, row-major, `A: [M,K] . B: [K,N] -> C: [M,N]`. Four rungs,
one shared header, a comparison against cuBLAS.

| | |
|---|---|
| Files | [`kernel.cuh`](kernel.cuh) all variants, [`main.cu`](main.cu) driver, [`binding.cu`](binding.cu) + [`op.py`](op.py) PyTorch ops |
| Correctness | [`test_matmul.py`](test_matmul.py), 30 cases (`tiled`, `v2` and `v3`); the numeric ones use an fp64 reference |
| Benchmark | [`bench.py`](bench.py) -> `results/matmul_*.csv` |

```bash
make run   KERNEL=matmul     # all four variants, correctness + GFLOP/s
make test                    # pytest vs fp64 reference
make bench KERNEL=matmul     # against cuBLAS
```

Tests and the driver use non-square shapes. The driver defaults to M=1024, K=768,
N=512; the suite includes `(17, 33, 65)`, `(100, 1, 100)` and `(129, 257, 63)`,
several of them smaller than one tile in some dimension and not multiples of
`TILE_WIDTH=16`, v2's `BM=BN=64`, or v3's `BM3=BN3=128`. `A` and `B` have
different strides, so square-only coverage would not exercise the indexing.

`make run KERNEL=matmul` (and the CPU reference it checks every rung against)
stays at problem sizes up to a few hundred thousand elements; the CPU reference
is an unaccelerated triple loop, and at M=K=N=4096 alone it is `2 * 4096^3`
flops single-threaded, several minutes before any GPU kernel even runs. Sizes
that large are covered by `bench.py` instead, which checks against PyTorch and
never runs that reference.

## The four rungs

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

**v3, wider register tile + vectorized shared-memory reads.** Same idea as v2,
scaled up: `BM=BN=128` (up from 64), `TM=TN=8` (up from 4), so each of the
block's 256 threads now does `TM*TN = 64` FMAs per `k` step against `TM+TN =
16` shared-memory reads -- the same 1:2 read:FMA ratio as v2 (both scale
together), but now spread over 4x fewer, 4x bigger blocks, and each read
arrives as a `float4` (4 values in one instruction) instead of one scalar at a
time. The `float4` reads need 16-byte alignment on a shared-memory tile whose
logical row (`TM` or `TN` consecutive output columns) has to be contiguous in
memory; for `Bs` that's already true, but `As`'s natural layout has `BK`, not
`BM`, as the contiguous axis, so v3 stores it transposed (`As_t[BK][BM]`,
written that way during staging) purely so the read side can vectorize. See
the comment above `matmul_v3_kernel` in [`kernel.cuh`](kernel.cuh).

From `make run KERNEL=matmul` (M=1024, K=768, N=512):

| variant | ms/launch | GFLOP/s | vs naive |
|---------|-----------|---------|----------|
| naive | 0.5887 | 1368 | 1.00x |
| tiled | 0.4344 | 1854 | 1.35x |
| v2 (register) | 0.1226 | 6567 | 4.80x |
| v3 (wide) | 0.1617 | 4980 | 3.64x |

v2's win over v1 is smallest at small or narrow problems: register tiling adds
fixed overhead (staging into registers, a bigger 64x64 output tile the last
block partially wastes) that only pays for itself once there is enough
arithmetic per tile to amortise it. At `M=1, K=512, N=1` v2 is *slower* than
naive (see the correctness sweep below); at the 1024x768x512 default it is
already 4.8x naive and 3.5x v1.

**v3 is not a strict win over v2 -- it trades small-problem performance for
large-problem performance, and loses that trade at this default shape.** A
128x128 tile means a `cdiv(512,128) x cdiv(1024,128) = 4 x 8 = 32`-block grid
for this problem, on a 48-SM card: most SMs get at most one block, some get
none, and `-Xptxas -v` puts v3 at 127 registers/thread against v2's 70, capping
how many of even those blocks can run concurrently per SM. v2's smaller tile
simply has more, cheaper blocks to spread across the card. See "Against
cuBLAS" below for where that flips.

## Against cuBLAS

`make bench KERNEL=matmul`, TF32 disabled on both sides:

| M=K=N | cuBLAS | tiled | v2 | v3 | v2 %peak | v3 %peak | v3/v2 |
|-------|--------|-------|-----|-----|----------|----------|-------|
| 512 | 3141 GFLOP/s | 1499 GFLOP/s | 3153 GFLOP/s | 1917 GFLOP/s | 10.2% | 6.2% | 0.61x |
| 1024 | 10213 | 1817 | 6296 | 5647 | 20.4% | 18.3% | 0.90x |
| 2048 | 15139 | 1887 | 7696 | 8567 | 24.9% | 27.7% | 1.11x |
| 4096 | 15520 | 1905 | 8032 | 9795 | 26.0% | 31.7% | 1.22x |

v2 and v3 cross over between 1024^3 and 2048^3. Below that, v3's 128x128 tile
means too few blocks to fill the card's 48 SMs (see the crossover paragraph
above) and it loses to v2 -- by 10% at 1024^3, the cleanest cell to read this
at (see the noise note below for 512^3). Above the crossover, there is enough
grid to saturate the SMs regardless of block size, and the wider tile's better
FMA:shared-memory-read ratio wins outright: 31.7% of peak at 4096^3, against
v2's 26.0% and cuBLAS's 50.2%. Whichever rung you'd actually want at a given
shape is a real engineering answer, not a constant -- this is why the ladder
keeps both rather than deleting v2 once v3 existed.

512^3 is the noisiest cell in this sweep for all variants -- cuBLAS alone has
been seen between 2852 and 3141 GFLOP/s across different runs in this repo's
history, a ~10% swing from launch overhead and clock/thermal state at a
problem this small, before the algorithm's own effect is counted. v3 loses to
v2 by 39% at 512^3 in the run behind this table; treat that specific number as
noisy (direction: v3 loses, consistent; exact margin: not verified across
repeats). The crossover between 1024^3 and 4096^3 is a bigger, more stable
effect and not in question.

The remaining gap at the top of the sweep is at least two separate things, not
one. The obvious candidate is structural: the tile is still loaded fresh every
phase, with no overlap between staging the next tile's global loads and
computing the current tile's FMAs, and the FMAs themselves are still scalar
fp32 rather than tensor cores -- closing that needs double-buffering across
`k0` phases and eventually tensor cores. But Nsight Compute on `matmul_v3_kernel`
(2048^3) says there is a cheaper problem sitting in front of it:
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` is 40% of the
kernel's shared-memory load wavefronts, and the store-side conflict count is
67% of its store wavefronts -- `Bs[kk][thread_col*TN3 + j]` puts every 4th
`thread_col` on the same bank. Padding the shared-memory tiles would likely be
worth more per line of code than double-buffering until that's fixed. Neither
is implemented here.
