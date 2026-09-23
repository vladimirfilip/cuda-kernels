# CUDA & Triton kernels

Hand-written GPU kernels. Each has a correctness suite checked against a
higher-precision reference and a benchmark against the PyTorch path. `vector_add`
is the exception and exists only to show the measurement harness.

All numbers below were measured on the reference machine in
[`docs/profiling.md`](docs/profiling.md): RTX 5070 (`sm_120`, 672 GB/s GDDR7,
48 MB L2), CUDA 13.0, PyTorch 2.14, Triton 3.8. They reproduce with the commands
shown plus the peak overrides listed in that file. Raw CSVs are in
[`results/`](results/).

## Kernels

| kernel | language | result | write-up |
|--------|----------|--------|----------|
| Fused RMSNorm + residual | CUDA | 1.23x (fp32) / 1.22x (bf16) over the PyTorch pair, ~85% of DRAM peak | [read](kernels/rmsnorm_fused/) |
| FlashAttention-2 | Triton | 82% of PyTorch SDPA, 2.9-9.3x over naive, O(N) forward memory | [read](kernels/flash_attention/) |
| Matmul (naive to register-tiled) | CUDA | 4.8x over naive; 26% of peak vs cuBLAS 50% | [read](kernels/matmul/) |
| Vector add | CUDA | measurement harness and the L2 benchmarking trap | [read](kernels/vector_add/) |

Each directory holds what applies to that kernel: device code, PyTorch binding,
tests, benchmark, write-up.

```
kernels/<name>/
  kernel.cuh | kernel.py    device code
  main.cu                   standalone driver: self-checks, self-benchmarks
  binding.cu + op.py        PyTorch custom op
  test_<name>.py            correctness
  bench.py                  benchmark -> results/*.csv
  README.md                 what it does, what the numbers mean
```

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

make                            # build every standalone driver
make run   KERNEL=rmsnorm_fused # one kernel: correctness + timings
make test                       # pytest across all slices
make bench KERNEL=matmul        # benchmark one slice (bench-all for every slice)
make nsys  KERNEL=vector_add    # Nsight Systems timeline
make sanitize KERNEL=rmsnorm_fused
```

Every standalone driver checks itself against a CPU reference and exits non-zero
on mismatch, so `make run` doubles as a smoke test.

## Notes

**RMSNorm: coalescing is worth 6.6x, and a register cache recovers the rest.**
Moving from one thread per row to one warp per row takes fp32 from 12.4% to
81.7% of DRAM bandwidth with no change to the arithmetic, only which thread
touches which address. One block per row barely moves fp32 (83.2% vs 81.7%) and
hurts bf16 badly (52.4% vs 96.7%): the cross-warp reduction costs more than the
added parallelism buys -- until the same kernel also vectorizes its loads and
keeps the row it just computed in registers instead of re-reading it from
global memory, which lands fp32 at 86.8% (the best in the ladder) and recovers
bf16 to 96.5% without giving up the cross-warp reduction.
[Details](kernels/rmsnorm_fused/)

**Matmul: register tiling turns one FMA per two shared-memory reads into
sixteen per eight.** Each thread computing a 4x4 patch of the output instead of
one element takes the kernel from 6% to 26% of the card's fp32 peak, 4.8x over
the naive kernel and briefly ahead of cuBLAS at 512^3 before cuBLAS's own
blocking scales past it. [Details](kernels/matmul/)

**FlashAttention: Triton's `tl.dot` runs fp32 on TF32 tensor cores by default.**
Against an fp64 reference the default was 2.3e-03, versus PyTorch fp32 SDPA at
5.5e-07. `input_precision="ieee"` brings it to 4.0e-07 and roughly doubles
shared-memory use, so the tile schedule depends on dtype.
[Details](kernels/flash_attention/)

**Benchmarks that fit in L2 measure L2.** This card has 48 MB. `vector_add` on a
12 MB working set reports 1518 GB/s, 226% of the card's bandwidth, because the
traffic never reaches DRAM. The rmsnorm CSV carries `footprint_mb` and
`l2_resident` columns so an L2-resident row is labelled rather than left to
inflate the result.
[Details](kernels/vector_add/)

## Correctness

Attention and matmul are tested against an fp64 reference. rmsnorm's pytest suite
compares against `F.rms_norm`; its standalone driver adds an fp64 CPU reference.
Comparing two fp32 kernels to each other hides a systematically worse one inside a
plausible tolerance.

Shapes are chosen to break things: sequence lengths that aren't multiples of the
tile size, head dimensions that aren't powers of two, hidden sizes that aren't
multiples of any vector width, non-square matmuls, degenerate `K=1` cases.

```bash
make test        # 94 cases
```

## Profiling

`nsys`, `ncu` and `compute-sanitizer` all work on the reference machine. See
[`docs/profiling.md`](docs/profiling.md) for what each tool needs and what
`ncu`'s counters found in the rmsnorm write-up.

## License

GPL-3.0, see [LICENSE](LICENSE).
