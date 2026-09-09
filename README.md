# CUDA & Triton kernels

Hand-written GPU kernels, each one paired with a correctness suite checked
against a higher-precision reference and a benchmark against the production
PyTorch path. The kernels are the point; the measurement discipline around them
is the part I'd want reviewed.

All numbers below were measured on the reference machine described in
[`docs/profiling.md`](docs/profiling.md) — **RTX 4070 Ti** (`sm_89`, 504 GB/s HBM,
48 MB L2), CUDA 13.0, PyTorch 2.14, Triton 3.8 — and are reproducible with the
commands shown. Raw CSVs are in [`results/`](results/).

## The kernels

| kernel | language | headline result | write-up |
|--------|----------|-----------------|----------|
| **Fused RMSNorm + residual** | CUDA | **1.28× faster than the PyTorch pair**, 85% of HBM peak | [read](kernels/rmsnorm_fused/) |
| **FlashAttention-2** | Triton | **69% of PyTorch SDPA**, 4.9–11.9× over naive, O(N) memory | [read](kernels/flash_attention/) |
| **Matmul** (naive → tiled) | CUDA | 1.30× over naive; **8% of peak vs cuBLAS's 68%** | [read](kernels/matmul/) |
| **Vector add** | CUDA | measurement harness + the L2 benchmarking trap | [read](kernels/vector_add/) |

Each directory is self-contained: kernel, PyTorch binding, tests, benchmark and
write-up together.

```
kernels/<name>/
  kernel.cuh | kernel.py    device code
  main.cu                   standalone driver: self-checks, self-benchmarks
  binding.cu + op.py        PyTorch custom op
  test_<name>.py            correctness
  bench.py                  benchmark -> results/*.csv
  README.md                 what it does and what the numbers mean
```

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

make                            # build every standalone driver
make run   KERNEL=rmsnorm_fused # one kernel: correctness + timings, no GPU setup needed
make test                       # pytest across all slices
make bench KERNEL=matmul        # benchmark one slice (bench-all for every slice)
make nsys  KERNEL=vector_add    # Nsight Systems timeline
make sanitize KERNEL=rmsnorm_fused
```

Every standalone driver checks itself against a CPU reference and exits non-zero
on mismatch, so `make run` doubles as a smoke test.

## Three things worth reading

**RMSNorm: coalescing is worth 3.5×, and the "best" variant depends on dtype.**
Moving from one thread per row to one warp per row takes fp32 from 23.5% to 81.4%
of HBM bandwidth without changing a single arithmetic operation — only which
thread touches which address. Going further to one block per row helps fp32
(84.8%) but *hurts* bf16 (85.7% vs 93.7%), because the cross-warp reduction costs
more than the added parallelism buys. [The ladder, with numbers →](kernels/rmsnorm_fused/)

**FlashAttention: Triton's `tl.dot` silently runs fp32 on TF32 tensor cores.**
Against an fp64 reference the default was 2.3e-03 — 3500× worse than PyTorch's
fp32 SDPA at 6.5e-07. Requesting `input_precision="ieee"` fixes it (4.0e-07) but
roughly doubles shared-memory use, so the tile schedule has to depend on dtype.
[The measurements →](kernels/flash_attention/)

**Benchmarks that fit in L2 measure L2.** This card has 48 MB of it. `vector_add`
on a 12 MB working set reports 2217 GB/s — 440% of the card's actual bandwidth.
Nothing is broken; the traffic never reaches HBM. Every sweep here starts above
the L2 footprint, and the rmsnorm CSV carries `footprint_mb` and `l2_resident`
columns so the claim can be checked rather than trusted.
[Why it matters →](kernels/vector_add/)

## Correctness

Every kernel is tested against a **higher-precision reference** — fp64 for
attention and matmul, an fp64 CPU implementation for rmsnorm — rather than
against another fp32 implementation. Comparing two fp32 kernels to each other
hides a systematically worse one inside a tolerance that looks reasonable; it is
how the TF32 problem above stayed invisible.

Shapes are chosen to break things: sequence lengths that aren't multiples of the
tile size, head dimensions that aren't powers of two, hidden sizes that aren't
multiples of any vector width, non-square matmuls, and degenerate `K=1` cases.

```bash
make test        # 78 cases
```

## Profiling

`nsys` and `compute-sanitizer` work on the reference box. `ncu` is installed but
its hardware counters are blocked by `RmProfilingAdminOnly: 1` with no
passwordless sudo, so counter-based analysis is unavailable there — the
workarounds, and the reason `ncu` timings can't be compared against normal runs,
are in [`docs/profiling.md`](docs/profiling.md).

## License

GPL-3.0 — see [LICENSE](LICENSE).
