"""Benchmark scaffold for the fused RMSNorm + residual-add kernel.

    make bench                                  # or:
    .venv/bin/python bench/bench_rmsnorm.py [--peak-gbps 1790] [--quick]

For every (N, H, dtype) cell it times three things:
  fused    - torch.ops.rmsnorm_kernels.rmsnorm_add (the custom kernel)
  unfused  - h = x + residual ; F.rms_norm(h, ...)   <- the headline baseline to beat
  norm     - F.rms_norm alone on a precomputed h     <- lower bound (no add, no h write)

Writes bench/results/rmsnorm_<timestamp>.csv (columns are plot-ready for Day 2)
and prints a table. No plotting here.
"""

import argparse
import csv
import datetime as _dt
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch_ext"))
from _timing import cuda_time_ms  # noqa: E402
from rmsnorm_fused import rmsnorm_add  # noqa: E402

N_VALUES = [512, 2048, 8192, 32768]
H_VALUES = [2048, 3072, 4096]
DTYPES = [("fp32", torch.float32), ("bf16", torch.bfloat16)]
EPS = 1e-5

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def bench_cell(n, h, dtype_name, dtype, peak_gbps):
    x = torch.randn(n, h, device="cuda", dtype=dtype)
    residual = torch.randn(n, h, device="cuda", dtype=dtype)
    weight = torch.randn(h, device="cuda", dtype=dtype)
    h_pre = (x + residual).contiguous()
    hd = (h,)  # normalized_shape

    variants = {
        "fused": lambda: rmsnorm_add(x, residual, weight, EPS),
        "unfused": lambda: F.rms_norm(x + residual, hd, weight, EPS),
        "norm": lambda: F.rms_norm(h_pre, hd, weight, EPS),
    }
    # bytes moved by the fused op: read x + residual, write h + out.
    fused_bytes = 4 * n * h * x.element_size()

    rows = []
    ms = {name: cuda_time_ms(fn) for name, fn in variants.items()}
    for name, t in ms.items():
        gbps = fused_bytes / (t * 1e6)
        rows.append(
            dict(
                N=n,
                H=h,
                dtype=dtype_name,
                variant=name,
                ms=round(t, 5),
                gbps=round(gbps, 1),
                pct_peak=round(100 * gbps / peak_gbps, 1),
                speedup_vs_unfused=round(ms["unfused"] / t, 2),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-gbps", type=float, default=1790.0,
                    help="HBM bandwidth for the %%-peak column (RTX 5090 ~1790)")
    ap.add_argument("--quick", action="store_true",
                    help="one small shape, for a fast sanity run")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"# {torch.cuda.get_device_name()}  torch {torch.__version__}")

    n_values, h_values = ([2048], [2048]) if args.quick else (N_VALUES, H_VALUES)

    all_rows = []
    hdr = f"{'N':>6} {'H':>5} {'dtype':>5} {'variant':>8} {'ms':>9} {'GB/s':>8} {'%peak':>6} {'x/unfused':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, dt in DTYPES:
        for n in n_values:
            for h in h_values:
                rows = bench_cell(n, h, name, dt, args.peak_gbps)
                for r in rows:
                    all_rows.append(r)
                    print(f"{r['N']:>6} {r['H']:>5} {name:>5} {r['variant']:>8} "
                          f"{r['ms']:>9.4f} {r['gbps']:>8.1f} {r['pct_peak']:>6.1f} "
                          f"{r['speedup_vs_unfused']:>10.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"rmsnorm_{stamp}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
