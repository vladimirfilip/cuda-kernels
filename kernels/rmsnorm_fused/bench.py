"""Benchmark the fused RMSNorm + residual-add kernel against PyTorch.

    make bench KERNEL=rmsnorm_fused             # or:
    python kernels/rmsnorm_fused/bench.py [--peak-gbps 504] [--quick]

For every (N, H, dtype) cell it times three things:
  fused    - torch.ops.rmsnorm_kernels.rmsnorm_add (the custom kernel)
  unfused  - h = x + residual ; F.rms_norm(h, ...)   <- the headline baseline to beat
  norm     - F.rms_norm alone on a precomputed h     <- lower bound (no add, no h write)

Writes results/rmsnorm_<timestamp>.csv and prints a table. Sizes are kept past
this card's 48 MB L2 so the %peak column measures DRAM, not cache.
"""

import argparse
import csv
import datetime as _dt
from pathlib import Path

import torch
import torch.nn.functional as F

from kernels._common.timing import cuda_time_ms
from kernels.rmsnorm_fused.op import rmsnorm_add

# The smallest cells here used to be a few MB, which fits entirely in this
# card's 48 MB L2 -- those rows reported "bandwidth" several times the card's
# DRAM limit. The sweep now starts past L2 so %peak means what it claims; the
# L2_MB column records the footprint so the reader can check.
N_VALUES = [4096, 8192, 16384, 32768]
H_VALUES = [2048, 3072, 4096]
DTYPES = [("fp32", torch.float32), ("bf16", torch.bfloat16)]
EPS = 1e-5
L2_MB = 48.0

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


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
    # Each variant moves a different amount, so each gets its own traffic model.
    # Charging all three the fused op's traffic (as an earlier version did) made
    # the cheapest variant look like it exceeded the card's DRAM bandwidth.
    elem = x.element_size()
    tensor = n * h * elem
    traffic = {
        # read x + residual, write h + out
        "fused": 4 * tensor,
        # read x + residual, write tmp; read tmp, write out
        "unfused": 5 * tensor,
        # read h, write out (no add, no h write) -- the lower bound
        "norm": 2 * tensor,
    }

    rows = []
    ms = {name: cuda_time_ms(fn) for name, fn in variants.items()}
    for name, t in ms.items():
        gbps = traffic[name] / (t * 1e6)
        rows.append(
            dict(
                N=n,
                H=h,
                dtype=dtype_name,
                variant=name,
                ms=round(t, 5),
                gbps=round(gbps, 1),
                pct_peak=round(100 * gbps / peak_gbps, 1),
                footprint_mb=round(traffic[name] / 1e6, 1),
                l2_resident="yes" if traffic[name] / 1e6 < L2_MB else "no",
                speedup_vs_unfused=round(ms["unfused"] / t, 2),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-gbps", type=float, default=504.0,
                    help="DRAM bandwidth for the %%-peak column (RTX 4070 Ti = 504)")
    ap.add_argument("--quick", action="store_true",
                    help="one small shape, for a fast sanity run")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"# {torch.cuda.get_device_name()}  torch {torch.__version__}")

    # Even the quick shape stays past L2, so a fast run is still a real
    # DRAM measurement rather than a cache benchmark.
    n_values, h_values = ([8192], [2048]) if args.quick else (N_VALUES, H_VALUES)

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"rmsnorm_{stamp}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
