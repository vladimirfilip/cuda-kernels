"""Benchmark the tiled matmul against cuBLAS.

    make bench KERNEL=matmul        # or:
    python kernels/matmul/bench.py [--quick]

cuBLAS (torch.matmul) is the honest baseline: a hand-tuned library kernel using
tensor cores and a far more sophisticated blocking scheme. A 16x16 shared-memory
tile is not going to reach it, and the point of the column is to show the size of
the gap, not to hide it. TF32 is disabled so both sides do the same arithmetic.
"""

import argparse
import csv
import datetime as _dt
from functools import partial
from pathlib import Path

import torch

from kernels._common.timing import cuda_time_ms
from kernels.matmul.op import matmul_tiled

# (M, K, N)
SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (1024, 768, 512),      # non-square
]
QUICK_SHAPES = SHAPES[:2]

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
# RTX 4070 Ti fp32 vector peak: 7680 cores * 2 FLOP/clk * ~2.61 GHz.
DEFAULT_PEAK_GFLOPS = 40100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--peak-gflops", type=float, default=DEFAULT_PEAK_GFLOPS)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    # Keep cuBLAS in true fp32 so the comparison is arithmetic-for-arithmetic
    # rather than tensor-core-vs-not.
    torch.backends.cuda.matmul.allow_tf32 = False

    print(f"# {torch.cuda.get_device_name(0)}  torch {torch.__version__}  (TF32 off)")
    hdr = (f"{'M':>5} {'K':>5} {'N':>5} {'variant':>8} {'ms':>9} "
           f"{'GFLOP/s':>9} {'%peak':>7} {'x/cublas':>9}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for M, K, N in (QUICK_SHAPES if args.quick else SHAPES):
        torch.manual_seed(0)
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # functools.partial rather than a lambda: a lambda defined in a loop
        # captures the loop variable by reference, which is a trap even when
        # (as here) the call happens in the same iteration.
        ms = {
            "cublas": cuda_time_ms(partial(torch.matmul, a, b)),
            "tiled": cuda_time_ms(partial(matmul_tiled, a, b)),
        }
        flops = 2.0 * M * N * K
        for name, t in ms.items():
            gflops = flops / (t * 1e6)
            rows.append(dict(M=M, K=K, N=N, variant=name, ms=round(t, 4),
                             gflops=round(gflops, 1),
                             pct_peak=round(100 * gflops / args.peak_gflops, 1),
                             speedup_vs_cublas=round(ms["cublas"] / t, 3)))
            r = rows[-1]
            print(f"{M:>5} {K:>5} {N:>5} {name:>8} {r['ms']:>9} "
                  f"{r['gflops']:>9} {r['pct_peak']:>6}% {r['speedup_vs_cublas']:>9}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"matmul_{stamp}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
