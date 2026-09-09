"""Benchmark scaffold for flash attention implementations.

    make bench                                  # or:
    .venv/bin/python bench/bench_flash_attention.py [--peak-gbps 1790] [--quick]

For every (B, H, N, D, causal) cell it times three things:
  sdpa     - F.scaled_dot_product_attention (PyTorch native, fp32)
  naive    - naive_attention (pure PyTorch, fp32)
  flash    - flash_attention_forward (Triton/CUDA, fp16)

Writes bench/results/flash_attention_<timestamp>.csv (columns are plot-ready)
and prints a table. No plotting here.
"""

import argparse
import csv
import datetime as _dt
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from flash_attention import naive_attention, flash_attention_forward  # noqa: E402
from _timing import cuda_time_ms  # noqa: E402

# (batch, heads, seq_len, head_dim)
SHAPES = [
    (1, 1, 128, 64),
    (1, 4, 256, 64),
    (2, 8, 512, 64),
    (1, 1, 1024, 128),
    (2, 8, 2048, 128),
]
CAUSAL_VALUES = [False, True]

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def bench_cell(batch, heads, seq_len, head_dim, causal, peak_gbps):
    # Create inputs in fp32 for SDPA and naive, will convert to fp16 for flash
    q_fp32 = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    k_fp32 = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    v_fp32 = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float32)

    q_fp16 = q_fp32.to(torch.float16)
    k_fp16 = k_fp32.to(torch.float16)
    v_fp16 = v_fp32.to(torch.float16)

    # Approximate bytes moved: 3 reads (Q, K, V) + 1 write (O), plus softmax overhead
    # Q: (batch, heads, seq_len, head_dim)
    total_elements = batch * heads * seq_len * head_dim
    # 3 reads + 1 write of outputs + intermediate softmax buffer
    approx_bytes = 5 * total_elements * 4  # conservative estimate in bytes

    variants = {
        "sdpa": lambda: F.scaled_dot_product_attention(q_fp32, k_fp32, v_fp32, is_causal=causal),
        "naive": lambda: naive_attention(q_fp32, k_fp32, v_fp32, is_causal=causal),
        "flash": lambda: flash_attention_forward(q_fp16, k_fp16, v_fp16, causal=causal)[0],
    }

    rows = []
    ms = {name: cuda_time_ms(fn) for name, fn in variants.items()}
    for name, t in ms.items():
        gbps = approx_bytes / (t * 1e6)
        rows.append(
            dict(
                batch=batch,
                heads=heads,
                seq_len=seq_len,
                head_dim=head_dim,
                causal="yes" if causal else "no",
                variant=name,
                ms=round(t, 4),
                gbps=round(gbps, 1),
                pct_peak=round(100 * gbps / peak_gbps, 1),
                speedup_vs_sdpa=round(ms["sdpa"] / t, 2),
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

    shapes = ([(1, 1, 128, 64)] if args.quick else SHAPES)
    causal_values = ([False] if args.quick else CAUSAL_VALUES)

    all_rows = []
    hdr = f"{'B':>2} {'H':>2} {'N':>5} {'D':>3} {'causal':>6} {'variant':>7} {'ms':>8} {'GB/s':>8} {'%peak':>6} {'x/sdpa':>8}"
    print(hdr)
    print("-" * len(hdr))

    for causal in causal_values:
        for batch, heads, seq_len, head_dim in shapes:
            rows = bench_cell(batch, heads, seq_len, head_dim, causal, args.peak_gbps)
            for r in rows:
                all_rows.append(r)
                causal_str = "yes" if causal else "no"
                print(f"{r['batch']:>2} {r['heads']:>2} {r['seq_len']:>5} {r['head_dim']:>3} "
                      f"{causal_str:>6} {r['variant']:>7} {r['ms']:>8.4f} {r['gbps']:>8.1f} "
                      f"{r['pct_peak']:>6.1f} {r['speedup_vs_sdpa']:>8.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"flash_attention_{stamp}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
