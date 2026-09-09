"""Latency benchmark for the Triton FlashAttention-2 forward.

    make bench KERNEL=flash_attention        # or:
    python kernels/flash_attention/bench.py [--quick] [--dtypes fp16,fp32]

Every variant in a cell runs at the SAME dtype. Comparing an fp16 kernel against
an fp32 baseline measures the dtype, not the kernel.

Attention is compute-bound, so the throughput column is TFLOP/s over the two
matmuls (QK^T and P@V), not GB/s -- there is no meaningful single traffic model
for a kernel whose whole point is not materialising the N x N intermediate.

Writes results/flash_attention_<timestamp>.csv and prints a table.
"""

import argparse
import csv
import datetime as _dt
from pathlib import Path

import torch
import torch.nn.functional as F

from kernels._common.timing import cuda_time_ms
from kernels.flash_attention.kernel import flash_attention_forward, naive_attention

# (batch, heads, seq_len, head_dim)
SHAPES = [
    (1, 4, 256, 64),
    (2, 8, 512, 64),
    (2, 8, 1024, 64),
    (2, 8, 2048, 64),
    (1, 8, 4096, 64),
]
QUICK_SHAPES = SHAPES[:2]
CAUSAL_VALUES = [False, True]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def attention_tflops(batch, heads, seq_len, head_dim, causal, ms):
    """QK^T and P@V are each 2*N*N*D MACs per (batch, head). Causal masking
    skips just under half the key tiles."""
    flops = 4.0 * batch * heads * seq_len * seq_len * head_dim
    if causal:
        flops *= 0.5
    return flops / (ms * 1e-3) / 1e12


def bench_cell(shape, causal, dtype_name, dtype):
    batch, heads, seq_len, head_dim = shape
    torch.manual_seed(0)
    q, k, v = (torch.randn(*shape, device="cuda", dtype=dtype) for _ in range(3))

    variants = {
        "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
        "naive": lambda: naive_attention(q, k, v, is_causal=causal),
        "flash": lambda: flash_attention_forward(q, k, v, is_causal=causal)[0],
    }

    ms = {}
    for name, fn in variants.items():
        try:
            ms[name] = cuda_time_ms(fn)
        except (torch.OutOfMemoryError, Exception) as e:
            if not isinstance(e, torch.OutOfMemoryError) and "out of resource" not in str(e):
                raise
            ms[name] = None
            torch.cuda.empty_cache()

    rows = []
    for name, t in ms.items():
        rows.append(dict(
            batch=batch, heads=heads, seq_len=seq_len, head_dim=head_dim,
            dtype=dtype_name, causal="yes" if causal else "no", variant=name,
            ms=round(t, 4) if t else "",
            tflops=round(attention_tflops(*shape, causal, t), 2) if t else "",
            speedup_vs_sdpa=(round(ms["sdpa"] / t, 2) if t and ms.get("sdpa") else ""),
            status="ok" if t else "OOM/unsupported",
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="two smallest shapes only")
    ap.add_argument("--dtypes", default="fp16,fp32",
                    help="comma-separated subset of fp16,bf16,fp32")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    shapes = QUICK_SHAPES if args.quick else SHAPES
    dtypes = [(n, DTYPES[n]) for n in args.dtypes.split(",")]

    print(f"# {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    hdr = (f"{'dtype':>5} {'B':>2} {'H':>2} {'N':>5} {'D':>3} {'causal':>6} "
           f"{'variant':>7} {'ms':>9} {'TFLOP/s':>8} {'x/sdpa':>7}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for dtype_name, dtype in dtypes:
        for causal in CAUSAL_VALUES:
            for shape in shapes:
                for r in bench_cell(shape, causal, dtype_name, dtype):
                    rows.append(r)
                    print(f"{r['dtype']:>5} {r['batch']:>2} {r['heads']:>2} "
                          f"{r['seq_len']:>5} {r['head_dim']:>3} {r['causal']:>6} "
                          f"{r['variant']:>7} {r['ms']!s:>9} "
                          f"{r['tflops']!s:>8} {r['speedup_vs_sdpa']!s:>7}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"flash_attention_{stamp}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
