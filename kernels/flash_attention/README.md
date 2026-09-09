# FlashAttention-2 (Triton)

Tiled attention with online softmax. The kernel streams over key/value tiles and
carries a running max and running sum per query row, so the `N × N` score matrix
is never materialised — intermediate storage drops from O(N²) to O(N).

| | |
|---|---|
| Files | [`kernel.py`](kernel.py) Triton kernel + autograd wrapper |
| Correctness | [`test_flash_attention.py`](test_flash_attention.py) — 41 cases against an fp64 reference |
| Benchmarks | [`bench.py`](bench.py) latency · [`latency_benchmark.ipynb`](latency_benchmark.ipynb) · [`memory_benchmark.ipynb`](memory_benchmark.ipynb) |

```python
from kernels.flash_attention.kernel import FlashAttentionTriton

# Q, K, V: (batch, heads, seq_len, head_dim), fp16/bf16/fp32, CUDA
out = FlashAttentionTriton.apply(Q, K, V, is_causal)
```

`flash_attention_forward(Q, K, V, is_causal)` is the lower-overhead entry point when
gradients aren't needed; it returns `(output, L)` where `L = m + log(l)` is the
log-sum-exp saved for the backward pass.

## Online softmax

For each key tile `j`, with running max `m`, running denominator `l` and
accumulator `O`:

```
S     = Q_i @ K_j^T / sqrt(d)
m_new = max(m, rowmax(S))
P     = exp(S - m_new)
alpha = exp(m - m_new)          # how much the previous accumulation must shrink
l     = alpha * l + sum(P, axis=1)
O     = alpha * O + P @ V_j
m     = m_new
```

then `O / l` at the end. `alpha` is the whole trick: when a later tile contains a
larger score than anything seen so far, everything accumulated under the old
maximum is rescaled to the new one, so the result is identical to a softmax over
the full row without ever holding the full row. `m` and `l` are always fp32, even
when Q/K/V are fp16 — the running max is what keeps `exp` from overflowing.

## Performance

RTX 4070 Ti, fp16, head_dim 64, from `make bench KERNEL=flash_attention`. `sdpa`
is `F.scaled_dot_product_attention`, which dispatches to the official
FlashAttention CUDA kernels; `naive` materialises the full score matrix.

| shape | causal | sdpa | naive | **flash** | vs sdpa | vs naive |
|-------|--------|------|-------|-----------|---------|----------|
| B2 H8 N2048 | no | 0.339 ms | 2.221 ms | **0.531 ms** | 0.64× | 4.2× |
| B1 H8 N4096 | no | 0.602 ms | 4.226 ms | **0.868 ms** | 0.69× | 4.9× |
| B2 H8 N2048 | yes | 0.242 ms | 3.567 ms | **0.404 ms** | 0.60× | 8.8× |
| B1 H8 N4096 | yes | 0.408 ms | 7.042 ms | **0.590 ms** | 0.69× | 11.9× |

At N=4096 that is 39.6 TFLOP/s against SDPA's 57.1 — **about 69% of the hand-tuned
CUDA implementation**, and 4.9–11.9× over naive. The remaining gap is roughly what
you would expect: SDPA uses CUTLASS kernels with a warp-specialised pipeline and
per-architecture tile tuning, and its backward is fused where ours is not.

Below N≈1024 the Triton kernel loses to both baselines — those cells are
dominated by launch overhead, not by the algorithm. The crossover is visible in
[`latency_benchmark.ipynb`](latency_benchmark.ipynb).

## Precision: fp32 means fp32

Triton's `tl.dot` defaults to **TF32** on fp32 inputs, which silently costs about
three decimal digits. Measured against an fp64 reference at (1,1,128,64) causal:

| `tl.dot` mode | max abs error |
|---------------|--------------|
| default (TF32) | 2.3e-03 |
| `input_precision="ieee"` | **4.0e-07** |
| PyTorch SDPA fp32, for scale | 6.5e-07 |

The default is 3500× worse than SDPA and would quietly fail any test written to a
real fp32 tolerance. The kernel now requests `ieee` explicitly for fp32 and keeps
tensor cores for fp16/bf16, where the question doesn't arise.

That accuracy is not free. IEEE fp32 makes Triton emit a multi-pass emulation
instead of one tensor-core op, which needs roughly twice the shared memory for
the same tile — the 128×64 tiles no longer launch. So the tile schedule depends
on the dtype:

| dtype | BLOCK_M × BLOCK_N | `tl.dot` precision |
|-------|-------------------|--------------------|
| fp16, bf16 | 128 × 64 | tensor cores |
| fp32 | 64 × 32 | IEEE |

## Testing

```bash
make test        # or: pytest kernels/flash_attention
```

41 cases, all against an **fp64** reference rather than against SDPA — comparing
two fp32 implementations to each other would hide a systematically worse kernel.

- forward, fp32, `rtol=atol=1e-5` (only reachable because of the IEEE fix above)
- forward, fp16 and bf16, at dtype-appropriate tolerance
- causal and non-causal
- `seq_len=200`, not a multiple of `BLOCK_M`
- `head_dim=48`, not a power of two
- backward vs autograd through the naive reference
- fp32 `head_dim=128` raises an error naming shared memory as the cause

Note on the backward test: `torch.autograd.gradcheck` **cannot** be used here. It
perturbs inputs in float64 and the Triton forward has no fp64 path, so a gradcheck
test fails on a dtype assertion before it tests anything. The analytic gradients
are instead compared against autograd differentiating the naive reference.

## Limitations

1. **Backward is not tiled.** It recomputes `P = exp(S - L)` in PyTorch, so
   backward memory is O(N²) even though forward is O(N). A tiled backward is
   substantially more involved and is not attempted here.
2. **fp32 with head_dim=128 does not fit** in 99 KB of shared memory on this
   class of GPU. It raises `OutOfResources` naming the cause. fp16/bf16 at
   head_dim=128 is fine.
3. **No attention dropout, no arbitrary masks** — causal or nothing.
4. `head_dim` is padded to a power of two internally. Non-power-of-two values
   work (there is a test at 48), at the cost of the padded lanes' bandwidth.

## References

- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
