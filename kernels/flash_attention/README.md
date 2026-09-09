# FlashAttention-2 (Triton)

Tiled attention with online softmax. The kernel streams over key/value tiles and
carries a running max and running sum per query row, so the `N x N` score matrix
is never materialised. Intermediate storage is O(N), not O(N^2).

| | |
|---|---|
| Files | [`kernel.py`](kernel.py) Triton kernel + autograd wrapper |
| Correctness | [`test_flash_attention.py`](test_flash_attention.py), 41 cases; numeric checks use an fp64 reference |
| Benchmarks | [`bench.py`](bench.py) latency, [`latency_benchmark.ipynb`](latency_benchmark.ipynb), [`memory_benchmark.ipynb`](memory_benchmark.ipynb) |

```python
from kernels.flash_attention.kernel import FlashAttentionTriton

# Q, K, V: (batch, heads, seq_len, head_dim), fp16/bf16/fp32, CUDA
out = FlashAttentionTriton.apply(Q, K, V, is_causal)
```

`flash_attention_forward(Q, K, V, is_causal)` is the lower-overhead entry point
when gradients aren't needed. It returns `(output, L)` where `L = m + log(l)` is
the log-sum-exp saved for the backward pass.

## Online softmax

For each key tile `j`, with running max `m`, running denominator `l`, accumulator
`O`:

```
S     = Q_i @ K_j^T / sqrt(d)
m_new = max(m, rowmax(S))
P     = exp(S - m_new)
alpha = exp(m - m_new)          # rescale factor for the previous accumulation
l     = alpha * l + sum(P, axis=1)
O     = alpha * O + P @ V_j
m     = m_new
```

then `O / l` at the end. When a later tile contains a larger score than anything
seen so far, `alpha` rescales everything accumulated under the old maximum to the
new one, so the result matches a softmax over the full row without holding the
full row. `m` and `l` are always fp32, even when Q/K/V are fp16.

## Performance

RTX 4070 Ti, fp16, head_dim 64, from `make bench KERNEL=flash_attention`. `sdpa`
is `F.scaled_dot_product_attention`; for fp16/bf16 inputs it dispatches to the
FlashAttention CUDA kernels. `naive` materialises the full score matrix.

| shape | causal | sdpa | naive | flash | vs sdpa | vs naive |
|-------|--------|------|-------|-------|---------|----------|
| B2 H8 N2048 | no | 0.339 ms | 2.221 ms | 0.531 ms | 0.64x | 4.2x |
| B1 H8 N4096 | no | 0.602 ms | 4.226 ms | 0.868 ms | 0.69x | 4.9x |
| B2 H8 N2048 | yes | 0.242 ms | 3.567 ms | 0.404 ms | 0.60x | 8.8x |
| B1 H8 N4096 | yes | 0.408 ms | 7.042 ms | 0.590 ms | 0.69x | 11.9x |

At N=4096 that is 39.6 TFLOP/s against SDPA's 57.1, about 69% of the hand-tuned
CUDA implementation and 4.9-11.9x over naive. SDPA uses hand-tuned CUTLASS
kernels with per-architecture tile tuning, and its backward is fused where this
one is not.

Below N~1024 the small cells are launch-overhead bound: the kernel still beats
naive by roughly 1.5x but trails SDPA at every size in the sweep. See
[`latency_benchmark.ipynb`](latency_benchmark.ipynb).

## Precision

Triton's `tl.dot` defaults to TF32 on fp32 inputs, which costs about three
decimal digits. Measured against an fp64 reference at (1,1,128,64) causal:

| `tl.dot` mode | max abs error |
|---------------|--------------|
| default (TF32) | 2.3e-03 |
| `input_precision="ieee"` | 4.0e-07 |
| PyTorch SDPA fp32, for scale | 6.5e-07 |

The default is 3500x worse than SDPA and fails any test written to a real fp32
tolerance. The kernel requests `ieee` for fp32 and keeps tensor cores for
fp16/bf16.

IEEE fp32 makes Triton emit a multi-pass emulation instead of one tensor-core op,
which needs roughly twice the shared memory for the same tile, so the 128x64
tiles no longer launch. The tile schedule depends on dtype:

| dtype | BLOCK_M x BLOCK_N | `tl.dot` precision |
|-------|-------------------|--------------------|
| fp16, bf16 | 128 x 64 | tensor cores |
| fp32 | 64 x 32 | IEEE |

## Testing

```bash
make test        # or: pytest kernels/flash_attention
```

41 cases. The forward and backward numeric checks use an fp64 reference, not
SDPA. A few cases check the naive reference against SDPA and the fp32
head_dim=128 error path.

- forward, fp32, `rtol=atol=1e-5`
- forward, fp16 and bf16, at dtype-appropriate tolerance
- causal and non-causal
- `seq_len=200`, not a multiple of `BLOCK_M`
- `head_dim=48`, not a power of two
- backward vs autograd through the naive reference
- fp32 `head_dim=128` raises an error naming shared memory as the cause

`torch.autograd.gradcheck` cannot be used here: it perturbs inputs in float64 and
the Triton forward has no fp64 path, so it fails on a dtype assertion before
testing anything. The analytic gradients are compared against autograd
differentiating the naive reference.

## Limitations

1. Backward is not tiled. It recomputes `P = exp(S - L)` in PyTorch, so backward
   memory is O(N^2) even though forward is O(N).
2. fp32 with head_dim=128 does not fit in 99 KB of shared memory on this class of
   GPU. It raises `OutOfResources` naming the cause. fp16/bf16 at head_dim=128 is
   fine.
3. No attention dropout, no arbitrary masks. Causal or nothing.
4. `head_dim` is padded to a power of two internally. Non-power-of-two values
   work (tested at 48), at the cost of the padded lanes' bandwidth.

## References

- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
