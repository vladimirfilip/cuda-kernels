"""Correctness tests for the Triton FlashAttention-2 forward and its backward.

    make test        # or: pytest kernels/flash_attention
"""

import math

import pytest
import torch
import torch.nn.functional as F

from kernels.flash_attention.kernel import (
    FlashAttentionTriton,
    flash_attention_forward,
    naive_attention,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

# (batch, heads, seq_len, head_dim)
SHAPES = [
    (1, 1, 128, 64),
    (2, 4, 256, 64),
    (1, 1, 200, 64),   # seq_len not a multiple of BLOCK_M
    (1, 2, 128, 48),   # head_dim not a power of 2 -> BLOCK_D padding path
]
# fp32 at head_dim=128 needs more shared memory than Ada/Ampere consumer parts
# have; see _TILES in kernel.py. Kept as its own list so the fp16 path still
# covers the large head dim.
LARGE_D_SHAPES = [(1, 2, 512, 128)]


def reference(q, k, v, causal):
    """fp64 softmax attention. Comparing against SDPA instead would compare one
    fp32 rounding order against another and hide a systematically worse kernel."""
    q, k, v = (t.double() for t in (q, k, v))
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        nq, nk = q.shape[-2], k.shape[-2]
        mask = torch.tril(torch.ones(nq, nk, dtype=torch.bool, device=q.device))
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def randn(shape, dtype):
    return torch.randn(*shape, device="cuda", dtype=dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_naive_matches_sdpa(shape, causal):
    """The naive reference is the baseline everything else is checked against,
    so it gets its own test."""
    torch.manual_seed(0)
    q, k, v = (randn(shape, torch.float32) for _ in range(3))
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(naive_attention(q, k, v, is_causal=causal), expected,
                               rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_forward_fp32(shape, causal):
    """fp32 runs tl.dot in IEEE mode, so it should land within a few ulp of the
    fp64 reference -- not merely within fp16-ish tolerance."""
    torch.manual_seed(0)
    q, k, v = (randn(shape, torch.float32) for _ in range(3))
    out, _ = flash_attention_forward(q, k, v, is_causal=causal)
    torch.testing.assert_close(out.double(), reference(q, k, v, causal),
                               rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES + LARGE_D_SHAPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_low_precision(shape, causal, dtype):
    torch.manual_seed(0)
    q, k, v = (randn(shape, dtype) for _ in range(3))
    out, _ = flash_attention_forward(q, k, v, is_causal=causal)
    # bf16 carries 8 mantissa bits to fp16's 11, hence the looser bound.
    tol = 2e-2 if dtype is torch.float16 else 1e-1
    torch.testing.assert_close(out.double(), reference(q, k, v, causal),
                               rtol=tol, atol=tol)


@pytest.mark.parametrize("shape", LARGE_D_SHAPES)
def test_fp32_large_head_dim_reports_shared_memory(shape):
    """head_dim=128 in fp32 exceeds shared memory on this class of GPU. That is
    a real limit, but it must surface as a message naming the cause."""
    import triton

    torch.manual_seed(0)
    q, k, v = (randn(shape, torch.float32) for _ in range(3))
    try:
        flash_attention_forward(q, k, v, is_causal=False)
    except triton.runtime.errors.OutOfResources as e:
        assert "head_dim" in str(e) and "shared memory" in str(e)
    else:
        pytest.skip("this GPU has enough shared memory for fp32 head_dim=128")


@pytest.mark.parametrize("shape", [(1, 1, 64, 32), (1, 2, 96, 48)])
@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_autograd(shape, causal):
    """torch.autograd.gradcheck cannot be used here: it perturbs inputs in
    float64 and the Triton forward has no fp64 path. Instead, differentiate the
    naive reference with autograd and compare the analytic gradients to it."""
    torch.manual_seed(0)
    q, k, v = (randn(shape, torch.float32).requires_grad_(True) for _ in range(3))
    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

    grad_out = randn(shape, torch.float32)

    FlashAttentionTriton.apply(q, k, v, causal).backward(grad_out)
    naive_attention(qr, kr, vr, is_causal=causal).backward(grad_out)

    for name, got, want in (("dQ", q.grad, qr.grad),
                            ("dK", k.grad, kr.grad),
                            ("dV", v.grad, vr.grad)):
        torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3,
                                   msg=lambda m, n=name: f"{n}: {m}")
