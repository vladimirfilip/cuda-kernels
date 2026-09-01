"""Correctness tests for torch.ops.rmsnorm_kernels.rmsnorm_add.

    make torch-test            # or: .venv/bin/pytest -q tests/
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch_ext"))

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from rmsnorm_fused import rmsnorm_add, ref_rmsnorm_add  # noqa: E402

SHAPES = [
    (1, 2048),        # single token
    (4096, 2048),     # Llama-3.2-1B hidden
    (4096, 3072),     # Llama-3.2-3B hidden
    (2, 4097),        # H not a multiple of any vector width
    (128, 4096),
    (8192, 2048),     # large token count
]
DTYPES = [torch.float32, torch.bfloat16]
EPS = [1e-5, 1e-6]

_TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", EPS)
def test_matches_reference(shape, dtype, eps):
    torch.manual_seed(0)
    n, h = shape
    x = torch.randn(n, h, device="cuda", dtype=dtype)
    residual = torch.randn(n, h, device="cuda", dtype=dtype)
    weight = torch.randn(h, device="cuda", dtype=dtype)

    got_h, got_out = rmsnorm_add(x, residual, weight, eps)
    ref_h, ref_out = ref_rmsnorm_add(x, residual, weight, eps)

    torch.testing.assert_close(got_h, ref_h, **_TOL[dtype])
    torch.testing.assert_close(got_out, ref_out, **_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_inputs_not_mutated(dtype):
    torch.manual_seed(1)
    x = torch.randn(64, 2048, device="cuda", dtype=dtype)
    residual = torch.randn(64, 2048, device="cuda", dtype=dtype)
    weight = torch.randn(2048, device="cuda", dtype=dtype)
    x0, r0, w0 = x.clone(), residual.clone(), weight.clone()

    rmsnorm_add(x, residual, weight, 1e-5)

    torch.testing.assert_close(x, x0, rtol=0, atol=0)
    torch.testing.assert_close(residual, r0, rtol=0, atol=0)
    torch.testing.assert_close(weight, w0, rtol=0, atol=0)


def test_non_contiguous_inputs():
    torch.manual_seed(2)
    # Transpose makes a non-contiguous view; the op should .contiguous() it.
    base = torch.randn(2048, 512, device="cuda", dtype=torch.float32)
    x = base.t()                      # [512, 2048], non-contiguous
    residual = torch.randn_like(x)
    weight = torch.randn(2048, device="cuda", dtype=torch.float32)

    got_h, got_out = rmsnorm_add(x, residual, weight, 1e-5)
    ref_h, ref_out = ref_rmsnorm_add(x, residual, weight, 1e-5)
    torch.testing.assert_close(got_h, ref_h, **_TOL[torch.float32])
    torch.testing.assert_close(got_out, ref_out, **_TOL[torch.float32])
