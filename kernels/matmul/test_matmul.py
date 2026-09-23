"""Correctness tests for torch.ops.cuda_kernels.matmul_tiled.

    make test        # or: pytest kernels/matmul
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from kernels.matmul.op import matmul_tiled, matmul_v2, matmul_v3

# All three ops share the same contract (and the same check_matmul_inputs on
# the C++ side), so every test below runs against all three.
OPS = [matmul_tiled, matmul_v2, matmul_v3]
OP_IDS = ["tiled", "v2", "v3"]

# (M, K, N). Deliberately mostly non-square and mostly not multiples of
# TILE_WIDTH=16, v2's BM=BN=64/BK=8 tile, or v3's BM3=BN3=128/BK3=8 tile: a
# square problem makes the [K, N] column stride equal to K, which hides
# row/column indexing bugs entirely.
SHAPES = [
    (512, 384, 256),
    (256, 256, 256),    # square, the easy case
    (17, 33, 65),       # smaller than one tile in every dimension
    (100, 1, 100),      # K = 1, a single accumulation step
    (1, 512, 1),        # degenerate outer dimensions
    (1024, 768, 512),
    (129, 257, 63),     # each dimension one off a tile boundary
]


@pytest.mark.parametrize("op", OPS, ids=OP_IDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_torch(op, shape):
    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # fp64 reference: comparing two fp32 accumulation orders would mask a
    # systematically wrong kernel as "close enough".
    expected = (a.double() @ b.double()).float()
    torch.testing.assert_close(op(a, b), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op", OPS, ids=OP_IDS)
def test_non_contiguous_inputs(op):
    """The kernel indexes with tight row-major strides; the binding is
    responsible for making that true."""
    torch.manual_seed(0)
    a = torch.randn(384, 512, device="cuda", dtype=torch.float32).T  # non-contiguous
    b = torch.randn(384, 256, device="cuda", dtype=torch.float32)
    assert not a.is_contiguous()
    torch.testing.assert_close(op(a, b), (a.double() @ b.double()).float(),
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op", OPS, ids=OP_IDS)
def test_shape_mismatch_raises(op):
    a = torch.randn(8, 16, device="cuda")
    b = torch.randn(32, 8, device="cuda")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        op(a, b)


@pytest.mark.parametrize("op", OPS, ids=OP_IDS)
def test_rejects_non_float32(op):
    a = torch.randn(8, 16, device="cuda", dtype=torch.float64)
    b = torch.randn(16, 8, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="float32"):
        op(a, b)
