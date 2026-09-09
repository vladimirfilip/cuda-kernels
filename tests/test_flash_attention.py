import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from flash_attention import naive_attention, flash_attention_forward, FlashAttentionTriton

SHAPES = [(1,1,128,64), (2,4,256,64), (1,2,512,128), (1,1,200,64)]
SMALL_SHAPES = [(1, 1, 32, 16)]  # Small shape for gradcheck

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_naive_forward(shape, causal):
    torch.manual_seed(0)
    Q, K, V = (torch.randn(*shape, device='cuda', dtype=torch.float32) for _ in range(3))
    expected = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    actual = naive_attention(Q, K, V, is_causal=causal)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_fp16(shape, causal):
    torch.manual_seed(0)
    Q, K, V = (torch.randn(*shape, device='cuda', dtype=torch.float32) for _ in range(3))
    expected = F.scaled_dot_product_attention(Q, K, V, is_causal=causal).to(torch.float16)
    actual, _ = flash_attention_forward(Q.to(torch.float16), K.to(torch.float16), V.to(torch.float16), causal=causal)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_fp32(shape, causal):
    torch.manual_seed(0)
    Q, K, V = (torch.randn(*shape, device='cuda', dtype=torch.float32) for _ in range(3))
    expected = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    actual, _ = flash_attention_forward(Q, K, V, causal=causal)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("shape", SMALL_SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_gradcheck(shape, causal):
    torch.manual_seed(0)
    Q = torch.randn(*shape, device='cuda', dtype=torch.float64, requires_grad=True)
    K = torch.randn(*shape, device='cuda', dtype=torch.float64, requires_grad=True)
    V = torch.randn(*shape, device='cuda', dtype=torch.float64, requires_grad=True)

    torch.autograd.gradcheck(
        lambda q, k, v: FlashAttentionTriton.apply(q, k, v, causal),
        (Q, K, V),
        eps=1e-4,
        atol=1e-3
    )