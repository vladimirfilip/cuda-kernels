"""JIT loader + reference + smoke test for the fused RMSNorm + residual-add op.

    python torch_ext/rmsnorm_fused.py        # build, check vs reference, time it

The compiled op is registered as ``torch.ops.rmsnorm_kernels.rmsnorm_add``.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_SRC = Path(__file__).with_suffix(".cu")

load(
    name="rmsnorm_kernels",
    sources=[str(_SRC)],
    extra_cuda_cflags=["-O2"],
    is_python_module=False,
    verbose=True,
)

rmsnorm_add = torch.ops.rmsnorm_kernels.rmsnorm_add


@torch.library.register_fake("rmsnorm_kernels::rmsnorm_add")
def _rmsnorm_add_fake(x, residual, weight, eps):
    torch._check(x.dim() == 2 and residual.shape == x.shape)
    torch._check(weight.dim() == 1 and weight.size(0) == x.size(1))
    return torch.empty_like(x), torch.empty_like(x)


def ref_rmsnorm_add(x, residual, weight, eps):
    """Unfused PyTorch reference: residual add, then RMSNorm over the last dim."""
    h = x + residual
    out = F.rms_norm(h, (h.size(-1),), weight, eps)
    return h, out


def _demo():
    torch.manual_seed(0)
    N, H, eps = 4096, 2048, 1e-5
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(N, H, device="cuda", dtype=dtype)
        residual = torch.randn(N, H, device="cuda", dtype=dtype)
        weight = torch.randn(H, device="cuda", dtype=dtype)

        h, out = rmsnorm_add(x, residual, weight, eps)
        ref_h, ref_out = ref_rmsnorm_add(x, residual, weight, eps)
        tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(
            rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(h, ref_h, **tol)
        torch.testing.assert_close(out, ref_out, **tol)

        iters = 200
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iters):
            rmsnorm_add(x, residual, weight, eps)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        gbps = 4 * N * H * x.element_size() / (ms * 1e6)
        print(f"{str(dtype):>16}: {ms:.4f} ms/call  {gbps:.1f} GB/s  (N={N} H={H})")
    print("OK")


if __name__ == "__main__":
    _demo()
