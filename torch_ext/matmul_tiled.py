from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_SRC = Path(__file__).with_suffix(".cu")

load(
    name="cuda_kernels",
    sources=[str(_SRC)],
    extra_cuda_cflags=["-O2"],
    is_python_module=False,
    verbose=True,
)

matmul_tiled = torch.ops.cuda_kernels.matmul_tiled

# A "fake" (meta) implementation lets torch.compile and other tracing machinery
# infer the output's shape/dtype/device without launching the kernel.
@torch.library.register_fake("cuda_kernels::matmul_tiled")
def _matmul_tiled_fake(a, b):
    torch._check(a.dim() == 2 and b.dim() == 2)
    torch._check(a.size(1) == b.size(0))
    return a.new_empty(a.size(0), b.size(1))


if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N = 512, 384, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    c = matmul_tiled(a, b)
    ref = a @ b
    max_err = (c - ref).abs().max().item()
    print(f"max error vs torch a@b: {max_err:.3e}")
    torch.testing.assert_close(c, ref, rtol=1e-3, atol=1e-3)

    # Quick timing (sync-bracketed; GPU work is async).
    iters = 100
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        matmul_tiled(a, b)
    end.record()
    torch.cuda.synchronize()
    print(f"{start.elapsed_time(end) / iters:.4f} ms/call "
          f"({M}x{K} @ {K}x{N})")
    print("OK")
