from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_SRC = Path(__file__).parent / "binding.cu"

load(
    name="cuda_kernels",
    sources=[str(_SRC)],
    extra_cuda_cflags=["-O2"],
    is_python_module=False,
    verbose=True,
)

matmul_tiled = torch.ops.cuda_kernels.matmul_tiled
matmul_v2 = torch.ops.cuda_kernels.matmul_v2

# A "fake" (meta) implementation lets torch.compile and other tracing machinery
# infer the output's shape/dtype/device without launching the kernel.
@torch.library.register_fake("cuda_kernels::matmul_tiled")
def _matmul_tiled_fake(a, b):
    torch._check(a.dim() == 2 and b.dim() == 2)
    torch._check(a.size(1) == b.size(0))
    return a.new_empty(a.size(0), b.size(1))


@torch.library.register_fake("cuda_kernels::matmul_v2")
def _matmul_v2_fake(a, b):
    torch._check(a.dim() == 2 and b.dim() == 2)
    torch._check(a.size(1) == b.size(0))
    return a.new_empty(a.size(0), b.size(1))


if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N = 512, 384, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    ref = a @ b

    iters = 100
    for name, fn in (("tiled", matmul_tiled), ("v2", matmul_v2)):
        c = fn(a, b)
        max_err = (c - ref).abs().max().item()
        print(f"{name}: max error vs torch a@b: {max_err:.3e}")
        torch.testing.assert_close(c, ref, rtol=1e-3, atol=1e-3)

        # Quick timing (sync-bracketed; GPU work is async).
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(a, b)
        end.record()
        torch.cuda.synchronize()
        print(f"{name}: {start.elapsed_time(end) / iters:.4f} ms/call "
              f"({M}x{K} @ {K}x{N})")
    print("OK")
