// PyTorch bindings for the tiled and register-tiled matmul kernels. The
// kernels themselves live in kernel.cuh, shared with the standalone driver,
// so there is exactly one copy of the device code and both paths are
// guaranteed to profile the same thing.

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "kernel.cuh"

namespace {

// Shared shape checks for both ops.
void check_matmul_inputs(const torch::Tensor &a, const torch::Tensor &b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                    b.scalar_type() == torch::kFloat32,
                "a and b must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0),
                "shape mismatch: a.size(1) (", a.size(1),
                ") != b.size(0) (", b.size(0), ")");
}

}  // namespace

// C = A @ B for row-major float32 CUDA tensors, A: MxK, B: KxN, C: MxN.
torch::Tensor matmul_tiled(torch::Tensor a, torch::Tensor b) {
    check_matmul_inputs(a, b);

    // The kernel indexes with tight row-major strides, so make that true.
    a = a.contiguous();
    b = b.contiguous();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    auto c = torch::empty({M, N}, a.options());

    launch_matmul_tiled(a.data_ptr<float>(), b.data_ptr<float>(),
                        c.data_ptr<float>(), M, K, N,
                        c10::cuda::getCurrentCUDAStream());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "matmul_tiled kernel launch failed");
    return c;
}

// Same contract as matmul_tiled, dispatching to the register-tiled v2 kernel.
torch::Tensor matmul_v2(torch::Tensor a, torch::Tensor b) {
    check_matmul_inputs(a, b);

    a = a.contiguous();
    b = b.contiguous();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    auto c = torch::empty({M, N}, a.options());

    launch_matmul_v2(a.data_ptr<float>(), b.data_ptr<float>(),
                     c.data_ptr<float>(), M, K, N,
                     c10::cuda::getCurrentCUDAStream());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "matmul_v2 kernel launch failed");
    return c;
}

// Declare the ops' schemas in the "cuda_kernels" namespace.
TORCH_LIBRARY(cuda_kernels, m) {
    m.def("matmul_tiled(Tensor a, Tensor b) -> Tensor");
    m.def("matmul_v2(Tensor a, Tensor b) -> Tensor");
}

// Bind the CUDA implementations to those schemas.
TORCH_LIBRARY_IMPL(cuda_kernels, CUDA, m) {
    m.impl("matmul_tiled", &matmul_tiled);
    m.impl("matmul_v2", &matmul_v2);
}
