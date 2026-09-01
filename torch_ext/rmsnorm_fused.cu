#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <tuple>

#include "../src/rmsnorm_fused.cuh"

// Fused RMSNorm + residual add for row-major CUDA tensors.
//   x, residual : [N, H]   weight : [H]
// returns (h, out) where
//   h   = residual + x                                   (the updated residual stream)
//   out = h * rsqrt(mean(h^2, dim=-1) + eps) * weight
std::tuple<torch::Tensor, torch::Tensor>
rmsnorm_add(torch::Tensor x, torch::Tensor residual, torch::Tensor weight,
            double eps) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [N, H]");
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual must match x shape");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1),
                "weight must be [H]");
    TORCH_CHECK(residual.scalar_type() == x.scalar_type() &&
                    weight.scalar_type() == x.scalar_type(),
                "x, residual, weight must share a dtype");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 ||
                    x.scalar_type() == torch::kBFloat16,
                "only float32 and bfloat16 are supported");

    x = x.contiguous();
    residual = residual.contiguous();
    weight = weight.contiguous();

    const int N = x.size(0);
    const int H = x.size(1);
    auto h = torch::empty_like(x);
    auto out = torch::empty_like(x);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const float epsf = static_cast<float>(eps);

    if (x.scalar_type() == torch::kFloat32) {
        launch_rmsnorm_fused_v0<float>(
            x.data_ptr<float>(), residual.data_ptr<float>(),
            weight.data_ptr<float>(), h.data_ptr<float>(), out.data_ptr<float>(),
            N, H, epsf, stream);
    } else {
        using bf16 = __nv_bfloat16;
        launch_rmsnorm_fused_v0<bf16>(
            reinterpret_cast<const bf16 *>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16 *>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16 *>(weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<bf16 *>(h.data_ptr<at::BFloat16>()),
            reinterpret_cast<bf16 *>(out.data_ptr<at::BFloat16>()), N, H, epsf,
            stream);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "rmsnorm_add launch failed");
    return {h, out};
}

// Separate library name so this can load alongside the matmul_tiled op
// (namespace cuda_kernels) in one process without a double registration.
TORCH_LIBRARY(rmsnorm_kernels, m) {
    m.def("rmsnorm_add(Tensor x, Tensor residual, Tensor weight, float eps) "
          "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(rmsnorm_kernels, CUDA, m) {
    m.impl("rmsnorm_add", &rmsnorm_add);
}
