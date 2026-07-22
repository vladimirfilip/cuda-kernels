#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define cdiv(a, b) ((a + b - 1) / b)

__global__ void matmul_tiled_kernel(const float *a, const float *b, float *c,
                                    int M, int K, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + tx;
    int y = blockIdx.y * TILE_WIDTH + ty;

    float acc = 0.0f;
    int phases = cdiv(K, TILE_WIDTH);

    for (int i = 0; i < phases; i++) {
        int _x = i * TILE_WIDTH + tx, _y = i * TILE_WIDTH + ty;
        As[ty][tx] = (y < M && _x < K) ? a[y * K + _x] : 0.0f;
        Bs[ty][tx] = (_y < K && x < N) ? b[_y * N + x] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }
    if (x < N && y < M)
        c[y * N + x] = acc;
}

// C = A @ B for row-major float32 CUDA tensors, A: MxK, B: KxN, C: MxN.
torch::Tensor matmul_tiled(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0),
                "shape mismatch: a.size(1) (", a.size(1),
                ") != b.size(0) (", b.size(0), ")");

    // The kernel indexes with tight row-major strides, so make that true.
    a = a.contiguous();
    b = b.contiguous();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    auto c = torch::empty({M, N}, a.options());

    const dim3 block(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid(cdiv(N, TILE_WIDTH), cdiv(M, TILE_WIDTH));

    matmul_tiled_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        M, K, N);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "matmul_tiled kernel launch failed");

    return c;
}

// Declare the op's schema in the "cuda_kernels" namespace.
TORCH_LIBRARY(cuda_kernels, m) {
    m.def("matmul_tiled(Tensor a, Tensor b) -> Tensor");
}

// Bind the CUDA implementation to that schema.
TORCH_LIBRARY_IMPL(cuda_kernels, CUDA, m) {
    m.impl("matmul_tiled", &matmul_tiled);
}
