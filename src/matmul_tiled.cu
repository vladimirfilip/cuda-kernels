#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define cdiv(a, b) ((a + b - 1) / b)
#define TILE_WIDTH 16

__global__ void matmul(const float *a, const float *b, float *c, int M, int K, int N) {

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
        As[ty][tx] = y < M && _x < K ? a[y * K + _x] : 0.0f;
        Bs[ty][tx] = _y < K && x < N ? b[_y * N + x] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }
    if (x < N && y < M)
        c[y * N + x] = acc;
}

int main() {
    const int n = 1 << 10;
    const size_t bytes = (n * n) * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    for (int i = 0; i < n * n; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(cdiv(n, TILE_WIDTH), cdiv(n, TILE_WIDTH));
    // Warm-up launch to pay one-time JIT/context costs before timing.
    matmul<<<grid, block>>>(d_a, d_b, d_c, n, n, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time the kernel with CUDA events, averaged over several iterations.
    const int iters = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        matmul<<<grid, block>>>(d_a, d_b, d_c, n, n, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());


    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double ms_per_iter = total_ms / iters;
    // Two reads (a, b) + one write (c) = 3 * bytes moved per launch.
    const double gbytes_per_sec = (3.0 * bytes) / (ms_per_iter * 1.0e6);
    printf("%s: %.4f ms/launch, %.1f GB/s effective bandwidth\n",
           __FILE__, ms_per_iter, gbytes_per_sec);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < n * n; ++i) {
        int cx = i % n, cy = i / n;
        float c = 0.0f;
        for (int d = 0; d < n; d++) {
            c += h_a[cy * n + d] * h_b[d * n + cx];
        }
        max_err = fmax(max_err, fabs(h_c[i] - c));
    }
    printf("%s: %dx%d elements, max error = %g\n", __FILE__, n, n, max_err);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return max_err <= 1e-4 ? EXIT_SUCCESS : EXIT_FAILURE;
}
