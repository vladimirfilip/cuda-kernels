// Standalone driver for the matmul kernels: C = A @ B, fp32, row-major.
//
//   make run  KERNEL=matmul GPU=0                 # build, self-check, self-benchmark
//   make ncu  KERNEL=matmul NCU_SET=full          # per-kernel hardware profile
//   ./bin/matmul [M] [K] [N] [iters] [peak_gflops]
//
// Runs the naive and the shared-memory-tiled variant over the SAME inputs,
// checks each against a CPU reference, and reports ms/launch + GFLOP/s. Exits
// non-zero if either variant is outside tolerance, so `make run` is a smoke test.
//
// Dimensions default to non-square on purpose: a square problem hides indexing
// bugs, because a [K,N] column stride of N and of K are then the same number.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kernel.cuh"

namespace {

constexpr int kDefaultM = 1024;
constexpr int kDefaultK = 768;
constexpr int kDefaultN = 512;
constexpr int kDefaultIters = 100;
// RTX 4070 Ti fp32 vector peak: 7680 CUDA cores * 2 FLOP/clk * ~2.61 GHz boost.
// Override via argv[5] on other hardware.
constexpr double kDefaultPeakGflops = 40100.0;

// Row-major CPU reference in fp64, so the tolerance check is not comparing
// one fp32 rounding order against another.
void matmul_cpu(const std::vector<float> &a, const std::vector<float> &b,
                std::vector<double> &c, int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)a[i * K + k] * b[k * N + j];
            c[i * N + j] = acc;
        }
}

struct Result { double ms; double max_rel_err; };

Result run_variant(void (*launch)(const float *, const float *, float *, int,
                                  int, int, cudaStream_t),
                   const float *d_a, const float *d_b, float *d_c,
                   const std::vector<double> &ref, int M, int K, int N,
                   int iters) {
    // Warm-up launch pays one-time context/JIT costs before timing.
    launch(d_a, d_b, d_c, M, K, N, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) launch(d_a, d_b, d_c, M, K, N, 0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    std::vector<float> h_c((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double worst = 0.0;
    for (size_t i = 0; i < h_c.size(); ++i) {
        double denom = std::fabs(ref[i]) > 1.0 ? std::fabs(ref[i]) : 1.0;
        worst = std::fmax(worst, std::fabs(h_c[i] - ref[i]) / denom);
    }
    return {total_ms / iters, worst};
}

}  // namespace

int main(int argc, char **argv) {
    const int M = argc > 1 ? atoi(argv[1]) : kDefaultM;
    const int K = argc > 2 ? atoi(argv[2]) : kDefaultK;
    const int N = argc > 3 ? atoi(argv[3]) : kDefaultN;
    const int iters = argc > 4 ? atoi(argv[4]) : kDefaultIters;
    const double peak = argc > 5 ? atof(argv[5]) : kDefaultPeakGflops;

    printf("matmul: M=%d K=%d N=%d iters=%d\n", M, K, N, iters);

    std::vector<float> h_a((size_t)M * K), h_b((size_t)K * N);
    srand(0);
    for (auto &v : h_a) v = (float)rand() / RAND_MAX;
    for (auto &v : h_b) v = (float)rand() / RAND_MAX;

    std::vector<double> ref((size_t)M * N);
    matmul_cpu(h_a, h_b, ref, M, K, N);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // 2 FLOP (one multiply, one add) per element of the K-length inner product.
    const double flops = 2.0 * M * N * K;
    const double kTol = 1e-4;
    bool ok = true;

    struct { const char *name; void (*fn)(const float *, const float *, float *,
                                          int, int, int, cudaStream_t); } variants[] = {
        {"naive", launch_matmul_naive},
        {"tiled", launch_matmul_tiled},
    };

    double naive_ms = 0.0;
    for (auto &v : variants) {
        Result r = run_variant(v.fn, d_a, d_b, d_c, ref, M, K, N, iters);
        const double gflops = flops / (r.ms * 1.0e6);
        if (naive_ms == 0.0) naive_ms = r.ms;
        printf("[%-5s] %.4f ms/launch  %7.1f GFLOP/s  (%.1f%% of %.0f peak)  %.2fx vs naive\n",
               v.name, r.ms, gflops, 100.0 * gflops / peak, peak, naive_ms / r.ms);
        printf("[%-5s] max rel err = %.2e  (tol %.0e)  %s\n", v.name,
               r.max_rel_err, kTol, r.max_rel_err <= kTol ? "OK" : "FAIL");
        ok = ok && r.max_rel_err <= kTol;
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
