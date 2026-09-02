// Standalone driver for the fused RMSNorm + residual-add kernel.
//
//   make run  KERNEL=rmsnorm_fused GPU=0            # build, self-check, self-benchmark
//   make ncu  KERNEL=rmsnorm_fused NCU_SET=full     # per-kernel hardware profile
//   make nsys KERNEL=rmsnorm_fused                  # timeline
//   ./bin/rmsnorm_fused [N] [H] [iters] [peak_gbps]
//
// Runs the op for fp32 and bf16, checks each against an fp64 CPU reference, and
// reports ms/launch + effective GB/s + % of peak HBM bandwidth. Exits non-zero
// if either dtype is outside tolerance, so `make run` doubles as a smoke test.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rmsnorm_fused.cuh"

namespace {

// Default problem size: Llama-3.2-1B hidden width, a decode-ish token count.
constexpr int kDefaultN = 4096;
constexpr int kDefaultH = 2048;
constexpr int kDefaultIters = 200;
// RTX 5090 spec HBM bandwidth (~1.79 TB/s); override via argv[4].
constexpr double kDefaultPeakGbps = 1790.0;

// Worst assert_close-style violation ratio: max |got - ref| / (atol + rtol|ref|).
// <= 1.0 means every element is within tolerance. Using a combined atol+rtol
// bound (not pure relative error) keeps the metric meaningful for h, which sees
// catastrophic cancellation and can be ~0.
template <typename T>
double worst_ratio(const std::vector<T> &got, const std::vector<double> &ref,
                   double atol, double rtol) {
    double m = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = static_cast<double>(to_f32<T>(got[i]));
        m = std::max(m, std::fabs(g - ref[i]) / (atol + rtol * std::fabs(ref[i])));
    }
    return m;
}

// fp64 reference: h = x + residual, out = h * rsqrt(mean(h^2) + eps) * weight.
void rmsnorm_ref(const std::vector<double> &x, const std::vector<double> &res,
                 const std::vector<double> &w, int N, int H, double eps,
                 std::vector<double> &h, std::vector<double> &out) {
    for (int r = 0; r < N; ++r) {
        double ss = 0.0;
        for (int j = 0; j < H; ++j) {
            const double hv = x[(size_t)r * H + j] + res[(size_t)r * H + j];
            h[(size_t)r * H + j] = hv;
            ss += hv * hv;
        }
        const double inv = 1.0 / std::sqrt(ss / H + eps);
        for (int j = 0; j < H; ++j)
            out[(size_t)r * H + j] = h[(size_t)r * H + j] * inv * w[j];
    }
}

template <typename T>
int run_dtype(const char *tag, const std::vector<double> &hx,
              const std::vector<double> &hres, const std::vector<double> &hw,
              int N, int H, float eps, int iters, double peak_gbps, double atol,
              double rtol) {
    const size_t n = (size_t)N * H;
    std::vector<T> hx_t(n), hres_t(n), hw_t(H), hh_t(n), hout_t(n);
    // Quantize inputs to T, then build the reference from those same quantized
    // values (round-tripped through double). This measures the kernel's own
    // error, not the unavoidable error of feeding it bf16 inputs.
    std::vector<double> qx(n), qres(n), qw(H), ref_h(n), ref_out(n);
    for (size_t i = 0; i < n; ++i) {
        hx_t[i] = from_f32<T>((float)hx[i]);
        hres_t[i] = from_f32<T>((float)hres[i]);
        qx[i] = to_f32<T>(hx_t[i]);
        qres[i] = to_f32<T>(hres_t[i]);
    }
    for (int j = 0; j < H; ++j) {
        hw_t[j] = from_f32<T>((float)hw[j]);
        qw[j] = to_f32<T>(hw_t[j]);
    }
    rmsnorm_ref(qx, qres, qw, N, H, eps, ref_h, ref_out);

    T *dx, *dres, *dw, *dh, *dout;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dres, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dw, H * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dh, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dout, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dx, hx_t.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dres, hres_t.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, hw_t.data(), H * sizeof(T), cudaMemcpyHostToDevice));

    // Warm-up (pays one-time JIT/context costs; profile with --launch-skip 1).
    launch_rmsnorm_fused<T>(dx, dres, dw, dh, dout, N, H, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it)
        launch_rmsnorm_fused<T>(dx, dres, dw, dh, dout, N, H, eps);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double ms = total_ms / iters;
    const double bytes = 4.0 * n * sizeof(T); // read x + residual, write h + out
    const double gbps = bytes / (ms * 1.0e6);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(hh_t.data(), dh, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hout_t.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dres));
    CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(dh));
    CUDA_CHECK(cudaFree(dout));

    const double r_h = worst_ratio<T>(hh_t, ref_h, atol, rtol);
    const double r_out = worst_ratio<T>(hout_t, ref_out, atol, rtol);
    const bool ok = r_h <= 1.0 && r_out <= 1.0;

    printf("[%-4s] %.4f ms/launch  %.1f GB/s  (%.1f%% of %.0f GB/s peak)\n", tag,
           ms, gbps, 100.0 * gbps / peak_gbps, peak_gbps);
    printf("[%-4s] tol ratio (<=1 ok): h=%.2f out=%.2f  (atol=%.1e rtol=%.1e)  %s\n",
           tag, r_h, r_out, atol, rtol, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : kDefaultN;
    const int H = argc > 2 ? std::atoi(argv[2]) : kDefaultH;
    const int iters = argc > 3 ? std::atoi(argv[3]) : kDefaultIters;
    const double peak_gbps = argc > 4 ? std::atof(argv[4]) : kDefaultPeakGbps;
    const float eps = 1e-5f;
    printf("rmsnorm_fused: N=%d H=%d iters=%d\n", N, H, iters);

    const size_t n = (size_t)N * H;
    std::vector<double> hx(n), hres(n), hw(H);
    srand(0);
    for (size_t i = 0; i < n; ++i) {
        hx[i] = (double)rand() / RAND_MAX - 0.5;
        hres[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (int j = 0; j < H; ++j)
        hw[j] = 0.5 + (double)rand() / RAND_MAX;

    int rc = 0;
    rc |= run_dtype<float>("fp32", hx, hres, hw, N, H, eps, iters, peak_gbps,
                           1e-4, 1e-4);
    rc |= run_dtype<__nv_bfloat16>("bf16", hx, hres, hw, N, H, eps, iters,
                                   peak_gbps, 5e-3, 2e-2);
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
