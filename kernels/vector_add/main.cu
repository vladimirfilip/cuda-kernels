// Standalone driver for the elementwise vector add.
//
//   make run  KERNEL=vector_add GPU=0      # sweep sizes across the L2 boundary
//   ./bin/vector_add [peak_gbps] [l2_mb]
//
// This kernel is trivial; the driver is the interesting part. It sweeps the
// working-set size from well inside L2 to well past it, which makes a
// measurement trap visible: at small n the "effective bandwidth" number is
// several times the card's theoretical DRAM bandwidth, because nothing is
// actually reaching DRAM. Only the largest sizes measure what the roofline
// model assumes they measure.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kernel.cuh"

namespace {

// RTX 4070 Ti: 21 Gbps GDDR6X on a 192-bit bus -> 504 GB/s. Override via argv[1].
constexpr double kDefaultPeakGbps = 504.0;
// Ada AD104 L2. Override via argv[2].
constexpr double kDefaultL2Mb = 48.0;
constexpr int kIters = 200;

}  // namespace

int main(int argc, char **argv) {
    const double peak_gbps = argc > 1 ? atof(argv[1]) : kDefaultPeakGbps;
    const double l2_mb = argc > 2 ? atof(argv[2]) : kDefaultL2Mb;

    printf("vector_add: peak %.0f GB/s, L2 %.0f MB, %d iters\n", peak_gbps,
           l2_mb, kIters);
    printf("%12s %10s %10s %9s %8s  %s\n", "n", "MB moved", "ms/launch", "GB/s",
           "%peak", "working set");
    printf("%s\n", "--------------------------------------------------------------------------");

    bool ok = true;
    for (int shift = 18; shift <= 26; shift += 2) {
        const int n = 1 << shift;
        const size_t bytes = (size_t)n * sizeof(float);
        // Two reads (a, b) + one write (c) = 3 * bytes moved per launch.
        const double mb_moved = 3.0 * bytes / 1.0e6;

        std::vector<float> h_a(n), h_b(n), h_c(n);
        for (int i = 0; i < n; ++i) {
            h_a[i] = (float)i;
            h_b[i] = (float)(2 * i);
        }

        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

        // Warm-up launch pays one-time context/JIT costs before timing.
        launch_vector_add(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < kIters; ++it) launch_vector_add(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        const double ms = total_ms / kIters;
        const double gbps = (3.0 * bytes) / (ms * 1.0e6);

        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < n; ++i)
            max_err = fmax(max_err, fabs(h_c[i] - (h_a[i] + h_b[i])));
        ok = ok && max_err == 0.0;

        // The crossover is gradual, not a cliff: a footprint a little over L2
        // still gets most of its traffic served by cache.
        const char *regime = mb_moved < l2_mb        ? "L2-resident (not a DRAM measurement)"
                             : mb_moved < 4 * l2_mb  ? "L2/DRAM transition"
                                                     : "DRAM-bound";
        printf("%12d %10.1f %10.4f %9.1f %7.1f%%  %s\n", n, mb_moved, ms, gbps,
               100.0 * gbps / peak_gbps, regime);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    printf("\ncorrectness: %s\n", ok ? "OK (exact for all sizes)" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
