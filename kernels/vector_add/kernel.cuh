// Elementwise vector add: c = a + b. The "hello world" kernel, kept because it
// is the cleanest place to demonstrate the measurement setup itself:
// CUDA-event timing, an effective-bandwidth model, and the cache effect below.
//
// Purely memory bound: 3 * n * sizeof(float) bytes moved, 1 FLOP per element.
// Note that at small n the whole working set fits in L2 (48 MB on Ada), so the
// reported "bandwidth" measures L2, not HBM, and can exceed the memory clock's
// theoretical limit. Size the problem past L2 to measure HBM.
#pragma once

#include "../_common/cuda_check.cuh"

__global__ void vector_add_kernel(const float *__restrict__ a,
                                  const float *__restrict__ b,
                                  float *__restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

inline void launch_vector_add(const float *a, const float *b, float *c, int n,
                              cudaStream_t stream = 0) {
    constexpr int kBlock = 256;
    vector_add_kernel<<<cdiv(n, kBlock), kBlock, 0, stream>>>(a, b, c, n);
}
