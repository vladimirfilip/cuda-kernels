// Fused RMSNorm + residual add — shared device code and host utilities.
//
// Pre-norm transformer block pattern, done in one kernel instead of a separate
// elementwise-add kernel + a norm kernel + the extra global-memory round trip:
//
//     h   = residual + x                                   // h is the NEW residual stream; it must be written out
//     out = h * rsqrt(mean(h^2, axis=-1) + eps) * weight   // RMSNorm over the hidden dim
//
//   x, residual : [N, H]  row-major, row = token, reduction axis = last (H)
//   weight      : [H]
//   h, out      : [N, H]  h = updated residual, out = normalized input to the next sublayer
//
// The sum of squares is ALWAYS accumulated in fp32, whatever the IO dtype.
//
// This kernel is memory bound. Traffic lower bound is 4 * N * H * sizeof(T)
// (read x + residual, write h + out; weight is negligible). The roofline goal is
// to approach HBM bandwidth (~1.7 TB/s class on an RTX 5090 / sm_120).
//
// v0 below is deliberately naive (one thread per row, scalar loop, no shared
// memory, no vectorization) so there is a correct kernel to compile, test and
// profile from the first hour. Add v1.. alongside it and log each rung in
// docs/day1_rmsnorm.md.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__host__ __device__ inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// --- dtype <-> fp32 bridges, usable from host and device -----------------------
// float is identity; __nv_bfloat16 goes through the header's conversion operators
// (host + device in CUDA 12.x).
template <typename T> __host__ __device__ inline float to_f32(T v);
template <typename T> __host__ __device__ inline T from_f32(float v);

template <> __host__ __device__ inline float to_f32<float>(float v) { return v; }
template <> __host__ __device__ inline float from_f32<float>(float v) { return v; }

// Named intrinsics (not the cast operators) so this still compiles under the
// PyTorch build, which defines -D__CUDA_NO_BFLOAT16_CONVERSIONS__.
template <> __host__ __device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <> __host__ __device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// --- v0: one thread per row, scalar, un-coalesced (the baseline) --------------
template <typename T>
__global__ void rmsnorm_fused_v0(const T *__restrict__ x,
                                 const T *__restrict__ residual,
                                 const T *__restrict__ weight,
                                 T *__restrict__ h, T *__restrict__ out,
                                 int N, int H, float eps) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;

    const T *xr = x + static_cast<size_t>(row) * H;
    const T *rr = residual + static_cast<size_t>(row) * H;
    T *hr = h + static_cast<size_t>(row) * H;
    T *or_ = out + static_cast<size_t>(row) * H;

    float ss = 0.0f;
    for (int j = 0; j < H; ++j) {
        float hv = to_f32<T>(xr[j]) + to_f32<T>(rr[j]);
        hr[j] = from_f32<T>(hv);
        ss += hv * hv;
    }

    const float inv_rms = rsqrtf(ss / static_cast<float>(H) + eps);
    for (int j = 0; j < H; ++j) {
        float hv = to_f32<T>(hr[j]);
        or_[j] = from_f32<T>(hv * inv_rms * to_f32<T>(weight[j]));
    }
}

// v1: one warp per row for coalesced reads and writes to memory.
template <typename T>
__global__ void rmsnorm_fused_v1(const T *__restrict__ x,
                                 const T *__restrict__ residual,
                                 const T *__restrict__ weight,
                                 T *__restrict__ h, T *__restrict__ out,
                                 int N, int H, float eps)
{
    const int lane = threadIdx.x % 32;
    const int row = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    if (row >= N) return;

    const T *xr = x + static_cast<size_t>(row) * H;
    const T *rr = residual + static_cast<size_t>(row) * H;
    T *hr = h + static_cast<size_t>(row) * H;
    T *or_ = out + static_cast<size_t>(row) * H;

    float sum = 0.0f;
    for (int i = lane; i < H; i += 32) {
        float hv = to_f32<T>(xr[i]) + to_f32<T>(rr[i]);
        hr[i] = from_f32<T>(hv);
        sum += hv * hv;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(__activemask(), sum, offset);
    sum = __shfl_sync(__activemask(), sum, 0);

    const float inv_rms = rsqrtf(sum / static_cast<float>(H) + eps);
    for (int i = lane; i < H; i += 32) {
        float hv = to_f32<T>(hr[i]);
        or_[i] = from_f32<T>(hv * inv_rms * to_f32<T>(weight[i]));
    }
}

// Single place to pick launch config; reused by the standalone binary and the
// PyTorch op so both exercise the same path.
template <typename T>
inline void launch_rmsnorm_fused(const T *x, const T *residual, const T *weight,
                                    T *h, T *out, int N, int H, float eps,
                                    cudaStream_t stream = 0) {
    const int block = 256;            // 8 warps per block => 8 rows per block
    const int grid = cdiv(N, block / 32);
    rmsnorm_fused_v1<T><<<grid, block, 0, stream>>>(x, residual, weight, h, out,
                                                    N, H, eps);
}
