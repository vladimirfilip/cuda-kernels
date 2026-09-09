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
// to approach HBM bandwidth (504 GB/s on the RTX 4070 Ti / sm_89 reference box;
// see docs/profiling.md).
//
// Three rungs are kept side by side so the optimization ladder is reproducible
// in one run rather than being copied out of three separate builds:
//   v0  one thread per row  -- deliberately naive; un-coalesced baseline
//   v1  one warp per row    -- coalesced; the big win
//   v2  one block per row   -- cross-warp reduction; helps fp32, costs bf16
// launch_rmsnorm_fused() below selects between them. Measured numbers for each
// are in this directory's README.md.

#pragma once

#include <cuda_bf16.h>

#include "../_common/cuda_check.cuh"

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
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    sum = __shfl_sync(__activemask(), sum, 0);

    const float inv_rms = rsqrtf(sum / static_cast<float>(H) + eps);
    for (int i = lane; i < H; i += 32) {
        float hv = to_f32<T>(hr[i]);
        or_[i] = from_f32<T>(hv * inv_rms * to_f32<T>(weight[i]));
    }
}

// v2: one block per row for coalesced reads and writes to memory.
template <typename T>
__global__ void rmsnorm_fused_v2(const T *__restrict__ x,
                                 const T *__restrict__ residual,
                                 const T *__restrict__ weight,
                                 T *__restrict__ h, T *__restrict__ out,
                                 int N, int H, float eps)
{
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x;
    if (row >= N) return;

    const T *xr = x + static_cast<size_t>(row) * H;
    const T *rr = residual + static_cast<size_t>(row) * H;
    T *hr = h + static_cast<size_t>(row) * H;
    T *or_ = out + static_cast<size_t>(row) * H;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float hv = to_f32<T>(xr[i]) + to_f32<T>(rr[i]);
        hr[i] = from_f32<T>(hv);
        sum += hv * hv;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    __shared__ float warp_sums[32];

    if (lane == 0) warp_sums[warp] = sum;

    __syncthreads();

    float block_sum = 0.0f;

    if (warp == 0) {
        int warps = blockDim.x / 32;

        block_sum = lane < warps ? warp_sums[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1)
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);

        if (lane == 0) {
            warp_sums[0] = block_sum;
        }
    }

    __syncthreads();

    const float inv_rms = rsqrtf(warp_sums[0] / static_cast<float>(H) + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float hv = to_f32<T>(hr[i]);
        or_[i] = from_f32<T>(hv * inv_rms * to_f32<T>(weight[i]));
    }
}


// Which rung of the optimization ladder to launch. Kept selectable so the
// driver can print the whole ladder in one run and the write-up's numbers stay
// reproducible instead of being copied from an old build.
enum class RmsNormVariant { kV0Thread, kV1Warp, kV2Block };

inline const char *variant_name(RmsNormVariant v) {
    switch (v) {
        case RmsNormVariant::kV0Thread: return "v0-thread";
        case RmsNormVariant::kV1Warp:   return "v1-warp";
        case RmsNormVariant::kV2Block:  return "v2-block";
    }
    return "?";
}

// Single place to pick launch config; reused by the standalone binary and the
// PyTorch op so both exercise the same path.
template <typename T>
inline void launch_rmsnorm_fused(const T *x, const T *residual, const T *weight,
                                 T *h, T *out, int N, int H, float eps,
                                 cudaStream_t stream = 0,
                                 RmsNormVariant variant = RmsNormVariant::kV2Block) {
    switch (variant) {
        case RmsNormVariant::kV0Thread: {
            // One thread per row: adjacent threads are H elements apart, so
            // every load is a separate memory transaction.
            constexpr int kBlock = 256;
            rmsnorm_fused_v0<T><<<cdiv(N, kBlock), kBlock, 0, stream>>>(
                x, residual, weight, h, out, N, H, eps);
            break;
        }
        case RmsNormVariant::kV1Warp: {
            // One warp per row: the 32 lanes walk the row together, so loads
            // coalesce. 8 warps per block => 8 rows per block.
            constexpr int kBlock = 256;
            rmsnorm_fused_v1<T><<<cdiv(N, kBlock / 32), kBlock, 0, stream>>>(
                x, residual, weight, h, out, N, H, eps);
            break;
        }
        case RmsNormVariant::kV2Block: {
            // One block per row. Size the block from H (the reduction width),
            // not from N: H is what the threads have to cover. Round to a whole
            // number of warps and cap at 1024 so warp_sums[32] is always large
            // enough for blockDim.x / 32 partial sums.
            const int block = min(1024, max(32, 32 * cdiv(H, 32)));
            rmsnorm_fused_v2<T><<<N, block, 0, stream>>>(
                x, residual, weight, h, out, N, H, eps);
            break;
        }
    }
}
