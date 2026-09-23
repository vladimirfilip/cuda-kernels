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
// to approach the reference card's peak DRAM bandwidth; see docs/profiling.md.
//
// Four rungs are kept side by side so the optimization ladder is reproducible
// in one run rather than being copied out of separate builds:
//   v0  one thread per row  -- deliberately naive; un-coalesced baseline
//   v1  one warp per row    -- coalesced; the big win
//   v2  one block per row   -- cross-warp reduction; helps fp32, costs bf16
//   v3  v2 + 128-bit vectorized loads + register-cached `h` -- see the
//       comment above rmsnorm_fused_v3 for both halves of that.
// launch_rmsnorm_fused() below selects between them. Measured numbers for each
// are in this directory's README.md.

#pragma once

#include <cstdint>

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

// --- v3: v2 + 128-bit vectorized loads + a per-thread h cache -----------------
//
// v2's remaining gap (see the rmsnorm_fused README) is that pass 2 re-reads
// `h` from global memory, and each of v0-v2's loads moves only sizeof(T) bytes
// at a time even though every access is already coalesced. v3 fixes both:
//
//   - each thread moves a 128-bit chunk per load/store (float4 for fp32,
//     4x packed bf16-pairs for bf16) instead of one element, so the same
//     coalesced traffic needs fewer, wider transactions;
//   - pass 1's `h` values are kept in a per-thread `cache` array and reused
//     directly in pass 2, so `h` is written once and read from global never.
//     `-Xptxas -v` shows this array as true registers for fp32 but spilled to
//     (L1-backed, still per-thread, still not the shared `h` tensor) local
//     memory for bf16 -- see the README for the register-pressure reason why.
//
// A 128-bit load/store requires the address to be 16-byte aligned. Every row
// starts at row * H * sizeof(T) bytes into whichever buffer the caller passed,
// so alignment needs two things: H * sizeof(T) a multiple of 16, so every
// row's OFFSET from that buffer's start is 16-aligned (equivalently, H a
// multiple of VEC, since VEC * sizeof(T) == 16 by construction for both
// dtypes below); and the buffer's own start address 16-aligned to begin with.
// The first holds structurally whenever H is right; the second does NOT
// follow from "contiguous" -- a PyTorch tensor can be a contiguous view into
// a storage at a non-16-aligned offset (e.g. one element sliced off a flat
// buffer). launch_rmsnorm_fused() checks both and falls back to v2 otherwise:
// this slice's H=4097 test case exercises the first, its misaligned-offset
// test the second.
//
// The h cache is sized for kMaxCacheVecsV3 vectors per thread, enough to fully
// cache any row this repo exercises (H <= 16384 for fp32, <= 32768 for bf16,
// at the block sizes launch_rmsnorm_fused picks). A row wide enough to
// overflow that falls back to re-reading `h` from global for the excess
// vectors only -- still correct, just without the full benefit, the same
// graceful-degradation shape as the fallback above.

// Maps T to a 16-byte vector load/store: how many elements it packs (width),
// the carrier type used for the actual load/store, and how to move between
// that carrier and VEC individual fp32 values.
template <typename T> struct VecTraits;

template <> struct VecTraits<float> {
    static constexpr int width = 4;
    using vec_t = float4;
    __device__ static void unpack(const vec_t &v, float out[4]) {
        out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
    }
    __device__ static vec_t pack(const float in[4]) {
        return make_float4(in[0], in[1], in[2], in[3]);
    }
};

template <> struct VecTraits<__nv_bfloat16> {
    static constexpr int width = 8;
    // float4 is just a convenient 16-byte carrier here; its lanes are
    // reinterpreted as 4 packed bf16 pairs, not as floats.
    using vec_t = float4;
    __device__ static void unpack(const vec_t &v, float out[8]) {
        const __nv_bfloat162 *p = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __bfloat1622float2(p[i]);
            out[2 * i] = f.x;
            out[2 * i + 1] = f.y;
        }
    }
    __device__ static vec_t pack(const float in[8]) {
        __nv_bfloat162 packed[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            packed[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
        return *reinterpret_cast<const vec_t *>(packed);
    }
};

constexpr int kMaxCacheVecsV3 = 4;

template <typename T>
__global__ void rmsnorm_fused_v3(const T *__restrict__ x,
                                 const T *__restrict__ residual,
                                 const T *__restrict__ weight,
                                 T *__restrict__ h, T *__restrict__ out,
                                 int N, int H, float eps)
{
    using VT = VecTraits<T>;
    using VecT = typename VT::vec_t;
    constexpr int VEC = VT::width;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x;
    if (row >= N) return;

    // H % VEC == 0 is the caller's contract (see the comment above), so nvec
    // is exact -- no scalar tail to special-case.
    const int nvec = H / VEC;

    const VecT *xv = reinterpret_cast<const VecT *>(x + static_cast<size_t>(row) * H);
    const VecT *rv = reinterpret_cast<const VecT *>(residual + static_cast<size_t>(row) * H);
    VecT *hv = reinterpret_cast<VecT *>(h + static_cast<size_t>(row) * H);
    VecT *ov = reinterpret_cast<VecT *>(out + static_cast<size_t>(row) * H);
    const VecT *wv = reinterpret_cast<const VecT *>(weight);

    float cache[kMaxCacheVecsV3][VEC];

    float sum = 0.0f;
    int slot = 0;
    for (int v = threadIdx.x; v < nvec; v += blockDim.x, ++slot) {
        float xf[VEC], rf[VEC];
        VT::unpack(xv[v], xf);
        VT::unpack(rv[v], rf);
        float hf[VEC];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            hf[i] = xf[i] + rf[i];
            sum += hf[i] * hf[i];
        }
        if (slot < kMaxCacheVecsV3) {
#pragma unroll
            for (int i = 0; i < VEC; ++i) cache[slot][i] = hf[i];
        }
        hv[v] = VT::pack(hf);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    __shared__ float warp_sums[32];

    if (lane == 0) warp_sums[warp] = sum;

    __syncthreads();

    if (warp == 0) {
        int warps = blockDim.x / 32;
        float block_sum = lane < warps ? warp_sums[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1)
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);

        if (lane == 0) warp_sums[0] = block_sum;
    }

    __syncthreads();

    const float inv_rms = rsqrtf(warp_sums[0] / static_cast<float>(H) + eps);

    slot = 0;
    for (int v = threadIdx.x; v < nvec; v += blockDim.x, ++slot) {
        float hf[VEC];
        if (slot < kMaxCacheVecsV3) {
#pragma unroll
            for (int i = 0; i < VEC; ++i) hf[i] = cache[slot][i];
        } else {
            // Beyond the register cache's capacity: re-read from global, same
            // as v2 does for every element. Only reachable for rows wider
            // than this rung's documented ceiling.
            VT::unpack(hv[v], hf);
        }
        float wf[VEC];
        VT::unpack(wv[v], wf);
        float of[VEC];
#pragma unroll
        for (int i = 0; i < VEC; ++i) of[i] = hf[i] * inv_rms * wf[i];
        ov[v] = VT::pack(of);
    }
}

// Which rung of the optimization ladder to launch. Kept selectable so the
// driver can print the whole ladder in one run and the write-up's numbers stay
// reproducible instead of being copied from an old build.
enum class RmsNormVariant { kV0Thread, kV1Warp, kV2Block, kV3Vector };

inline const char *variant_name(RmsNormVariant v) {
    switch (v) {
        case RmsNormVariant::kV0Thread: return "v0-thread";
        case RmsNormVariant::kV1Warp:   return "v1-warp";
        case RmsNormVariant::kV2Block:  return "v2-block";
        case RmsNormVariant::kV3Vector: return "v3-vector";
    }
    return "?";
}

// Single place to pick launch config; reused by the standalone binary and the
// PyTorch op so both exercise the same path.
template <typename T>
inline void launch_rmsnorm_fused(const T *x, const T *residual, const T *weight,
                                 T *h, T *out, int N, int H, float eps,
                                 cudaStream_t stream = 0,
                                 RmsNormVariant variant = RmsNormVariant::kV3Vector) {
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
        case RmsNormVariant::kV3Vector: {
            using VT = VecTraits<T>;
            constexpr int VEC = VT::width;
            // v3's 128-bit loads need every row 16-byte aligned. H % VEC == 0
            // makes the row STRIDE a multiple of 16, which is sufficient when
            // the base pointer is itself 16-byte aligned -- true for the
            // standalone driver's cudaMalloc buffers, but not guaranteed for
            // a PyTorch tensor: a contiguous view can still start at a
            // non-16-aligned storage offset (e.g. a 1-element slice off a
            // flat buffer), and .contiguous() does not fix that up. So both
            // halves of the precondition are checked explicitly; either
            // failing falls back to the next-best rung, which has neither
            // requirement.
            const bool aligned =
                reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(residual) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(weight) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(h) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(out) % 16 == 0;
            if (H % VEC != 0 || !aligned) {
                launch_rmsnorm_fused<T>(x, residual, weight, h, out, N, H, eps,
                                        stream, RmsNormVariant::kV2Block);
                break;
            }
            // Same block-sizing logic as v2, but over nvec = H / VEC: each
            // thread should own one vector, not one element.
            const int nvec = H / VEC;
            const int block = min(1024, max(32, 32 * cdiv(nvec, 32)));
            rmsnorm_fused_v3<T><<<N, block, 0, stream>>>(
                x, residual, weight, h, out, N, H, eps);
            break;
        }
    }
}
