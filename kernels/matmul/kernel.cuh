// Single-precision matmul: C = A @ B, row-major.
//   A : [M, K]   B : [K, N]   C : [M, N]
//
// Four variants, sharing one launch surface so the standalone driver and the
// PyTorch binding exercise exactly the same device code:
//
//   v0 "naive"    : one thread per output element, K global loads per thread.
//                   Every thread in a warp re-reads the same column of B, so
//                   the kernel is bound by redundant global traffic, not by
//                   FLOPs.
//   v1 "tiled"    : TILE_WIDTH x TILE_WIDTH shared-memory tiling. Each B
//                   element is loaded once per tile instead of once per
//                   thread, cutting global traffic by a factor of TILE_WIDTH.
//   v2 "register" : v1's tile, plus each thread computes a TM x TN patch of C
//                   instead of one element. Every value pulled from shared
//                   memory now feeds TM (or TN) FMAs instead of one, so the
//                   kernel moves from shared-memory-bandwidth-bound toward
//                   FMA-bound. See the comment above matmul_v2_kernel.
//   v3 "wide"     : v2's idea at a bigger tile (8x8 per thread instead of
//                   4x4), with the shared-memory reads that feed each phase's
//                   FMAs done as float4 loads instead of scalar ones. See the
//                   comment above matmul_v3_kernel.
#pragma once

#include "../_common/cuda_check.cuh"

constexpr int TILE_WIDTH = 16;

// v2 tile shape: a block computes a BM x BN patch of C, reducing BK columns
// of A / rows of B per phase. Each of the block's threads owns a TM x TN
// patch of that output tile, so THREADS_V2 = (BM/TM) * (BN/TN).
constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
constexpr int THREADS_V2 = (BM / TM) * (BN / TN);  // 256
// Elements each thread stages per phase, filling the two shared tiles with
// THREADS_V2 threads. BM*BK and BK*BN both divide THREADS_V2 for this choice
// of tile shape -- static_assert rather than a runtime remainder check, since
// this is a compile-time tuning choice, not something callers can vary.
static_assert((BM * BK) % THREADS_V2 == 0, "BM*BK must divide THREADS_V2");
static_assert((BK * BN) % THREADS_V2 == 0, "BK*BN must divide THREADS_V2");
constexpr int LOADS_A_V2 = (BM * BK) / THREADS_V2;
constexpr int LOADS_B_V2 = (BK * BN) / THREADS_V2;

// v3 tile shape: same idea as v2's, doubled in every dimension (128x128 output
// tile, 8x8 per thread). THREADS_V3 stays 256 -- a bigger tile means fewer,
// bigger blocks, not more threads per block.
constexpr int BM3 = 128, BN3 = 128, BK3 = 8, TM3 = 8, TN3 = 8;
constexpr int THREADS_V3 = (BM3 / TM3) * (BN3 / TN3);  // 256
static_assert((BM3 * BK3) % THREADS_V3 == 0, "BM3*BK3 must divide THREADS_V3");
static_assert((BK3 * BN3) % THREADS_V3 == 0, "BK3*BN3 must divide THREADS_V3");
constexpr int LOADS_A_V3 = (BM3 * BK3) / THREADS_V3;
constexpr int LOADS_B_V3 = (BK3 * BN3) / THREADS_V3;
// The compute loop below reads TM3 (or TN3) values per phase-step as float4s;
// both need to be a multiple of 4 for that split to cover them exactly.
static_assert(TM3 % 4 == 0 && TN3 % 4 == 0, "TM3, TN3 must be multiples of 4");

// v0: one thread per output element.
__global__ void matmul_naive_kernel(const float *__restrict__ a,
                                    const float *__restrict__ b,
                                    float *__restrict__ c, int M, int K, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column of C, in [0, N)
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row of C,    in [0, M)
    if (x < N && y < M) {
        float acc = 0.0f;
        // b is [K, N], so stepping down a column of b strides by N, not K.
        for (int k = 0; k < K; k++) acc += a[y * K + k] * b[k * N + x];
        c[y * N + x] = acc;
    }
}

// v1: shared-memory tiling.
__global__ void matmul_tiled_kernel(const float *__restrict__ a,
                                    const float *__restrict__ b,
                                    float *__restrict__ c, int M, int K, int N) {
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
        // Out-of-range lanes stage a zero so the inner product stays correct
        // for M, N, K that are not multiples of TILE_WIDTH.
        As[ty][tx] = (y < M && _x < K) ? a[y * K + _x] : 0.0f;
        Bs[ty][tx] = (_y < K && x < N) ? b[_y * N + x] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) acc += As[ty][k] * Bs[k][tx];

        // Guard the next iteration's writes against threads still reading.
        __syncthreads();
    }
    if (x < N && y < M) c[y * N + x] = acc;
}

// v2: shared-memory tiling + per-thread register blocking.
//
// v1 gives every thread one FMA per two shared-memory reads (one from As, one
// from Bs). Shared memory has far less bandwidth than the register file or
// the FMA units, so v1 is bound by those two reads, not by arithmetic. v2
// keeps the same shared-memory tile but has each thread own a TM x TN patch
// of the output: it stages TM values from As and TN from Bs into registers
// (TM + TN reads), then does TM * TN FMAs against them, so the read:FMA ratio
// goes from 1:0.5 to 1:(TM*TN/(TM+TN)) = 1:2 at TM=TN=4. The same shared-
// memory traffic now feeds 8x the arithmetic.
//
// Loading the BM x BK / BK x BN tiles is a separate, smaller-grained pass:
// with THREADS_V2 < BM*BK, each thread stages LOADS_A_V2 elements of As
// (and LOADS_B_V2 of Bs) per phase, one contiguous run per thread so the
// global loads across a warp stay coalesced.
__global__ void matmul_v2_kernel(const float *__restrict__ a,
                                 const float *__restrict__ b,
                                 float *__restrict__ c, int M, int K, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int tid = threadIdx.x;              // linear thread id, [0, THREADS_V2)
    const int thread_row = tid / (BN / TN);    // this thread's patch, in [0, BM/TM)
    const int thread_col = tid % (BN / TN);    // this thread's patch, in [0, BN/TN)

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Stage A's [BM, BK] tile. Each thread's LOADS_A_V2 elements are
        // consecutive in the tile's row-major linear index, so consecutive
        // threads touch consecutive addresses -- coalesced despite the
        // per-thread loop.
#pragma unroll
        for (int i = 0; i < LOADS_A_V2; ++i) {
            const int idx = tid * LOADS_A_V2 + i;
            const int r = idx / BK, cc = idx % BK;
            const int gr = block_row + r, gc = k0 + cc;
            As[r][cc] = (gr < M && gc < K) ? a[(size_t)gr * K + gc] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < LOADS_B_V2; ++i) {
            const int idx = tid * LOADS_B_V2 + i;
            const int r = idx / BN, cc = idx % BN;
            const int gr = k0 + r, gc = block_col + cc;
            Bs[r][cc] = (gr < K && gc < N) ? b[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float regM[TM], regN[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[thread_row * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[kk][thread_col * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        // Guard the next phase's writes against threads still reading.
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = block_row + thread_row * TM + i;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = block_col + thread_col * TN + j;
            if (gc < N) c[(size_t)gr * N + gc] = acc[i][j];
        }
    }
}

// v3: v2's idea, doubled tile, vectorized shared-memory reads.
//
// v2's own remaining gap (see the matmul README): a 4x4 register tile gives
// each thread 16 FMAs per 8 shared-memory reads -- better than v1's 1 FMA per
// 2 reads, but each of those 8 reads is still a separate scalar instruction.
// v3 widens the per-thread tile to 8x8 (64 FMAs per phase-step, same 1:2
// read:FMA ratio as v2 since read count scales with FMA count too, but now
// spread over far fewer, far bigger blocks) and reads regM/regN as float4s
// instead of one float at a time, so the same 16 values arrive in 4 wide
// instructions instead of 16 scalar ones.
//
// float4 needs the address 16-byte aligned. Bs[kk][thread_col*TN3 + i] is:
// Bs is [BK3][BN3] row-major, thread_col*TN3 is always a multiple of TN3=8
// floats (32 bytes), and each Bs row is BN3=128 floats (512 bytes) from the
// last, both multiples of 16 -- so every float4 read lands on a 16-byte
// boundary regardless of M, K, N. That only works because BN3 and TN3 are
// compile-time constants this kernel controls; it says nothing about the
// caller's A, B, C, which is why the vectorizing stops here rather than
// reaching into the global loads below (see the README for the trade-off).
//
// Reading a TM3-wide run of A the same way needs A's shared-memory tile
// TRANSPOSED: the natural [BM3][BK3] layout has BK3 (not BM3) as the fast
// axis, so TM3 consecutive rows at a fixed k are BK3 floats apart, not
// contiguous. Staging A into As_t[BK3][BM3] instead -- writing element (r, cc)
// of the logical BM3 x BK3 tile to As_t[cc][r] -- makes a fixed k's row of
// As_t exactly as contiguous as Bs's, at the cost of one extra index swap in
// the staging loop below. The global read pattern doesn't change, only where
// each value lands in shared memory.
__global__ void matmul_v3_kernel(const float *__restrict__ a,
                                 const float *__restrict__ b,
                                 float *__restrict__ c, int M, int K, int N) {
    __shared__ __align__(16) float As_t[BK3][BM3];  // transposed: [k][m]
    __shared__ __align__(16) float Bs[BK3][BN3];

    const int block_row = blockIdx.y * BM3;
    const int block_col = blockIdx.x * BN3;

    const int tid = threadIdx.x;
    const int thread_row = tid / (BN3 / TN3);
    const int thread_col = tid % (BN3 / TN3);

    float acc[TM3][TN3] = {};

    for (int k0 = 0; k0 < K; k0 += BK3) {
        // Global reads stay scalar and bounds-checked, same as v2 -- see the
        // comment above this kernel for why. Only the write side transposes.
#pragma unroll
        for (int i = 0; i < LOADS_A_V3; ++i) {
            const int idx = tid * LOADS_A_V3 + i;
            const int r = idx / BK3, cc = idx % BK3;
            const int gr = block_row + r, gc = k0 + cc;
            As_t[cc][r] = (gr < M && gc < K) ? a[(size_t)gr * K + gc] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < LOADS_B_V3; ++i) {
            const int idx = tid * LOADS_B_V3 + i;
            const int r = idx / BN3, cc = idx % BN3;
            const int gr = k0 + r, gc = block_col + cc;
            Bs[r][cc] = (gr < K && gc < N) ? b[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK3; ++kk) {
            float regM[TM3], regN[TN3];
#pragma unroll
            for (int i = 0; i < TM3; i += 4) {
                float4 v = *reinterpret_cast<const float4 *>(&As_t[kk][thread_row * TM3 + i]);
                regM[i] = v.x; regM[i + 1] = v.y; regM[i + 2] = v.z; regM[i + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < TN3; j += 4) {
                float4 v = *reinterpret_cast<const float4 *>(&Bs[kk][thread_col * TN3 + j]);
                regN[j] = v.x; regN[j + 1] = v.y; regN[j + 2] = v.z; regN[j + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM3; ++i)
#pragma unroll
                for (int j = 0; j < TN3; ++j) acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    // Output write stays scalar and bounds-checked, same reason as the global
    // reads above: C's row stride is N * sizeof(float), which this kernel
    // does not control and cannot assume is 16-byte aligned.
#pragma unroll
    for (int i = 0; i < TM3; ++i) {
        const int gr = block_row + thread_row * TM3 + i;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < TN3; ++j) {
            const int gc = block_col + thread_col * TN3 + j;
            if (gc < N) c[(size_t)gr * N + gc] = acc[i][j];
        }
    }
}

inline void launch_matmul_naive(const float *a, const float *b, float *c, int M,
                                int K, int N, cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid(cdiv(N, block.x), cdiv(M, block.y));
    matmul_naive_kernel<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}

inline void launch_matmul_tiled(const float *a, const float *b, float *c, int M,
                                int K, int N, cudaStream_t stream = 0) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(cdiv(N, TILE_WIDTH), cdiv(M, TILE_WIDTH));
    matmul_tiled_kernel<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}

inline void launch_matmul_v2(const float *a, const float *b, float *c, int M,
                             int K, int N, cudaStream_t stream = 0) {
    dim3 block(THREADS_V2);
    dim3 grid(cdiv(N, BN), cdiv(M, BM));
    matmul_v2_kernel<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}

inline void launch_matmul_v3(const float *a, const float *b, float *c, int M,
                             int K, int N, cudaStream_t stream = 0) {
    dim3 block(THREADS_V3);
    dim3 grid(cdiv(N, BN3), cdiv(M, BM3));
    matmul_v3_kernel<<<grid, block, 0, stream>>>(a, b, c, M, K, N);
}
