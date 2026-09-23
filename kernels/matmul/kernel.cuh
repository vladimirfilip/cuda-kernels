// Single-precision matmul: C = A @ B, row-major.
//   A : [M, K]   B : [K, N]   C : [M, N]
//
// Three variants, sharing one launch surface so the standalone driver and the
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
