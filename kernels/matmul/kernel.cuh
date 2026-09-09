// Single-precision matmul: C = A @ B, row-major.
//   A : [M, K]   B : [K, N]   C : [M, N]
//
// Two variants, sharing one launch surface so the standalone driver and the
// PyTorch binding exercise exactly the same device code:
//
//   v0 "naive" : one thread per output element, K global loads per thread.
//                Every thread in a warp re-reads the same column of B, so the
//                kernel is bound by redundant global traffic, not by FLOPs.
//   v1 "tiled" : TILE_WIDTH x TILE_WIDTH shared-memory tiling. Each B element
//                is loaded once per tile instead of once per thread, cutting
//                global traffic by a factor of TILE_WIDTH.
#pragma once

#include "../_common/cuda_check.cuh"

constexpr int TILE_WIDTH = 16;

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
