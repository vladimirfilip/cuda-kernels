"""FlashAttention-2 in Triton: tiled attention with online softmax.

The forward kernel streams over key/value tiles carrying a running max and sum
per query row, so the N x N score matrix is never materialised and intermediate
storage is O(N) rather than O(N^2).

Notation follows the FlashAttention-2 paper: Q, K, V, O for the tensors, L for
the saved log-sum-exp, m and l for the running max and denominator.

    from kernels.flash_attention.kernel import FlashAttentionTriton
    out = FlashAttentionTriton.apply(Q, K, V, is_causal)

See README.md in this directory for measured accuracy and throughput.
"""

import math

import torch
import triton
import triton.language as tl


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    is_causal: bool = False) -> torch.Tensor:
    """Q, K, V: (batch, n_heads, seq, head_dim)"""
    d = Q.shape[-1]
    S = (Q @ K.transpose(-1, -2)) / math.sqrt(d)
    if is_causal:
        nq, nk = Q.shape[-2], K.shape[-2]
        mask = torch.tril(torch.ones(nq, nk, dtype=torch.bool, device=Q.device))
        S = S.masked_fill(~mask, float('-inf'))
    return torch.softmax(S, dim=-1) @ V

# Launched with grid size (N // BLOCK_M, Z * H)
# Z - batches, H - heads, N - number of tokens
@triton.jit
def flash_attention_2(
    Q, K, V, O,
    L,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N,
    D,                       # true head dimension (BLOCK_D is it rounded up)
    softmax_scale,
    BLOCK_M: tl.constexpr,   # query rows
    BLOCK_N: tl.constexpr,   # KVs
    BLOCK_D: tl.constexpr,   # head dim, rounded up to a power of 2
    IS_CAUSAL: tl.constexpr,
    PRECISION: tl.constexpr, # "ieee" for true fp32, "tf32" for tensor cores
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    m_mask = offs_m < N

    offs_d = tl.arange(0, BLOCK_D)
    # BLOCK_D is D rounded up to a power of 2. Lanes in [D, BLOCK_D) address
    # memory belonging to the next row, so every access is masked on d_mask and
    # the padding is zero-filled -- it then contributes nothing to any dot.
    d_mask = offs_d < D

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    running_max = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    running_denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    end_n = (start_m + 1) * BLOCK_M if IS_CAUSAL else N

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # softmax_scale is 1/sqrt(D) from the host: BLOCK_D is the PADDED width,
        # so scaling by it silently changes the attention temperature whenever D
        # is not a power of 2.
        s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * softmax_scale

        score_mask = m_mask[:, None] & n_mask[None, :]

        if IS_CAUSAL:
            score_mask = score_mask & (offs_m[:, None] >= offs_n[None, :])

        s = tl.where(score_mask, s, float('-inf'))

        row_max = tl.max(s, axis=1)
        new_max = tl.maximum(running_max, row_max)

        alpha = tl.exp(running_max - new_max)

        p = tl.exp(s - new_max[:, None]) # softmax numerator for this block

        running_denom = alpha * running_denom + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)

        running_max = new_max

    acc = acc / running_denom[:, None]

    L_ptrs = L + off_hz * N + offs_m
    tl.store(L_ptrs, running_max + tl.log(running_denom), mask=m_mask)

    o_ptrs = O + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])

# Tile schedule. fp16/bf16 feed the tensor cores directly and fit the large
# tiles. True IEEE fp32 needs roughly twice the shared memory for the same tile
# -- Triton emits a 3-pass split-float emulation instead of one tensor-core op --
# so the fp32 path trades tile size for accuracy. Measured on an RTX 4070 Ti
# (99 KB shared/SM): fp32 at 128x64 needs 131 KB and does not launch.
_TILES = {
    torch.float16:  (128, 64),
    torch.bfloat16: (128, 64),
    torch.float32:  (64, 32),
}


def _softmax_precision(dtype: torch.dtype) -> str:
    """fp32 defaults to TF32 tensor cores in Triton, which costs ~3 decimal
    digits. Ask for IEEE explicitly so the fp32 path means fp32."""
    return "ieee" if dtype == torch.float32 else "tf32"


def flash_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            is_causal: bool = False):
    Z, H, N, D = q.shape
    assert q.shape == k.shape == v.shape, "Q, K, V shapes don't match"
    assert q.dtype == k.dtype == v.dtype, "Q, K, V must share a dtype"
    assert q.dtype in _TILES, (
        f"unsupported dtype {q.dtype}; expected one of {tuple(_TILES)}"
    )
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Q, K, V not all CUDA tensors"

    o = torch.empty_like(q)
    L = torch.empty((Z, H, N), device=q.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N = _TILES[q.dtype]
    BLOCK_D = triton.next_power_of_2(D)

    grid = (triton.cdiv(N, BLOCK_M), Z * H)

    try:
        flash_attention_2[grid](
            q, k, v, o, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            Z, H, N,
            D,
            1.0 / math.sqrt(D),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            IS_CAUSAL=is_causal,
            PRECISION=_softmax_precision(q.dtype),
        )
    except triton.runtime.errors.OutOfResources as e:
        # The raw Triton message names byte counts but not the cause, which is
        # always the same: this (dtype, head_dim) needs a tile that does not fit
        # in this GPU's shared memory.
        raise triton.runtime.errors.OutOfResources(
            e.required, e.limit,
            f"shared memory for head_dim={D} in {q.dtype} at "
            f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}. Use fp16/bf16, a smaller "
            f"head_dim, or a GPU with more shared memory per SM"
        ) from None

    return o, L


def flash_attention_backward(Q, K, V, O, L, dO, is_causal, scale):
    S = (Q @ K.transpose(-1, -2)) * scale
    if is_causal:
        nq, nk = Q.shape[-2], K.shape[-2]
        mask = torch.tril(torch.ones(nq, nk, dtype=torch.bool, device=Q.device))
        S = S.masked_fill(~mask, -1e6)
    P = torch.exp(S - L.unsqueeze(-1))
    dV = P.transpose(-1, -2) @ dO
    dP = dO @ V.transpose(-1, -2)
    Dv = (dO * O).sum(dim=-1, keepdim=True)
    dS = P * (dP - Dv)
    dQ = (dS @ K) * scale
    dK = (dS.transpose(-1, -2) @ Q) * scale
    return dQ, dK, dV


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, L = flash_attention_forward(Q, K, V, is_causal)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = 1.0 / math.sqrt(Q.shape[-1])
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward(Q, K, V, O, L, dO.contiguous(),
                                              ctx.is_causal, ctx.scale)
        return dQ, dK, dV, None