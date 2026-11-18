# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 fused per-head QK RMSNorm + partial split-half RoPE (Triton, one launch for Q and K)."""

import torch
import triton
import triton.language as tl

# A head row is read as chunks of this many dims so every load is vectorized.
_CHUNK = 16
# Launch shape: rows per program = TOKENS_PER_PROGRAM * HEADS_PER_PROGRAM must be a power of two.
# One token x eight heads per program with one warp measured fastest on B200; the head group
# falls back to 4, 2 or 1 for head counts that eight does not divide.
_TOKENS_PER_PROGRAM = 1
_NUM_WARPS = 1


@triton.jit
def _minimax_h3_qk_norm_rope_kernel(
    qkv_ptr,
    q_out_ptr,
    k_out_ptr,
    weight_q_ptr,
    weight_k_ptr,
    cos_ptr,
    sin_ptr,
    NUM_TOKENS,
    SEQ_LEN,
    qkv_row_stride,
    k_col_offset,
    out_row_stride,
    cos_row_stride,
    cos_col_stride,
    sin_row_stride,
    sin_col_stride,
    eps,
    HEAD_DIM: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNK: tl.constexpr,
    NUM_ROT_HALF_CHUNKS: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    """Per-head RMSNorm + partial split-half RoPE for Q and K, read from the packed QKV GEMM output.

    One program handles TOKENS_PER_PROGRAM tokens x HEADS_PER_PROGRAM heads of either Q (axis 2 == 0)
    or K (axis 2 == 1). A head row is viewed as NUM_CHUNKS chunks of CHUNK dims so every load is
    vectorized; the rotate-half partner of chunk j is a second load of chunk j +- NUM_ROT_HALF_CHUNKS,
    and chunks at or beyond 2 * NUM_ROT_HALF_CHUNKS pass through unrotated.

    All math is FP32 (BF16 inputs and tables upcast on load) with a single rounding to BF16 at the
    store, the same contract as the other VisualGen fused QK-norm + RoPE kernels.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    which = tl.program_id(2)
    rows = tl.arange(0, TOKENS_PER_PROGRAM * HEADS_PER_PROGRAM)
    tok = pid_t * TOKENS_PER_PROGRAM + rows // HEADS_PER_PROGRAM
    head = pid_h * HEADS_PER_PROGRAM + rows % HEADS_PER_PROGRAM
    j = tl.arange(0, NUM_CHUNKS)
    kk = tl.arange(0, CHUNK)
    partner_chunk = tl.where(
        j < NUM_ROT_HALF_CHUNKS,
        j + NUM_ROT_HALF_CHUNKS,
        tl.where(j < 2 * NUM_ROT_HALF_CHUNKS, j - NUM_ROT_HALF_CHUNKS, j),
    )
    tok_ok = (tok < NUM_TOKENS)[:, None, None]
    base = (tok.to(tl.int64) * qkv_row_stride + head * HEAD_DIM + which * k_col_offset)[
        :, None, None
    ]
    off = (j * CHUNK)[None, :, None] + kk[None, None, :]
    partner_off = (partner_chunk * CHUNK)[None, :, None] + kk[None, None, :]
    x = tl.load(qkv_ptr + base + off, mask=tok_ok, other=0.0).to(tl.float32)
    xp = tl.load(qkv_ptr + base + partner_off, mask=tok_ok, other=0.0).to(tl.float32)
    weight_ptr = tl.where(which == 0, weight_q_ptr, weight_k_ptr)
    w = tl.load(weight_ptr + off).to(tl.float32)
    wp = tl.load(weight_ptr + partner_off).to(tl.float32)

    sum_sq = tl.sum(tl.sum(x * x, axis=2), axis=1)
    inv_rms = tl.rsqrt(sum_sq / HEAD_DIM + eps)[:, None, None]
    y = w * (x * inv_rms)
    yp = wp * (xp * inv_rms)

    rot_chunk = (j < 2 * NUM_ROT_HALF_CHUNKS)[None, :, None]
    rot_mask = tok_ok & rot_chunk
    seq = (tok % SEQ_LEN).to(tl.int64)
    c = tl.load(
        cos_ptr + (seq * cos_row_stride)[:, None, None] + off * cos_col_stride,
        mask=rot_mask,
        other=1.0,
    ).to(tl.float32)
    sn = tl.load(
        sin_ptr + (seq * sin_row_stride)[:, None, None] + off * sin_col_stride,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)
    rotated = tl.where((j < NUM_ROT_HALF_CHUNKS)[None, :, None], -yp, yp)
    out = tl.where(rot_chunk, y * c + rotated * sn, y)
    out_base = (tok.to(tl.int64) * out_row_stride + head * HEAD_DIM)[:, None, None]
    out_ptr = tl.where(which == 0, q_out_ptr, k_out_ptr)
    tl.store(out_ptr + out_base + off, out.to(tl.bfloat16), mask=tok_ok)


def launch_minimax_h3_qk_norm_rope(
    qkv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
    *,
    tokens_per_program: int = _TOKENS_PER_PROGRAM,
    heads_per_program: int | None = None,
    num_warps: int = _NUM_WARPS,
) -> None:
    """Launch the Triton kernel on validated inputs, writing ``q`` and ``k`` in place.

    The launch shape is a keyword so tests can exercise tail-token masking with shapes that do not
    divide the token count; the model uses the defaults through the custom op.
    """
    if heads_per_program is None:
        # Largest power of two up to _MAX_HEADS_PER_PROGRAM that divides the head count.
        heads_per_program = next(h for h in (8, 4, 2, 1) if num_heads % h == 0)
    if (
        heads_per_program < 1
        or heads_per_program & (heads_per_program - 1)
        or num_heads % heads_per_program
    ):
        raise ValueError("heads_per_program must be a power of two dividing num_heads")
    rows = tokens_per_program * heads_per_program
    if tokens_per_program < 1 or rows & (rows - 1):
        raise ValueError("tokens_per_program * heads_per_program must be a power of two")
    tokens = qkv.shape[0] * qkv.shape[1]
    if tokens == 0:
        return
    hd = num_heads * head_dim
    qkv2 = qkv.view(tokens, qkv.shape[-1])
    grid = (triton.cdiv(tokens, tokens_per_program), num_heads // heads_per_program, 2)
    with torch.cuda.device(qkv.device):
        _minimax_h3_qk_norm_rope_kernel[grid](
            qkv2,
            q,
            k,
            weight_q,
            weight_k,
            cos,
            sin,
            tokens,
            qkv.shape[1],
            qkv2.stride(0),
            hd,
            hd,
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1),
            float(eps),
            HEAD_DIM=head_dim,
            NUM_CHUNKS=head_dim // _CHUNK,
            CHUNK=_CHUNK,
            NUM_ROT_HALF_CHUNKS=cos.shape[1] // 2 // _CHUNK,
            TOKENS_PER_PROGRAM=tokens_per_program,
            HEADS_PER_PROGRAM=heads_per_program,
            num_warps=num_warps,
        )


def validate_minimax_h3_qk_norm_rope_inputs(
    qkv: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> None:
    """Raise ``ValueError`` unless the tensors satisfy the kernel's contract.

    Runs inside the custom op and its fake kernel. Autograd is rejected by the wrapper
    because grad mode is already off by the time the op body runs.
    """
    if qkv.ndim != 3:
        raise ValueError("H3 fused QK-norm RoPE expects packed qkv [batch, sequence, columns]")
    if not qkv.is_cuda or qkv.dtype != torch.bfloat16:
        raise ValueError("H3 fused QK-norm RoPE requires CUDA BF16 qkv")
    if not qkv.is_contiguous():
        raise ValueError("qkv must be contiguous")
    if head_dim != 128:
        raise ValueError("H3 fused QK-norm RoPE is implemented for head dimension 128 only")
    hd = num_heads * head_dim
    if num_heads < 1 or qkv.shape[-1] < 2 * hd:
        raise ValueError("qkv must hold Q and K columns in its first 2 * heads * dim columns")
    for weight in (weight_q, weight_k):
        if (
            weight.shape != (head_dim,)
            or weight.dtype != torch.bfloat16
            or weight.device != qkv.device
            or not weight.is_contiguous()
        ):
            raise ValueError(
                "H3 QK-norm weights must be contiguous BF16 [dim] tensors on the qkv device"
            )
    seq = qkv.shape[1]
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != seq:
        raise ValueError("H3 fused RoPE tables must have shape [sequence, rotary_dim]")
    rotary_dim = cos.shape[1]
    if rotary_dim > head_dim or rotary_dim % (2 * _CHUNK):
        raise ValueError(
            "H3 rotary dimension must be a multiple of 32 and no larger than the head dimension"
        )
    for table in (cos, sin):
        if table.device != qkv.device or table.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("H3 RoPE tables must be FP32 or BF16 on the qkv device")


@torch.library.custom_op("trtllm::minimax_h3_qk_norm_rope", mutates_args=(), device_types="cuda")
def _minimax_h3_qk_norm_rope_op(
    qkv: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque custom op around the Triton launch (one node under torch.compile).

    Validates the tensor contract so direct ``torch.ops.trtllm`` callers get a ``ValueError``
    instead of a kernel fault.
    """
    validate_minimax_h3_qk_norm_rope_inputs(qkv, weight_q, weight_k, cos, sin, num_heads, head_dim)
    hd = num_heads * head_dim
    batch, seq = qkv.shape[:2]
    q = torch.empty((batch, seq, hd), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((batch, seq, hd), dtype=qkv.dtype, device=qkv.device)
    launch_minimax_h3_qk_norm_rope(
        qkv, q, k, weight_q, weight_k, cos, sin, eps, num_heads, head_dim
    )
    return q, k


@_minimax_h3_qk_norm_rope_op.register_fake
def _(qkv, weight_q, weight_k, cos, sin, eps, num_heads, head_dim):
    validate_minimax_h3_qk_norm_rope_inputs(qkv, weight_q, weight_k, cos, sin, num_heads, head_dim)
    hd = num_heads * head_dim
    return qkv.new_empty((*qkv.shape[:2], hd)), qkv.new_empty((*qkv.shape[:2], hd))


def apply_minimax_h3_qk_norm_rope_bf16(
    qkv: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm + partial split-half RoPE for Q and K in one launch.

    Args:
        qkv: CUDA BF16 packed projection output ``[batch, sequence, >= 2 * heads * dim]`` with
            Q in columns ``[0, heads*dim)`` and K in ``[heads*dim, 2*heads*dim)``; V is not read.
        weight_q, weight_k: BF16 ``[dim]`` RMSNorm weights.
        cos, sin: FP32 or BF16 tables ``[sequence, rotary_dim]`` shared across batch and heads;
            arbitrary strides are accepted (contiguous tables vectorize best).
        eps: RMSNorm epsilon.
        num_heads, head_dim: local head geometry; ``head_dim`` must be 128.

    Returns:
        Contiguous BF16 ``q`` and ``k`` shaped ``[batch, sequence, heads * dim]``, computed in FP32
        and rounded once; not bit-identical to the eager module, which rounds at every step.
    """
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (qkv, weight_q, weight_k, cos, sin)
    ):
        raise ValueError("H3 fused QK-norm RoPE does not support autograd")
    return torch.ops.trtllm.minimax_h3_qk_norm_rope(
        qkv, weight_q, weight_k, cos, sin, float(eps), num_heads, head_dim
    )
