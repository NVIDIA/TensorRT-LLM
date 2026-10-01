# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 split-half RoPE with eager BF16 rounding boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _minimax_h3_rope_kernel(
    hidden_states_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    NUM_ELEMENTS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_sequence: tl.constexpr,
    stride_head: tl.constexpr,
    stride_dim: tl.constexpr,
    cos_stride_sequence: tl.constexpr,
    cos_stride_dim: tl.constexpr,
    sin_stride_sequence: tl.constexpr,
    sin_stride_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    d = i % HEAD_DIM
    h = i // HEAD_DIM % NUM_HEADS
    s = i // (HEAD_DIM * NUM_HEADS) % SEQ_LEN
    b = i // (HEAD_DIM * NUM_HEADS * SEQ_LEN)
    x = tl.load(
        hidden_states_ptr
        + b * stride_batch
        + s * stride_sequence
        + h * stride_head
        + d * stride_dim,
        i < NUM_ELEMENTS,
        0,
    ).to(tl.float32)
    partner = tl.where(d < ROTARY_DIM // 2, d + ROTARY_DIM // 2, d - ROTARY_DIM // 2)
    xp = tl.load(
        hidden_states_ptr
        + b * stride_batch
        + s * stride_sequence
        + h * stride_head
        + partner * stride_dim,
        (i < NUM_ELEMENTS) & (d < ROTARY_DIM),
        0,
    ).to(tl.float32)
    rotated = tl.where(d < ROTARY_DIM // 2, -xp, xp)
    c = tl.load(
        cos_ptr + s * cos_stride_sequence + d * cos_stride_dim,
        (i < NUM_ELEMENTS) & (d < ROTARY_DIM),
        0,
    )
    sn = tl.load(
        sin_ptr + s * sin_stride_sequence + d * sin_stride_dim,
        (i < NUM_ELEMENTS) & (d < ROTARY_DIM),
        0,
    )
    # Eager casts the tables and materializes both products in BF16 before
    # adding them. Keeping these boundaries prevents output drift from fusion.
    c = c.to(tl.bfloat16).to(tl.float32)
    sn = sn.to(tl.bfloat16).to(tl.float32)
    left = (x * c).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sn).to(tl.bfloat16).to(tl.float32)
    y = tl.where(d < ROTARY_DIM, left + right, x)
    tl.store(output_ptr + i, y, i < NUM_ELEMENTS)


def apply_minimax_h3_rope_bf16(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply partial split-half RoPE with eager BF16 rounding.

    Args:
        hidden_states: CUDA BF16 tensor shaped ``[batch, sequence, heads, dim]``.
        cos: FP32 or BF16 table shaped ``[sequence, rotary_dim]`` on the same device.
        sin: FP32 or BF16 table with the same shape/device as cos.

    Returns:
        Contiguous BF16 output with the unrotated tail copied unchanged.
        Arbitrary input/table strides are supported. Tables and both products
        round to BF16 before addition; multiply-add contraction is disabled.

    Gradient-bearing inputs are rejected when autograd is enabled; the model
    dispatcher retains eager execution for these inputs.
    """
    if hidden_states.ndim != 4:
        raise ValueError("H3 fused RoPE expects [batch, sequence, heads, dim]")
    if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        raise ValueError("H3 fused RoPE requires CUDA BF16 hidden states")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != hidden_states.shape[1]:
        raise ValueError("H3 fused RoPE tables must have shape [sequence, rotary_dim]")
    if cos.shape[1] > hidden_states.shape[-1] or cos.shape[1] % 2:
        raise ValueError("H3 rotary dimension must be even and no larger than the head dimension")
    for table in (cos, sin):
        if table.device != hidden_states.device or table.dtype not in (
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError("H3 RoPE tables must be FP32 or BF16 on the hidden-state device")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (hidden_states, cos, sin)):
        raise ValueError("H3 fused RoPE does not support autograd")
    _, seq, heads, dim = hidden_states.shape
    out = torch.empty(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
    if out.numel() == 0:
        return out
    with torch.cuda.device(hidden_states.device):
        _minimax_h3_rope_kernel[(triton.cdiv(out.numel(), 1024),)](
            hidden_states,
            cos,
            sin,
            out,
            out.numel(),
            seq,
            heads,
            dim,
            cos.shape[-1],
            *hidden_states.stride(),
            *cos.stride(),
            *sin.stride(),
            BLOCK_SIZE=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


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
    EXACT_ROUNDING: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    """Per-head RMSNorm + partial split-half RoPE for Q and K, read from the packed QKV GEMM output.

    One program handles TOKENS_PER_PROGRAM tokens x HEADS_PER_PROGRAM heads of either Q (axis 2 == 0)
    or K (axis 2 == 1). A head row is viewed as NUM_CHUNKS chunks of CHUNK dims so every load is
    vectorized; the rotate-half partner of chunk j is a second load of chunk j +- NUM_ROT_HALF_CHUNKS,
    and chunks at or beyond 2 * NUM_ROT_HALF_CHUNKS pass through unrotated.

    EXACT_ROUNDING reproduces the eager module bit for bit: torch's FP32 summation order for the
    variance of a 128-wide row (32 lanes x 4 sequential squares, then a shuffle-down tree), x * rsqrt
    rounded to BF16, weight multiply rounded to BF16, BF16 tables, each RoPE product rounded to BF16.
    Otherwise everything is computed in FP32 with a single rounding at the store.
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

    if EXACT_ROUNDING:
        squares = tl.reshape(x * x, [TOKENS_PER_PROGRAM * HEADS_PER_PROGRAM, 32, 2, 2])
        even, odd = tl.split(squares)
        e0, e2 = tl.split(even)
        e1, e3 = tl.split(odd)
        partial = ((e0 + e1) + e2) + e3
        partial = tl.sum(
            tl.reshape(partial, [TOKENS_PER_PROGRAM * HEADS_PER_PROGRAM, 2, 16]), axis=1
        )
        partial = tl.sum(
            tl.reshape(partial, [TOKENS_PER_PROGRAM * HEADS_PER_PROGRAM, 2, 8]), axis=1
        )
        partial = tl.sum(
            tl.reshape(partial, [TOKENS_PER_PROGRAM * HEADS_PER_PROGRAM, 2, 4]), axis=1
        )
        partial = tl.sum(
            tl.reshape(partial, [TOKENS_PER_PROGRAM * HEADS_PER_PROGRAM, 2, 2]), axis=1
        )
        sum_sq = tl.sum(partial, axis=1)
    else:
        sum_sq = tl.sum(tl.sum(x * x, axis=2), axis=1)
    inv_rms = tl.rsqrt(sum_sq / HEAD_DIM + eps)[:, None, None]

    if EXACT_ROUNDING:
        y = (w * (x * inv_rms).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        yp = (wp * (xp * inv_rms).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    else:
        y = w * (x * inv_rms)
        yp = wp * (xp * inv_rms)

    rot_chunk = (j < 2 * NUM_ROT_HALF_CHUNKS)[None, :, None]
    rot_mask = tok_ok & rot_chunk
    seq = (tok % SEQ_LEN).to(tl.int64)
    c = tl.load(
        cos_ptr + (seq * cos_row_stride)[:, None, None] + off * cos_col_stride,
        mask=rot_mask,
        other=1.0,
    )
    sn = tl.load(
        sin_ptr + (seq * sin_row_stride)[:, None, None] + off * sin_col_stride,
        mask=rot_mask,
        other=0.0,
    )
    rotated = tl.where((j < NUM_ROT_HALF_CHUNKS)[None, :, None], -yp, yp)
    if EXACT_ROUNDING:
        c = c.to(tl.bfloat16).to(tl.float32)
        sn = sn.to(tl.bfloat16).to(tl.float32)
        left = (y * c).to(tl.bfloat16).to(tl.float32)
        right = (rotated * sn).to(tl.bfloat16).to(tl.float32)
        out = tl.where(rot_chunk, left + right, y)
    else:
        out = tl.where(rot_chunk, y * c + rotated * sn, y)
    out_base = (tok.to(tl.int64) * out_row_stride + head * HEAD_DIM)[:, None, None]
    out_ptr = tl.where(which == 0, q_out_ptr, k_out_ptr)
    tl.store(out_ptr + out_base + off, out.to(tl.bfloat16), mask=tok_ok)


_CHUNK = 16


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
    exact_rounding: bool,
    tokens_per_program: int,
    heads_per_program: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque custom op around the Triton launch.

    Registering the launch as a custom op keeps torch.compile from functionalizing
    the kernel's output stores into extra clones and lets Inductor schedule it as
    one opaque node between the QKV projection and attention.
    """
    hd = num_heads * head_dim
    batch, seq = qkv.shape[:2]
    tokens = batch * seq
    q = torch.empty((batch, seq, hd), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((batch, seq, hd), dtype=qkv.dtype, device=qkv.device)
    if tokens == 0:
        return q, k
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
            seq,
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
            EXACT_ROUNDING=bool(exact_rounding),
            TOKENS_PER_PROGRAM=tokens_per_program,
            HEADS_PER_PROGRAM=heads_per_program,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    return q, k


@_minimax_h3_qk_norm_rope_op.register_fake
def _(
    qkv,
    weight_q,
    weight_k,
    cos,
    sin,
    eps,
    num_heads,
    head_dim,
    exact_rounding,
    tokens_per_program,
    heads_per_program,
    num_warps,
):
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
    exact_rounding: bool = True,
    tokens_per_program: int = 1,
    heads_per_program: int | None = None,
    num_warps: int = 1,
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
        exact_rounding: reproduce the eager module bit for bit (default); ``False`` computes in
            FP32 with one final rounding, closer to an FP64 oracle but not eager-identical.
        tokens_per_program, heads_per_program, num_warps: launch shape; the defaults are the
            fastest measured on B200 (``heads_per_program`` adapts to small head counts).

    Returns:
        Contiguous BF16 ``q`` and ``k`` shaped ``[batch, sequence, heads * dim]``.
    """
    if qkv.ndim != 3:
        raise ValueError("H3 fused QK-norm RoPE expects packed qkv [batch, sequence, columns]")
    if not qkv.is_cuda or qkv.dtype != torch.bfloat16:
        raise ValueError("H3 fused QK-norm RoPE requires CUDA BF16 qkv")
    if head_dim != 128:
        raise ValueError("H3 fused QK-norm RoPE is implemented for head dimension 128 only")
    hd = num_heads * head_dim
    if qkv.shape[-1] < 2 * hd or qkv.stride(-1) != 1:
        raise ValueError("qkv must hold Q and K columns contiguously along the last dimension")
    for weight in (weight_q, weight_k):
        if (
            weight.shape != (head_dim,)
            or weight.dtype != torch.bfloat16
            or weight.device != qkv.device
        ):
            raise ValueError("H3 QK-norm weights must be BF16 [dim] tensors on the qkv device")
    batch, seq = qkv.shape[:2]
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
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (qkv, weight_q, weight_k, cos, sin)
    ):
        raise ValueError("H3 fused QK-norm RoPE does not support autograd")
    if heads_per_program is None:
        # Largest power of two up to 8 that divides the head count (8 measured best on B200).
        heads_per_program = next(h for h in (8, 4, 2, 1) if num_heads % h == 0)
    if heads_per_program & (heads_per_program - 1) or num_heads % heads_per_program:
        raise ValueError("heads_per_program must be a power of two dividing num_heads")
    return torch.ops.trtllm.minimax_h3_qk_norm_rope(
        qkv,
        weight_q,
        weight_k,
        cos,
        sin,
        float(eps),
        num_heads,
        head_dim,
        bool(exact_rounding),
        tokens_per_program,
        heads_per_program,
        num_warps,
    )
