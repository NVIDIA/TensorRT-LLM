# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA CuTe DSL prefill / verification, native decode and portable FLA dispatch.

Kimi K3 (k3_rubin) copy of ``modules/kimi_kda/_kda_kernels.py``: main's dispatch, plus the
Rubin development branch's KDA kernels through the package's own ops:

* prefill: the branch's CuTe DSL pipeline (``KdaPrefillRunner``, ``trtllm::k3r_kda_prefill`` in
  ``prefill_ops.py``) on L2-normalized Q/K, with its mid-chunk snapshot outputs. Main's FlashInfer
  prefill (``trtllm::kda_prefill`` / ``kda_prefill_conv`` on raw Q/K/V) runs only when the
  package's op cannot be imported, and FLA when neither can;
* multi-token verify: the branch's v1 / v2 / two-CTA kernels (``trtllm::k3r_kda_mtp_decode*`` in
  ``mtp_ops.py``) on the replay caches main's cache manager allocates. Main's
  ``trtllm::kda_mtp_decode`` / ``_fp8`` run only when the package's module cannot be imported
  (then the branch-only variants degrade, see ``KDAKernelDispatch.has_mtp_op``);
* the branch's Triton prefill convolution (``fused_kda_conv_prefill``,
  TRTLLM_KDA_FUSED_CONV_PREFILL) and post-convolution layout (``fused_kda_post_conv``, with the
  chunk padding of TLLM_KDA_PREPAD_PREFILL).

Single-token decode stays main's ``trtllm::kda_decode`` (a C++ op). The replay helpers below
(``commit_kda_pending_drafts`` and its kernels) are main's; main's cache manager calls main's
copies of them. ``plan_kda_replay_seed`` / ``seed_kda_replay_layers`` are the package's: main's
``KDAReplayState`` loads them to seed the replay caches of transferred requests in all layers
with one launch per buffer kind (TRTLLM_KDA_BATCHED_REPLAY_SEED).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib
import os
from types import ModuleType
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._utils import prefer_pinned

from . import _kda_decode

try:
    from tensorrt_llm._utils import get_sm_version as _tllm_get_sm_version
except ImportError:  # pragma: no cover — source-loader stub path
    _tllm_get_sm_version = None


def _default_get_sm_version() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return -1
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor


def get_kda_sm_version() -> int:
    """Return the runtime SM version used for KDA kernel selection.

    Prefers ``tensorrt_llm._utils.get_sm_version`` when the real package is
    importable so environment-side overrides propagate. Falls back to a
    plain CUDA-property probe when we are executing under the source-loader
    stub subtree.
    """
    if _tllm_get_sm_version is not None:
        try:
            return int(_tllm_get_sm_version())
        except RuntimeError:
            # torch raises RuntimeError when no CUDA device is usable;
            # the property probe below handles that case itself.
            return _default_get_sm_version()
    return _default_get_sm_version()


def is_kda_optimized_supported() -> bool:
    """The optimized prefill/decode kernels need Blackwell sm_100 or Rubin sm_107."""
    return get_kda_sm_version() in (100, 103, 107)


@triton.jit(do_not_specialize=["num_tokens", "num_out_tokens", "beta_row_stride"])
def _fused_kda_post_conv_kernel(
    packed_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    beta_ptr,
    beta_out_ptr,
    num_tokens,
    num_out_tokens,
    beta_row_stride,
    l2_norm_eps,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """Normalize and transpose channel-major packed Q/K/V in one launch.

    Rows in ``[num_tokens, num_out_tokens)`` are written as exact zeros, so
    the outputs are already zero-padded for the KDA prefill op. When
    ``HAS_BETA``, beta is cast to fp32 and zero-padded the same way.
    """
    token_offsets = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_DIM)
    token_mask = token_offsets < num_tokens
    out_token_mask = token_offsets < num_out_tokens
    dim_mask = dim_offsets < HEAD_DIM
    mask = token_mask[:, None] & dim_mask[None, :]
    out_mask = out_token_mask[:, None] & dim_mask[None, :]

    feature_offsets = head_idx * HEAD_DIM + dim_offsets
    projection_size = NUM_HEADS * HEAD_DIM
    num_tokens_i64 = num_tokens.to(tl.int64)
    source_offsets = feature_offsets[None, :].to(tl.int64) * num_tokens_i64 + token_offsets[
        :, None
    ].to(tl.int64)
    section_stride = projection_size * num_tokens_i64
    output_offsets = (
        token_offsets[:, None].to(tl.int64) * projection_size + feature_offsets[None, :]
    )

    # Masked-off rows load 0 and normalize to exactly +0 (0 / sqrt(eps)).
    q = tl.load(packed_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    q /= tl.sqrt(tl.sum(q * q, axis=1) + l2_norm_eps)[:, None]
    tl.store(q_out_ptr + output_offsets, q, mask=out_mask)

    k = tl.load(
        packed_ptr + section_stride + source_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    k /= tl.sqrt(tl.sum(k * k, axis=1) + l2_norm_eps)[:, None]
    tl.store(k_out_ptr + output_offsets, k, mask=out_mask)

    v = tl.load(
        packed_ptr + 2 * section_stride + source_offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(v_out_ptr + output_offsets, v, mask=out_mask)

    if HAS_BETA:
        token_i64 = token_offsets.to(tl.int64)
        beta = tl.load(
            beta_ptr + token_i64 * beta_row_stride + head_idx, mask=token_mask, other=0.0
        ).to(tl.float32)
        tl.store(beta_out_ptr + token_i64 * NUM_HEADS + head_idx, beta, mask=out_token_mask)


def fused_kda_post_conv(
    packed: torch.Tensor,
    num_heads: int,
    head_dim: int,
    l2_norm_eps: float = 1e-6,
    padded_tokens: Optional[int] = None,
    beta: Optional[torch.Tensor] = None,
    *,
    normalize_qk: bool = True,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
):
    """Convert packed channel-major ``[3 * H * D, T]`` convolution output to KDA Q/K/V.

    A contiguous ``packed`` with ``normalize_qk`` runs the branch's Triton kernel (the CuTe DSL
    prefill path): contiguous ``[1, out_tokens, H, D]`` Q/K/V with Q and K L2-normalized, where
    ``out_tokens`` is ``padded_tokens`` when given (rows past ``T`` are exact zeros, the chunk
    padding of TLLM_KDA_PREPAD_PREFILL), else ``T``; with ``beta`` (``[..., T, H]``, unit head
    stride) a fourth output holds it cast to fp32 and zero-padded to ``[1, out_tokens, H]``.

    Otherwise main's portable layout conversion runs (Q/K normalized only with
    ``normalize_qk``, into ``out`` when given); it takes no padding.
    """
    if normalize_qk and out is None and packed.is_contiguous():
        return _fused_kda_post_conv_triton(
            packed, num_heads, head_dim, l2_norm_eps, padded_tokens, beta
        )
    if padded_tokens is not None or beta is not None:
        raise ValueError(
            "Padded KDA post-conv outputs need a contiguous packed input, Q/K normalization "
            "and no caller-provided outputs"
        )
    if packed.ndim != 2 or packed.shape[0] != 3 * num_heads * head_dim:
        raise ValueError("Expected packed [3 * H * D, T] convolution output")
    if packed.shape[1] > 1 and 1 not in packed.stride():
        raise ValueError("Packed KDA post-conv requires a unit stride along tokens or features")
    qkv = tuple(
        t.t().reshape(1, packed.shape[1], num_heads, head_dim).contiguous()
        for t in packed.chunk(3, dim=0)
    )
    if normalize_qk:
        q, k, v = qkv

        def normalize(t):
            f = t.float()
            return (f * torch.rsqrt((f * f).sum(-1, keepdim=True) + l2_norm_eps)).to(t.dtype)

        qkv = (normalize(q), normalize(k), v)
    if out is not None:
        if len(out) != 3 or any(
            dst.shape != src.shape
            or dst.dtype != src.dtype
            or dst.device != src.device
            or not dst.is_contiguous()
            for dst, src in zip(out, qkv)
        ):
            raise ValueError("Invalid KDA layout output buffers")
        for dst, src in zip(out, qkv):
            dst.copy_(src)
        return out
    return qkv


def _beta_rows_and_output(
    beta: Optional[torch.Tensor],
    num_tokens: int,
    out_tokens: int,
    num_heads: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Validate an optional ``beta`` and allocate its fp32 ``[1, out_tokens, heads]`` output.

    Returns ``(beta_rows, beta_out)``: the ``[tokens, heads]`` view of ``beta``
    the kernels read and the output they write, or ``(None, None)``.
    """
    if beta is None:
        return None, None
    beta_2d = None
    if beta.ndim == 2 and beta.shape == (num_tokens, num_heads):
        beta_2d = beta
    elif beta.ndim == 3 and beta.shape == (1, num_tokens, num_heads):
        beta_2d = beta[0]
    if beta_2d is None or beta_2d.stride(-1) != 1 or beta_2d.device != device:
        raise ValueError(
            f"beta must be [tokens={num_tokens}, heads={num_heads}] or "
            f"[1, tokens={num_tokens}, heads={num_heads}] on {device} "
            f"with unit head stride, got shape {tuple(beta.shape)}, "
            f"stride {beta.stride()}, device {beta.device}"
        )
    beta_out = torch.empty((1, out_tokens, num_heads), dtype=torch.float32, device=device)
    return beta_2d, beta_out


def _fused_kda_post_conv_triton(
    packed: torch.Tensor,
    num_heads: int,
    head_dim: int,
    l2_norm_eps: float,
    padded_tokens: Optional[int],
    beta: Optional[torch.Tensor],
):
    """The branch's ``fused_kda_post_conv`` (see :func:`fused_kda_post_conv`)."""
    projection_size = num_heads * head_dim
    if packed.ndim != 2 or packed.shape[0] != 3 * projection_size:
        raise ValueError(
            "Packed KDA post-conv expected shape "
            f"[{3 * projection_size}, tokens], got {tuple(packed.shape)}"
        )
    if not packed.is_contiguous():
        raise ValueError("Packed KDA post-conv requires a contiguous tensor")

    num_tokens = packed.shape[1]
    out_tokens = num_tokens if padded_tokens is None else padded_tokens
    if out_tokens < num_tokens:
        raise ValueError(f"padded_tokens={out_tokens} is smaller than tokens={num_tokens}")
    output_shape = (1, out_tokens, num_heads, head_dim)
    q_out = torch.empty(output_shape, dtype=packed.dtype, device=packed.device)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    beta_2d, beta_out = _beta_rows_and_output(
        beta, num_tokens, out_tokens, num_heads, packed.device
    )
    if out_tokens == 0:
        return (q_out, k_out, v_out) if beta is None else (q_out, k_out, v_out, beta_out)

    block_tokens = 16
    block_dim = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(out_tokens, block_tokens), num_heads)
    _fused_kda_post_conv_kernel[grid](
        packed,
        q_out,
        k_out,
        v_out,
        beta_2d if beta_2d is not None else q_out,
        beta_out if beta_out is not None else q_out,
        num_tokens,
        out_tokens,
        beta_2d.stride(0) if beta_2d is not None else 0,
        l2_norm_eps,
        num_heads,
        head_dim,
        block_tokens,
        block_dim,
        beta is not None,
        num_warps=8,
        num_stages=3,
    )
    return (q_out, k_out, v_out) if beta is None else (q_out, k_out, v_out, beta_out)


# Tile of the fused prefill convolution: tokens per program and warps per program.
_FUSED_CONV_BLOCK_TOKENS = 32
_FUSED_CONV_NUM_WARPS = 4


@triton.jit
def _kda_conv_silu(acc, out_dtype):
    """``causal_conv1d_fwd``'s SiLU, ``y * __frcp_rn(1 + __expf(-y))``, rounded to ``out_dtype``.

    ``__expf`` is spelled as the ``ex2.approx.ftz`` it compiles to. The
    reciprocal is ``rcp.rn``'s in-range sequence (MUFU.RCP and one FMA
    correction) without its out-of-range branch: ``1 + exp(-y)`` is at least
    1, and from 2**126 on (``y`` below about -87.3) the reciprocal is taken as
    0 instead of a subnormal. Results are bitwise ``causal_conv1d_fwd``'s
    except for those ``y``, where SiLU gives -0 instead of a value of
    magnitude below 1e-36.
    """
    exp_neg = tl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;",
        "=r,r",
        [acc * -1.4426950408889634],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    reciprocal = tl.inline_asm_elementwise(
        """{
        .reg .f32 r0, t;
        .reg .pred large;
        rcp.approx.ftz.f32 r0, $1;
        fma.rn.f32 t, $1, r0, 0fBF800000;
        neg.f32 t, t;
        fma.rn.f32 $0, r0, t, r0;
        setp.ge.f32 large, $1, 0f7E800000;
        selp.f32 $0, 0f00000000, $0, large;
        }""",
        "=r,r",
        [1.0 + exp_neg],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return _kda_round(acc * reciprocal, out_dtype)


@triton.jit
def _kda_round(x, out_dtype: tl.constexpr):
    """Round fp32 ``x`` to nearest-even ``out_dtype``; bf16 two elements per instruction."""
    if out_dtype == tl.bfloat16:
        return tl.inline_asm_elementwise(
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,r,r",
            [x],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=2,
        )
    return x.to(out_dtype)


@triton.jit
def _kda_widen(x, MIXED_BF16_FMA: tl.constexpr):
    """Exact fp32 value of ``x``; bf16 with one ``add.rn.f32.bf16`` of -0 (SM100+)."""
    if MIXED_BF16_FMA and x.dtype == tl.bfloat16:
        return tl.inline_asm_elementwise(
            "add.rn.f32.bf16 $0, $1, 0f80000000;",
            "=r,h",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    return x.to(tl.float32)


@triton.jit
def _kda_conv_tap(acc, weight, x, MIXED_BF16_FMA: tl.constexpr):
    """``acc + weight * x`` with an fp32 accumulator, one rounding per tap.

    With ``MIXED_BF16_FMA`` (SM100+), bf16 operands use the mixed-precision
    ``fma.rn.f32.bf16`` (no conversion instructions); the bf16 product is
    exact in fp32 either way.
    """
    if MIXED_BF16_FMA and weight.dtype == tl.bfloat16 and x.dtype == tl.bfloat16:
        return tl.inline_asm_elementwise(
            "fma.rn.f32.bf16 $0, $1, $2, $3;",
            "=r,h,h,r",
            [weight, x, acc],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    return acc + weight.to(tl.float32) * x.to(tl.float32)


@triton.jit
def _kda_store_conv_rows(
    out,
    offsets,
    mask,
    section,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    l2_norm_eps,
    MIXED_BF16_FMA: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
):
    """Store ``[rows, head_dim]`` conv outputs: V as is, Q and K L2-normalized per row
    with ``NORMALIZE_QK``, else as is.

    The norm is fp32 over the rounded values (the branch's Triton post-conv). Its
    division compiles to a multiplication by ``rcp.approx`` of the norm for
    norms in ``[2**-126, 2**126]``; the reciprocal is taken once per row here.
    """
    if section == 2:
        tl.store(v_out_ptr + offsets, out, mask=mask)
    elif not NORMALIZE_QK:
        if section == 0:
            tl.store(q_out_ptr + offsets, out, mask=mask)
        else:
            tl.store(k_out_ptr + offsets, out, mask=mask)
    else:
        normalized = _kda_widen(out, MIXED_BF16_FMA)
        norm = tl.sqrt(tl.sum(normalized * normalized, axis=1) + l2_norm_eps)
        inv_norm = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=r,r",
            [norm],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        normalized = _kda_round(normalized * inv_norm[:, None], out.dtype)
        if section == 0:
            tl.store(q_out_ptr + offsets, normalized, mask=mask)
        else:
            tl.store(k_out_ptr + offsets, normalized, mask=mask)


@triton.jit(do_not_specialize=["num_tokens", "num_out_tokens", "beta_row_stride"])
def _fused_kda_conv_prefill_kernel(
    src_ptr,
    weight_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    beta_ptr,
    beta_out_ptr,
    num_tokens,
    num_out_tokens,
    src_stride_token,
    src_stride_channel,
    weight_stride_channel,
    weight_stride_tap,
    beta_row_stride,
    l2_norm_eps,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    CHANNEL_CONTIGUOUS: tl.constexpr,
    WEIGHT_CHANNEL_CONTIGUOUS: tl.constexpr,
    HAS_BETA: tl.constexpr,
    MIXED_BF16_FMA: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
):
    """Causal conv + SiLU (+ L2 norm for Q/K) of one head of one Q/K/V section.

    Program ``(tile, section * NUM_HEADS + head)`` produces tokens
    ``[tile * BLOCK_TOKENS, (tile + 1) * BLOCK_TOKENS)`` of that head as if
    all tokens formed one sequence without history. The first ``WIDTH - 1``
    tokens of every sequence are recomputed by
    ``_kda_conv_prefill_edges_kernel``. Tokens in
    ``[num_tokens, num_out_tokens)`` are zeros.

    The arithmetic follows ``causal_conv1d_fwd`` (fp32 taps accumulated
    oldest first, SiLU, rounding to the activation dtype) and
    ``fused_kda_post_conv``. The taps run as a loop: the rows of a tap load
    are mostly cache hits on the previous tap's, and only one tap's tile is
    live in registers.
    """
    channel_block = tl.program_id(1)
    section = channel_block // NUM_HEADS
    head = channel_block % NUM_HEADS
    first_token = tl.program_id(0) * BLOCK_TOKENS
    rows = tl.arange(0, BLOCK_TOKENS)
    tokens = first_token + rows
    dims = tl.arange(0, BLOCK_DIM)
    dim_mask = dims < HEAD_DIM
    token_mask = tokens < num_tokens
    channels = channel_block * HEAD_DIM + dims

    # 64-bit tile base, 32-bit offsets within the tile.
    src_tile = (
        src_ptr
        + first_token.to(tl.int64) * src_stride_token
        + (channel_block * HEAD_DIM).to(tl.int64) * src_stride_channel
    )
    if CHANNEL_CONTIGUOUS:
        dim_offsets = dims
    else:
        dim_offsets = dims * src_stride_channel
    if WEIGHT_CHANNEL_CONTIGUOUS:
        weight_offsets = channels
    else:
        weight_offsets = channels * weight_stride_channel
    # Weights are loaded per tile row so that they share the input tile's layout.
    weight_offsets = weight_offsets[None, :] + rows[:, None] * 0
    acc = tl.zeros([BLOCK_TOKENS, BLOCK_DIM], dtype=tl.float32)
    for tap in tl.range(WIDTH, loop_unroll_factor=1):
        shift = WIDTH - 1 - tap
        weight = tl.load(
            weight_ptr + tap * weight_stride_tap + weight_offsets,
            mask=dim_mask[None, :],
            other=0.0,
        )
        x = tl.load(
            src_tile + ((rows - shift) * src_stride_token)[:, None] + dim_offsets[None, :],
            mask=(token_mask & (tokens >= shift))[:, None] & dim_mask[None, :],
            other=0.0,
        )
        acc = _kda_conv_tap(acc, weight, x, MIXED_BF16_FMA)

    out_tile = first_token.to(tl.int64) * (NUM_HEADS * HEAD_DIM) + head * HEAD_DIM
    _kda_store_conv_rows(
        _kda_conv_silu(acc, v_out_ptr.dtype.element_ty),
        out_tile + (rows * (NUM_HEADS * HEAD_DIM))[:, None] + dims[None, :],
        (tokens < num_out_tokens)[:, None] & dim_mask[None, :],
        section,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        l2_norm_eps,
        MIXED_BF16_FMA,
        NORMALIZE_QK,
    )

    if HAS_BETA:
        if section == 0:
            beta_tokens = tokens.to(tl.int64)
            beta = tl.load(
                beta_ptr + beta_tokens * beta_row_stride + head, mask=token_mask, other=0.0
            ).to(tl.float32)
            tl.store(
                beta_out_ptr + beta_tokens * NUM_HEADS + head,
                beta,
                mask=tokens < num_out_tokens,
            )


@triton.jit
def _kda_conv_prefill_edges_kernel(
    src_ptr,
    weight_ptr,
    conv_state_ptr,
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    src_stride_token,
    src_stride_channel,
    weight_stride_channel,
    weight_stride_tap,
    state_stride_slot,
    state_stride_channel,
    state_stride_tap,
    l2_norm_eps,
    pad_slot_id,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_EDGE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    MIXED_BF16_FMA: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
):
    """Sequence edges of the fused conv for one sequence and one head of one section.

    Recomputes the sequence's first ``WIDTH - 1`` tokens with tap inputs
    before its start taken from its conv-state slot (zeros without an
    initial state or with a ``pad_slot_id`` slot), then stores its last
    ``WIDTH - 1`` inputs in the slot. Sequences shorter than ``WIDTH - 1``
    keep the newest entries of their initial state (zeros without one) in
    front of their inputs. A ``pad_slot_id`` slot is left untouched.
    """
    seq = tl.program_id(0)
    channel_block = tl.program_id(1)
    section = channel_block // NUM_HEADS
    head = channel_block % NUM_HEADS
    dims = tl.arange(0, BLOCK_DIM)
    dim_mask = dims < HEAD_DIM
    channels = channel_block * HEAD_DIM + dims
    rows = tl.arange(0, BLOCK_EDGE)

    seq_start = tl.load(query_start_loc_ptr + seq).to(tl.int64)
    seq_len = tl.load(query_start_loc_ptr + seq + 1).to(tl.int64) - seq_start
    slot = tl.load(cache_indices_ptr + seq).to(tl.int64)
    use_state = (tl.load(has_initial_state_ptr + seq) != 0) & (slot != pad_slot_id)
    src_channels = channels.to(tl.int64) * src_stride_channel
    state = conv_state_ptr + slot * state_stride_slot + channels.to(tl.int64) * state_stride_channel

    # The first WIDTH - 1 tokens.
    row_mask = rows < tl.minimum(seq_len, WIDTH - 1)
    acc = tl.zeros([BLOCK_EDGE, BLOCK_DIM], dtype=tl.float32)
    for tap in tl.static_range(WIDTH):
        shift = WIDTH - 1 - tap
        weight = tl.load(
            weight_ptr + channels * weight_stride_channel + tap * weight_stride_tap,
            mask=dim_mask,
            other=0.0,
        )
        from_input = rows >= shift
        x = tl.load(
            src_ptr
            + ((seq_start + rows - shift) * src_stride_token)[:, None]
            + src_channels[None, :],
            mask=(row_mask & from_input)[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if shift > 0:
            history = tl.load(
                state[None, :] + ((WIDTH - 1 - shift + rows) * state_stride_tap)[:, None],
                mask=(row_mask & (rows < shift) & use_state)[:, None] & dim_mask[None, :],
                other=0.0,
            )
            x = tl.where(from_input[:, None], x, history)
        acc = _kda_conv_tap(acc, weight[None, :], x, MIXED_BF16_FMA)
    _kda_store_conv_rows(
        _kda_conv_silu(acc, v_out_ptr.dtype.element_ty),
        (seq_start + rows)[:, None] * (NUM_HEADS * HEAD_DIM) + (head * HEAD_DIM + dims)[None, :],
        row_mask[:, None] & dim_mask[None, :],
        section,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        l2_norm_eps,
        MIXED_BF16_FMA,
        NORMALIZE_QK,
    )

    if slot != pad_slot_id:
        tap_mask = rows < WIDTH - 1
        position = seq_len - (WIDTH - 1) + rows
        from_input = position >= 0
        x = tl.load(
            src_ptr + ((seq_start + position) * src_stride_token)[:, None] + src_channels[None, :],
            mask=(tap_mask & from_input)[:, None] & dim_mask[None, :],
            other=0.0,
        )
        kept = tl.load(
            state[None, :] + ((rows + seq_len) * state_stride_tap)[:, None],
            mask=(tap_mask & (position < 0) & use_state)[:, None] & dim_mask[None, :],
            other=0.0,
        )
        new_state = tl.where(from_input[:, None], x, kept)
        # Every conv-state read of this program comes before the stores.
        tl.debug_barrier()
        tl.store(
            state[None, :] + (rows * state_stride_tap)[:, None],
            new_state,
            mask=tap_mask[:, None] & dim_mask[None, :],
        )


def fused_kda_conv_prefill(
    qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    num_heads: int,
    head_dim: int,
    l2_norm_eps: float = 1e-6,
    padded_tokens: Optional[int] = None,
    beta: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    normalize_qk: bool = True,
):
    """Short convolution, SiLU and Q/K L2 norm of packed KDA Q/K/V in one pass.

    Equivalent to ``causal_conv1d_fn`` (SiLU, varlen, indexed conv states)
    on the channel-major packed input followed by ``fused_kda_post_conv``,
    without materializing the packed tensor.

    ``qkv`` is the ``[tokens, 3 * num_heads * head_dim]`` convolution input
    (Q, K, V sections), read in place; a unit channel stride (rows of the
    QKVG projection output) gives vectorized loads. Sequences are the
    ``query_start_loc`` ranges (``[batch + 1]``); sequence ``b`` reads its
    initial conv state from ``conv_states[cache_indices[b]]`` when
    ``has_initial_state[b]`` and its last ``W - 1`` inputs are written back
    there. ``conv_states`` is ``[slots, 3 * num_heads * head_dim, W - 1]`` for
    ``conv_weight`` ``[3 * num_heads * head_dim, W]``. A sequence whose slot is
    ``pad_slot_id`` reads no conv state and leaves the pool untouched.

    Returns contiguous ``[1, out_tokens, num_heads, head_dim]`` Q, K, V (Q and
    K L2-normalized unless ``normalize_qk`` is False, for cores that normalize
    them in the kernel) and, if ``beta`` is given, its fp32
    ``[1, out_tokens, num_heads]`` copy. ``out_tokens`` is ``padded_tokens``
    when given (rows past ``tokens`` are exact zeros), else ``tokens``.
    """
    projection_size = num_heads * head_dim
    num_channels = 3 * projection_size
    if qkv.ndim != 2 or qkv.shape[1] != num_channels:
        raise ValueError(
            f"Fused KDA conv expected input shape [tokens, {num_channels}], got {tuple(qkv.shape)}"
        )
    if (
        conv_weight.ndim != 2
        or conv_weight.shape[0] != num_channels
        or not 2 <= conv_weight.shape[1] <= 4
    ):
        raise ValueError(
            f"Fused KDA conv expected weight shape [{num_channels}, 2..4], "
            f"got {tuple(conv_weight.shape)}"
        )
    width = conv_weight.shape[1]
    if (
        conv_states.ndim != 3
        or tuple(conv_states.shape[1:]) != (num_channels, width - 1)
        or conv_states.dtype != qkv.dtype
    ):
        raise ValueError(
            f"Fused KDA conv expected {qkv.dtype} conv states [slots, {num_channels}, "
            f"{width - 1}], got {conv_states.dtype} {tuple(conv_states.shape)}"
        )
    num_seqs = cache_indices.numel()
    if (
        cache_indices.ndim != 1
        or tuple(query_start_loc.shape) != (num_seqs + 1,)
        or tuple(has_initial_state.shape) != (num_seqs,)
        or has_initial_state.dtype != torch.bool
    ):
        raise ValueError(
            "Fused KDA conv expected query_start_loc [batch + 1], cache_indices [batch] and "
            f"bool has_initial_state [batch], got {tuple(query_start_loc.shape)}, "
            f"{tuple(cache_indices.shape)} and {has_initial_state.dtype} "
            f"{tuple(has_initial_state.shape)}"
        )
    tensors = (conv_weight, conv_states, query_start_loc, cache_indices, has_initial_state)
    if any(tensor.device != qkv.device for tensor in tensors):
        raise ValueError("Fused KDA conv tensors must be on one device")

    num_tokens = qkv.shape[0]
    out_tokens = num_tokens if padded_tokens is None else padded_tokens
    if out_tokens < num_tokens:
        raise ValueError(f"padded_tokens={out_tokens} is smaller than tokens={num_tokens}")
    block_tokens = _FUSED_CONV_BLOCK_TOKENS
    output_shape = (1, out_tokens, num_heads, head_dim)
    q_out = torch.empty(output_shape, dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    beta_2d, beta_out = _beta_rows_and_output(beta, num_tokens, out_tokens, num_heads, qkv.device)
    block_dim = triton.next_power_of_2(head_dim)
    mixed_bf16_fma = torch.cuda.get_device_capability(qkv.device)[0] >= 10

    with torch.cuda.device(qkv.device):
        if out_tokens > 0:
            _fused_kda_conv_prefill_kernel[(triton.cdiv(out_tokens, block_tokens), 3 * num_heads)](
                qkv,
                conv_weight,
                q_out,
                k_out,
                v_out,
                beta_2d if beta_2d is not None else q_out,
                beta_out if beta_out is not None else q_out,
                num_tokens,
                out_tokens,
                qkv.stride(0),
                qkv.stride(1),
                conv_weight.stride(0),
                conv_weight.stride(1),
                beta_2d.stride(0) if beta_2d is not None else 0,
                l2_norm_eps,
                NUM_HEADS=num_heads,
                HEAD_DIM=head_dim,
                WIDTH=width,
                BLOCK_TOKENS=block_tokens,
                BLOCK_DIM=block_dim,
                CHANNEL_CONTIGUOUS=qkv.stride(1) == 1,
                WEIGHT_CHANNEL_CONTIGUOUS=conv_weight.stride(0) == 1,
                HAS_BETA=beta is not None,
                MIXED_BF16_FMA=mixed_bf16_fma,
                NORMALIZE_QK=normalize_qk,
                num_warps=_FUSED_CONV_NUM_WARPS,
            )
        if num_seqs > 0:
            # Runs after the kernel above: it overwrites that kernel's sequence-edge
            # tokens, and it updates the conv states only after reading them.
            _kda_conv_prefill_edges_kernel[(num_seqs, 3 * num_heads)](
                qkv,
                conv_weight,
                conv_states,
                query_start_loc,
                cache_indices,
                has_initial_state,
                q_out,
                k_out,
                v_out,
                qkv.stride(0),
                qkv.stride(1),
                conv_weight.stride(0),
                conv_weight.stride(1),
                conv_states.stride(0),
                conv_states.stride(1),
                conv_states.stride(2),
                l2_norm_eps,
                pad_slot_id,
                NUM_HEADS=num_heads,
                HEAD_DIM=head_dim,
                WIDTH=width,
                BLOCK_EDGE=max(2, triton.next_power_of_2(width - 1)),
                BLOCK_DIM=block_dim,
                MIXED_BF16_FMA=mixed_bf16_fma,
                NORMALIZE_QK=normalize_qk,
                num_warps=4,
            )
    return (q_out, k_out, v_out) if beta is None else (q_out, k_out, v_out, beta_out)


@triton.jit
def _copy_kda_replay_conv_window_kernel(
    conv_ptr,
    q_cache_ptr,
    k_cache_ptr,
    v_cache_ptr,
    state_indices_ptr,
    conv_stride_slot,
    conv_stride_dim,
    conv_stride_window,
    cache_stride_slot,
    cache_stride_dim,
    cache_stride_window,
    PROJECTION_SIZE: tl.constexpr,
    COMMITTED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < PROJECTION_SIZE * COMMITTED
    dim = offsets // COMMITTED
    window = offsets % COMMITTED
    slot = tl.load(state_indices_ptr + request).to(tl.int64)

    conv_offset = slot * conv_stride_slot + dim * conv_stride_dim + window * conv_stride_window
    cache_offset = slot * cache_stride_slot + dim * cache_stride_dim + window * cache_stride_window
    section_offset = PROJECTION_SIZE * conv_stride_dim
    tl.store(q_cache_ptr + cache_offset, tl.load(conv_ptr + conv_offset, mask=mask), mask=mask)
    tl.store(
        k_cache_ptr + cache_offset,
        tl.load(conv_ptr + conv_offset + section_offset, mask=mask),
        mask=mask,
    )
    tl.store(
        v_cache_ptr + cache_offset,
        tl.load(conv_ptr + conv_offset + 2 * section_offset, mask=mask),
        mask=mask,
    )


def copy_kda_replay_conv_window(
    conv_pool: torch.Tensor,
    q_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    """Copy selected packed ``W - 1`` rows into KDA replay caches.

    The live convolution pool is ``[slots, 3D, W - 1]``. Replay caches are
    ``[slots, D, W - 1 + num_spec]`` with a dim-contiguous inner layout.
    """
    if conv_pool.ndim != 3 or conv_pool.shape[1] % 3:
        raise ValueError(f"Expected packed rank-3 KDA convolution pool, got {conv_pool.shape}")
    projection_size = conv_pool.shape[1] // 3
    committed = conv_pool.shape[2]
    caches = (q_cache, k_cache, v_cache)
    expected_prefix = (conv_pool.shape[0], projection_size)
    if any(cache.ndim != 3 or cache.shape[:2] != expected_prefix for cache in caches):
        raise ValueError("KDA replay convolution caches do not match the live pool geometry")
    if any(cache.shape[2] < committed for cache in caches):
        raise ValueError("KDA replay convolution caches are shorter than the committed window")
    if any(cache.stride() != q_cache.stride() for cache in caches[1:]):
        raise ValueError("KDA replay convolution caches must share one layout")
    if state_indices.ndim != 1 or state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA replay state indices must be a rank-1 int32 or int64 tensor")
    if any(tensor.device != conv_pool.device for tensor in (*caches, state_indices)):
        raise ValueError("KDA replay convolution tensors must be on one device")
    if state_indices.numel() == 0:
        return

    block_size = 256
    grid = (
        state_indices.numel(),
        triton.cdiv(projection_size * committed, block_size),
    )
    with torch.cuda.device(conv_pool.device.index):
        _copy_kda_replay_conv_window_kernel[grid](
            conv_pool,
            q_cache,
            k_cache,
            v_cache,
            state_indices,
            conv_pool.stride(0),
            conv_pool.stride(1),
            conv_pool.stride(2),
            q_cache.stride(0),
            q_cache.stride(1),
            q_cache.stride(2),
            PROJECTION_SIZE=projection_size,
            COMMITTED=committed,
            BLOCK_SIZE=block_size,
        )


_TORCH_TO_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _seed_kda_replay_conv_layers_kernel(
    conv_ptrs,
    cache_ptrs,
    state_indices_ptr,
    conv_stride_slot,
    conv_stride_dim,
    conv_stride_window,
    cache_stride_slot,
    cache_stride_dim,
    cache_stride_window,
    CONV_DTYPE: tl.constexpr,
    CACHE_DTYPE: tl.constexpr,
    PROJECTION_SIZE: tl.constexpr,
    COMMITTED: tl.constexpr,
    EXTENDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Seed one slot's q/k/v replay conv windows in one layer.

    ``conv_ptrs[layer]`` is the base address of that layer's live conv pool and
    ``cache_ptrs[3 * layer + section]`` of its replay conv cache for ``section``.
    The committed window is copied and the draft tail zeroed.
    """
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    layer = tl.program_id(2)
    mask = offsets < PROJECTION_SIZE * EXTENDED
    dim = offsets // EXTENDED
    window = offsets % EXTENDED
    committed = mask & (window < COMMITTED)
    slot = tl.load(state_indices_ptr + request).to(tl.int64)

    conv = tl.load(conv_ptrs + layer).to(tl.pointer_type(CONV_DTYPE))
    conv_offset = slot * conv_stride_slot + dim * conv_stride_dim + window * conv_stride_window
    cache_offset = slot * cache_stride_slot + dim * cache_stride_dim + window * cache_stride_window
    section_offset = PROJECTION_SIZE * conv_stride_dim
    for section in tl.static_range(3):
        cache = tl.load(cache_ptrs + 3 * layer + section).to(tl.pointer_type(CACHE_DTYPE))
        values = tl.load(conv + conv_offset + section * section_offset, mask=committed, other=0.0)
        tl.store(cache + cache_offset, values.to(CACHE_DTYPE), mask=mask)


@triton.jit
def _zero_kda_replay_slots_kernel(
    cache_ptrs,
    state_indices_ptr,
    stride_slot,
    stride_row,
    stride_col,
    DTYPE: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero one slot's ``[ROWS, COLS]`` entry of the layer at ``cache_ptrs[layer]``."""
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    layer = tl.program_id(2)
    mask = offsets < ROWS * COLS
    row = offsets // COLS
    col = offsets % COLS
    slot = tl.load(state_indices_ptr + request).to(tl.int64)
    cache = tl.load(cache_ptrs + layer).to(tl.pointer_type(DTYPE))
    tl.store(
        cache + slot * stride_slot + row * stride_row + col * stride_col,
        tl.zeros([BLOCK_SIZE], dtype=DTYPE),
        mask=mask,
    )


@dataclasses.dataclass(frozen=True)
class _KDAReplayZeroGroup:
    """One replay buffer kind, ``[slots, ROWS, COLS]`` in every layer."""

    ptrs: torch.Tensor
    strides: Tuple[int, int, int]
    dtype: torch.dtype
    rows: int
    cols: int


@dataclasses.dataclass(frozen=True)
class KDAReplaySeedPlan:
    """Device pointer tables and the shared geometry of every layer's replay buffers."""

    num_layers: int
    conv_ptrs: torch.Tensor
    conv_cache_ptrs: torch.Tensor
    conv_strides: Tuple[int, int, int]
    cache_strides: Tuple[int, int, int]
    conv_dtype: torch.dtype
    cache_dtype: torch.dtype
    projection_size: int
    committed: int
    extended: int
    zero_groups: Tuple[_KDAReplayZeroGroup, ...]


def _shared_layout(tensors: Sequence[torch.Tensor]) -> bool:
    reference = tensors[0]
    return (
        reference.ndim == 3
        and reference.is_cuda
        and reference.dtype in _TORCH_TO_TL_DTYPE
        and all(
            tensor.shape == reference.shape
            and tensor.stride() == reference.stride()
            and tensor.dtype == reference.dtype
            and tensor.device == reference.device
            for tensor in tensors
        )
    )


def _pointer_table(tensors: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Device int64 table of base addresses, copied without waiting for queued GPU work."""
    host = torch.tensor(
        [tensor.data_ptr() for tensor in tensors], dtype=torch.int64, pin_memory=prefer_pinned()
    )
    return host.to(device, non_blocking=True)


def plan_kda_replay_seed(
    conv_pools: Sequence[torch.Tensor],
    conv_caches: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    zero_buffers: Sequence[Sequence[torch.Tensor]],
) -> Optional[KDAReplaySeedPlan]:
    """Plan seeding every layer's replay caches with one launch per buffer kind.

    ``conv_pools[layer]`` is the live ``[slots, 3D, W - 1]`` conv pool view of a
    layer and ``conv_caches[layer]`` its q/k/v ``[slots, D, W - 1 + num_spec]``
    replay conv caches. Each entry of ``zero_buffers`` lists one replay buffer
    kind (``[slots, rows, cols]``) across the layers; seeding zeroes it. The
    views may live anywhere (e.g. per-layer views into KV cache pool pages) as
    long as each kind shares one shape, stride and dtype across layers.

    Returns None, so the caller seeds layer by layer, when they do not.
    """
    num_layers = len(conv_pools)
    if num_layers == 0 or len(conv_caches) != num_layers:
        return None
    flat_caches = [cache for caches in conv_caches for cache in caches]
    if not _shared_layout(conv_pools) or not _shared_layout(flat_caches):
        return None
    conv_pool = conv_pools[0]
    cache = flat_caches[0]
    if conv_pool.shape[1] % 3 or cache.device != conv_pool.device:
        return None
    projection_size = conv_pool.shape[1] // 3
    committed = conv_pool.shape[2]
    if cache.shape[:2] != (conv_pool.shape[0], projection_size) or cache.shape[2] < committed:
        return None
    zero_groups = []
    for buffers in zero_buffers:
        if len(buffers) != num_layers or not _shared_layout(buffers):
            return None
        reference = buffers[0]
        if reference.device != conv_pool.device or reference.shape[0] != conv_pool.shape[0]:
            return None
        zero_groups.append(
            _KDAReplayZeroGroup(
                ptrs=_pointer_table(buffers, conv_pool.device),
                strides=tuple(reference.stride()),
                dtype=reference.dtype,
                rows=reference.shape[1],
                cols=reference.shape[2],
            )
        )
    return KDAReplaySeedPlan(
        num_layers=num_layers,
        conv_ptrs=_pointer_table(conv_pools, conv_pool.device),
        conv_cache_ptrs=_pointer_table(flat_caches, conv_pool.device),
        conv_strides=tuple(conv_pool.stride()),
        cache_strides=tuple(cache.stride()),
        conv_dtype=conv_pool.dtype,
        cache_dtype=cache.dtype,
        projection_size=projection_size,
        committed=committed,
        extended=cache.shape[2],
        zero_groups=tuple(zero_groups),
    )


def seed_kda_replay_layers(plan: KDAReplaySeedPlan, state_indices: torch.Tensor) -> None:
    """Seed the replay caches of ``state_indices`` in every planned layer.

    Copies each layer's committed conv window into its q/k/v replay conv
    caches, zeroes their draft tails and zeroes the planned buffer kinds; the
    same result as doing so layer by layer.
    """
    if state_indices.ndim != 1 or state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA replay state indices must be a rank-1 int32 or int64 tensor")
    if state_indices.numel() == 0:
        return
    block_size = 256
    num_slots = state_indices.numel()
    with torch.cuda.device(plan.conv_ptrs.device.index):
        _seed_kda_replay_conv_layers_kernel[
            (
                num_slots,
                triton.cdiv(plan.projection_size * plan.extended, block_size),
                plan.num_layers,
            )
        ](
            plan.conv_ptrs,
            plan.conv_cache_ptrs,
            state_indices,
            *plan.conv_strides,
            *plan.cache_strides,
            CONV_DTYPE=_TORCH_TO_TL_DTYPE[plan.conv_dtype],
            CACHE_DTYPE=_TORCH_TO_TL_DTYPE[plan.cache_dtype],
            PROJECTION_SIZE=plan.projection_size,
            COMMITTED=plan.committed,
            EXTENDED=plan.extended,
            BLOCK_SIZE=block_size,
        )
        for group in plan.zero_groups:
            _zero_kda_replay_slots_kernel[
                (num_slots, triton.cdiv(group.rows * group.cols, block_size), plan.num_layers)
            ](
                group.ptrs,
                state_indices,
                *group.strides,
                DTYPE=_TORCH_TO_TL_DTYPE[group.dtype],
                ROWS=group.rows,
                COLS=group.cols,
                BLOCK_SIZE=block_size,
            )


@triton.jit
def _store_kda_replay_conv_window_kernel(
    conv_ptr,
    q_cache_ptr,
    k_cache_ptr,
    v_cache_ptr,
    state_indices_ptr,
    num_accepted_ptr,
    conv_stride_slot,
    conv_stride_dim,
    conv_stride_window,
    cache_stride_slot,
    cache_stride_dim,
    cache_stride_window,
    PROJECTION_SIZE: tl.constexpr,
    COMMITTED: tl.constexpr,
    NUM_SPEC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < PROJECTION_SIZE * COMMITTED
    dim = offsets // COMMITTED
    window = offsets % COMMITTED
    slot = tl.load(state_indices_ptr + request).to(tl.int64)
    # The window after the accepted drafts starts at their count, clamped
    # like the verify kernels clamp it.
    shift = tl.minimum(tl.maximum(tl.load(num_accepted_ptr + request), 0), NUM_SPEC)

    conv_offset = slot * conv_stride_slot + dim * conv_stride_dim + window * conv_stride_window
    cache_offset = (
        slot * cache_stride_slot + dim * cache_stride_dim + (window + shift) * cache_stride_window
    )
    section_offset = PROJECTION_SIZE * conv_stride_dim
    tl.store(conv_ptr + conv_offset, tl.load(q_cache_ptr + cache_offset, mask=mask), mask=mask)
    tl.store(
        conv_ptr + conv_offset + section_offset,
        tl.load(k_cache_ptr + cache_offset, mask=mask),
        mask=mask,
    )
    tl.store(
        conv_ptr + conv_offset + 2 * section_offset,
        tl.load(v_cache_ptr + cache_offset, mask=mask),
        mask=mask,
    )


def store_kda_replay_conv_window(
    conv_pool: torch.Tensor,
    q_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Store each selected slot's conv window after its accepted drafts in
    the live convolution pool.

    The replay caches hold the window before the golden token in columns
    ``[0, W - 1)`` and the raw inputs of the round's drafts after it, so the
    ``W - 1`` columns starting at the request's ``num_accepted_tokens`` entry
    are the window the next verify would start from. They are raw
    convolution inputs, so storing them in a bf16 pool is exact.
    """
    projection_size = conv_pool.shape[1] // 3
    committed = conv_pool.shape[2]
    caches = (q_cache, k_cache, v_cache)
    if any(cache.shape[1] != projection_size or cache.shape[2] < committed for cache in caches):
        raise ValueError("KDA replay convolution caches do not match the live pool geometry")
    if any(cache.stride() != q_cache.stride() for cache in caches[1:]):
        raise ValueError("KDA replay convolution caches must share one layout")
    if num_accepted_tokens.shape != state_indices.shape:
        raise ValueError("Expected one accepted-draft count per state index")
    if state_indices.numel() == 0:
        return

    block_size = 256
    grid = (
        state_indices.numel(),
        triton.cdiv(projection_size * committed, block_size),
    )
    with torch.cuda.device(conv_pool.device.index):
        _store_kda_replay_conv_window_kernel[grid](
            conv_pool,
            q_cache,
            k_cache,
            v_cache,
            state_indices,
            num_accepted_tokens,
            conv_pool.stride(0),
            conv_pool.stride(1),
            conv_pool.stride(2),
            q_cache.stride(0),
            q_cache.stride(1),
            q_cache.stride(2),
            PROJECTION_SIZE=projection_size,
            COMMITTED=committed,
            NUM_SPEC=q_cache.shape[2] - committed,
            BLOCK_SIZE=block_size,
        )


@triton.jit
def _replay_kda_token_records_kernel(
    state_ptr,
    records_ptr,
    state_indices_ptr,
    num_accepted_ptr,
    state_stride_slot,
    records_stride_slot,
    NUM_SPEC: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.program_id(2) * BLOCK_V + tl.arange(0, BLOCK_V)
    keys = tl.arange(0, K)
    slot = tl.load(state_indices_ptr + request).to(tl.int64)
    accepted = tl.minimum(tl.maximum(tl.load(num_accepted_ptr + request), 0), NUM_SPEC)
    state_ptrs = (
        state_ptr + slot * state_stride_slot + (head * V + rows[:, None]) * K + keys[None, :]
    )
    state = tl.load(state_ptrs)
    records = records_ptr + slot * records_stride_slot
    for t in range(0, accepted):
        vn = tl.load(records + (t * H + head) * V + rows)
        beta_k = tl.load(records + NUM_SPEC * H * V + (t * H + head) * K + keys)
        decay = tl.load(records + NUM_SPEC * H * (V + K) + (t * H + head) * K + keys)
        state = tl.fma(decay[None, :], state, vn[:, None] * beta_k[None, :])
    tl.store(state_ptrs, state)


def replay_kda_token_records(
    recurrent_state: torch.Tensor,
    state_tok: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Replay each selected slot's accepted drafts onto its pool state from
    the per-draft records the Kimi K3 verify kernels keep in ``state_tok``.

    A slot of ``state_tok`` (``[slots, 3, num_spec, H, K]`` fp32) holds the
    records of the last round's drafts: the row innovations ``vn``
    ``[num_spec][H][V]``, then ``beta * k`` and the decay
    ``[num_spec][H][K]`` (V = K). Draft ``t`` updates the state as the
    kernels do, ``S = fma(decay, S, vn (beta * k))``, so the result is
    bit-identical to the state their next verify starts from.
    """
    _, num_spec, num_heads, k_dim = state_tok.shape[1:]
    v_dim = recurrent_state.shape[-2]
    if state_tok.shape[1] != 3 or v_dim != k_dim:
        raise ValueError(
            "replay_kda_token_records: expected the drafts' records "
            "[slots, 3, num_spec, H, K] with V = K, got "
            f"{tuple(state_tok.shape)} for a state of "
            f"{tuple(recurrent_state.shape)}"
        )
    if state_indices.numel() == 0:
        return
    block_v = 32
    grid = (state_indices.numel(), num_heads, v_dim // block_v)
    with torch.cuda.device(recurrent_state.device.index):
        _replay_kda_token_records_kernel[grid](
            recurrent_state,
            state_tok,
            state_indices,
            num_accepted_tokens,
            recurrent_state.stride(0),
            state_tok.stride(0),
            NUM_SPEC=num_spec,
            H=num_heads,
            V=v_dim,
            K=k_dim,
            BLOCK_V=block_v,
        )


def commit_kda_pending_drafts(
    recurrent_state: torch.Tensor,
    conv_pool: torch.Tensor,
    conv_q: torch.Tensor,
    conv_k: torch.Tensor,
    conv_v: torch.Tensor,
    k_cache: Optional[torch.Tensor],
    g_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    beta_cache: Optional[torch.Tensor],
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_tok: Optional[torch.Tensor] = None,
) -> None:
    """Commit one layer's pending accepted drafts into its live pools.

    ``num_accepted_tokens[n]`` is the number of drafts of slot
    ``state_indices[n]`` the last verify accepted, which the next verify
    would take up first. Afterwards ``recurrent_state`` and ``conv_pool``
    hold the state and the conv window that verify would start from (the
    window bit for bit), and the caller clears the slot's count:

    * with the Kimi K3 verify kernels' draft records (``state_tok``), the
      drafts are replayed from those (``replay_kda_token_records``), and the
      replay caches, which are then not allocated, may be None;
    * otherwise they are replayed from the replay caches by the fused verify
      kernel's own replay steps (``kda_mtp_commit_pending_drafts``). The
      tensor-core verify computes a chunk in one pass, so this commit
      rounds the state differently from a verify that replays the drafts
      itself.

    The replay-only commit is main's (``REPLAY_ONLY`` in main's verify kernel); the branch has
    none. It reads the same replay caches the package's verify ops write.
    """
    num_spec = conv_q.shape[2] - conv_pool.shape[2]
    if state_tok is not None:
        replay_kda_token_records(recurrent_state, state_tok, state_indices, num_accepted_tokens)
    else:
        assert (
            k_cache is not None
            and g_cache is not None
            and v_cache is not None
            and beta_cache is not None
        )
        _load_main_mtp_module().kda_mtp_commit_pending_drafts(
            recurrent_state,
            conv_q,
            conv_k,
            conv_v,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            state_indices,
            num_accepted_tokens,
            num_spec,
            conv_width=conv_pool.shape[2] + 1,
        )
    store_kda_replay_conv_window(
        conv_pool, conv_q, conv_k, conv_v, state_indices, num_accepted_tokens
    )


# ---------------------------------------------------------------------------
# KDA op modules: the package's own ops over the branch's CuTe DSL kernels
# (prefill_ops.py: trtllm::k3r_kda_prefill; mtp_ops.py: trtllm::k3r_kda_mtp_decode*), and
# main's ops (trtllm::kda_prefill / kda_mtp_decode*) where a package module cannot be imported.
# ---------------------------------------------------------------------------

_PRIVATE_PREFILL_MODULE = ".prefill_ops"
_MAIN_PREFILL_MODULE = "tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops"
_PRIVATE_MTP_MODULE = ".mtp_ops"
_MAIN_MTP_MODULE = "tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops"

_OP_MODULES: dict[str, ModuleType] = {}
_OP_IMPORT_ERRORS: dict[str, Exception] = {}


def _load_op_module(name: str) -> ModuleType:
    """Import an op module once (registering its ops); a failed import is cached and re-raised.

    ``name`` is a module of this package (leading dot) or an absolute module of main.
    """
    module = _OP_MODULES.get(name)
    if module is not None:
        return module
    error = _OP_IMPORT_ERRORS.get(name)
    if error is not None:
        raise error
    try:
        module = importlib.import_module(name, __package__)
    except Exception as exc:  # typically ImportError when CuTe DSL / FlashInfer is unavailable
        _OP_IMPORT_ERRORS[name] = exc
        raise
    _OP_MODULES[name] = module
    return module


def _try_load_op_module(name: str) -> Optional[ModuleType]:
    """``_load_op_module`` or None; the first failure of a package module is logged with its
    error (a kernel module that cannot be imported, e.g. for a CuTe DSL API it lacks)."""
    try:
        return _load_op_module(name)
    except Exception as exc:  # see _load_op_module
        if name.startswith("."):
            from tensorrt_llm.logger import logger

            logger.warning_once(
                f"Kimi K3 KDA: cannot import the private op module {name[1:]} "
                f"({type(exc).__name__}: {exc}); its kernels do not run.",
                key=f"k3_kda_op_module_{name}",
            )
        return None


@functools.cache
def kda_prefill_backend() -> Optional[str]:
    """The op of the optimized indexed prefill path, imported (registered) on first call.

    ``"cute"``: the package's ``trtllm::k3r_kda_prefill`` (the branch's CuTe DSL
    ``KdaPrefillRunner`` on L2-normalized Q/K, fp32 state rows, mid-chunk snapshots).
    ``"flashinfer"``: main's ``trtllm::kda_prefill`` and ``kda_prefill_conv`` (raw Q/K/V), only
    when the package's module cannot be imported. None: neither (the FLA fallback runs).
    """
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        return None
    if _try_load_op_module(_PRIVATE_PREFILL_MODULE) is not None:
        return "cute"
    if _try_load_op_module(_MAIN_PREFILL_MODULE) is not None:
        return "flashinfer"
    return None


def is_intree_prefill_available() -> bool:
    """True when an optimized prefill op (the package's CuTe DSL or main's FlashInfer) is available."""
    return kda_prefill_backend() is not None


def kda_prefill_supports_state_snapshots() -> bool:
    """True when Kimi K3 KDA layers prefill with the indexed kernel, which takes state snapshots.

    Mirrors the prefill path ``KimiKDALinearAttention`` selects: only the package's CuTe DSL op
    stores mid-chunk snapshots (``snapshot_rows`` / ``snapshot_chunks``,
    TRTLLM_KDA_MIDCHUNK_SNAPSHOT). ``KimiK3HybridCacheManager`` asks this before planning them.
    """
    return (
        os.getenv("TLLM_KDA_ENABLE_OPT_PREFILL", "1") == "1"
        and is_kda_optimized_supported()
        and kda_prefill_backend() == "cute"
    )


@functools.cache
def kda_mtp_backend() -> Optional[str]:
    """The ops of the fused multi-token verify path, imported (registered) on first call.

    ``"private"``: ``trtllm::k3r_kda_mtp_decode`` / ``_fp8`` / ``_fused_fb`` (the branch's v1,
    v2 and two-CTA kernels). ``"main"``: main's ``trtllm::kda_mtp_decode`` /
    ``_fp8``, only when the package's module cannot be imported. None: neither.
    """
    if _try_load_op_module(_PRIVATE_MTP_MODULE) is not None:
        return "private"
    if _try_load_op_module(_MAIN_MTP_MODULE) is not None:
        return "main"
    return None


# Prefix of the registered verify ops, by backend.
_MTP_OP_PREFIX = {"private": "k3r_", "main": ""}


def _load_mtp_module() -> ModuleType:
    """The verify op module the dispatch runs (see ``kda_mtp_backend``)."""
    backend = kda_mtp_backend()
    if backend is None:
        raise _OP_IMPORT_ERRORS[_PRIVATE_MTP_MODULE]
    return _OP_MODULES[_PRIVATE_MTP_MODULE if backend == "private" else _MAIN_MTP_MODULE]


def _load_main_mtp_module() -> ModuleType:
    """Main's verify op module (its replay-only draft commit, ``kda_mtp_commit_pending_drafts``)."""
    return _load_op_module(_MAIN_MTP_MODULE)


def is_intree_mtp_available() -> bool:
    """True when a CuTe DSL MTP verify op (the package's or main's) can be imported."""
    return kda_mtp_backend() is not None


def is_kda_mtp_verify_available() -> bool:
    """True when the fused multi-token verify kernel can run here.

    Used by the executor (cache-manager sizing) to decide whether to
    allocate the KDA replay caches instead of the legacy intermediate
    verification buffers. (Main's executor asks main's copy of this module.)
    """
    return is_kda_optimized_supported() and is_intree_mtp_available()


def kda_mtp_max_num_spec() -> int:
    """Largest draft count the fused multi-token verify kernels support: the replayed drafts
    and the ``1 + num_spec`` new tokens fit the padded chunk (``T_PAD``)."""
    if kda_mtp_backend() == "private":
        from ..kernels.kda.kda_mtp_decode import T_PAD

        return (T_PAD - 1) // 2
    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda.kda_mtp_decode import (
        MAX_NUM_SPEC,
    )

    return MAX_NUM_SPEC


def kda_mtp_replay_num_spec(num_spec: int) -> Optional[int]:
    """Draft count to size the KDA replay caches for, or ``None``.

    ``None`` keeps the legacy intermediate verification buffers, so the
    module runs the sequential FLA verify: either the fused kernel cannot
    run here, or ``num_spec`` exceeds what it supports.
    """
    if not is_kda_mtp_verify_available():
        return None
    max_num_spec = kda_mtp_max_num_spec()
    if num_spec > max_num_spec:
        from tensorrt_llm.logger import logger

        logger.warning_once(
            f"KDA fused multi-token verify supports at most {max_num_spec} draft "
            f"tokens, got {num_spec}; using the sequential FLA verify.",
            key="kda_mtp_num_spec_fallback",
        )
        return None
    return num_spec


def kda_mtp_verify_ctas(num_heads: int, num_requests: int, device: torch.device) -> Optional[int]:
    """CTAs of the v2 verify kernel's grid for these tiles (the package ops' ``verify_ctas``:
    cluster pairs up to ``SPLIT_MAX_TILES`` tiles, or one CTA per tile), or None when the verify
    ops are main's, which have no v2 kernel."""
    if kda_mtp_backend() != "private":
        return None
    return _OP_MODULES[_PRIVATE_MTP_MODULE].verify_ctas(num_heads, num_requests, device)


# Argument names of the MTP verify ops, by op name.
_MTP_OP_ARGUMENTS: dict[str, frozenset[str]] = {}
# Registered verify ops, by logical name (the package's module registers more than main's).
_MTP_OP_PRESENT: dict[str, bool] = {}


def _supported_op_kwargs(op, kwargs: dict) -> dict:
    """``kwargs`` restricted to the arguments ``op`` takes.

    The package's verify ops take every argument the mixer passes. Main's ops (the fallback when
    the package's module cannot be imported) take no ``use_pdl`` (the PDL launch of
    TRTLLM_K3_KDA_DECODE_FUSED_FAB) and run without it; the first time a dropped argument asks
    for a feature (a truthy value), a warning names it.
    """
    schema = op.default._schema
    names = _MTP_OP_ARGUMENTS.get(schema.name)
    if names is None:
        names = frozenset(argument.name for argument in schema.arguments)
        _MTP_OP_ARGUMENTS[schema.name] = names
    if all(key in names for key in kwargs):
        return kwargs
    requested = sorted(key for key, value in kwargs.items() if key not in names and value)
    if requested:
        from tensorrt_llm.logger import logger

        logger.warning_once(
            f"Kimi K3 KDA: {schema.name} takes no {', '.join(requested)}; the verify kernel "
            "runs without them (they need the private KDA MTP ops).",
            key=f"k3_kda_mtp_dropped_{schema.name}_{'_'.join(requested)}",
        )
    return {key: value for key, value in kwargs.items() if key in names}


# ---------------------------------------------------------------------------
# Dispatch API used by the KimiKDALinearAttention module.
# ---------------------------------------------------------------------------


class KDAKernelDispatch:
    """Kernel dispatch state for one ``KimiKDALinearAttention`` instance.

    Attributes
    ----------
    prefill_kernel_path : str
        Selected prefill path: ``"optimized"`` or ``"fla"``.
    prefill_backend : Optional[str]
        Op of the optimized prefill path: ``"cute"`` (the package's
        ``trtllm::k3r_kda_prefill``, the branch's CuTe DSL kernels) or
        ``"flashinfer"`` (main's ``trtllm::kda_prefill``); None on the FLA path.
    decode_kernel_path : str
        Selected decode path: ``"optimized"`` or ``"fla"``.
    verify_kernel_path : str
        Selected multi-token verify path: ``"optimized"`` (fused replay
        kernel) or ``"fla"`` (sequential per-step ``fused_recurrent_kda``
        with intermediate-buffer state promotion).
    mtp_backend : Optional[str]
        Ops of the optimized verify path: ``"private"``
        (``trtllm::k3r_kda_mtp_decode*``, the branch's kernels) or ``"main"``
        (main's ``trtllm::kda_mtp_decode`` / ``_fp8``); None on the FLA path.
    Notes
    -----
    Prefill, decode, and verify dispatch are decided independently. All
    require a supported GPU; the CuTe DSL ops additionally require their
    modules to be importable.
    """

    _selection_logged = False

    def __init__(
        self,
        use_optimized_prefill: bool = True,
        use_optimized_decode: bool = True,
        use_optimized_verify: bool = True,
    ) -> None:
        optimized_supported = is_kda_optimized_supported()
        self.prefill_backend: Optional[str] = None
        if use_optimized_prefill and optimized_supported:
            self.prefill_backend = kda_prefill_backend()
        self.prefill_kernel_path = "fla" if self.prefill_backend is None else "optimized"
        self.decode_kernel_path = (
            "optimized" if use_optimized_decode and optimized_supported else "fla"
        )
        self.mtp_backend: Optional[str] = None
        if use_optimized_verify and optimized_supported:
            self.mtp_backend = kda_mtp_backend()
        self.verify_kernel_path = "fla" if self.mtp_backend is None else "optimized"
        # One line per process so runs record which paths actually executed
        # (the fallback is otherwise silent).
        if not KDAKernelDispatch._selection_logged:
            KDAKernelDispatch._selection_logged = True
            try:
                from tensorrt_llm.logger import logger
            except ImportError:  # pragma: no cover — source-loader stub path
                pass
            else:
                logger.info(
                    f"KDA kernel dispatch: prefill={self.prefill_kernel_path}"
                    f"[{self.prefill_backend}] decode={self.decode_kernel_path} "
                    f"verify={self.verify_kernel_path}[{self.mtp_backend}]"
                )
                if self.prefill_backend == "flashinfer" or self.mtp_backend == "main":
                    logger.warning(
                        "Kimi K3 KDA: the private CuTe DSL ops (k3_rubin/kda/prefill_ops.py, "
                        "mtp_ops.py) did not import; main's KDA kernels run instead "
                        f"(prefill={self.prefill_backend}, verify={self.mtp_backend})."
                    )

    def _mtp_op(self, name: str):
        """The registered verify op of logical name ``name`` (e.g. ``"kda_mtp_decode"``)."""
        _load_mtp_module()  # registers the ops
        return getattr(torch.ops.trtllm, _MTP_OP_PREFIX[self.mtp_backend] + name)

    def has_mtp_op(self, name: str) -> bool:
        """Whether the optimized verify path has the verify op of logical name ``name``.

        The package's ops module registers all three: ``kda_mtp_decode``, ``kda_mtp_decode_fp8``
        and ``kda_mtp_decode_fused_fb`` (TRTLLM_K3_KDA_DECODE_FUSED_FAB), as
        ``trtllm::k3r_<name>``.
        Main's module (the fallback) registers the first two only, and callers keep to those.
        """
        if self.verify_kernel_path != "optimized":
            return False
        present = _MTP_OP_PRESENT.get(name)
        if present is None:
            try:
                self._mtp_op(name)
            except (AttributeError, RuntimeError):
                present = False
            else:
                present = True
            _MTP_OP_PRESENT[name] = present
        return present

    def mtp_verify(
        self, *, mxfp8_output: bool = False, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the fused KDA multi-token verify kernel.

        Thin passthrough to ``kda_mtp_decode``, or with ``mxfp8_output`` to
        ``kda_mtp_decode_fp8``, which always fuses the output norm and returns
        the E4M3 core and its R128c4 scales (see ``mtp_ops.py`` for the full
        argument and state-management contract). Only defined on the
        optimized path; the FLA fallback is the module's sequential
        per-step loop with intermediate-buffer promotion.

        The MXFP8-output op takes no ``use_pdl`` (as on the branch); main's ops (the fallback)
        drop the other arguments they do not take (``_supported_op_kwargs``).
        """
        if self.verify_kernel_path != "optimized":
            raise RuntimeError(
                "mtp_verify called on non-optimized path; use the module's "
                "sequential FLA verify fallback instead."
            )
        if mxfp8_output:
            if not kwargs.pop("fuse_output_norm", True):
                raise ValueError("the MXFP8-output KDA MTP op always fuses the output norm")
            kwargs.pop("use_pdl", None)
            op = self._mtp_op("kda_mtp_decode_fp8")
        else:
            op = self._mtp_op("kda_mtp_decode")
        return op(**_supported_op_kwargs(op, kwargs))

    def mtp_verify_fused_fb(self, **kwargs) -> torch.Tensor:
        """Fused KDA multi-token verify that computes its forget gate from ``f_a``.

        Thin passthrough to ``trtllm::k3r_kda_mtp_decode_fused_fb`` (packed rows, fused gated
        RMSNorm, BF16 output). Callers check ``has_mtp_op("kda_mtp_decode_fused_fb")`` first.
        """
        if not self.has_mtp_op("kda_mtp_decode_fused_fb"):
            raise RuntimeError(
                "kda_mtp_decode_fused_fb needs the optimized verify path and the private KDA "
                "MTP ops (TRTLLM_K3_KDA_DECODE_FUSED_FAB)"
            )
        return self._mtp_op("kda_mtp_decode_fused_fb")(**kwargs)

    def mxfp8_pdl_early_trigger(self, enabled: bool):
        """Scope in which an MXFP8-output verify launch releases its dependents right after its
        own ``griddepcontrol.wait`` (TRTLLM_K3_KDA_MX_PDL).

        The package's ops module's ``mxfp8_pdl_early_trigger``. Main's ops module (the fallback)
        has no such scope, so there this is a no-op and the launch is an ordinary one.
        """
        if not enabled:
            return contextlib.nullcontext()
        trigger = getattr(_load_mtp_module(), "mxfp8_pdl_early_trigger", None)
        if trigger is None:
            from tensorrt_llm.logger import logger

            logger.warning_once(
                "Kimi K3 KDA: TRTLLM_K3_KDA_MX_PDL needs the private KDA MTP ops "
                "(mxfp8_pdl_early_trigger); the MXFP8 verify launches without it.",
                key="k3_kda_mx_pdl_unavailable",
            )
            return contextlib.nullcontext()
        return trigger(True)

    def can_use_indexed_prefill(
        self,
        *,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        num_sequences: int,
        num_tokens: int,
    ) -> bool:
        """Return whether prefill can update this V-first pool directly.

        The package's CuTe DSL op takes fp32 rows (a bf16 pool is staged through an fp32 copy
        first) and int32 / int64 16-byte-aligned indices; main's FlashInfer op takes BF16 or
        FP32 rows, int32 indices and int64 offsets.
        """
        if self.prefill_kernel_path != "optimized" or num_tokens == 0:
            return False
        if self.prefill_backend == "cute":
            return self._can_use_cute_indexed_prefill(
                state_pool, state_indices, has_initial_states, cu_seqlens, num_sequences
            )
        if (
            cu_seqlens is None
            or cu_seqlens.dtype != torch.int64
            or cu_seqlens.shape != (num_sequences + 1,)
            or not cu_seqlens.is_contiguous()
        ):
            return False
        if state_pool.dtype not in (torch.bfloat16, torch.float32) or state_pool.ndim != 4:
            return False
        _, H, V, K = state_pool.shape
        if state_pool.stride()[1:] != (V * K, K, 1):
            return False
        if (
            (V, K) != (128, 128)
            or state_pool.stride(0) < H * V * K
            or state_pool.stride(0) * state_pool.element_size() % 16
        ):
            return False
        if state_pool.data_ptr() % 16:
            return False
        if (
            state_indices.ndim != 1
            or state_indices.shape[0] != num_sequences
            or state_indices.dtype != torch.int32
            or not state_indices.is_contiguous()
        ):
            return False
        if (
            has_initial_states.ndim != 1
            or has_initial_states.shape[0] != num_sequences
            or has_initial_states.dtype != torch.bool
            or not has_initial_states.is_contiguous()
        ):
            return False
        if (
            state_pool.device != state_indices.device
            or state_pool.device != has_initial_states.device
        ):
            return False
        return True

    @staticmethod
    def _can_use_cute_indexed_prefill(
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        num_sequences: int,
    ) -> bool:
        """The branch's ``can_use_indexed_prefill`` (the package's CuTe DSL op's layouts)."""
        if cu_seqlens is not None:
            if cu_seqlens.shape[0] - 1 != num_sequences:
                return False
        if state_pool.dtype != torch.float32 or state_pool.ndim != 4:
            return False
        _, H, V, K = state_pool.shape
        if state_pool.stride()[1:] != (V * K, K, 1):
            return False
        if state_pool.stride(0) < H * V * K or state_pool.stride(0) % 4:
            return False
        if state_pool.data_ptr() % 16:
            return False
        if (
            state_indices.ndim != 1
            or state_indices.shape[0] != num_sequences
            or state_indices.dtype not in (torch.int32, torch.int64)
            or not state_indices.is_contiguous()
        ):
            return False
        if state_indices.data_ptr() % 16:
            return False
        if (
            has_initial_states.ndim != 1
            or has_initial_states.shape[0] != num_sequences
            or has_initial_states.dtype != torch.bool
            or not has_initial_states.is_contiguous()
        ):
            return False
        if (
            state_pool.device != state_indices.device
            or state_pool.device != has_initial_states.device
        ):
            return False
        return True

    def prefill_chunk_kda(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        safe_gate: bool,
        lower_bound: Optional[float],
        cu_seqlens: Optional[torch.Tensor],
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
        state_pool: Optional[torch.Tensor] = None,
        state_indices: Optional[torch.Tensor] = None,
        varlen_is_aligned: Optional[bool] = None,
        single_sequence_length: Optional[int] = None,
        snapshot_rows: Optional[torch.Tensor] = None,
        snapshot_chunks: Optional[torch.Tensor] = None,
        max_seqlen: int = 0,
        seq_order: Optional[torch.Tensor] = None,
        cu_chunks: Optional[torch.Tensor] = None,
        qk_normalized: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run KDA chunked prefill.

        * ``prefill_backend == "cute"`` with ``state_pool``: the package's
          ``trtllm::k3r_kda_prefill`` (the branch's ``KdaPrefillRunner``) on
          L2-normalized Q/K, updating indexed fp32 V-first ``[slots, H, V, K]`` rows in place.
          ``chunk_indices``, ``varlen_is_aligned`` and ``single_sequence_length`` are prepared
          once from Kimi K3 runtime metadata so no layer reads ``cu_seqlens`` back from the GPU.
          ``snapshot_rows`` / ``snapshot_chunks`` (int32 ``[num_sequences]``) also store each
          selected sequence's state after its first ``64 * snapshot_chunks[b]`` tokens into
          ``state_pool[snapshot_rows[b]]``; ``-1`` skips a sequence.
        * ``prefill_backend == "flashinfer"`` with ``state_pool``: main's ``trtllm::kda_prefill``
          on raw Q/K/V and gate/beta logits (``max_seqlen`` / ``seq_order`` / ``cu_chunks``
          are its scheduling metadata), updating indexed BF16/FP32 rows directly.
        * otherwise FLA ``chunk_kda`` with dense FP32 initial/final states; it normalizes Q/K
          itself unless ``qk_normalized``.
        """
        if state_pool is not None and self.prefill_backend == "cute":
            if initial_state is not None or state_indices is None:
                raise ValueError(
                    "Indexed KDA prefill requires state_indices and does not accept initial_state."
                )
            return self._prefill_cute(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
                state_pool=state_pool,
                state_indices=state_indices,
                varlen_is_aligned=varlen_is_aligned,
                single_sequence_length=single_sequence_length,
                snapshot_rows=snapshot_rows,
                snapshot_chunks=snapshot_chunks,
            )
        if snapshot_rows is not None:
            raise RuntimeError(
                "KDA prefill state snapshots require the private CuTe DSL prefill op "
                "(trtllm::k3r_kda_prefill)."
            )
        if state_pool is not None:
            if self.prefill_kernel_path != "optimized" or initial_state is not None:
                raise ValueError("Indexed prefill requires FlashInfer and no dense initial state")
            if not safe_gate or lower_bound is None:
                raise ValueError("FlashInfer requires a finite negative gate bound")
            _load_op_module(_MAIN_PREFILL_MODULE)  # registers trtllm::kda_prefill
            out = torch.ops.trtllm.kda_prefill(
                q,
                k,
                v,
                g,
                beta.to(torch.bfloat16),
                state_pool,
                state_indices,
                cu_seqlens,
                A_log.float().contiguous(),
                dt_bias.float().contiguous(),
                scale,
                lower_bound,
                getattr(self, "prefill_mode", "auto"),
                max_seqlen,
                seq_order,
                cu_chunks,
            )
            return out, None

        from fla.ops.kda import chunk_kda

        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=not qk_normalized,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )

    @staticmethod
    def _prefill_cute(
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        safe_gate: bool,
        lower_bound: Optional[float],
        cu_seqlens: Optional[torch.Tensor],
        chunk_indices: Optional[torch.Tensor],
        chunk_size: int,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        varlen_is_aligned: Optional[bool],
        single_sequence_length: Optional[int],
        snapshot_rows: Optional[torch.Tensor],
        snapshot_chunks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, None]:
        """The branch's optimized ``prefill_chunk_kda`` on ``trtllm::k3r_kda_prefill``."""
        import torch.nn.functional as F

        snapshot_kwargs = (
            {}
            if snapshot_rows is None
            else {"snapshot_rows": snapshot_rows, "snapshot_chunks": snapshot_chunks}
        )
        real_T = q.shape[1]
        if (
            cu_seqlens is not None
            and single_sequence_length is not None
            and q.shape[1] != single_sequence_length
        ):
            # Caller already produced chunk-padded inputs (zero tails);
            # only the output needs trimming back to the real length.
            real_T = single_sequence_length
            padded_T = q.shape[1]
            if (
                cu_seqlens.shape[0] != 2
                or padded_T % chunk_size != 0
                or not 0 < padded_T - real_T < chunk_size
                or any(t.shape[1] != padded_T for t in (k, v, beta))
                or g.shape[1] not in (real_T, padded_T)
            ):
                raise ValueError(
                    "Pre-padded KDA prefill inputs must be one sequence padded to the "
                    f"next {chunk_size}-token boundary (real_T={real_T}, "
                    f"q/k/v/g/beta lengths={[t.shape[1] for t in (q, k, v, g, beta)]})."
                )
        elif cu_seqlens is not None:
            # The op's varlen single-seq path (Phase 2.1) expects the
            # caller to zero-pad the packed tensors to a chunk multiple
            # (FLA convention) while cu_seqlens keeps the real length;
            # the op re-sentinels g's tail itself. Without this the op
            # would run the mask-free kernel on a partial final chunk.
            # Multi-seq varlen runs the masked path and needs no pad.
            if cu_seqlens.shape[0] == 2 and real_T % chunk_size != 0:
                pad = chunk_size - real_T % chunk_size
                q = F.pad(q, (0, 0, 0, 0, 0, pad))
                k = F.pad(k, (0, 0, 0, 0, 0, pad))
                v = F.pad(v, (0, 0, 0, 0, 0, pad))
                g = F.pad(g, (0, 0, 0, 0, 0, pad))
                beta = F.pad(beta, (0, 0, 0, pad))

        A_log_kernel = A_log.detach() if A_log is not None else None
        dt_bias_kernel = dt_bias.detach() if dt_bias is not None else None

        _load_op_module(_PRIVATE_PREFILL_MODULE)  # registers trtllm::k3r_kda_prefill
        out = torch.ops.trtllm.k3r_kda_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            state_pool=state_pool,
            state_indices=state_indices,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            A_log=A_log_kernel,
            dt_bias=dt_bias_kernel,
            varlen_is_aligned=varlen_is_aligned,
            single_sequence_length=single_sequence_length,
            **snapshot_kwargs,
        )
        if out.shape[1] != real_T:
            out = out[:, :real_T]
        return out, None

    def decode_kda(self, **kwargs) -> torch.Tensor:
        """Run the fused KDA single-token decode kernel.

        This is only defined on the optimized path; the FLA fallback runs
        ``fla.ops.kda.fused_recurrent_kda`` directly and does not use this
        wrapper (see ``_decode_via_fla`` on the module).
        """
        if self.decode_kernel_path != "optimized":
            raise RuntimeError(
                "decode_kda called on non-optimized path; use FLA path via "
                "the module's fallback handling instead."
            )
        return _kda_decode.run_kda_decode_fusion_cuda(**kwargs)
