# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton kernels for DeepSeek V4 ratio-0 cached sparse attention."""

from typing import Optional

import torch
import triton
import triton.language as tl

_RATIO0_TOPK_WIDTH = 128
_RATIO0_LOCAL_HEADS = 8
_RATIO0_HEAD_DIM = 512
_RATIO0_NOPE_DIM = 448
_RATIO0_ROPE_DIM = 64
_RATIO0_FP8_BLOCK_SIZE = 128
_RATIO0_SCALE_BLOCKS = 4
_FP8_E4M3_DTYPE = torch.float8_e4m3fn


def _device_skip_reason(name: str, tensor: torch.Tensor) -> str | None:
    if not tensor.is_cuda:
        return f"{name} must be a CUDA tensor for the Triton ratio-0 SWA path"
    return None


def _e8m0_dtype() -> torch.dtype | None:
    return getattr(torch, "float8_e8m0fnu", None)


def _scale_cache_raw_bytes(scale_cache: torch.Tensor) -> torch.Tensor:
    if scale_cache.dtype == torch.uint8:
        return scale_cache

    e8m0_dtype = _e8m0_dtype()
    if e8m0_dtype is not None and scale_cache.dtype == e8m0_dtype:
        return scale_cache.view(torch.uint8)

    raise TypeError(
        "scale_cache must contain raw E8M0 bytes as torch.uint8 or "
        f"{e8m0_dtype}; got {scale_cache.dtype}"
    )


def _cache_layout_for_triton(cache: torch.Tensor) -> tuple[int, int, int]:
    """Return ``(tokens_per_block, page_stride, token_stride)`` for supported cache layouts."""
    if cache.dim() == 5:
        if int(cache.shape[1]) == 1:
            token_dim = 2 if int(cache.shape[2]) >= int(cache.shape[3]) else 3
        else:
            token_dim = 1
    elif cache.dim() in (3, 4):
        token_dim = 1
    else:
        raise ValueError(f"swa_cache must have rank 3, 4, or 5, got rank {cache.dim()}")

    return int(cache.shape[token_dim]), int(cache.stride(0)), int(cache.stride(token_dim))


def deepseek_v4_ratio0_swa_triton_skip_reason(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    out: Optional[torch.Tensor],
    active_tokens: Optional[int] = None,
) -> str | None:
    """Return why the ratio-0 Triton SWA path cannot run, or ``None`` if it can."""
    if compress_ratio != 0:
        return f"compress_ratio must be 0 for the Triton SWA path, got {compress_ratio}"
    if window_size != _RATIO0_TOPK_WIDTH:
        return f"window_size must be {_RATIO0_TOPK_WIDTH}, got {window_size}"
    if q.dim() != 4:
        return f"q must have rank 4, got rank {q.dim()}"
    if kv.dim() != 3:
        return f"kv must have rank 3, got rank {kv.dim()}"
    if topk_idxs.dim() != 3:
        return f"topk_idxs must have rank 3, got rank {topk_idxs.dim()}"
    if q.shape[1] != 1:
        return "the Triton ratio-0 SWA path currently supports decode-only q.shape[1] == 1"
    if q.shape[2] != _RATIO0_LOCAL_HEADS:
        return f"q local head count must be {_RATIO0_LOCAL_HEADS}, got {q.shape[2]}"
    if q.shape[3] != _RATIO0_HEAD_DIM:
        return f"q head dimension must be {_RATIO0_HEAD_DIM}, got {q.shape[3]}"
    if kv.shape != (q.shape[0], q.shape[1], _RATIO0_HEAD_DIM):
        return (
            f"kv must have shape {(q.shape[0], q.shape[1], _RATIO0_HEAD_DIM)}, "
            f"got {tuple(kv.shape)}"
        )
    if attn_sink.shape != (_RATIO0_LOCAL_HEADS,):
        return f"attn_sink must have shape ({_RATIO0_LOCAL_HEADS},), got {tuple(attn_sink.shape)}"
    if topk_idxs.shape != (q.shape[0], q.shape[1], _RATIO0_TOPK_WIDTH):
        return (
            f"topk_idxs must have shape {(q.shape[0], q.shape[1], _RATIO0_TOPK_WIDTH)}, "
            f"got {tuple(topk_idxs.shape)}"
        )
    if q.dtype != torch.bfloat16:
        return f"q must be bfloat16, got {q.dtype}"
    if kv.dtype != torch.bfloat16:
        return f"kv must be bfloat16, got {kv.dtype}"
    if attn_sink.dtype != torch.float32:
        return f"attn_sink must be float32, got {attn_sink.dtype}"
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        return f"topk_idxs must be int32 or int64, got {topk_idxs.dtype}"
    if swa_cache.dtype != torch.bfloat16:
        return f"swa_cache must be bfloat16, got {swa_cache.dtype}"
    if swa_cache.shape[-1] != _RATIO0_HEAD_DIM:
        return f"swa_cache last dimension must be {_RATIO0_HEAD_DIM}, got {swa_cache.shape[-1]}"
    if swa_cache.stride(-1) != 1:
        return f"swa_cache last dimension must be contiguous, got stride {swa_cache.stride(-1)}"
    if q.stride(-1) != 1:
        return f"q last dimension must be contiguous, got stride {q.stride(-1)}"
    if kv.stride(-1) != 1:
        return f"kv last dimension must be contiguous, got stride {kv.stride(-1)}"
    if out is not None:
        if out.shape != q.shape:
            return f"out shape must be {tuple(q.shape)}, got {tuple(out.shape)}"
        if out.dtype != q.dtype:
            return f"out dtype must be {q.dtype}, got {out.dtype}"
        if out.stride(-1) != 1:
            return f"out last dimension must be contiguous, got stride {out.stride(-1)}"

    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("attn_sink", attn_sink),
        ("topk_idxs", topk_idxs),
        ("seq_len_host", seq_len_host),
        ("input_pos_host", input_pos_host),
        ("cu_seqlen_host", cu_seqlen_host),
        ("cache_loc_host", cache_loc_host),
        ("cu_num_pages_host", cu_num_pages_host),
        ("swa_cache", swa_cache),
    ):
        reason = _device_skip_reason(name, tensor)
        if reason is not None:
            return reason
    if out is not None:
        reason = _device_skip_reason("out", out)
        if reason is not None:
            return reason

    try:
        tokens_per_block, _, _ = _cache_layout_for_triton(swa_cache)
    except ValueError as exc:
        return str(exc)
    if tokens_per_block <= 0:
        return f"swa_cache tokens_per_block must be positive, got {tokens_per_block}"
    if active_tokens is not None:
        if active_tokens <= 0:
            return f"active_tokens must be positive, got {active_tokens}"
        if active_tokens > q.shape[0]:
            return f"active_tokens ({active_tokens}) exceed decode batch capacity ({q.shape[0]})"
        if input_pos_host.numel() < active_tokens:
            return (
                f"input_pos_host needs at least {active_tokens} entries, "
                f"got {input_pos_host.numel()}"
            )
        if cu_num_pages_host.numel() < active_tokens + 1:
            return (
                f"cu_num_pages_host needs at least {active_tokens + 1} entries, "
                f"got {cu_num_pages_host.numel()}"
            )

    return None


def deepseek_v4_ratio0_swa_fp8_triton_skip_reason(
    q: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    window_size: int,
    out: Optional[torch.Tensor],
    active_tokens: Optional[int] = None,
) -> str | None:
    """Return why the ratio-0 split FP8 SWA cache path cannot run, or ``None``."""
    if window_size != _RATIO0_TOPK_WIDTH:
        return f"window_size must be {_RATIO0_TOPK_WIDTH}, got {window_size}"
    if q.dim() != 4:
        return f"q must have rank 4, got rank {q.dim()}"
    if topk_idxs.dim() != 3:
        return f"topk_idxs must have rank 3, got rank {topk_idxs.dim()}"
    if q.shape[1] != 1:
        return (
            "the Triton ratio-0 split FP8 SWA path currently supports decode-only q.shape[1] == 1"
        )
    if q.shape[2] != _RATIO0_LOCAL_HEADS:
        return f"q local head count must be {_RATIO0_LOCAL_HEADS}, got {q.shape[2]}"
    if q.shape[3] != _RATIO0_HEAD_DIM:
        return f"q head dimension must be {_RATIO0_HEAD_DIM}, got {q.shape[3]}"
    if attn_sink.shape != (_RATIO0_LOCAL_HEADS,):
        return f"attn_sink must have shape ({_RATIO0_LOCAL_HEADS},), got {tuple(attn_sink.shape)}"
    if topk_idxs.shape != (q.shape[0], q.shape[1], _RATIO0_TOPK_WIDTH):
        return (
            f"topk_idxs must have shape {(q.shape[0], q.shape[1], _RATIO0_TOPK_WIDTH)}, "
            f"got {tuple(topk_idxs.shape)}"
        )
    if q.dtype != torch.bfloat16:
        return f"q must be bfloat16, got {q.dtype}"
    if attn_sink.dtype != torch.float32:
        return f"attn_sink must be float32, got {attn_sink.dtype}"
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        return f"topk_idxs must be int32 or int64, got {topk_idxs.dtype}"
    if nope_cache.dtype != _FP8_E4M3_DTYPE:
        return f"nope_cache must be {_FP8_E4M3_DTYPE}, got {nope_cache.dtype}"
    if rope_cache.dtype != torch.bfloat16:
        return f"rope_cache must be bfloat16, got {rope_cache.dtype}"
    try:
        _scale_cache_raw_bytes(scale_cache)
    except TypeError as exc:
        return str(exc)
    if nope_cache.shape[-1] != _RATIO0_NOPE_DIM:
        return f"nope_cache last dimension must be {_RATIO0_NOPE_DIM}, got {nope_cache.shape[-1]}"
    if rope_cache.shape[-1] != _RATIO0_ROPE_DIM:
        return f"rope_cache last dimension must be {_RATIO0_ROPE_DIM}, got {rope_cache.shape[-1]}"
    if scale_cache.shape[-1] != _RATIO0_SCALE_BLOCKS:
        return (
            f"scale_cache last dimension must be {_RATIO0_SCALE_BLOCKS}, "
            f"got {scale_cache.shape[-1]}"
        )
    if q.stride(-1) != 1:
        return f"q last dimension must be contiguous, got stride {q.stride(-1)}"
    if nope_cache.stride(-1) != 1:
        return f"nope_cache last dimension must be contiguous, got stride {nope_cache.stride(-1)}"
    if rope_cache.stride(-1) != 1:
        return f"rope_cache last dimension must be contiguous, got stride {rope_cache.stride(-1)}"
    if scale_cache.stride(-1) != 1:
        return f"scale_cache last dimension must be contiguous, got stride {scale_cache.stride(-1)}"
    if out is not None:
        if out.shape != q.shape:
            return f"out shape must be {tuple(q.shape)}, got {tuple(out.shape)}"
        if out.dtype != q.dtype:
            return f"out dtype must be {q.dtype}, got {out.dtype}"
        if out.stride(-1) != 1:
            return f"out last dimension must be contiguous, got stride {out.stride(-1)}"

    for name, tensor in (
        ("q", q),
        ("attn_sink", attn_sink),
        ("topk_idxs", topk_idxs),
        ("seq_len_host", seq_len_host),
        ("input_pos_host", input_pos_host),
        ("cu_seqlen_host", cu_seqlen_host),
        ("cache_loc_host", cache_loc_host),
        ("cu_num_pages_host", cu_num_pages_host),
        ("nope_cache", nope_cache),
        ("rope_cache", rope_cache),
        ("scale_cache", scale_cache),
    ):
        reason = _device_skip_reason(name, tensor)
        if reason is not None:
            return reason
    if out is not None:
        reason = _device_skip_reason("out", out)
        if reason is not None:
            return reason

    try:
        tokens_per_block, _, _ = _cache_layout_for_triton(nope_cache)
        rope_tokens_per_block, _, _ = _cache_layout_for_triton(rope_cache)
        scale_tokens_per_block, _, _ = _cache_layout_for_triton(scale_cache)
    except ValueError as exc:
        return str(exc)
    if tokens_per_block <= 0:
        return f"nope_cache tokens_per_block must be positive, got {tokens_per_block}"
    if rope_tokens_per_block != tokens_per_block:
        return "rope_cache tokens_per_block must match nope_cache"
    if scale_tokens_per_block != tokens_per_block:
        return "scale_cache tokens_per_block must match nope_cache"
    if active_tokens is not None:
        if active_tokens <= 0:
            return f"active_tokens must be positive, got {active_tokens}"
        if active_tokens > q.shape[0]:
            return f"active_tokens ({active_tokens}) exceed decode batch capacity ({q.shape[0]})"
        if input_pos_host.numel() < active_tokens:
            return (
                f"input_pos_host needs at least {active_tokens} entries, "
                f"got {input_pos_host.numel()}"
            )
        if cu_num_pages_host.numel() < active_tokens + 1:
            return (
                f"cu_num_pages_host needs at least {active_tokens + 1} entries, "
                f"got {cu_num_pages_host.numel()}"
            )

    return None


@triton.jit
def _update_ratio0_swa_cache_kernel(
    kv_ptr,
    input_pos_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    swa_cache_ptr,
    kv_stride_batch: tl.constexpr,
    kv_stride_seq: tl.constexpr,
    kv_stride_dim: tl.constexpr,
    cache_page_stride: tl.constexpr,
    cache_token_stride: tl.constexpr,
    tokens_per_block: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim

    input_pos = tl.load(input_pos_ptr + token_idx)
    page_ordinal = input_pos // tokens_per_block
    token_offset = input_pos - page_ordinal * tokens_per_block
    page_table_start = tl.load(cu_num_pages_ptr + token_idx)
    page_idx = tl.load(cache_loc_ptr + page_table_start + page_ordinal).to(tl.int64)

    kv_offsets = token_idx * kv_stride_batch + dim_offsets * kv_stride_dim
    kv_values = tl.load(kv_ptr + kv_offsets, mask=dim_mask, other=0.0)
    cache_offsets = page_idx * cache_page_stride + token_offset * cache_token_stride + dim_offsets
    tl.store(swa_cache_ptr + cache_offsets, kv_values, mask=dim_mask)


@triton.jit
def _ratio0_swa_attention_kernel(
    q_ptr,
    attn_sink_ptr,
    topk_idxs_ptr,
    input_pos_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    swa_cache_ptr,
    out_ptr,
    q_stride_batch: tl.constexpr,
    q_stride_seq: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    topk_stride_batch: tl.constexpr,
    topk_stride_seq: tl.constexpr,
    topk_stride_idx: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_seq: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    cache_page_stride: tl.constexpr,
    cache_token_stride: tl.constexpr,
    tokens_per_block: tl.constexpr,
    active_tokens: tl.constexpr,
    softmax_scale: tl.constexpr,
    head_dim: tl.constexpr,
    topk_width: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    token_is_active = token_idx < active_tokens

    out_offsets = (
        token_idx * out_stride_batch + head_idx * out_stride_head + dim_offsets * out_stride_dim
    )
    if not token_is_active:
        tl.store(out_ptr + out_offsets, tl.zeros([block_dim], dtype=tl.float32), mask=dim_mask)
        return

    q_offsets = token_idx * q_stride_batch + head_idx * q_stride_head + dim_offsets * q_stride_dim
    q_vec = tl.load(q_ptr + q_offsets, mask=dim_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([block_dim], dtype=tl.float32)
    m_i = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
    l_i = tl.full((), 1.0, dtype=tl.float32)
    page_table_start = tl.load(cu_num_pages_ptr + token_idx)

    for topk_offset in range(topk_width):
        topk = tl.load(
            topk_idxs_ptr + token_idx * topk_stride_batch + topk_offset * topk_stride_idx
        )
        valid_topk = topk >= 0
        row_pos = topk.to(tl.int64)
        page_ordinal = row_pos // tokens_per_block
        token_offset = row_pos - page_ordinal * tokens_per_block
        page_idx = tl.load(
            cache_loc_ptr + page_table_start + page_ordinal,
            mask=valid_topk,
            other=0,
        ).to(tl.int64)
        cache_offsets = (
            page_idx * cache_page_stride + token_offset * cache_token_stride + dim_offsets
        )
        kv_vec = tl.load(swa_cache_ptr + cache_offsets, mask=dim_mask & valid_topk, other=0.0).to(
            tl.float32
        )
        logit = tl.sum(q_vec * kv_vec, axis=0) * softmax_scale
        logit = tl.where(valid_topk, logit, float("-inf"))

        m_next = tl.maximum(m_i, logit)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(logit - m_next)
        acc = acc * alpha + kv_vec * beta
        l_i = l_i * alpha + beta
        m_i = m_next

    output = acc / l_i
    tl.store(out_ptr + out_offsets, output, mask=dim_mask)


@triton.jit
def _ratio0_swa_fp8_attention_kernel(
    q_ptr,
    attn_sink_ptr,
    topk_idxs_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    nope_cache_ptr,
    rope_cache_ptr,
    scale_cache_ptr,
    out_ptr,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    topk_stride_batch: tl.constexpr,
    topk_stride_idx: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    nope_cache_page_stride: tl.constexpr,
    nope_cache_token_stride: tl.constexpr,
    rope_cache_page_stride: tl.constexpr,
    rope_cache_token_stride: tl.constexpr,
    scale_cache_page_stride: tl.constexpr,
    scale_cache_token_stride: tl.constexpr,
    tokens_per_block: tl.constexpr,
    active_tokens: tl.constexpr,
    softmax_scale: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    fp8_block_size: tl.constexpr,
    topk_width: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    nope_dim_mask = dim_offsets < nope_dim
    rope_dim_mask = (dim_offsets >= nope_dim) & dim_mask
    token_is_active = token_idx < active_tokens

    out_offsets = (
        token_idx * out_stride_batch + head_idx * out_stride_head + dim_offsets * out_stride_dim
    )
    if not token_is_active:
        tl.store(out_ptr + out_offsets, tl.zeros([block_dim], dtype=tl.float32), mask=dim_mask)
        return

    q_offsets = token_idx * q_stride_batch + head_idx * q_stride_head + dim_offsets * q_stride_dim
    q_vec = tl.load(q_ptr + q_offsets, mask=dim_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([block_dim], dtype=tl.float32)
    m_i = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
    l_i = tl.full((), 1.0, dtype=tl.float32)
    page_table_start = tl.load(cu_num_pages_ptr + token_idx)

    for topk_offset in range(topk_width):
        topk = tl.load(
            topk_idxs_ptr + token_idx * topk_stride_batch + topk_offset * topk_stride_idx
        )
        valid_topk = topk >= 0
        row_pos = topk.to(tl.int64)
        page_ordinal = row_pos // tokens_per_block
        token_offset = row_pos - page_ordinal * tokens_per_block
        page_idx = tl.load(
            cache_loc_ptr + page_table_start + page_ordinal,
            mask=valid_topk,
            other=0,
        ).to(tl.int64)

        scale_offsets = (
            page_idx * scale_cache_page_stride
            + token_offset * scale_cache_token_stride
            + dim_offsets // fp8_block_size
        )
        scale_exp_bits = tl.load(
            scale_cache_ptr + scale_offsets,
            mask=nope_dim_mask & valid_topk,
            other=127,
        ).to(tl.float32)
        scale = tl.exp2(scale_exp_bits - 127.0)

        nope_offsets = (
            page_idx * nope_cache_page_stride + token_offset * nope_cache_token_stride + dim_offsets
        )
        nope_vec = tl.load(
            nope_cache_ptr + nope_offsets,
            mask=nope_dim_mask & valid_topk,
            other=0.0,
        ).to(tl.float32)
        nope_vec = nope_vec * scale

        rope_offsets = (
            page_idx * rope_cache_page_stride
            + token_offset * rope_cache_token_stride
            + dim_offsets
            - nope_dim
        )
        rope_vec = tl.load(
            rope_cache_ptr + rope_offsets,
            mask=rope_dim_mask & valid_topk,
            other=0.0,
        ).to(tl.float32)
        kv_vec = tl.where(nope_dim_mask, nope_vec, rope_vec)

        logit = tl.sum(q_vec * kv_vec, axis=0) * softmax_scale
        logit = tl.where(valid_topk, logit, float("-inf"))

        m_next = tl.maximum(m_i, logit)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(logit - m_next)
        acc = acc * alpha + kv_vec * beta
        l_i = l_i * alpha + beta
        m_i = m_next

    output = acc / l_i
    tl.store(out_ptr + out_offsets, output, mask=dim_mask)


def triton_deepseek_v4_ratio0_swa_attention_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    active_tokens: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run ratio-0 DeepSeek V4 cached SWA attention through Triton.

    The path is intentionally decode-only: one query/KV token per active
    sequence, caller-owned SWA cache pages, and caller-provided local
    ``topk_idxs``. Unsupported shapes and metadata placements should be filtered
    by :func:`deepseek_v4_ratio0_swa_triton_skip_reason` before calling here.
    """
    reason = deepseek_v4_ratio0_swa_triton_skip_reason(
        q,
        kv,
        attn_sink,
        topk_idxs,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        window_size,
        0,
        out,
        active_tokens,
    )
    if reason is not None:
        raise RuntimeError(reason)

    tokens_per_block, cache_page_stride, cache_token_stride = _cache_layout_for_triton(swa_cache)
    output = torch.empty_like(q) if out is None else out
    block_dim = triton.next_power_of_2(_RATIO0_HEAD_DIM)

    _update_ratio0_swa_cache_kernel[(active_tokens,)](
        kv,
        input_pos_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        cache_page_stride,
        cache_token_stride,
        tokens_per_block,
        _RATIO0_HEAD_DIM,
        block_dim,
        num_warps=8,
    )
    _ratio0_swa_attention_kernel[(q.shape[0], _RATIO0_LOCAL_HEADS)](
        q,
        attn_sink,
        topk_idxs,
        input_pos_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        cache_page_stride,
        cache_token_stride,
        tokens_per_block,
        active_tokens,
        softmax_scale,
        _RATIO0_HEAD_DIM,
        _RATIO0_TOPK_WIDTH,
        block_dim,
        num_warps=8,
    )

    if out is not None:
        return out.new_empty(0)
    return output


def triton_deepseek_v4_ratio0_swa_attention_with_fp8_cache(
    q: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    active_tokens: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run ratio-0 DeepSeek V4 cached SWA attention from split FP8 NoPE cache rows.

    This narrow path consumes caller-populated ``nope_cache`` (FP8 E4M3),
    ``rope_cache`` (BF16), and ``scale_cache`` (raw E8M0 bytes or E8M0 dtype).
    It does not update the cache from ``kv``; unsupported inputs raise instead
    of falling back to the Torch cached reference.
    """
    reason = deepseek_v4_ratio0_swa_fp8_triton_skip_reason(
        q,
        attn_sink,
        topk_idxs,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        nope_cache,
        rope_cache,
        scale_cache,
        window_size,
        out,
        active_tokens,
    )
    if reason is not None:
        raise RuntimeError(reason)

    tokens_per_block, nope_cache_page_stride, nope_cache_token_stride = _cache_layout_for_triton(
        nope_cache
    )
    _, rope_cache_page_stride, rope_cache_token_stride = _cache_layout_for_triton(rope_cache)
    scale_cache_bytes = _scale_cache_raw_bytes(scale_cache)
    _, scale_cache_page_stride, scale_cache_token_stride = _cache_layout_for_triton(
        scale_cache_bytes
    )

    output = torch.empty_like(q) if out is None else out
    block_dim = triton.next_power_of_2(_RATIO0_HEAD_DIM)
    _ratio0_swa_fp8_attention_kernel[(q.shape[0], _RATIO0_LOCAL_HEADS)](
        q,
        attn_sink,
        topk_idxs,
        cache_loc_host,
        cu_num_pages_host,
        nope_cache,
        rope_cache,
        scale_cache_bytes,
        output,
        q.stride(0),
        q.stride(2),
        q.stride(3),
        topk_idxs.stride(0),
        topk_idxs.stride(2),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        nope_cache_page_stride,
        nope_cache_token_stride,
        rope_cache_page_stride,
        rope_cache_token_stride,
        scale_cache_page_stride,
        scale_cache_token_stride,
        tokens_per_block,
        active_tokens,
        softmax_scale,
        _RATIO0_HEAD_DIM,
        _RATIO0_NOPE_DIM,
        _RATIO0_FP8_BLOCK_SIZE,
        _RATIO0_TOPK_WIDTH,
        block_dim,
        num_warps=8,
    )

    if out is not None:
        return out.new_empty(0)
    return output
