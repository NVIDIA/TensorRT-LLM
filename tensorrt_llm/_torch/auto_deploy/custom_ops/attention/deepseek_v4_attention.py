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

"""DeepSeek V4 sparse/HMA attention source and cached reference ops."""

import os
import warnings
from typing import List, Optional

import torch
import triton
import triton.language as tl
from torch.fx import Node

from ..._compat import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BatchInfo,
    Constant,
    KVPagedResourceHandler,
    MHACallable,
    ResourceHandlerDict,
)
from .deepseek_v4_fp8_cache import DSV4_FP8_NOPE_ATTENTION_UNSUPPORTED_REASON, FP8_E4M3_DTYPE
from .triton_deepseek_v4_sparse_attention import (
    _cache_layout_for_triton,
    deepseek_v4_ratio0_swa_triton_skip_reason,
    triton_deepseek_v4_ratio0_swa_attention_with_cache,
)

_SPARSE_ATTENTION_CHUNK_TARGET_BYTES = 512 * 1024 * 1024
_SPARSE_ATTENTION_MAX_CHUNK_TOKENS = 64
_DSV4_TRITON_LOCAL_NUM_HEADS = 8
_DSV4_TRITON_HEAD_DIM = 512
_DSV4_TRITON_ROPE_DIM = 64
_DSV4_TRITON_WINDOW_SIZE = 128
_DSV4_TRITON_RATIO4_MAX_COMPRESSED_LEN = 2048
_DSV4_TRITON_RATIO4_COMPRESSED_TOPK = 512
_DSV4_TRITON_RATIO128_MAX_COMPRESSED_LEN = 64
_DSV4_TRITON_TOPK_WIDTH_BY_RATIO = {
    0: 128,
    4: 640,
    128: 192,
}
_DSV4_TRITON_COMPRESSOR_DIM_BY_RATIO = {
    0: 0,
    4: 1024,
    128: 512,
}
_DSV4_FORCE_TORCH_REFERENCE_ENV = "TRTLLM_AD_DSV4_SPARSE_ATTENTION_FORCE_TORCH"
_DSV4_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _force_torch_cached_reference() -> bool:
    return os.getenv(_DSV4_FORCE_TORCH_REFERENCE_ENV, "").strip().lower() in _DSV4_TRUE_ENV_VALUES


def _warn_torch_cached_reference_fallback(reason: str) -> None:
    warnings.warn(
        "auto_deploy::triton_deepseek_v4_sparse_attention_v2_with_cache is falling back to "
        f"torch_deepseek_v4_sparse_attention_v2_with_cache: {reason}",
        RuntimeWarning,
        stacklevel=2,
    )


def _validate_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}, got rank {tensor.dim()}")


def _validate_int_metadata(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"{name} must have rank 1, got rank {tensor.dim()}")
    if tensor.dtype not in (torch.int32, torch.int64, torch.int):
        raise TypeError(f"{name} must be an int32/int64 tensor, got {tensor.dtype}")


def _validate_contract_cache(
    name: str,
    cache: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    if cache.dim() not in (3, 4, 5):
        raise ValueError(f"{name} must have rank 3, 4, or 5, got rank {cache.dim()}")
    if cache.shape[-1] != head_dim:
        raise ValueError(f"{name} last dimension must be {head_dim}, got {cache.shape[-1]}")
    if cache.dtype != dtype:
        if name == "swa_cache" and cache.dtype == FP8_E4M3_DTYPE:
            raise TypeError(
                f"{name} must have dtype {dtype}, got {cache.dtype}. "
                f"{DSV4_FP8_NOPE_ATTENTION_UNSUPPORTED_REASON}"
            )
        raise TypeError(f"{name} must have dtype {dtype}, got {cache.dtype}")


def _validate_contract_resource_placeholder(
    name: str,
    cache: torch.Tensor,
) -> None:
    if cache.dim() not in (3, 4, 5):
        raise ValueError(f"{name} must have rank 3, 4, or 5, got rank {cache.dim()}")
    if cache.shape[-1] not in (0, _DSV4_TRITON_HEAD_DIM, 2 * _DSV4_TRITON_HEAD_DIM):
        raise ValueError(
            f"{name} last dimension must be 0, {_DSV4_TRITON_HEAD_DIM}, or "
            f"{2 * _DSV4_TRITON_HEAD_DIM}; got {cache.shape[-1]}"
        )
    if not (cache.is_floating_point() or cache.dtype == getattr(torch, "float8_e4m3fn", None)):
        raise TypeError(f"{name} must be a floating-point cache placeholder, got {cache.dtype}")


def _validate_deepseek_v4_sparse_attention_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> None:
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)

    if not q.is_floating_point():
        raise TypeError(f"q must be floating point, got {q.dtype}")
    if not kv.is_floating_point():
        raise TypeError(f"kv must be floating point, got {kv.dtype}")
    if not attn_sink.is_floating_point():
        raise TypeError(f"attn_sink must be floating point, got {attn_sink.dtype}")
    if q.dtype != kv.dtype:
        raise TypeError(f"q and kv must have the same dtype, got {q.dtype} and {kv.dtype}")
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_idxs must be int32 or int64, got {topk_idxs.dtype}")

    if kv.device != q.device:
        raise ValueError(f"kv must be on {q.device}, got {kv.device}")
    if attn_sink.device != q.device:
        raise ValueError(f"attn_sink must be on {q.device}, got {attn_sink.device}")
    if topk_idxs.device != q.device:
        raise ValueError(f"topk_idxs must be on {q.device}, got {topk_idxs.device}")

    batch_size, seq_len, num_heads, head_dim = q.shape
    kv_batch_size, _, kv_head_dim = kv.shape
    topk_batch_size, topk_seq_len, _ = topk_idxs.shape

    if kv_batch_size != batch_size:
        raise ValueError(f"kv batch dimension must be {batch_size}, got {kv_batch_size}")
    if topk_batch_size != batch_size:
        raise ValueError(f"topk_idxs batch dimension must be {batch_size}, got {topk_batch_size}")
    if topk_seq_len != seq_len:
        raise ValueError(f"topk_idxs sequence dimension must be {seq_len}, got {topk_seq_len}")
    if kv_head_dim != head_dim:
        raise ValueError(f"kv head dimension must be {head_dim}, got {kv_head_dim}")
    if attn_sink.shape[0] != num_heads:
        raise ValueError(f"attn_sink length must be {num_heads}, got {attn_sink.shape[0]}")
    if out is not None:
        if out.shape != q.shape:
            raise ValueError(f"out shape must be {q.shape}, got {out.shape}")
        if out.dtype != q.dtype:
            raise TypeError(f"out dtype must be {q.dtype}, got {out.dtype}")
        if out.device != q.device:
            raise ValueError(f"out must be on {q.device}, got {out.device}")


def _gather_selected_kv(
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_idxs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    gather_topk_idxs = topk_idxs.to(torch.long).clamp(min=0)
    if batch_idxs is not None:
        return kv[batch_idxs.to(torch.long).unsqueeze(1), gather_topk_idxs]

    batch_size, seq_len, k_select = topk_idxs.shape
    head_dim = kv.shape[-1]
    gather_idx = gather_topk_idxs.unsqueeze(-1).expand(batch_size, seq_len, k_select, head_dim)
    expanded_kv = kv.unsqueeze(1).expand(batch_size, seq_len, kv.shape[1], head_dim)
    return torch.gather(expanded_kv, dim=2, index=gather_idx)


def _sparse_attention_query_chunk_size(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    k_select: int,
    compute_dtype: torch.dtype,
) -> int:
    compute_element_size = torch.empty((), dtype=compute_dtype).element_size()
    logits_element_size = torch.empty((), dtype=torch.float32).element_size()
    bytes_per_token = (
        k_select * head_dim * compute_element_size
        + 3 * num_heads * (k_select + 1) * logits_element_size
        + num_heads * head_dim * compute_element_size
    )
    if bytes_per_token <= 0:
        return 1
    chunk_size = _SPARSE_ATTENTION_CHUNK_TARGET_BYTES // bytes_per_token
    chunk_size = max(1, int(chunk_size))
    chunk_size = min(chunk_size, _SPARSE_ATTENTION_MAX_CHUNK_TOKENS)
    return min(num_tokens, chunk_size)


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_sparse_attention", mutates_args=())
def torch_deepseek_v4_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    enable_sharding: bool = False,
    layer_type: str = "unknown",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
) -> torch.Tensor:
    """Reference DeepSeek V4 sparse/HMA attention source op.

    Args:
        q: Query states with shape ``[batch, seq_len, num_heads, head_dim]``.
        kv: Shared sparse key/value rows with shape ``[batch, kv_rows, head_dim]``.
        attn_sink: Per-head sink logits with shape ``[num_heads]``.
        topk_idxs: Selected row indices into ``kv`` with shape
            ``[batch, seq_len, k_select]``. Duplicate indices are preserved and
            receive independent probability mass. Negative indices are masked
            slots and receive zero probability.
        softmax_scale: Scale applied to query/key logits before adding the sink
            logit.

    Returns:
        Attention output with shape ``[batch, seq_len, num_heads, head_dim]``.
        The sink participates in softmax normalization but contributes no value
        vector.
    """
    del (
        enable_sharding,
        layer_type,
        layer_idx,
        window_size,
        compress_ratio,
        max_compressed_len,
        head_dim,
        rope_dim,
    )

    _validate_deepseek_v4_sparse_attention_inputs(q, kv, attn_sink, topk_idxs, out)

    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    batch_size, seq_len, num_heads, q_head_dim = q.shape
    _, _, k_select = topk_idxs.shape
    num_tokens = batch_size * seq_len
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if num_tokens == 0:
        if out is not None:
            out.copy_(output)
            return out.new_empty(0)
        return output

    chunk_size = _sparse_attention_query_chunk_size(
        num_tokens, num_heads, q_head_dim, k_select, compute_dtype
    )

    q_flat = q.reshape(num_tokens, num_heads, q_head_dim)
    topk_flat = topk_idxs.reshape(num_tokens, k_select)
    batch_idxs = torch.arange(batch_size, device=q.device).view(batch_size, 1)
    batch_idxs = batch_idxs.expand(batch_size, seq_len).reshape(num_tokens)
    output_flat = output.reshape(num_tokens, num_heads, q_head_dim)
    sink_logits = attn_sink.to(dtype=compute_dtype).reshape(1, num_heads, 1)

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        topk_chunk = topk_flat[start:end]
        selected_kv_compute = _gather_selected_kv(kv, topk_chunk, batch_idxs[start:end]).to(
            compute_dtype
        )
        q_compute = q_flat[start:end].to(compute_dtype)

        logits = torch.einsum("thd,tkd->thk", q_compute, selected_kv_compute)
        logits = logits * softmax_scale
        logits = logits.masked_fill((topk_chunk < 0).unsqueeze(1), float("-inf"))
        chunk_sink_logits = sink_logits.expand(end - start, num_heads, 1)
        logits_with_sink = torch.cat([logits, chunk_sink_logits], dim=-1)

        weights_with_sink = torch.softmax(logits_with_sink, dim=-1, dtype=torch.float32)
        weights = weights_with_sink[..., :-1].to(compute_dtype)
        chunk_output = torch.einsum("thk,tkd->thd", weights, selected_kv_compute)
        output_flat[start:end].copy_(chunk_output.to(q.dtype))

    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


@torch_deepseek_v4_sparse_attention.register_fake
def torch_deepseek_v4_sparse_attention_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    enable_sharding: bool = False,
    layer_type: str = "unknown",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    del (
        softmax_scale,
        enable_sharding,
        layer_type,
        layer_idx,
        window_size,
        compress_ratio,
        max_compressed_len,
        head_dim,
        rope_dim,
    )
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    if out is not None:
        return out.new_empty(0)
    return q.new_empty(q.shape).contiguous()


def _to_host_long(tensor: torch.Tensor, length: int) -> torch.Tensor:
    return tensor.detach().cpu().to(torch.long).flatten()[:length]


def _cache_token_view(cache: torch.Tensor, page_idx: int, token_offset: int) -> torch.Tensor:
    """Return a mutable view for one DeepSeek V4 SWA cache row."""
    if cache.dim() == 5:
        if int(cache.shape[1]) == 1:
            # KVCacheManager.get_buffers(..., kv_layout="NHD"):
            # [num_pages, kv_factor, tokens_per_block, num_kv_heads, head_dim].
            if int(cache.shape[2]) >= int(cache.shape[3]):
                return cache[page_idx, 0, token_offset, 0]
            # KVCacheManager.get_buffers(..., kv_layout="HND"):
            # [num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim].
            return cache[page_idx, 0, 0, token_offset]
        # Local unit tests use [num_pages, tokens_per_block, kv_factor, num_kv_heads, head_dim].
        return cache[page_idx, token_offset, 0, 0]
    if cache.dim() == 4:
        # Local unit tests often use [num_pages, tokens_per_block, num_heads, head_dim].
        return cache[page_idx, token_offset, 0]
    if cache.dim() == 3:
        return cache[page_idx, token_offset]
    raise ValueError(f"swa_cache must have rank 3, 4, or 5, got rank {cache.dim()}")


def _cache_tokens_per_block(cache: torch.Tensor) -> int:
    if cache.dim() not in (3, 4, 5):
        raise ValueError(f"swa_cache must have rank 3, 4, or 5, got rank {cache.dim()}")
    if cache.dim() == 5 and int(cache.shape[1]) == 1:
        return int(cache.shape[2] if int(cache.shape[2]) >= int(cache.shape[3]) else cache.shape[3])
    return int(cache.shape[1])


def _page_for_position(
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    position: int,
    tokens_per_block: int,
) -> tuple[int, int]:
    page_ordinal = position // tokens_per_block
    page_table_start = int(cu_num_pages_host[seq_idx].item())
    page_table_end = int(cu_num_pages_host[seq_idx + 1].item())
    page_table_idx = page_table_start + page_ordinal
    if page_table_idx >= page_table_end:
        raise ValueError(
            f"Sequence {seq_idx} position {position} needs page ordinal {page_ordinal}, "
            f"but only {page_table_end - page_table_start} page(s) are available."
        )
    page_idx = int(cache_loc_host[page_table_idx].item())
    return page_idx, position % tokens_per_block


def _write_swa_cache(
    kv_seq: torch.Tensor,
    cache: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    input_pos: int,
) -> None:
    tokens_per_block = _cache_tokens_per_block(cache)
    for token_offset in range(kv_seq.shape[0]):
        page_idx, page_offset = _page_for_position(
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            input_pos + token_offset,
            tokens_per_block,
        )
        _cache_token_view(cache, page_idx, page_offset).copy_(kv_seq[token_offset].to(cache.dtype))


def _gather_swa_rows(
    cache: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    start_pos: int,
    end_pos: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    tokens_per_block = _cache_tokens_per_block(cache)
    rows = []
    for position in range(start_pos, end_pos):
        page_idx, page_offset = _page_for_position(
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            position,
            tokens_per_block,
        )
        rows.append(_cache_token_view(cache, page_idx, page_offset).to(dtype))
    if not rows:
        head_dim = int(cache.shape[-1])
        return torch.empty(0, head_dim, dtype=dtype, device=cache.device)
    return torch.stack(rows, dim=0)


def _slice_sequence_tokens(
    tensor: torch.Tensor,
    seq_idx: int,
    flat_start: int,
    seq_len: int,
) -> torch.Tensor:
    if tensor.shape[0] > seq_idx and tensor.shape[0] != 1:
        return tensor[seq_idx, :seq_len]
    return tensor.reshape(-1, *tensor.shape[2:])[flat_start : flat_start + seq_len]


def _cached_local_window_attention(
    q_seq: torch.Tensor,
    attn_sink: torch.Tensor,
    cache: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    input_pos: int,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    for token_offset in range(q_seq.shape[0]):
        query_pos = input_pos + token_offset
        start_pos = max(0, query_pos - window_size + 1)
        kv_seq = _gather_swa_rows(
            cache,
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            start_pos,
            query_pos + 1,
            q_seq.dtype,
        )
        topk = torch.arange(kv_seq.shape[0], dtype=torch.int64, device=q_seq.device).view(1, 1, -1)
        out = torch_deepseek_v4_sparse_attention(
            q_seq[token_offset : token_offset + 1].unsqueeze(0),
            kv_seq.unsqueeze(0),
            attn_sink,
            topk,
            softmax_scale,
        )
        outputs.append(out.squeeze(0).squeeze(0))
    if not outputs:
        return q_seq.new_empty(q_seq.shape)
    return torch.stack(outputs, dim=0)


def _rms_norm_ref(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    compute = x.to(torch.float32)
    output = compute * torch.rsqrt(compute.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None and weight.numel() != 0:
        output = output * weight.to(device=x.device, dtype=torch.float32)
    return output.to(x.dtype)


def _apply_rope_ref(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == 0:
        return x.contiguous()
    nope = x[..., : x.shape[-1] - rope_dim]
    rope = x[..., -rope_dim:]
    rope_complex = torch.view_as_complex(rope.float().reshape(*rope.shape[:-1], -1, 2))
    if freqs_cis.dim() == rope_complex.dim() - 1:
        freqs_cis = freqs_cis.unsqueeze(-2)
    rope_out = torch.view_as_real(rope_complex * freqs_cis).flatten(-2).to(x.dtype)
    return torch.cat([nope, rope_out], dim=-1).contiguous()


def _overlap_transform_projected(
    tensor: torch.Tensor,
    ratio: int,
    head_dim: int,
    value: float,
) -> torch.Tensor:
    batch_size, compressed_len, _, _ = tensor.shape
    out = tensor.new_full((batch_size, compressed_len, 2 * ratio, head_dim), value)
    out[:, :, ratio:] = tensor[:, :, :, head_dim:]
    out[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
    return out


def _build_full_compressed_kv(
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    eps: float,
    rope_dim: int,
    compress_ratio: int,
    max_compressed_len: int,
) -> torch.Tensor:
    """Build full-context compressed rows from source-op compressor projections."""
    if compress_ratio <= 0:
        return compressor_kv.new_empty(compressor_kv.shape[0], 0, compressor_kv.shape[-1])

    _validate_rank("compressor_kv", compressor_kv, 3)
    _validate_rank("compressor_gate", compressor_gate, 3)
    if compressor_kv.shape != compressor_gate.shape:
        raise ValueError(
            "compressor_kv and compressor_gate must have matching shapes, "
            f"got {tuple(compressor_kv.shape)} and {tuple(compressor_gate.shape)}"
        )
    if max_compressed_len <= 0:
        raise ValueError(f"max_compressed_len must be positive, got {max_compressed_len}")

    batch_size, seq_len, state_dim = compressor_kv.shape
    overlap = compress_ratio == 4
    channels = 2 if overlap else 1
    if state_dim % channels != 0:
        raise ValueError(f"compressor state dim {state_dim} is not divisible by {channels}")
    head_dim = state_dim // channels

    max_compressed_tokens = max_compressed_len * compress_ratio
    pad_len = max_compressed_tokens - seq_len
    if pad_len < 0:
        raise ValueError(f"seq_len {seq_len} exceeds compressed capacity {max_compressed_tokens}")

    kv = torch.nn.functional.pad(compressor_kv, (0, 0, 0, pad_len))
    gate = torch.nn.functional.pad(compressor_gate, (0, 0, 0, pad_len), value=float("-inf"))
    kv = kv.view(batch_size, max_compressed_len, compress_ratio, state_dim)
    gate = gate.view(batch_size, max_compressed_len, compress_ratio, state_dim)
    gate = gate + compressor_ape.to(device=gate.device, dtype=gate.dtype)
    if overlap:
        kv = _overlap_transform_projected(kv, compress_ratio, head_dim, 0.0)
        gate = _overlap_transform_projected(gate, compress_ratio, head_dim, float("-inf"))

    pooled = (kv * gate.softmax(dim=2)).sum(dim=2)
    pooled = _rms_norm_ref(pooled, compressor_norm_weight, eps)

    row_idx = torch.arange(max_compressed_len, device=compressor_kv.device) * compress_ratio
    row_idx = torch.minimum(row_idx, torch.full_like(row_idx, seq_len - 1))
    row_idx = row_idx.unsqueeze(0).expand(batch_size, -1)
    compressed_position_ids = torch.gather(position_ids, 1, row_idx)
    compressed_freqs = freqs_cis_table[compressed_position_ids]
    return _apply_rope_ref(pooled, compressed_freqs, rope_dim)


def _write_paged_rows(
    rows: torch.Tensor,
    cache: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    input_pos: int,
) -> None:
    tokens_per_block = _cache_tokens_per_block(cache)
    for token_offset in range(rows.shape[0]):
        page_idx, page_offset = _page_for_position(
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            input_pos + token_offset,
            tokens_per_block,
        )
        row_view = _cache_token_view(cache, page_idx, page_offset)
        row = rows[token_offset].to(cache.dtype)
        row_view[..., : row.shape[-1]].copy_(row)


def _gather_paged_rows(
    cache: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    start_pos: int,
    end_pos: int,
    dtype: torch.dtype,
    width: Optional[int] = None,
) -> torch.Tensor:
    tokens_per_block = _cache_tokens_per_block(cache)
    rows = []
    for position in range(start_pos, end_pos):
        page_idx, page_offset = _page_for_position(
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            position,
            tokens_per_block,
        )
        row = _cache_token_view(cache, page_idx, page_offset)
        if width is not None:
            row = row[..., :width]
        rows.append(row.to(dtype))
    head_dim = int(cache.shape[-1] if width is None else width)
    if not rows:
        return torch.empty(0, head_dim, dtype=dtype, device=cache.device)
    return torch.stack(rows, dim=0)


def _compressed_row_from_state(
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    row_idx: int,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    eps: float,
    rope_dim: int,
    compress_ratio: int,
    head_dim: int,
    state_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    anchor = row_idx * compress_ratio
    kv_rows = []
    gate_rows = []
    if compress_ratio == 4:
        for offset in range(compress_ratio):
            position = anchor - compress_ratio + offset
            if position < 0:
                kv_rows.append(
                    torch.zeros(head_dim, dtype=dtype, device=compressor_kv_cache.device)
                )
                gate_rows.append(
                    torch.full(
                        (head_dim,),
                        float("-inf"),
                        dtype=dtype,
                        device=compressor_gate_cache.device,
                    )
                )
                continue
            kv_state = _gather_paged_rows(
                compressor_kv_cache,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                position,
                position + 1,
                dtype,
                state_dim,
            )[0]
            gate_state = _gather_paged_rows(
                compressor_gate_cache,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                position,
                position + 1,
                dtype,
                state_dim,
            )[0]
            kv_rows.append(kv_state[:head_dim])
            gate_rows.append(gate_state[:head_dim] + compressor_ape[offset, :head_dim].to(dtype))

        for offset in range(compress_ratio):
            position = anchor + offset
            kv_state = _gather_paged_rows(
                compressor_kv_cache,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                position,
                position + 1,
                dtype,
                state_dim,
            )[0]
            gate_state = _gather_paged_rows(
                compressor_gate_cache,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                position,
                position + 1,
                dtype,
                state_dim,
            )[0]
            kv_rows.append(kv_state[head_dim : 2 * head_dim])
            gate_rows.append(
                gate_state[head_dim : 2 * head_dim]
                + compressor_ape[offset, head_dim : 2 * head_dim].to(dtype)
            )
    else:
        for offset in range(compress_ratio):
            position = anchor + offset
            kv_state = _gather_paged_rows(
                compressor_kv_cache,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                position,
                position + 1,
                dtype,
                state_dim,
            )[0]
            gate_state = _gather_paged_rows(
                compressor_gate_cache,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                position,
                position + 1,
                dtype,
                state_dim,
            )[0]
            kv_rows.append(kv_state[:head_dim])
            gate_rows.append(gate_state[:head_dim] + compressor_ape[offset, :head_dim].to(dtype))

    kv = torch.stack(kv_rows, dim=0)
    gate = torch.stack(gate_rows, dim=0)
    pooled = (kv * gate.softmax(dim=0)).sum(dim=0)
    pooled = _rms_norm_ref(pooled.unsqueeze(0), compressor_norm_weight, eps).squeeze(0)
    freqs = freqs_cis_table[anchor].unsqueeze(0)
    return _apply_rope_ref(pooled.unsqueeze(0), freqs, rope_dim).squeeze(0)


def _update_compressed_caches(
    compressor_kv_seq: torch.Tensor,
    compressor_gate_seq: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    input_pos: int,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    eps: float,
    rope_dim: int,
    compress_ratio: int,
    max_compressed_len: int,
) -> None:
    if compress_ratio <= 0 or compressor_kv_seq.numel() == 0:
        return

    state_dim = int(compressor_kv_seq.shape[-1])
    head_dim = state_dim // (2 if compress_ratio == 4 else 1)
    _write_paged_rows(
        compressor_kv_seq,
        compressor_kv_cache,
        cache_loc_host,
        cu_num_pages_host,
        seq_idx,
        input_pos,
    )
    _write_paged_rows(
        compressor_gate_seq,
        compressor_gate_cache,
        cache_loc_host,
        cu_num_pages_host,
        seq_idx,
        input_pos,
    )

    old_completed = min(input_pos // compress_ratio, max_compressed_len)
    new_completed = min(
        (input_pos + compressor_kv_seq.shape[0]) // compress_ratio,
        max_compressed_len,
    )
    compressed_rows = []
    for row_idx in range(old_completed, new_completed):
        compressed_rows.append(
            _compressed_row_from_state(
                compressor_kv_cache,
                compressor_gate_cache,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                row_idx,
                compressor_ape,
                compressor_norm_weight,
                freqs_cis_table,
                eps,
                rope_dim,
                compress_ratio,
                head_dim,
                state_dim,
                compressor_kv_seq.dtype,
            )
        )
    if compressed_rows:
        _write_paged_rows(
            torch.stack(compressed_rows, dim=0),
            mhc_cache,
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            old_completed,
        )


def _cached_compressed_attention(
    q_seq: torch.Tensor,
    attn_sink: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    seq_idx: int,
    input_pos: int,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    for token_offset in range(q_seq.shape[0]):
        query_pos = input_pos + token_offset
        local_start = max(0, query_pos - window_size + 1)
        local_kv = _gather_swa_rows(
            swa_cache,
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            local_start,
            query_pos + 1,
            q_seq.dtype,
        )
        compressed_len = min((query_pos + 1) // compress_ratio, max_compressed_len)
        compressed_kv = _gather_paged_rows(
            mhc_cache,
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            0,
            compressed_len,
            q_seq.dtype,
        )
        local_idxs = torch.arange(local_kv.shape[0], dtype=torch.int64, device=q_seq.device)
        compressed_idxs = torch.arange(
            compressed_kv.shape[0], dtype=torch.int64, device=q_seq.device
        )
        compressed_idxs = compressed_idxs + local_kv.shape[0]
        topk = torch.cat([local_idxs, compressed_idxs], dim=0).view(1, 1, -1)
        kv = torch.cat([local_kv, compressed_kv], dim=0)
        out = torch_deepseek_v4_sparse_attention(
            q_seq[token_offset : token_offset + 1].unsqueeze(0),
            kv.unsqueeze(0),
            attn_sink,
            topk,
            softmax_scale,
        )
        outputs.append(out.squeeze(0).squeeze(0))
    if not outputs:
        return q_seq.new_empty(q_seq.shape)
    return torch.stack(outputs, dim=0)


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_sparse_attention_v2", mutates_args=())
def torch_deepseek_v4_sparse_attention_v2(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    enable_sharding: bool = False,
    layer_type: str = "unknown",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
) -> torch.Tensor:
    """DSV4 sparse source op with explicit compressor projections for cache insertion."""
    del enable_sharding, layer_type, layer_idx, window_size, head_dim
    if compress_ratio:
        if max_compressed_len is None:
            raise ValueError("max_compressed_len is required for compressed DeepSeek V4 attention.")
        if rope_dim is None:
            raise ValueError("rope_dim is required for compressed DeepSeek V4 attention.")
        compressed_kv = _build_full_compressed_kv(
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            freqs_cis_table,
            position_ids,
            rms_norm_eps,
            rope_dim,
            compress_ratio,
            max_compressed_len,
        )
        kv = torch.cat([kv, compressed_kv], dim=1)
    result = torch_deepseek_v4_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)
    if out is not None:
        out.copy_(result)
        return out.new_empty(0)
    return result


@torch_deepseek_v4_sparse_attention_v2.register_fake
def torch_deepseek_v4_sparse_attention_v2_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    enable_sharding: bool = False,
    layer_type: str = "unknown",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
) -> torch.Tensor:
    del (
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        softmax_scale,
        enable_sharding,
        layer_type,
        layer_idx,
        window_size,
        compress_ratio,
        max_compressed_len,
        head_dim,
        rope_dim,
        rms_norm_eps,
    )
    _validate_rank("q", q, 4)
    if out is not None:
        return out.new_empty(0)
    return q.new_empty(q.shape).contiguous()


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_sparse_attention_with_cache",
    mutates_args=(),
)
def torch_deepseek_v4_sparse_attention_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    compress_ratio: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference cached DeepSeek V4 sparse attention.

    The executable decode path implemented here covers uncompressed DeepSeek V4
    layers, i.e. the SWA/local-window path.  Compressed layers still use the
    full source op during prefill, but decode requires incremental compressor
    and indexer state that is not available at this source-op boundary.
    """
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode

    seq_len_host = _to_host_long(seq_len_host, num_seq)
    input_pos_host = _to_host_long(input_pos_host, num_seq)
    cu_seqlen_host = _to_host_long(cu_seqlen_host, num_seq + 1)
    cache_loc_host = cache_loc_host.detach().cpu().to(torch.long).flatten()
    cu_num_pages_host = _to_host_long(cu_num_pages_host, num_seq + 1)

    if compress_ratio != 0 and any(int(pos.item()) > 0 for pos in input_pos_host):
        raise NotImplementedError(
            "DeepSeek V4 compressed sparse-attention decode is not implemented yet. "
            "The current cached reference backend supports ratio-0 SWA/local-window decode only; "
            "ratio-4 and ratio-128 layers need incremental compressor/indexer cache state."
        )

    if compress_ratio != 0:
        result = torch_deepseek_v4_sparse_attention(
            q, kv, attn_sink, topk_idxs, softmax_scale, out=out
        )
        local_flat_start = 0
        for seq_idx in range(num_seq):
            seq_len_i = int(seq_len_host[seq_idx].item())
            kv_seq = _slice_sequence_tokens(kv, seq_idx, local_flat_start, seq_len_i)
            _write_swa_cache(
                kv_seq,
                swa_cache,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                int(input_pos_host[seq_idx].item()),
            )
            local_flat_start += seq_len_i
        return result

    q_flat = q.reshape(-1, *q.shape[2:])
    output_flat = torch.zeros_like(q_flat)

    for seq_idx in range(num_seq):
        seq_len_i = int(seq_len_host[seq_idx].item())
        if seq_len_i == 0:
            continue
        flat_start = int(cu_seqlen_host[seq_idx].item())
        input_pos_i = int(input_pos_host[seq_idx].item())
        q_seq = q_flat[flat_start : flat_start + seq_len_i]
        kv_seq = _slice_sequence_tokens(kv, seq_idx, flat_start, seq_len_i)
        _write_swa_cache(
            kv_seq,
            swa_cache,
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            input_pos_i,
        )
        output_flat[flat_start : flat_start + seq_len_i] = _cached_local_window_attention(
            q_seq,
            attn_sink,
            swa_cache,
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            input_pos_i,
            window_size,
            softmax_scale,
        )

    output = output_flat.view_as(q)
    if active_tokens < q_flat.shape[0]:
        output.reshape(-1, *q.shape[2:])[active_tokens:].zero_()
    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


@torch_deepseek_v4_sparse_attention_with_cache.register_fake
def torch_deepseek_v4_sparse_attention_with_cache_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    compress_ratio: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del (
        kv,
        attn_sink,
        topk_idxs,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        softmax_scale,
        window_size,
        compress_ratio,
    )
    _validate_rank("q", q, 4)
    if out is not None:
        return out.new_empty(0)
    return q.new_empty(q.shape).contiguous()


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_sparse_attention_v2_with_cache",
    mutates_args=(),
)
def torch_deepseek_v4_sparse_attention_v2_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: Optional[int],
    rms_norm_eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference cached DSV4 sparse attention with incremental compressed rows.

    The compressed path stores high-resolution SWA rows, raw compressor
    projections, and emitted compressed MHC rows in paged tensors. A c4/c128 row
    is emitted when its final source token has arrived, matching the visibility
    rule used by the full source op.
    """
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if compress_ratio not in (0, 4, 128):
        raise ValueError(f"compress_ratio must be one of 0, 4, or 128; got {compress_ratio}")

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode

    seq_len_host = _to_host_long(seq_len_host, num_seq)
    input_pos_host = _to_host_long(input_pos_host, num_seq)
    cu_seqlen_host = _to_host_long(cu_seqlen_host, num_seq + 1)
    cache_loc_host = cache_loc_host.detach().cpu().to(torch.long).flatten()
    cu_num_pages_host = _to_host_long(cu_num_pages_host, num_seq + 1)

    q_flat = q.reshape(-1, *q.shape[2:])
    output_flat = torch.zeros_like(q_flat)

    if compress_ratio:
        if max_compressed_len is None or max_compressed_len <= 0:
            raise ValueError(
                f"max_compressed_len must be positive for compress_ratio={compress_ratio}, "
                f"got {max_compressed_len}"
            )
        compressed_capacity = int(max_compressed_len)
    else:
        compressed_capacity = 0

    for seq_idx in range(num_seq):
        seq_len_i = int(seq_len_host[seq_idx].item())
        if seq_len_i == 0:
            continue
        flat_start = int(cu_seqlen_host[seq_idx].item())
        input_pos_i = int(input_pos_host[seq_idx].item())
        q_seq = q_flat[flat_start : flat_start + seq_len_i]
        kv_seq = _slice_sequence_tokens(kv, seq_idx, flat_start, seq_len_i)
        _write_swa_cache(
            kv_seq,
            swa_cache,
            cache_loc_host,
            cu_num_pages_host,
            seq_idx,
            input_pos_i,
        )

        if compress_ratio:
            compressor_kv_seq = _slice_sequence_tokens(
                compressor_kv, seq_idx, flat_start, seq_len_i
            )
            compressor_gate_seq = _slice_sequence_tokens(
                compressor_gate, seq_idx, flat_start, seq_len_i
            )
            _update_compressed_caches(
                compressor_kv_seq,
                compressor_gate_seq,
                compressor_ape,
                compressor_norm_weight,
                freqs_cis_table,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                input_pos_i,
                mhc_cache,
                compressor_kv_cache,
                compressor_gate_cache,
                rms_norm_eps,
                rope_dim,
                compress_ratio,
                compressed_capacity,
            )

        if compress_ratio and input_pos_i > 0:
            output_flat[flat_start : flat_start + seq_len_i] = _cached_compressed_attention(
                q_seq,
                attn_sink,
                swa_cache,
                mhc_cache,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                input_pos_i,
                window_size,
                compress_ratio,
                compressed_capacity,
                softmax_scale,
            )
        elif compress_ratio:
            full = torch_deepseek_v4_sparse_attention_v2(
                q,
                kv,
                attn_sink,
                topk_idxs,
                compressor_kv,
                compressor_gate,
                compressor_ape,
                compressor_norm_weight,
                freqs_cis_table,
                position_ids,
                softmax_scale,
                window_size=window_size,
                compress_ratio=compress_ratio,
                max_compressed_len=max_compressed_len,
                rope_dim=rope_dim,
                rms_norm_eps=rms_norm_eps,
            )
            output_flat = full.reshape_as(output_flat)
            break
        else:
            output_flat[flat_start : flat_start + seq_len_i] = _cached_local_window_attention(
                q_seq,
                attn_sink,
                swa_cache,
                cache_loc_host,
                cu_num_pages_host,
                seq_idx,
                input_pos_i,
                window_size,
                softmax_scale,
            )

    output = output_flat.view_as(q)
    if active_tokens < q_flat.shape[0]:
        output.reshape(-1, *q.shape[2:])[active_tokens:].zero_()
    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


@torch_deepseek_v4_sparse_attention_v2_with_cache.register_fake
def torch_deepseek_v4_sparse_attention_v2_with_cache_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: Optional[int],
    rms_norm_eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del (
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        max_compressed_len,
        rms_norm_eps,
        rope_dim,
    )
    _validate_rank("q", q, 4)
    if out is not None:
        return out.new_empty(0)
    return q.new_empty(q.shape).contiguous()


def _device_skip_reason(name: str, tensor: torch.Tensor, path: str) -> str | None:
    if not tensor.is_cuda:
        return f"{name} must be a CUDA tensor for the Triton {path} path"
    return None


def _triton_metadata_on_q_device(tensor: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if q.is_cuda and tensor.device != q.device:
        return tensor.to(device=q.device, non_blocking=True)
    return tensor


def _deepseek_v4_ratio0_prefill_mixed_triton_skip_reason(
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
    num_seq: int,
    active_tokens: int,
) -> str | None:
    """Return why the ratio-0 prefill/mixed Triton path cannot run, or ``None``."""
    path = "ratio-0 prefill/mixed"
    if compress_ratio != 0:
        return f"compress_ratio must be 0 for the Triton {path} path, got {compress_ratio}"
    if window_size != _DSV4_TRITON_WINDOW_SIZE:
        return f"window_size must be {_DSV4_TRITON_WINDOW_SIZE}, got {window_size}"
    if q.shape[2] != _DSV4_TRITON_LOCAL_NUM_HEADS:
        return f"q local head count must be {_DSV4_TRITON_LOCAL_NUM_HEADS}, got {q.shape[2]}"
    if q.shape[3] != _DSV4_TRITON_HEAD_DIM:
        return f"q head dimension must be {_DSV4_TRITON_HEAD_DIM}, got {q.shape[3]}"
    if kv.shape[:2] != q.shape[:2] or kv.shape[2] != _DSV4_TRITON_HEAD_DIM:
        return (
            f"kv must have shape {(q.shape[0], q.shape[1], _DSV4_TRITON_HEAD_DIM)}, "
            f"got {tuple(kv.shape)}"
        )
    if topk_idxs.shape != (
        q.shape[0],
        q.shape[1],
        _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[0],
    ):
        return (
            "topk_idxs must have shape "
            f"{(q.shape[0], q.shape[1], _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[0])}, "
            f"got {tuple(topk_idxs.shape)}"
        )
    if attn_sink.shape != (_DSV4_TRITON_LOCAL_NUM_HEADS,):
        return (
            f"attn_sink must have shape ({_DSV4_TRITON_LOCAL_NUM_HEADS},), "
            f"got {tuple(attn_sink.shape)}"
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
    if swa_cache.shape[-1] != _DSV4_TRITON_HEAD_DIM:
        return (
            f"swa_cache last dimension must be {_DSV4_TRITON_HEAD_DIM}, got {swa_cache.shape[-1]}"
        )
    if active_tokens <= 0:
        return f"active_tokens must be positive, got {active_tokens}"
    flat_capacity = int(q.shape[0] * q.shape[1])
    if active_tokens > flat_capacity:
        return f"active_tokens ({active_tokens}) exceed q capacity ({flat_capacity})"
    if num_seq <= 0:
        return f"num_seq must be positive, got {num_seq}"
    if seq_len_host.numel() < num_seq:
        return f"seq_len_host needs at least {num_seq} entries, got {seq_len_host.numel()}"
    if input_pos_host.numel() < num_seq:
        return f"input_pos_host needs at least {num_seq} entries, got {input_pos_host.numel()}"
    if cu_seqlen_host.numel() < num_seq + 1:
        return f"cu_seqlen_host needs at least {num_seq + 1} entries, got {cu_seqlen_host.numel()}"
    if cu_num_pages_host.numel() < num_seq + 1:
        return (
            f"cu_num_pages_host needs at least {num_seq + 1} entries, "
            f"got {cu_num_pages_host.numel()}"
        )

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
        reason = _device_skip_reason(name, tensor, path)
        if reason is not None:
            return reason
        if tensor.device != q.device:
            return f"{name} must be on {q.device}, got {tensor.device}"
    if out is not None:
        reason = _device_skip_reason("out", out, path)
        if reason is not None:
            return reason
        if out.device != q.device:
            return f"out must be on {q.device}, got {out.device}"
        if out.shape != q.shape:
            return f"out shape must be {tuple(q.shape)}, got {tuple(out.shape)}"
        if out.dtype != q.dtype:
            return f"out dtype must be {q.dtype}, got {out.dtype}"
        if not out.is_contiguous():
            return "out must be contiguous for the Triton ratio-0 prefill/mixed path"

    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("topk_idxs", topk_idxs),
    ):
        if not tensor.is_contiguous():
            return f"{name} must be contiguous for the Triton ratio-0 prefill/mixed path"
    if swa_cache.stride(-1) != 1:
        return f"swa_cache last dimension must be contiguous, got stride {swa_cache.stride(-1)}"

    try:
        tokens_per_block, _, _ = _cache_layout_for_triton(swa_cache)
    except ValueError as exc:
        return str(exc)
    if tokens_per_block <= 0:
        return f"swa_cache tokens_per_block must be positive, got {tokens_per_block}"
    return None


@triton.jit
def _update_ratio0_prefill_mixed_swa_cache_kernel(
    kv_ptr,
    input_pos_ptr,
    cu_seqlen_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    swa_cache_ptr,
    kv_stride_token: tl.constexpr,
    kv_stride_dim: tl.constexpr,
    cache_page_stride: tl.constexpr,
    cache_token_stride: tl.constexpr,
    tokens_per_block: tl.constexpr,
    num_seq: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    seq_idx = tl.full((), 0, dtype=tl.int64)
    seq_start = tl.full((), 0, dtype=tl.int64)

    for idx in range(num_seq):
        start = tl.load(cu_seqlen_ptr + idx).to(tl.int64)
        end = tl.load(cu_seqlen_ptr + idx + 1).to(tl.int64)
        in_seq = (token_idx >= start) & (token_idx < end)
        seq_idx = tl.where(in_seq, idx, seq_idx)
        seq_start = tl.where(in_seq, start, seq_start)

    token_offset = token_idx - seq_start
    input_pos = tl.load(input_pos_ptr + seq_idx).to(tl.int64) + token_offset
    page_ordinal = input_pos // tokens_per_block
    page_offset = input_pos - page_ordinal * tokens_per_block
    page_table_start = tl.load(cu_num_pages_ptr + seq_idx).to(tl.int64)
    page_idx = tl.load(cache_loc_ptr + page_table_start + page_ordinal).to(tl.int64)

    kv_offsets = token_idx * kv_stride_token + dim_offsets * kv_stride_dim
    cache_offsets = page_idx * cache_page_stride + page_offset * cache_token_stride + dim_offsets
    tl.store(
        swa_cache_ptr + cache_offsets,
        tl.load(kv_ptr + kv_offsets, mask=dim_mask, other=0.0),
        mask=dim_mask,
    )


@triton.jit
def _ratio0_prefill_mixed_swa_attention_kernel(
    q_ptr,
    attn_sink_ptr,
    topk_idxs_ptr,
    input_pos_ptr,
    cu_seqlen_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    swa_cache_ptr,
    out_ptr,
    q_stride_token: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    topk_stride_token: tl.constexpr,
    topk_stride_topk: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    cache_page_stride: tl.constexpr,
    cache_token_stride: tl.constexpr,
    tokens_per_block: tl.constexpr,
    num_seq: tl.constexpr,
    active_tokens: tl.constexpr,
    softmax_scale: tl.constexpr,
    topk_width: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    token_is_active = token_idx < active_tokens

    out_offsets = (
        token_idx * out_stride_token + head_idx * out_stride_head + dim_offsets * out_stride_dim
    )
    if not token_is_active:
        tl.store(out_ptr + out_offsets, tl.zeros([block_dim], dtype=tl.float32), mask=dim_mask)
        return

    seq_idx = tl.full((), 0, dtype=tl.int64)
    seq_start = tl.full((), 0, dtype=tl.int64)
    for idx in range(num_seq):
        start = tl.load(cu_seqlen_ptr + idx).to(tl.int64)
        end = tl.load(cu_seqlen_ptr + idx + 1).to(tl.int64)
        in_seq = (token_idx >= start) & (token_idx < end)
        seq_idx = tl.where(in_seq, idx, seq_idx)
        seq_start = tl.where(in_seq, start, seq_start)

    token_offset = token_idx - seq_start
    query_pos = tl.load(input_pos_ptr + seq_idx).to(tl.int64) + token_offset
    page_table_start = tl.load(cu_num_pages_ptr + seq_idx).to(tl.int64)
    q_offsets = token_idx * q_stride_token + head_idx * q_stride_head + dim_offsets * q_stride_dim
    q_vec = tl.load(q_ptr + q_offsets, mask=dim_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([block_dim], dtype=tl.float32)
    m_i = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
    l_i = tl.full((), 1.0, dtype=tl.float32)

    for topk_offset in range(topk_width):
        selected = tl.load(
            topk_idxs_ptr + token_idx * topk_stride_token + topk_offset * topk_stride_topk
        )
        valid_topk = (selected >= 0) & (selected <= query_pos)
        row_pos = selected.to(tl.int64)
        page_ordinal = row_pos // tokens_per_block
        page_offset = row_pos - page_ordinal * tokens_per_block
        page_idx = tl.load(
            cache_loc_ptr + page_table_start + page_ordinal,
            mask=valid_topk,
            other=0,
        ).to(tl.int64)
        cache_offsets = (
            page_idx * cache_page_stride + page_offset * cache_token_stride + dim_offsets
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

    tl.store(out_ptr + out_offsets, acc / l_i, mask=dim_mask)


def triton_deepseek_v4_ratio0_swa_prefill_mixed_attention_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run ratio-0 cached DSV4 prefill/mixed attention through Triton kernels."""
    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode

    reason = _deepseek_v4_ratio0_prefill_mixed_triton_skip_reason(
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
        num_seq,
        active_tokens,
    )
    if reason is not None:
        raise RuntimeError(reason)

    tokens_per_block, cache_page_stride, cache_token_stride = _cache_layout_for_triton(swa_cache)
    output = torch.empty_like(q) if out is None else out
    flat_capacity = int(q.shape[0] * q.shape[1])
    q_flat = q.reshape(flat_capacity, q.shape[2], q.shape[3])
    kv_flat = kv.reshape(flat_capacity, kv.shape[2])
    topk_flat = topk_idxs.reshape(flat_capacity, topk_idxs.shape[2])
    output_flat = output.reshape(flat_capacity, output.shape[2], output.shape[3])
    block_dim = triton.next_power_of_2(_DSV4_TRITON_HEAD_DIM)

    _update_ratio0_prefill_mixed_swa_cache_kernel[(active_tokens,)](
        kv_flat,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        kv_flat.stride(0),
        kv_flat.stride(1),
        cache_page_stride,
        cache_token_stride,
        tokens_per_block,
        num_seq,
        _DSV4_TRITON_HEAD_DIM,
        block_dim,
        num_warps=8,
    )
    _ratio0_prefill_mixed_swa_attention_kernel[(flat_capacity, _DSV4_TRITON_LOCAL_NUM_HEADS)](
        q_flat,
        attn_sink,
        topk_flat,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        output_flat,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        topk_flat.stride(0),
        topk_flat.stride(1),
        output_flat.stride(0),
        output_flat.stride(1),
        output_flat.stride(2),
        cache_page_stride,
        cache_token_stride,
        tokens_per_block,
        num_seq,
        active_tokens,
        softmax_scale,
        _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[0],
        _DSV4_TRITON_HEAD_DIM,
        block_dim,
        num_warps=8,
    )

    if out is not None:
        return out.new_empty(0)
    return output


def _deepseek_v4_ratio4_triton_skip_reason(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: Optional[int],
    rope_dim: int,
    out: Optional[torch.Tensor],
    active_tokens: int,
) -> str | None:
    """Return why the ratio-4 Triton decode path cannot run, or ``None``."""
    path = "ratio-4 compressed decode"
    if compress_ratio != 4:
        return f"compress_ratio must be 4 for the Triton {path} path, got {compress_ratio}"
    if max_compressed_len != _DSV4_TRITON_RATIO4_MAX_COMPRESSED_LEN:
        return (
            "max_compressed_len must be "
            f"{_DSV4_TRITON_RATIO4_MAX_COMPRESSED_LEN} for the Triton {path} path, "
            f"got {max_compressed_len}"
        )
    if window_size != _DSV4_TRITON_WINDOW_SIZE:
        return f"window_size must be {_DSV4_TRITON_WINDOW_SIZE}, got {window_size}"
    if rope_dim != _DSV4_TRITON_ROPE_DIM:
        return f"rope_dim must be {_DSV4_TRITON_ROPE_DIM}, got {rope_dim}"
    if q.shape[1] != 1:
        return "the Triton ratio-4 compressed path currently supports decode-only q.shape[1] == 1"
    if topk_idxs.shape != (q.shape[0], q.shape[1], _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[4]):
        return (
            "topk_idxs must have shape "
            f"{(q.shape[0], q.shape[1], _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[4])}, "
            f"got {tuple(topk_idxs.shape)}"
        )
    if compressor_kv.shape != (q.shape[0], q.shape[1], 2 * _DSV4_TRITON_HEAD_DIM):
        return (
            "compressor_kv must have shape "
            f"{(q.shape[0], q.shape[1], 2 * _DSV4_TRITON_HEAD_DIM)}, "
            f"got {tuple(compressor_kv.shape)}"
        )
    if compressor_gate.shape != compressor_kv.shape:
        return (
            "compressor_gate must match compressor_kv shape, "
            f"got {tuple(compressor_gate.shape)} and {tuple(compressor_kv.shape)}"
        )
    if compressor_ape.shape != (4, 2 * _DSV4_TRITON_HEAD_DIM):
        return (
            f"compressor_ape must have shape {(4, 2 * _DSV4_TRITON_HEAD_DIM)}, "
            f"got {tuple(compressor_ape.shape)}"
        )
    if compressor_norm_weight.shape != (_DSV4_TRITON_HEAD_DIM,):
        return (
            f"compressor_norm_weight must have shape ({_DSV4_TRITON_HEAD_DIM},), "
            f"got {tuple(compressor_norm_weight.shape)}"
        )
    if active_tokens <= 0:
        return f"active_tokens must be positive, got {active_tokens}"
    if active_tokens > q.shape[0]:
        return f"active_tokens ({active_tokens}) exceed decode batch capacity ({q.shape[0]})"
    if seq_len_host.numel() < active_tokens:
        return f"seq_len_host needs at least {active_tokens} entries, got {seq_len_host.numel()}"
    if input_pos_host.numel() < active_tokens:
        return (
            f"input_pos_host needs at least {active_tokens} entries, got {input_pos_host.numel()}"
        )
    if cu_seqlen_host.numel() < active_tokens + 1:
        return (
            f"cu_seqlen_host needs at least {active_tokens + 1} entries, "
            f"got {cu_seqlen_host.numel()}"
        )
    if cu_num_pages_host.numel() < active_tokens + 1:
        return (
            f"cu_num_pages_host needs at least {active_tokens + 1} entries, "
            f"got {cu_num_pages_host.numel()}"
        )
    if not freqs_cis_table.is_complex():
        return f"freqs_cis_table must be complex, got {freqs_cis_table.dtype}"
    if freqs_cis_table.shape[-1] != rope_dim // 2:
        return (
            f"freqs_cis_table last dimension must be {rope_dim // 2}, "
            f"got {freqs_cis_table.shape[-1]}"
        )

    dtype_checks = (
        ("q", q, torch.bfloat16),
        ("kv", kv, torch.bfloat16),
        ("attn_sink", attn_sink, torch.float32),
        ("compressor_kv", compressor_kv, torch.bfloat16),
        ("compressor_gate", compressor_gate, torch.bfloat16),
        ("compressor_norm_weight", compressor_norm_weight, torch.float32),
        ("swa_cache", swa_cache, torch.bfloat16),
        ("mhc_cache", mhc_cache, torch.bfloat16),
    )
    for name, tensor, dtype in dtype_checks:
        if tensor.dtype != dtype:
            return f"{name} must be {dtype}, got {tensor.dtype}"
    if compressor_ape.dtype not in (torch.bfloat16, torch.float32):
        return f"compressor_ape must be bfloat16 or float32, got {compressor_ape.dtype}"
    if compressor_kv_cache.dtype not in (torch.bfloat16, torch.float32):
        return f"compressor_kv_cache must be bfloat16 or float32, got {compressor_kv_cache.dtype}"
    if compressor_gate_cache.dtype not in (torch.bfloat16, torch.float32):
        return (
            f"compressor_gate_cache must be bfloat16 or float32, got {compressor_gate_cache.dtype}"
        )
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        return f"topk_idxs must be int32 or int64, got {topk_idxs.dtype}"

    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("attn_sink", attn_sink),
        ("topk_idxs", topk_idxs),
        ("compressor_kv", compressor_kv),
        ("compressor_gate", compressor_gate),
        ("compressor_ape", compressor_ape),
        ("compressor_norm_weight", compressor_norm_weight),
        ("freqs_cis_table", freqs_cis_table),
        ("seq_len_host", seq_len_host),
        ("input_pos_host", input_pos_host),
        ("cu_seqlen_host", cu_seqlen_host),
        ("cache_loc_host", cache_loc_host),
        ("cu_num_pages_host", cu_num_pages_host),
        ("swa_cache", swa_cache),
        ("mhc_cache", mhc_cache),
        ("compressor_kv_cache", compressor_kv_cache),
        ("compressor_gate_cache", compressor_gate_cache),
    ):
        reason = _device_skip_reason(name, tensor, path)
        if reason is not None:
            return reason
    if out is not None:
        reason = _device_skip_reason("out", out, path)
        if reason is not None:
            return reason

    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("compressor_kv", compressor_kv),
        ("compressor_gate", compressor_gate),
        ("swa_cache", swa_cache),
        ("mhc_cache", mhc_cache),
        ("compressor_kv_cache", compressor_kv_cache),
        ("compressor_gate_cache", compressor_gate_cache),
    ):
        if tensor.stride(-1) != 1:
            return f"{name} last dimension must be contiguous, got stride {tensor.stride(-1)}"
    if compressor_kv_cache.shape[-1] < 2 * _DSV4_TRITON_HEAD_DIM:
        return (
            f"compressor_kv_cache last dimension must be at least {2 * _DSV4_TRITON_HEAD_DIM}, "
            f"got {compressor_kv_cache.shape[-1]}"
        )
    if compressor_gate_cache.shape[-1] < 2 * _DSV4_TRITON_HEAD_DIM:
        return (
            f"compressor_gate_cache last dimension must be at least {2 * _DSV4_TRITON_HEAD_DIM}, "
            f"got {compressor_gate_cache.shape[-1]}"
        )
    try:
        for cache_name, cache in (
            ("swa_cache", swa_cache),
            ("mhc_cache", mhc_cache),
            ("compressor_kv_cache", compressor_kv_cache),
            ("compressor_gate_cache", compressor_gate_cache),
        ):
            tokens_per_block, _, _ = _cache_layout_for_triton(cache)
            if tokens_per_block <= 0:
                return f"{cache_name} tokens_per_block must be positive, got {tokens_per_block}"
    except ValueError as exc:
        return str(exc)
    return None


def _deepseek_v4_ratio128_triton_skip_reason(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: Optional[int],
    rope_dim: int,
    out: Optional[torch.Tensor],
    active_tokens: int,
) -> str | None:
    """Return why the ratio-128 Triton decode path cannot run, or ``None``."""
    path = "ratio-128 compressed decode"
    if compress_ratio != 128:
        return f"compress_ratio must be 128 for the Triton {path} path, got {compress_ratio}"
    if max_compressed_len != _DSV4_TRITON_RATIO128_MAX_COMPRESSED_LEN:
        return (
            "max_compressed_len must be "
            f"{_DSV4_TRITON_RATIO128_MAX_COMPRESSED_LEN} for the Triton {path} path, "
            f"got {max_compressed_len}"
        )
    if window_size != _DSV4_TRITON_WINDOW_SIZE:
        return f"window_size must be {_DSV4_TRITON_WINDOW_SIZE}, got {window_size}"
    if rope_dim != _DSV4_TRITON_ROPE_DIM:
        return f"rope_dim must be {_DSV4_TRITON_ROPE_DIM}, got {rope_dim}"
    if q.shape[1] != 1:
        return "the Triton ratio-128 compressed path currently supports decode-only q.shape[1] == 1"
    if topk_idxs.shape != (q.shape[0], q.shape[1], _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[128]):
        return (
            "topk_idxs must have shape "
            f"{(q.shape[0], q.shape[1], _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[128])}, "
            f"got {tuple(topk_idxs.shape)}"
        )
    if compressor_kv.shape != (q.shape[0], q.shape[1], _DSV4_TRITON_HEAD_DIM):
        return (
            f"compressor_kv must have shape {(q.shape[0], q.shape[1], _DSV4_TRITON_HEAD_DIM)}, "
            f"got {tuple(compressor_kv.shape)}"
        )
    if compressor_gate.shape != compressor_kv.shape:
        return (
            "compressor_gate must match compressor_kv shape, "
            f"got {tuple(compressor_gate.shape)} and {tuple(compressor_kv.shape)}"
        )
    if compressor_ape.shape != (128, _DSV4_TRITON_HEAD_DIM):
        return (
            f"compressor_ape must have shape {(128, _DSV4_TRITON_HEAD_DIM)}, "
            f"got {tuple(compressor_ape.shape)}"
        )
    if compressor_norm_weight.shape != (_DSV4_TRITON_HEAD_DIM,):
        return (
            f"compressor_norm_weight must have shape ({_DSV4_TRITON_HEAD_DIM},), "
            f"got {tuple(compressor_norm_weight.shape)}"
        )
    if active_tokens <= 0:
        return f"active_tokens must be positive, got {active_tokens}"
    if active_tokens > q.shape[0]:
        return f"active_tokens ({active_tokens}) exceed decode batch capacity ({q.shape[0]})"
    if seq_len_host.numel() < active_tokens:
        return f"seq_len_host needs at least {active_tokens} entries, got {seq_len_host.numel()}"
    if input_pos_host.numel() < active_tokens:
        return (
            f"input_pos_host needs at least {active_tokens} entries, got {input_pos_host.numel()}"
        )
    if cu_seqlen_host.numel() < active_tokens + 1:
        return (
            f"cu_seqlen_host needs at least {active_tokens + 1} entries, "
            f"got {cu_seqlen_host.numel()}"
        )
    if cu_num_pages_host.numel() < active_tokens + 1:
        return (
            f"cu_num_pages_host needs at least {active_tokens + 1} entries, "
            f"got {cu_num_pages_host.numel()}"
        )
    if not freqs_cis_table.is_complex():
        return f"freqs_cis_table must be complex, got {freqs_cis_table.dtype}"
    if freqs_cis_table.shape[-1] != rope_dim // 2:
        return (
            f"freqs_cis_table last dimension must be {rope_dim // 2}, "
            f"got {freqs_cis_table.shape[-1]}"
        )

    dtype_checks = (
        ("q", q, torch.bfloat16),
        ("kv", kv, torch.bfloat16),
        ("attn_sink", attn_sink, torch.float32),
        ("compressor_kv", compressor_kv, torch.bfloat16),
        ("compressor_gate", compressor_gate, torch.bfloat16),
        ("compressor_norm_weight", compressor_norm_weight, torch.float32),
        ("swa_cache", swa_cache, torch.bfloat16),
        ("mhc_cache", mhc_cache, torch.bfloat16),
    )
    for name, tensor, dtype in dtype_checks:
        if tensor.dtype != dtype:
            return f"{name} must be {dtype}, got {tensor.dtype}"
    if compressor_ape.dtype not in (torch.bfloat16, torch.float32):
        return f"compressor_ape must be bfloat16 or float32, got {compressor_ape.dtype}"
    if compressor_kv_cache.dtype not in (torch.bfloat16, torch.float32):
        return f"compressor_kv_cache must be bfloat16 or float32, got {compressor_kv_cache.dtype}"
    if compressor_gate_cache.dtype not in (torch.bfloat16, torch.float32):
        return (
            f"compressor_gate_cache must be bfloat16 or float32, got {compressor_gate_cache.dtype}"
        )
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        return f"topk_idxs must be int32 or int64, got {topk_idxs.dtype}"

    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("attn_sink", attn_sink),
        ("topk_idxs", topk_idxs),
        ("compressor_kv", compressor_kv),
        ("compressor_gate", compressor_gate),
        ("compressor_ape", compressor_ape),
        ("compressor_norm_weight", compressor_norm_weight),
        ("freqs_cis_table", freqs_cis_table),
        ("seq_len_host", seq_len_host),
        ("input_pos_host", input_pos_host),
        ("cu_seqlen_host", cu_seqlen_host),
        ("cache_loc_host", cache_loc_host),
        ("cu_num_pages_host", cu_num_pages_host),
        ("swa_cache", swa_cache),
        ("mhc_cache", mhc_cache),
        ("compressor_kv_cache", compressor_kv_cache),
        ("compressor_gate_cache", compressor_gate_cache),
    ):
        reason = _device_skip_reason(name, tensor, path)
        if reason is not None:
            return reason
    if out is not None:
        reason = _device_skip_reason("out", out, path)
        if reason is not None:
            return reason

    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("compressor_kv", compressor_kv),
        ("compressor_gate", compressor_gate),
        ("swa_cache", swa_cache),
        ("mhc_cache", mhc_cache),
        ("compressor_kv_cache", compressor_kv_cache),
        ("compressor_gate_cache", compressor_gate_cache),
    ):
        if tensor.stride(-1) != 1:
            return f"{name} last dimension must be contiguous, got stride {tensor.stride(-1)}"
    if compressor_kv_cache.shape[-1] < _DSV4_TRITON_HEAD_DIM:
        return (
            f"compressor_kv_cache last dimension must be at least {_DSV4_TRITON_HEAD_DIM}, "
            f"got {compressor_kv_cache.shape[-1]}"
        )
    if compressor_gate_cache.shape[-1] < _DSV4_TRITON_HEAD_DIM:
        return (
            f"compressor_gate_cache last dimension must be at least {_DSV4_TRITON_HEAD_DIM}, "
            f"got {compressor_gate_cache.shape[-1]}"
        )
    try:
        for cache_name, cache in (
            ("swa_cache", swa_cache),
            ("mhc_cache", mhc_cache),
            ("compressor_kv_cache", compressor_kv_cache),
            ("compressor_gate_cache", compressor_gate_cache),
        ):
            tokens_per_block, _, _ = _cache_layout_for_triton(cache)
            if tokens_per_block <= 0:
                return f"{cache_name} tokens_per_block must be positive, got {tokens_per_block}"
    except ValueError as exc:
        return str(exc)
    return None


def _deepseek_v4_compressed_prefill_mixed_triton_skip_reason(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: Optional[int],
    rope_dim: int,
    out: Optional[torch.Tensor],
    num_seq: int,
    active_tokens: int,
) -> str | None:
    """Return why the compressed prefill/mixed Triton path cannot run, or ``None``."""
    path = f"ratio-{compress_ratio} compressed prefill/mixed"
    if compress_ratio not in (4, 128):
        return f"compress_ratio must be 4 or 128 for the Triton {path} path, got {compress_ratio}"
    expected_max_compressed_len = (
        _DSV4_TRITON_RATIO4_MAX_COMPRESSED_LEN
        if compress_ratio == 4
        else _DSV4_TRITON_RATIO128_MAX_COMPRESSED_LEN
    )
    if max_compressed_len != expected_max_compressed_len:
        return (
            f"max_compressed_len must be {expected_max_compressed_len} for the Triton "
            f"{path} path, got {max_compressed_len}"
        )
    if window_size != _DSV4_TRITON_WINDOW_SIZE:
        return f"window_size must be {_DSV4_TRITON_WINDOW_SIZE}, got {window_size}"
    if rope_dim != _DSV4_TRITON_ROPE_DIM:
        return f"rope_dim must be {_DSV4_TRITON_ROPE_DIM}, got {rope_dim}"
    if q.shape[2] != _DSV4_TRITON_LOCAL_NUM_HEADS:
        return f"q local head count must be {_DSV4_TRITON_LOCAL_NUM_HEADS}, got {q.shape[2]}"
    if q.shape[3] != _DSV4_TRITON_HEAD_DIM:
        return f"q head dimension must be {_DSV4_TRITON_HEAD_DIM}, got {q.shape[3]}"
    if kv.shape != (q.shape[0], q.shape[1], _DSV4_TRITON_HEAD_DIM):
        return (
            f"kv must have shape {(q.shape[0], q.shape[1], _DSV4_TRITON_HEAD_DIM)}, "
            f"got {tuple(kv.shape)}"
        )
    expected_topk_shape = (
        q.shape[0],
        q.shape[1],
        _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[compress_ratio],
    )
    if topk_idxs.shape != expected_topk_shape:
        return f"topk_idxs must have shape {expected_topk_shape}, got {tuple(topk_idxs.shape)}"
    state_dim = _DSV4_TRITON_COMPRESSOR_DIM_BY_RATIO[compress_ratio]
    expected_state_shape = (q.shape[0], q.shape[1], state_dim)
    if compressor_kv.shape != expected_state_shape:
        return f"compressor_kv must have shape {expected_state_shape}, got {tuple(compressor_kv.shape)}"
    if compressor_gate.shape != expected_state_shape:
        return (
            f"compressor_gate must have shape {expected_state_shape}, "
            f"got {tuple(compressor_gate.shape)}"
        )
    if compressor_ape.shape != (compress_ratio, state_dim):
        return (
            f"compressor_ape must have shape {(compress_ratio, state_dim)}, "
            f"got {tuple(compressor_ape.shape)}"
        )
    if compressor_norm_weight.shape != (_DSV4_TRITON_HEAD_DIM,):
        return (
            f"compressor_norm_weight must have shape ({_DSV4_TRITON_HEAD_DIM},), "
            f"got {tuple(compressor_norm_weight.shape)}"
        )
    if attn_sink.shape != (_DSV4_TRITON_LOCAL_NUM_HEADS,):
        return (
            f"attn_sink must have shape ({_DSV4_TRITON_LOCAL_NUM_HEADS},), "
            f"got {tuple(attn_sink.shape)}"
        )
    if active_tokens <= 0:
        return f"active_tokens must be positive, got {active_tokens}"
    flat_capacity = int(q.shape[0] * q.shape[1])
    if active_tokens > flat_capacity:
        return f"active_tokens ({active_tokens}) exceed q capacity ({flat_capacity})"
    if num_seq <= 0:
        return f"num_seq must be positive, got {num_seq}"
    if seq_len_host.numel() < num_seq:
        return f"seq_len_host needs at least {num_seq} entries, got {seq_len_host.numel()}"
    if input_pos_host.numel() < num_seq:
        return f"input_pos_host needs at least {num_seq} entries, got {input_pos_host.numel()}"
    if cu_seqlen_host.numel() < num_seq + 1:
        return f"cu_seqlen_host needs at least {num_seq + 1} entries, got {cu_seqlen_host.numel()}"
    if cu_num_pages_host.numel() < num_seq + 1:
        return (
            f"cu_num_pages_host needs at least {num_seq + 1} entries, "
            f"got {cu_num_pages_host.numel()}"
        )
    if not freqs_cis_table.is_complex():
        return f"freqs_cis_table must be complex, got {freqs_cis_table.dtype}"
    if freqs_cis_table.shape[-1] != rope_dim // 2:
        return (
            f"freqs_cis_table last dimension must be {rope_dim // 2}, "
            f"got {freqs_cis_table.shape[-1]}"
        )

    dtype_checks = (
        ("q", q, torch.bfloat16),
        ("kv", kv, torch.bfloat16),
        ("attn_sink", attn_sink, torch.float32),
        ("compressor_kv", compressor_kv, torch.bfloat16),
        ("compressor_gate", compressor_gate, torch.bfloat16),
        ("compressor_norm_weight", compressor_norm_weight, torch.float32),
        ("swa_cache", swa_cache, torch.bfloat16),
        ("mhc_cache", mhc_cache, torch.bfloat16),
    )
    for name, tensor, dtype in dtype_checks:
        if tensor.dtype != dtype:
            return f"{name} must be {dtype}, got {tensor.dtype}"
    if compressor_ape.dtype not in (torch.bfloat16, torch.float32):
        return f"compressor_ape must be bfloat16 or float32, got {compressor_ape.dtype}"
    if compressor_kv_cache.dtype not in (torch.bfloat16, torch.float32):
        return f"compressor_kv_cache must be bfloat16 or float32, got {compressor_kv_cache.dtype}"
    if compressor_gate_cache.dtype not in (torch.bfloat16, torch.float32):
        return (
            f"compressor_gate_cache must be bfloat16 or float32, got {compressor_gate_cache.dtype}"
        )
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        return f"topk_idxs must be int32 or int64, got {topk_idxs.dtype}"

    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("attn_sink", attn_sink),
        ("topk_idxs", topk_idxs),
        ("compressor_kv", compressor_kv),
        ("compressor_gate", compressor_gate),
        ("compressor_ape", compressor_ape),
        ("compressor_norm_weight", compressor_norm_weight),
        ("freqs_cis_table", freqs_cis_table),
        ("seq_len_host", seq_len_host),
        ("input_pos_host", input_pos_host),
        ("cu_seqlen_host", cu_seqlen_host),
        ("cache_loc_host", cache_loc_host),
        ("cu_num_pages_host", cu_num_pages_host),
        ("swa_cache", swa_cache),
        ("mhc_cache", mhc_cache),
        ("compressor_kv_cache", compressor_kv_cache),
        ("compressor_gate_cache", compressor_gate_cache),
    ):
        reason = _device_skip_reason(name, tensor, path)
        if reason is not None:
            return reason
        if tensor.device != q.device:
            return f"{name} must be on {q.device}, got {tensor.device}"
    if out is not None:
        reason = _device_skip_reason("out", out, path)
        if reason is not None:
            return reason
        if out.device != q.device:
            return f"out must be on {q.device}, got {out.device}"
        if out.shape != q.shape:
            return f"out shape must be {tuple(q.shape)}, got {tuple(out.shape)}"
        if out.dtype != q.dtype:
            return f"out dtype must be {q.dtype}, got {out.dtype}"
        if not out.is_contiguous():
            return f"out must be contiguous for the Triton {path} path"

    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("topk_idxs", topk_idxs),
        ("compressor_kv", compressor_kv),
        ("compressor_gate", compressor_gate),
    ):
        if not tensor.is_contiguous():
            return f"{name} must be contiguous for the Triton {path} path"
    for name, tensor in (
        ("swa_cache", swa_cache),
        ("mhc_cache", mhc_cache),
        ("compressor_kv_cache", compressor_kv_cache),
        ("compressor_gate_cache", compressor_gate_cache),
    ):
        if tensor.stride(-1) != 1:
            return f"{name} last dimension must be contiguous, got stride {tensor.stride(-1)}"
    if compressor_kv_cache.shape[-1] < state_dim:
        return (
            f"compressor_kv_cache last dimension must be at least {state_dim}, "
            f"got {compressor_kv_cache.shape[-1]}"
        )
    if compressor_gate_cache.shape[-1] < state_dim:
        return (
            f"compressor_gate_cache last dimension must be at least {state_dim}, "
            f"got {compressor_gate_cache.shape[-1]}"
        )
    try:
        for cache_name, cache in (
            ("swa_cache", swa_cache),
            ("mhc_cache", mhc_cache),
            ("compressor_kv_cache", compressor_kv_cache),
            ("compressor_gate_cache", compressor_gate_cache),
        ):
            tokens_per_block, _, _ = _cache_layout_for_triton(cache)
            if tokens_per_block <= 0:
                return f"{cache_name} tokens_per_block must be positive, got {tokens_per_block}"
    except ValueError as exc:
        return str(exc)
    return None


@triton.jit
def _build_prefill_mixed_token_metadata_kernel(
    seq_len_ptr,
    input_pos_ptr,
    cu_seqlen_ptr,
    cu_num_pages_ptr,
    token_input_pos_ptr,
    token_cu_num_pages_ptr,
    num_seq: tl.constexpr,
):
    token_idx = tl.program_id(0)
    seq_idx = tl.full((), 0, dtype=tl.int64)
    seq_start = tl.full((), 0, dtype=tl.int64)

    for idx in range(num_seq):
        start = tl.load(cu_seqlen_ptr + idx).to(tl.int64)
        end = tl.load(cu_seqlen_ptr + idx + 1).to(tl.int64)
        in_seq = (token_idx >= start) & (token_idx < end)
        seq_idx = tl.where(in_seq, idx, seq_idx)
        seq_start = tl.where(in_seq, start, seq_start)

    token_offset = token_idx - seq_start
    input_pos = tl.load(input_pos_ptr + seq_idx).to(tl.int64) + token_offset
    page_table_start = tl.load(cu_num_pages_ptr + seq_idx).to(tl.int64)
    tl.store(token_input_pos_ptr + token_idx, input_pos)
    tl.store(token_cu_num_pages_ptr + token_idx, page_table_start)


@triton.jit
def _update_ratio4_decode_caches_kernel(
    kv_ptr,
    compressor_kv_ptr,
    compressor_gate_ptr,
    input_pos_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    swa_cache_ptr,
    compressor_kv_cache_ptr,
    compressor_gate_cache_ptr,
    kv_stride_batch: tl.constexpr,
    kv_stride_dim: tl.constexpr,
    compressor_kv_stride_batch: tl.constexpr,
    compressor_kv_stride_dim: tl.constexpr,
    compressor_gate_stride_batch: tl.constexpr,
    compressor_gate_stride_dim: tl.constexpr,
    swa_cache_page_stride: tl.constexpr,
    swa_cache_token_stride: tl.constexpr,
    compressor_cache_page_stride: tl.constexpr,
    compressor_cache_token_stride: tl.constexpr,
    swa_tokens_per_block: tl.constexpr,
    compressor_tokens_per_block: tl.constexpr,
    head_dim: tl.constexpr,
    state_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dim_offsets = tl.arange(0, block_dim)
    head_mask = dim_offsets < head_dim
    state_mask = dim_offsets < state_dim
    input_pos = tl.load(input_pos_ptr + token_idx)
    page_table_start = tl.load(cu_num_pages_ptr + token_idx)

    swa_page_ordinal = input_pos // swa_tokens_per_block
    swa_token_offset = input_pos - swa_page_ordinal * swa_tokens_per_block
    swa_page_idx = tl.load(cache_loc_ptr + page_table_start + swa_page_ordinal).to(tl.int64)
    swa_offsets = swa_page_idx * swa_cache_page_stride + swa_token_offset * swa_cache_token_stride
    kv_offsets = token_idx * kv_stride_batch + dim_offsets * kv_stride_dim
    tl.store(
        swa_cache_ptr + swa_offsets + dim_offsets,
        tl.load(kv_ptr + kv_offsets, mask=head_mask),
        mask=head_mask,
    )

    compressor_page_ordinal = input_pos // compressor_tokens_per_block
    compressor_token_offset = input_pos - compressor_page_ordinal * compressor_tokens_per_block
    compressor_page_idx = tl.load(cache_loc_ptr + page_table_start + compressor_page_ordinal).to(
        tl.int64
    )
    compressor_offsets = (
        compressor_page_idx * compressor_cache_page_stride
        + compressor_token_offset * compressor_cache_token_stride
        + dim_offsets
    )
    compressor_kv_offsets = (
        token_idx * compressor_kv_stride_batch + dim_offsets * compressor_kv_stride_dim
    )
    compressor_gate_offsets = (
        token_idx * compressor_gate_stride_batch + dim_offsets * compressor_gate_stride_dim
    )
    tl.store(
        compressor_kv_cache_ptr + compressor_offsets,
        tl.load(compressor_kv_ptr + compressor_kv_offsets, mask=state_mask),
        mask=state_mask,
    )
    tl.store(
        compressor_gate_cache_ptr + compressor_offsets,
        tl.load(compressor_gate_ptr + compressor_gate_offsets, mask=state_mask),
        mask=state_mask,
    )


@triton.jit
def _emit_ratio4_mhc_rows_kernel(
    compressor_kv_cache_ptr,
    compressor_gate_cache_ptr,
    compressor_ape_ptr,
    compressor_norm_weight_ptr,
    freqs_real_ptr,
    input_pos_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    mhc_cache_ptr,
    compressor_cache_page_stride: tl.constexpr,
    compressor_cache_token_stride: tl.constexpr,
    mhc_cache_page_stride: tl.constexpr,
    mhc_cache_token_stride: tl.constexpr,
    compressor_tokens_per_block: tl.constexpr,
    mhc_tokens_per_block: tl.constexpr,
    compressor_ape_stride_row: tl.constexpr,
    compressor_ape_stride_dim: tl.constexpr,
    compressor_norm_weight_stride: tl.constexpr,
    freqs_stride_pos: tl.constexpr,
    freqs_stride_pair: tl.constexpr,
    freqs_stride_component: tl.constexpr,
    max_compressed_len: tl.constexpr,
    eps: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    input_pos = tl.load(input_pos_ptr + token_idx)
    row_idx = input_pos // 4
    emits_row = ((input_pos + 1) % 4 == 0) & (row_idx < max_compressed_len)
    anchor = row_idx * 4
    page_table_start = tl.load(cu_num_pages_ptr + token_idx)

    m_i = tl.full([block_dim], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([block_dim], dtype=tl.float32)
    acc = tl.zeros([block_dim], dtype=tl.float32)
    paired_m_i = tl.full([block_dim], float("-inf"), dtype=tl.float32)
    paired_l_i = tl.zeros([block_dim], dtype=tl.float32)
    paired_acc = tl.zeros([block_dim], dtype=tl.float32)

    rope_start = head_dim - rope_dim
    rope_offsets = dim_offsets - rope_start
    is_rope_dim = dim_offsets >= rope_start
    is_odd_rope_dim = (rope_offsets % 2) == 1
    pair_offsets = (rope_offsets // 2) * 2 + tl.where(is_odd_rope_dim, 0, 1)
    paired_dim_offsets = tl.where(is_rope_dim, rope_start + pair_offsets, dim_offsets)

    for ratio_offset in range(4):
        position = anchor - 4 + ratio_offset
        valid_position = emits_row & (position >= 0)
        safe_position = tl.maximum(position, 0)
        page_ordinal = safe_position // compressor_tokens_per_block
        token_offset = safe_position - page_ordinal * compressor_tokens_per_block
        page_idx = tl.load(
            cache_loc_ptr + page_table_start + page_ordinal,
            mask=valid_position,
            other=0,
        ).to(tl.int64)
        cache_base = (
            page_idx * compressor_cache_page_stride + token_offset * compressor_cache_token_stride
        )
        ape_base = ratio_offset * compressor_ape_stride_row

        valid_vec = valid_position & dim_mask
        kv_vec = tl.load(
            compressor_kv_cache_ptr + cache_base + dim_offsets,
            mask=valid_vec,
            other=0.0,
        ).to(tl.float32)
        gate_vec = tl.load(
            compressor_gate_cache_ptr + cache_base + dim_offsets,
            mask=valid_vec,
            other=0.0,
        ).to(tl.float32)
        gate_vec += tl.load(
            compressor_ape_ptr + ape_base + dim_offsets * compressor_ape_stride_dim,
            mask=valid_vec,
            other=0.0,
        ).to(tl.float32)

        safe_m = tl.where(valid_vec, m_i, 0.0)
        safe_gate = tl.where(valid_vec, gate_vec, 0.0)
        m_next_candidate = tl.maximum(safe_m, safe_gate)
        alpha = tl.where(valid_vec, tl.exp(safe_m - m_next_candidate), 1.0)
        beta = tl.where(valid_vec, tl.exp(safe_gate - m_next_candidate), 0.0)
        acc = acc * alpha + kv_vec * beta
        l_i = l_i * alpha + beta
        m_i = tl.where(valid_vec, m_next_candidate, m_i)

        paired_valid_vec = valid_position & dim_mask
        paired_kv_vec = tl.load(
            compressor_kv_cache_ptr + cache_base + paired_dim_offsets,
            mask=paired_valid_vec,
            other=0.0,
        ).to(tl.float32)
        paired_gate_vec = tl.load(
            compressor_gate_cache_ptr + cache_base + paired_dim_offsets,
            mask=paired_valid_vec,
            other=0.0,
        ).to(tl.float32)
        paired_gate_vec += tl.load(
            compressor_ape_ptr + ape_base + paired_dim_offsets * compressor_ape_stride_dim,
            mask=paired_valid_vec,
            other=0.0,
        ).to(tl.float32)

        safe_paired_m = tl.where(paired_valid_vec, paired_m_i, 0.0)
        safe_paired_gate = tl.where(paired_valid_vec, paired_gate_vec, 0.0)
        paired_m_next_candidate = tl.maximum(safe_paired_m, safe_paired_gate)
        paired_alpha = tl.where(
            paired_valid_vec,
            tl.exp(safe_paired_m - paired_m_next_candidate),
            1.0,
        )
        paired_beta = tl.where(
            paired_valid_vec,
            tl.exp(safe_paired_gate - paired_m_next_candidate),
            0.0,
        )
        paired_acc = paired_acc * paired_alpha + paired_kv_vec * paired_beta
        paired_l_i = paired_l_i * paired_alpha + paired_beta
        paired_m_i = tl.where(paired_valid_vec, paired_m_next_candidate, paired_m_i)

    for ratio_offset in range(4):
        position = anchor + ratio_offset
        page_ordinal = position // compressor_tokens_per_block
        token_offset = position - page_ordinal * compressor_tokens_per_block
        page_idx = tl.load(
            cache_loc_ptr + page_table_start + page_ordinal,
            mask=emits_row,
            other=0,
        ).to(tl.int64)
        cache_base = (
            page_idx * compressor_cache_page_stride + token_offset * compressor_cache_token_stride
        )
        state_base = head_dim
        ape_base = ratio_offset * compressor_ape_stride_row + head_dim * compressor_ape_stride_dim

        valid_vec = emits_row & dim_mask
        kv_vec = tl.load(
            compressor_kv_cache_ptr + cache_base + state_base + dim_offsets,
            mask=valid_vec,
            other=0.0,
        ).to(tl.float32)
        gate_vec = tl.load(
            compressor_gate_cache_ptr + cache_base + state_base + dim_offsets,
            mask=valid_vec,
            other=0.0,
        ).to(tl.float32)
        gate_vec += tl.load(
            compressor_ape_ptr + ape_base + dim_offsets * compressor_ape_stride_dim,
            mask=valid_vec,
            other=0.0,
        ).to(tl.float32)

        safe_m = tl.where(valid_vec, m_i, 0.0)
        safe_gate = tl.where(valid_vec, gate_vec, 0.0)
        m_next_candidate = tl.maximum(safe_m, safe_gate)
        alpha = tl.where(valid_vec, tl.exp(safe_m - m_next_candidate), 1.0)
        beta = tl.where(valid_vec, tl.exp(safe_gate - m_next_candidate), 0.0)
        acc = acc * alpha + kv_vec * beta
        l_i = l_i * alpha + beta
        m_i = tl.where(valid_vec, m_next_candidate, m_i)

        paired_kv_vec = tl.load(
            compressor_kv_cache_ptr + cache_base + state_base + paired_dim_offsets,
            mask=valid_vec,
            other=0.0,
        ).to(tl.float32)
        paired_gate_vec = tl.load(
            compressor_gate_cache_ptr + cache_base + state_base + paired_dim_offsets,
            mask=valid_vec,
            other=0.0,
        ).to(tl.float32)
        paired_gate_vec += tl.load(
            compressor_ape_ptr + ape_base + paired_dim_offsets * compressor_ape_stride_dim,
            mask=valid_vec,
            other=0.0,
        ).to(tl.float32)

        safe_paired_m = tl.where(valid_vec, paired_m_i, 0.0)
        safe_paired_gate = tl.where(valid_vec, paired_gate_vec, 0.0)
        paired_m_next_candidate = tl.maximum(safe_paired_m, safe_paired_gate)
        paired_alpha = tl.where(valid_vec, tl.exp(safe_paired_m - paired_m_next_candidate), 1.0)
        paired_beta = tl.where(valid_vec, tl.exp(safe_paired_gate - paired_m_next_candidate), 0.0)
        paired_acc = paired_acc * paired_alpha + paired_kv_vec * paired_beta
        paired_l_i = paired_l_i * paired_alpha + paired_beta
        paired_m_i = tl.where(valid_vec, paired_m_next_candidate, paired_m_i)

    pooled = acc / tl.maximum(l_i, 1.0e-20)
    paired_pooled = paired_acc / tl.maximum(paired_l_i, 1.0e-20)
    sum_sq = tl.sum(pooled * pooled, axis=0)
    norm_scale = tl.rsqrt(sum_sq / head_dim + eps)
    norm_weight = tl.load(
        compressor_norm_weight_ptr + dim_offsets * compressor_norm_weight_stride,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    paired_norm_weight = tl.load(
        compressor_norm_weight_ptr + paired_dim_offsets * compressor_norm_weight_stride,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    normed = pooled * norm_scale * norm_weight
    paired_normed = paired_pooled * norm_scale * paired_norm_weight

    safe_pair_idx = tl.where(is_rope_dim, rope_offsets // 2, 0)
    freqs_base = anchor * freqs_stride_pos + safe_pair_idx * freqs_stride_pair
    cos = tl.load(
        freqs_real_ptr + freqs_base,
        mask=dim_mask & is_rope_dim & emits_row,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        freqs_real_ptr + freqs_base + freqs_stride_component,
        mask=dim_mask & is_rope_dim & emits_row,
        other=0.0,
    ).to(tl.float32)
    rope_value = tl.where(
        is_odd_rope_dim,
        paired_normed * sin + normed * cos,
        normed * cos - paired_normed * sin,
    )
    output = tl.where(is_rope_dim, rope_value, normed)

    mhc_page_ordinal = row_idx // mhc_tokens_per_block
    mhc_token_offset = row_idx - mhc_page_ordinal * mhc_tokens_per_block
    mhc_page_idx = tl.load(
        cache_loc_ptr + page_table_start + mhc_page_ordinal,
        mask=emits_row,
        other=0,
    ).to(tl.int64)
    mhc_offsets = mhc_page_idx * mhc_cache_page_stride + mhc_token_offset * mhc_cache_token_stride
    tl.store(mhc_cache_ptr + mhc_offsets + dim_offsets, output, mask=dim_mask & emits_row)


@triton.jit
def _ratio4_selected_attention_kernel(
    q_ptr,
    attn_sink_ptr,
    topk_idxs_ptr,
    input_pos_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    swa_cache_ptr,
    mhc_cache_ptr,
    out_ptr,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    topk_stride_batch: tl.constexpr,
    topk_stride_topk: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    swa_cache_page_stride: tl.constexpr,
    swa_cache_token_stride: tl.constexpr,
    mhc_cache_page_stride: tl.constexpr,
    mhc_cache_token_stride: tl.constexpr,
    swa_tokens_per_block: tl.constexpr,
    mhc_tokens_per_block: tl.constexpr,
    active_tokens: tl.constexpr,
    softmax_scale: tl.constexpr,
    max_compressed_len: tl.constexpr,
    topk_width: tl.constexpr,
    head_dim: tl.constexpr,
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
    input_pos = tl.load(input_pos_ptr + token_idx, mask=token_is_active, other=0)
    source_seq_len = input_pos + 1
    page_table_start = tl.load(cu_num_pages_ptr + token_idx, mask=token_is_active, other=0)
    q_offsets = token_idx * q_stride_batch + head_idx * q_stride_head + dim_offsets * q_stride_dim
    q_vec = tl.load(q_ptr + q_offsets, mask=dim_mask & token_is_active, other=0.0).to(tl.float32)

    acc = tl.zeros([block_dim], dtype=tl.float32)
    m_i = tl.load(attn_sink_ptr + head_idx, mask=token_is_active, other=0.0).to(tl.float32)
    l_i = tl.full((), 1.0, dtype=tl.float32)
    compressed_len = tl.minimum(source_seq_len // 4, max_compressed_len)

    for topk_offset in range(topk_width):
        selected = tl.load(
            topk_idxs_ptr + token_idx * topk_stride_batch + topk_offset * topk_stride_topk,
            mask=token_is_active,
            other=-1,
        )
        selected_is_valid = token_is_active & (selected >= 0)
        selected_is_local = selected_is_valid & (selected < source_seq_len)
        compressed_row = selected - source_seq_len
        selected_is_compressed = (
            selected_is_valid & (selected >= source_seq_len) & (compressed_row < compressed_len)
        )

        local_page_ordinal = selected // swa_tokens_per_block
        local_token_offset = selected - local_page_ordinal * swa_tokens_per_block
        local_page_idx = tl.load(
            cache_loc_ptr + page_table_start + local_page_ordinal,
            mask=selected_is_local,
            other=0,
        ).to(tl.int64)
        local_offsets = (
            local_page_idx * swa_cache_page_stride + local_token_offset * swa_cache_token_stride
        )
        local_vec = tl.load(
            swa_cache_ptr + local_offsets + dim_offsets,
            mask=dim_mask & selected_is_local,
            other=0.0,
        ).to(tl.float32)

        compressed_page_ordinal = compressed_row // mhc_tokens_per_block
        compressed_token_offset = compressed_row - compressed_page_ordinal * mhc_tokens_per_block
        compressed_page_idx = tl.load(
            cache_loc_ptr + page_table_start + compressed_page_ordinal,
            mask=selected_is_compressed,
            other=0,
        ).to(tl.int64)
        compressed_offsets = (
            compressed_page_idx * mhc_cache_page_stride
            + compressed_token_offset * mhc_cache_token_stride
        )
        compressed_vec = tl.load(
            mhc_cache_ptr + compressed_offsets + dim_offsets,
            mask=dim_mask & selected_is_compressed,
            other=0.0,
        ).to(tl.float32)

        valid_entry = selected_is_local | selected_is_compressed
        kv_vec = tl.where(selected_is_local, local_vec, compressed_vec)
        logit = tl.sum(q_vec * kv_vec, axis=0) * softmax_scale
        m_next_candidate = tl.maximum(m_i, logit)
        alpha = tl.where(valid_entry, tl.exp(m_i - m_next_candidate), 1.0)
        beta = tl.where(valid_entry, tl.exp(logit - m_next_candidate), 0.0)
        acc = acc * alpha + kv_vec * beta
        l_i = l_i * alpha + beta
        m_i = tl.where(valid_entry, m_next_candidate, m_i)

    output = tl.where(token_is_active, acc / l_i, tl.zeros([block_dim], dtype=tl.float32))
    tl.store(out_ptr + out_offsets, output, mask=dim_mask)


@triton.jit
def _ratio4_prefill_mixed_selected_attention_kernel(
    q_ptr,
    attn_sink_ptr,
    topk_idxs_ptr,
    seq_len_ptr,
    input_pos_ptr,
    cu_seqlen_ptr,
    token_input_pos_ptr,
    token_cu_num_pages_ptr,
    cache_loc_ptr,
    swa_cache_ptr,
    mhc_cache_ptr,
    out_ptr,
    q_stride_token: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    topk_stride_token: tl.constexpr,
    topk_stride_topk: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    swa_cache_page_stride: tl.constexpr,
    swa_cache_token_stride: tl.constexpr,
    mhc_cache_page_stride: tl.constexpr,
    mhc_cache_token_stride: tl.constexpr,
    swa_tokens_per_block: tl.constexpr,
    mhc_tokens_per_block: tl.constexpr,
    num_seq: tl.constexpr,
    active_tokens: tl.constexpr,
    softmax_scale: tl.constexpr,
    max_compressed_len: tl.constexpr,
    topk_width: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    token_is_active = token_idx < active_tokens

    seq_idx = tl.full((), 0, dtype=tl.int64)
    for idx in range(num_seq):
        start = tl.load(cu_seqlen_ptr + idx).to(tl.int64)
        end = tl.load(cu_seqlen_ptr + idx + 1).to(tl.int64)
        in_seq = (token_idx >= start) & (token_idx < end)
        seq_idx = tl.where(in_seq, idx, seq_idx)

    input_pos = tl.load(token_input_pos_ptr + token_idx, mask=token_is_active, other=0)
    seq_len = tl.load(seq_len_ptr + seq_idx, mask=token_is_active, other=0)
    compressed_offset = tl.maximum(seq_len, input_pos + 1)
    compressed_len = tl.minimum((input_pos + 1) // 4, max_compressed_len)
    page_table_start = tl.load(token_cu_num_pages_ptr + token_idx, mask=token_is_active, other=0)

    q_offsets = token_idx * q_stride_token + head_idx * q_stride_head + dim_offsets * q_stride_dim
    q_vec = tl.load(q_ptr + q_offsets, mask=dim_mask & token_is_active, other=0.0).to(tl.float32)
    out_offsets = (
        token_idx * out_stride_token + head_idx * out_stride_head + dim_offsets * out_stride_dim
    )

    acc = tl.zeros([block_dim], dtype=tl.float32)
    m_i = tl.load(attn_sink_ptr + head_idx, mask=token_is_active, other=0.0).to(tl.float32)
    l_i = tl.full((), 1.0, dtype=tl.float32)

    for topk_offset in range(topk_width):
        selected = tl.load(
            topk_idxs_ptr + token_idx * topk_stride_token + topk_offset * topk_stride_topk,
            mask=token_is_active,
            other=-1,
        ).to(tl.int64)
        selected_is_valid = token_is_active & (selected >= 0)
        selected_is_local = selected_is_valid & (selected <= input_pos)
        compressed_row = selected - compressed_offset
        selected_is_compressed = (
            selected_is_valid & (selected >= compressed_offset) & (compressed_row < compressed_len)
        )

        local_page_ordinal = selected // swa_tokens_per_block
        local_token_offset = selected - local_page_ordinal * swa_tokens_per_block
        local_page_idx = tl.load(
            cache_loc_ptr + page_table_start + local_page_ordinal,
            mask=selected_is_local,
            other=0,
        ).to(tl.int64)
        local_offsets = (
            local_page_idx * swa_cache_page_stride + local_token_offset * swa_cache_token_stride
        )
        local_vec = tl.load(
            swa_cache_ptr + local_offsets + dim_offsets,
            mask=dim_mask & selected_is_local,
            other=0.0,
        ).to(tl.float32)

        compressed_page_ordinal = compressed_row // mhc_tokens_per_block
        compressed_token_offset = compressed_row - compressed_page_ordinal * mhc_tokens_per_block
        compressed_page_idx = tl.load(
            cache_loc_ptr + page_table_start + compressed_page_ordinal,
            mask=selected_is_compressed,
            other=0,
        ).to(tl.int64)
        compressed_offsets = (
            compressed_page_idx * mhc_cache_page_stride
            + compressed_token_offset * mhc_cache_token_stride
        )
        compressed_vec = tl.load(
            mhc_cache_ptr + compressed_offsets + dim_offsets,
            mask=dim_mask & selected_is_compressed,
            other=0.0,
        ).to(tl.float32)

        valid_entry = selected_is_local | selected_is_compressed
        kv_vec = tl.where(selected_is_local, local_vec, compressed_vec)
        logit = tl.sum(q_vec * kv_vec, axis=0) * softmax_scale
        m_next_candidate = tl.maximum(m_i, logit)
        alpha = tl.where(valid_entry, tl.exp(m_i - m_next_candidate), 1.0)
        beta = tl.where(valid_entry, tl.exp(logit - m_next_candidate), 0.0)
        acc = acc * alpha + kv_vec * beta
        l_i = l_i * alpha + beta
        m_i = tl.where(valid_entry, m_next_candidate, m_i)

    output = tl.where(token_is_active, acc / l_i, tl.zeros([block_dim], dtype=tl.float32))
    tl.store(out_ptr + out_offsets, output, mask=dim_mask)


@triton.jit
def _update_ratio128_decode_caches_kernel(
    kv_ptr,
    compressor_kv_ptr,
    compressor_gate_ptr,
    input_pos_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    swa_cache_ptr,
    compressor_kv_cache_ptr,
    compressor_gate_cache_ptr,
    kv_stride_batch: tl.constexpr,
    kv_stride_dim: tl.constexpr,
    compressor_kv_stride_batch: tl.constexpr,
    compressor_kv_stride_dim: tl.constexpr,
    compressor_gate_stride_batch: tl.constexpr,
    compressor_gate_stride_dim: tl.constexpr,
    swa_cache_page_stride: tl.constexpr,
    swa_cache_token_stride: tl.constexpr,
    compressor_cache_page_stride: tl.constexpr,
    compressor_cache_token_stride: tl.constexpr,
    swa_tokens_per_block: tl.constexpr,
    compressor_tokens_per_block: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    input_pos = tl.load(input_pos_ptr + token_idx)

    swa_page_ordinal = input_pos // swa_tokens_per_block
    swa_token_offset = input_pos - swa_page_ordinal * swa_tokens_per_block
    page_table_start = tl.load(cu_num_pages_ptr + token_idx)
    swa_page_idx = tl.load(cache_loc_ptr + page_table_start + swa_page_ordinal).to(tl.int64)
    swa_offsets = swa_page_idx * swa_cache_page_stride + swa_token_offset * swa_cache_token_stride
    kv_offsets = token_idx * kv_stride_batch + dim_offsets * kv_stride_dim
    tl.store(
        swa_cache_ptr + swa_offsets + dim_offsets,
        tl.load(kv_ptr + kv_offsets, mask=dim_mask),
        mask=dim_mask,
    )

    compressor_page_ordinal = input_pos // compressor_tokens_per_block
    compressor_token_offset = input_pos - compressor_page_ordinal * compressor_tokens_per_block
    compressor_page_idx = tl.load(cache_loc_ptr + page_table_start + compressor_page_ordinal).to(
        tl.int64
    )
    compressor_offsets = (
        compressor_page_idx * compressor_cache_page_stride
        + compressor_token_offset * compressor_cache_token_stride
        + dim_offsets
    )
    compressor_kv_offsets = (
        token_idx * compressor_kv_stride_batch + dim_offsets * compressor_kv_stride_dim
    )
    compressor_gate_offsets = (
        token_idx * compressor_gate_stride_batch + dim_offsets * compressor_gate_stride_dim
    )
    tl.store(
        compressor_kv_cache_ptr + compressor_offsets,
        tl.load(compressor_kv_ptr + compressor_kv_offsets, mask=dim_mask),
        mask=dim_mask,
    )
    tl.store(
        compressor_gate_cache_ptr + compressor_offsets,
        tl.load(compressor_gate_ptr + compressor_gate_offsets, mask=dim_mask),
        mask=dim_mask,
    )


@triton.jit
def _emit_ratio128_mhc_rows_kernel(
    compressor_kv_cache_ptr,
    compressor_gate_cache_ptr,
    compressor_ape_ptr,
    compressor_norm_weight_ptr,
    freqs_real_ptr,
    input_pos_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    mhc_cache_ptr,
    compressor_cache_page_stride: tl.constexpr,
    compressor_cache_token_stride: tl.constexpr,
    mhc_cache_page_stride: tl.constexpr,
    mhc_cache_token_stride: tl.constexpr,
    compressor_tokens_per_block: tl.constexpr,
    mhc_tokens_per_block: tl.constexpr,
    compressor_ape_stride_row: tl.constexpr,
    compressor_ape_stride_dim: tl.constexpr,
    compressor_norm_weight_stride: tl.constexpr,
    freqs_stride_pos: tl.constexpr,
    freqs_stride_pair: tl.constexpr,
    freqs_stride_component: tl.constexpr,
    max_compressed_len: tl.constexpr,
    eps: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    input_pos = tl.load(input_pos_ptr + token_idx)
    row_idx = input_pos // 128
    emits_row = ((input_pos + 1) % 128 == 0) & (row_idx < max_compressed_len)
    anchor = row_idx * 128
    page_table_start = tl.load(cu_num_pages_ptr + token_idx)

    m_i = tl.full([block_dim], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([block_dim], dtype=tl.float32)
    acc = tl.zeros([block_dim], dtype=tl.float32)
    paired_m_i = tl.full([block_dim], float("-inf"), dtype=tl.float32)
    paired_l_i = tl.zeros([block_dim], dtype=tl.float32)
    paired_acc = tl.zeros([block_dim], dtype=tl.float32)

    rope_start = head_dim - rope_dim
    rope_offsets = dim_offsets - rope_start
    is_rope_dim = dim_offsets >= rope_start
    is_odd_rope_dim = (rope_offsets % 2) == 1
    pair_offsets = (rope_offsets // 2) * 2 + tl.where(is_odd_rope_dim, 0, 1)
    paired_dim_offsets = tl.where(is_rope_dim, rope_start + pair_offsets, dim_offsets)

    for ratio_offset in range(128):
        position = anchor + ratio_offset
        page_ordinal = position // compressor_tokens_per_block
        token_offset = position - page_ordinal * compressor_tokens_per_block
        page_idx = tl.load(
            cache_loc_ptr + page_table_start + page_ordinal,
            mask=emits_row,
            other=0,
        ).to(tl.int64)
        cache_base = (
            page_idx * compressor_cache_page_stride + token_offset * compressor_cache_token_stride
        )
        ape_base = ratio_offset * compressor_ape_stride_row

        kv_vec = tl.load(
            compressor_kv_cache_ptr + cache_base + dim_offsets,
            mask=dim_mask & emits_row,
            other=0.0,
        ).to(tl.float32)
        gate_vec = tl.load(
            compressor_gate_cache_ptr + cache_base + dim_offsets,
            mask=dim_mask & emits_row,
            other=float("-inf"),
        ).to(tl.float32)
        gate_vec += tl.load(
            compressor_ape_ptr + ape_base + dim_offsets * compressor_ape_stride_dim,
            mask=dim_mask & emits_row,
            other=0.0,
        ).to(tl.float32)

        m_next = tl.maximum(m_i, gate_vec)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(gate_vec - m_next)
        acc = acc * alpha + kv_vec * beta
        l_i = l_i * alpha + beta
        m_i = m_next

        paired_kv_vec = tl.load(
            compressor_kv_cache_ptr + cache_base + paired_dim_offsets,
            mask=dim_mask & emits_row,
            other=0.0,
        ).to(tl.float32)
        paired_gate_vec = tl.load(
            compressor_gate_cache_ptr + cache_base + paired_dim_offsets,
            mask=dim_mask & emits_row,
            other=float("-inf"),
        ).to(tl.float32)
        paired_gate_vec += tl.load(
            compressor_ape_ptr + ape_base + paired_dim_offsets * compressor_ape_stride_dim,
            mask=dim_mask & emits_row,
            other=0.0,
        ).to(tl.float32)

        paired_m_next = tl.maximum(paired_m_i, paired_gate_vec)
        paired_alpha = tl.exp(paired_m_i - paired_m_next)
        paired_beta = tl.exp(paired_gate_vec - paired_m_next)
        paired_acc = paired_acc * paired_alpha + paired_kv_vec * paired_beta
        paired_l_i = paired_l_i * paired_alpha + paired_beta
        paired_m_i = paired_m_next

    pooled = acc / tl.maximum(l_i, 1.0e-20)
    paired_pooled = paired_acc / tl.maximum(paired_l_i, 1.0e-20)
    sum_sq = tl.sum(pooled * pooled, axis=0)
    norm_scale = tl.rsqrt(sum_sq / head_dim + eps)
    norm_weight = tl.load(
        compressor_norm_weight_ptr + dim_offsets * compressor_norm_weight_stride,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    paired_norm_weight = tl.load(
        compressor_norm_weight_ptr + paired_dim_offsets * compressor_norm_weight_stride,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    normed = pooled * norm_scale * norm_weight
    paired_normed = paired_pooled * norm_scale * paired_norm_weight

    safe_pair_idx = tl.where(is_rope_dim, rope_offsets // 2, 0)
    freqs_base = anchor * freqs_stride_pos + safe_pair_idx * freqs_stride_pair
    cos = tl.load(
        freqs_real_ptr + freqs_base,
        mask=dim_mask & is_rope_dim & emits_row,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        freqs_real_ptr + freqs_base + freqs_stride_component,
        mask=dim_mask & is_rope_dim & emits_row,
        other=0.0,
    ).to(tl.float32)
    rope_value = tl.where(
        is_odd_rope_dim,
        paired_normed * sin + normed * cos,
        normed * cos - paired_normed * sin,
    )
    output = tl.where(is_rope_dim, rope_value, normed)

    mhc_page_ordinal = row_idx // mhc_tokens_per_block
    mhc_token_offset = row_idx - mhc_page_ordinal * mhc_tokens_per_block
    mhc_page_idx = tl.load(
        cache_loc_ptr + page_table_start + mhc_page_ordinal,
        mask=emits_row,
        other=0,
    ).to(tl.int64)
    mhc_offsets = mhc_page_idx * mhc_cache_page_stride + mhc_token_offset * mhc_cache_token_stride
    tl.store(mhc_cache_ptr + mhc_offsets + dim_offsets, output, mask=dim_mask & emits_row)


@triton.jit
def _ratio128_compressed_attention_kernel(
    q_ptr,
    attn_sink_ptr,
    input_pos_ptr,
    cache_loc_ptr,
    cu_num_pages_ptr,
    swa_cache_ptr,
    mhc_cache_ptr,
    out_ptr,
    q_stride_batch: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    out_stride_batch: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    swa_cache_page_stride: tl.constexpr,
    swa_cache_token_stride: tl.constexpr,
    mhc_cache_page_stride: tl.constexpr,
    mhc_cache_token_stride: tl.constexpr,
    swa_tokens_per_block: tl.constexpr,
    mhc_tokens_per_block: tl.constexpr,
    active_tokens: tl.constexpr,
    softmax_scale: tl.constexpr,
    window_size: tl.constexpr,
    max_compressed_len: tl.constexpr,
    head_dim: tl.constexpr,
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
    input_pos = tl.load(input_pos_ptr + token_idx, mask=token_is_active, other=0)
    q_offsets = token_idx * q_stride_batch + head_idx * q_stride_head + dim_offsets * q_stride_dim
    q_vec = tl.load(q_ptr + q_offsets, mask=dim_mask & token_is_active, other=0.0).to(tl.float32)

    acc = tl.zeros([block_dim], dtype=tl.float32)
    m_i = tl.load(attn_sink_ptr + head_idx, mask=token_is_active, other=0.0).to(tl.float32)
    l_i = tl.full((), 1.0, dtype=tl.float32)
    page_table_start = tl.load(cu_num_pages_ptr + token_idx, mask=token_is_active, other=0)
    local_len = tl.minimum(input_pos + 1, window_size)
    local_start = input_pos + 1 - local_len

    for local_offset in range(window_size):
        valid_local = token_is_active & (local_offset < local_len)
        row_pos = local_start + local_offset
        page_ordinal = row_pos // swa_tokens_per_block
        token_offset = row_pos - page_ordinal * swa_tokens_per_block
        page_idx = tl.load(
            cache_loc_ptr + page_table_start + page_ordinal,
            mask=valid_local,
            other=0,
        ).to(tl.int64)
        cache_offsets = (
            page_idx * swa_cache_page_stride + token_offset * swa_cache_token_stride + dim_offsets
        )
        kv_vec = tl.load(
            swa_cache_ptr + cache_offsets,
            mask=dim_mask & valid_local,
            other=0.0,
        ).to(tl.float32)
        logit = tl.sum(q_vec * kv_vec, axis=0) * softmax_scale
        logit = tl.where(valid_local, logit, float("-inf"))
        m_next = tl.maximum(m_i, logit)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(logit - m_next)
        acc = acc * alpha + kv_vec * beta
        l_i = l_i * alpha + beta
        m_i = m_next

    compressed_len = tl.minimum((input_pos + 1) // 128, max_compressed_len)
    for compressed_offset in range(max_compressed_len):
        valid_compressed = token_is_active & (compressed_offset < compressed_len)
        page_ordinal = compressed_offset // mhc_tokens_per_block
        token_offset = compressed_offset - page_ordinal * mhc_tokens_per_block
        page_idx = tl.load(
            cache_loc_ptr + page_table_start + page_ordinal,
            mask=valid_compressed,
            other=0,
        ).to(tl.int64)
        cache_offsets = (
            page_idx * mhc_cache_page_stride + token_offset * mhc_cache_token_stride + dim_offsets
        )
        kv_vec = tl.load(
            mhc_cache_ptr + cache_offsets,
            mask=dim_mask & valid_compressed,
            other=0.0,
        ).to(tl.float32)
        logit = tl.sum(q_vec * kv_vec, axis=0) * softmax_scale
        logit = tl.where(valid_compressed, logit, float("-inf"))
        m_next = tl.maximum(m_i, logit)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(logit - m_next)
        acc = acc * alpha + kv_vec * beta
        l_i = l_i * alpha + beta
        m_i = m_next

    output = tl.where(token_is_active, acc / l_i, tl.zeros([block_dim], dtype=tl.float32))
    tl.store(out_ptr + out_offsets, output, mask=dim_mask)


def triton_deepseek_v4_ratio4_attention_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    max_compressed_len: int,
    rms_norm_eps: float,
    rope_dim: int,
    active_tokens: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run ratio-4 cached compressed DSV4 decode through Triton kernels."""
    reason = _deepseek_v4_ratio4_triton_skip_reason(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size,
        4,
        max_compressed_len,
        rope_dim,
        out,
        active_tokens,
    )
    if reason is not None:
        raise RuntimeError(reason)

    swa_tokens_per_block, swa_page_stride, swa_token_stride = _cache_layout_for_triton(swa_cache)
    mhc_tokens_per_block, mhc_page_stride, mhc_token_stride = _cache_layout_for_triton(mhc_cache)
    compressor_tokens_per_block, compressor_page_stride, compressor_token_stride = (
        _cache_layout_for_triton(compressor_kv_cache)
    )
    gate_tokens_per_block, gate_page_stride, gate_token_stride = _cache_layout_for_triton(
        compressor_gate_cache
    )
    if gate_tokens_per_block != compressor_tokens_per_block:
        raise RuntimeError(
            "compressor_kv_cache and compressor_gate_cache must use the same tokens_per_block"
        )
    if gate_page_stride != compressor_page_stride or gate_token_stride != compressor_token_stride:
        raise RuntimeError(
            "compressor_kv_cache and compressor_gate_cache must share layout strides"
        )

    output = torch.empty_like(q) if out is None else out
    freqs_real = torch.view_as_real(freqs_cis_table)
    head_block_dim = triton.next_power_of_2(_DSV4_TRITON_HEAD_DIM)
    state_dim = 2 * _DSV4_TRITON_HEAD_DIM
    state_block_dim = triton.next_power_of_2(state_dim)

    _update_ratio4_decode_caches_kernel[(active_tokens,)](
        kv,
        compressor_kv,
        compressor_gate,
        input_pos_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        kv.stride(0),
        kv.stride(2),
        compressor_kv.stride(0),
        compressor_kv.stride(2),
        compressor_gate.stride(0),
        compressor_gate.stride(2),
        swa_page_stride,
        swa_token_stride,
        compressor_page_stride,
        compressor_token_stride,
        swa_tokens_per_block,
        compressor_tokens_per_block,
        _DSV4_TRITON_HEAD_DIM,
        state_dim,
        state_block_dim,
        num_warps=8,
    )
    _emit_ratio4_mhc_rows_kernel[(active_tokens,)](
        compressor_kv_cache,
        compressor_gate_cache,
        compressor_ape,
        compressor_norm_weight,
        freqs_real,
        input_pos_host,
        cache_loc_host,
        cu_num_pages_host,
        mhc_cache,
        compressor_page_stride,
        compressor_token_stride,
        mhc_page_stride,
        mhc_token_stride,
        compressor_tokens_per_block,
        mhc_tokens_per_block,
        compressor_ape.stride(0),
        compressor_ape.stride(1),
        compressor_norm_weight.stride(0),
        freqs_real.stride(0),
        freqs_real.stride(1),
        freqs_real.stride(2),
        max_compressed_len,
        rms_norm_eps,
        _DSV4_TRITON_HEAD_DIM,
        rope_dim,
        head_block_dim,
        num_warps=8,
    )
    _ratio4_selected_attention_kernel[(q.shape[0], _DSV4_TRITON_LOCAL_NUM_HEADS)](
        q,
        attn_sink,
        topk_idxs,
        input_pos_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        output,
        q.stride(0),
        q.stride(2),
        q.stride(3),
        topk_idxs.stride(0),
        topk_idxs.stride(2),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        swa_page_stride,
        swa_token_stride,
        mhc_page_stride,
        mhc_token_stride,
        swa_tokens_per_block,
        mhc_tokens_per_block,
        active_tokens,
        softmax_scale,
        max_compressed_len,
        _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[4],
        _DSV4_TRITON_HEAD_DIM,
        head_block_dim,
        num_warps=8,
    )

    if out is not None:
        return out.new_empty(0)
    return output


def triton_deepseek_v4_ratio128_attention_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    max_compressed_len: int,
    rms_norm_eps: float,
    rope_dim: int,
    active_tokens: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run ratio-128 cached compressed DSV4 decode through Triton kernels."""
    reason = _deepseek_v4_ratio128_triton_skip_reason(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size,
        128,
        max_compressed_len,
        rope_dim,
        out,
        active_tokens,
    )
    if reason is not None:
        raise RuntimeError(reason)

    swa_tokens_per_block, swa_page_stride, swa_token_stride = _cache_layout_for_triton(swa_cache)
    mhc_tokens_per_block, mhc_page_stride, mhc_token_stride = _cache_layout_for_triton(mhc_cache)
    compressor_tokens_per_block, compressor_page_stride, compressor_token_stride = (
        _cache_layout_for_triton(compressor_kv_cache)
    )
    gate_tokens_per_block, gate_page_stride, gate_token_stride = _cache_layout_for_triton(
        compressor_gate_cache
    )
    if gate_tokens_per_block != compressor_tokens_per_block:
        raise RuntimeError(
            "compressor_kv_cache and compressor_gate_cache must use the same tokens_per_block"
        )
    if gate_page_stride != compressor_page_stride or gate_token_stride != compressor_token_stride:
        raise RuntimeError(
            "compressor_kv_cache and compressor_gate_cache must share layout strides"
        )

    output = torch.empty_like(q) if out is None else out
    freqs_real = torch.view_as_real(freqs_cis_table)
    block_dim = triton.next_power_of_2(_DSV4_TRITON_HEAD_DIM)

    _update_ratio128_decode_caches_kernel[(active_tokens,)](
        kv,
        compressor_kv,
        compressor_gate,
        input_pos_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        kv.stride(0),
        kv.stride(2),
        compressor_kv.stride(0),
        compressor_kv.stride(2),
        compressor_gate.stride(0),
        compressor_gate.stride(2),
        swa_page_stride,
        swa_token_stride,
        compressor_page_stride,
        compressor_token_stride,
        swa_tokens_per_block,
        compressor_tokens_per_block,
        _DSV4_TRITON_HEAD_DIM,
        block_dim,
        num_warps=8,
    )
    _emit_ratio128_mhc_rows_kernel[(active_tokens,)](
        compressor_kv_cache,
        compressor_gate_cache,
        compressor_ape,
        compressor_norm_weight,
        freqs_real,
        input_pos_host,
        cache_loc_host,
        cu_num_pages_host,
        mhc_cache,
        compressor_page_stride,
        compressor_token_stride,
        mhc_page_stride,
        mhc_token_stride,
        compressor_tokens_per_block,
        mhc_tokens_per_block,
        compressor_ape.stride(0),
        compressor_ape.stride(1),
        compressor_norm_weight.stride(0),
        freqs_real.stride(0),
        freqs_real.stride(1),
        freqs_real.stride(2),
        max_compressed_len,
        rms_norm_eps,
        _DSV4_TRITON_HEAD_DIM,
        rope_dim,
        block_dim,
        num_warps=8,
    )
    _ratio128_compressed_attention_kernel[(q.shape[0], _DSV4_TRITON_LOCAL_NUM_HEADS)](
        q,
        attn_sink,
        input_pos_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        output,
        q.stride(0),
        q.stride(2),
        q.stride(3),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        swa_page_stride,
        swa_token_stride,
        mhc_page_stride,
        mhc_token_stride,
        swa_tokens_per_block,
        mhc_tokens_per_block,
        active_tokens,
        softmax_scale,
        window_size,
        max_compressed_len,
        _DSV4_TRITON_HEAD_DIM,
        block_dim,
        num_warps=8,
    )

    if out is not None:
        return out.new_empty(0)
    return output


def triton_deepseek_v4_compressed_prefill_mixed_attention_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: int,
    rms_norm_eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run compressed cached DSV4 prefill/mixed attention through Triton kernels."""
    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode

    reason = _deepseek_v4_compressed_prefill_mixed_triton_skip_reason(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size,
        compress_ratio,
        max_compressed_len,
        rope_dim,
        out,
        num_seq,
        active_tokens,
    )
    if reason is not None:
        raise RuntimeError(reason)

    swa_tokens_per_block, swa_page_stride, swa_token_stride = _cache_layout_for_triton(swa_cache)
    mhc_tokens_per_block, mhc_page_stride, mhc_token_stride = _cache_layout_for_triton(mhc_cache)
    compressor_tokens_per_block, compressor_page_stride, compressor_token_stride = (
        _cache_layout_for_triton(compressor_kv_cache)
    )
    gate_tokens_per_block, gate_page_stride, gate_token_stride = _cache_layout_for_triton(
        compressor_gate_cache
    )
    if gate_tokens_per_block != compressor_tokens_per_block:
        raise RuntimeError(
            "compressor_kv_cache and compressor_gate_cache must use the same tokens_per_block"
        )
    if gate_page_stride != compressor_page_stride or gate_token_stride != compressor_token_stride:
        raise RuntimeError(
            "compressor_kv_cache and compressor_gate_cache must share layout strides"
        )

    flat_capacity = int(q.shape[0] * q.shape[1])
    q_flat = q.reshape(flat_capacity, q.shape[2], q.shape[3])
    kv_flat = kv.reshape(flat_capacity, kv.shape[2])
    compressor_kv_flat = compressor_kv.reshape(flat_capacity, compressor_kv.shape[2])
    compressor_gate_flat = compressor_gate.reshape(flat_capacity, compressor_gate.shape[2])
    topk_flat = topk_idxs.reshape(flat_capacity, topk_idxs.shape[2])
    output = torch.empty_like(q) if out is None else out
    output_flat = output.reshape(flat_capacity, output.shape[2], output.shape[3])
    token_input_pos = input_pos_host.new_empty((active_tokens,))
    token_cu_num_pages = cu_num_pages_host.new_empty((active_tokens,))
    freqs_real = torch.view_as_real(freqs_cis_table)
    block_dim = triton.next_power_of_2(_DSV4_TRITON_HEAD_DIM)
    state_dim = _DSV4_TRITON_COMPRESSOR_DIM_BY_RATIO[compress_ratio]
    state_block_dim = triton.next_power_of_2(state_dim)

    _build_prefill_mixed_token_metadata_kernel[(active_tokens,)](
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cu_num_pages_host,
        token_input_pos,
        token_cu_num_pages,
        num_seq,
        num_warps=1,
    )

    if compress_ratio == 4:
        _update_ratio4_decode_caches_kernel[(active_tokens,)](
            kv_flat,
            compressor_kv_flat,
            compressor_gate_flat,
            token_input_pos,
            cache_loc_host,
            token_cu_num_pages,
            swa_cache,
            compressor_kv_cache,
            compressor_gate_cache,
            kv_flat.stride(0),
            kv_flat.stride(1),
            compressor_kv_flat.stride(0),
            compressor_kv_flat.stride(1),
            compressor_gate_flat.stride(0),
            compressor_gate_flat.stride(1),
            swa_page_stride,
            swa_token_stride,
            compressor_page_stride,
            compressor_token_stride,
            swa_tokens_per_block,
            compressor_tokens_per_block,
            _DSV4_TRITON_HEAD_DIM,
            state_dim,
            state_block_dim,
            num_warps=8,
        )
        _emit_ratio4_mhc_rows_kernel[(active_tokens,)](
            compressor_kv_cache,
            compressor_gate_cache,
            compressor_ape,
            compressor_norm_weight,
            freqs_real,
            token_input_pos,
            cache_loc_host,
            token_cu_num_pages,
            mhc_cache,
            compressor_page_stride,
            compressor_token_stride,
            mhc_page_stride,
            mhc_token_stride,
            compressor_tokens_per_block,
            mhc_tokens_per_block,
            compressor_ape.stride(0),
            compressor_ape.stride(1),
            compressor_norm_weight.stride(0),
            freqs_real.stride(0),
            freqs_real.stride(1),
            freqs_real.stride(2),
            max_compressed_len,
            rms_norm_eps,
            _DSV4_TRITON_HEAD_DIM,
            rope_dim,
            block_dim,
            num_warps=8,
        )
        _ratio4_prefill_mixed_selected_attention_kernel[
            (flat_capacity, _DSV4_TRITON_LOCAL_NUM_HEADS)
        ](
            q_flat,
            attn_sink,
            topk_flat,
            seq_len_host,
            token_input_pos,
            cu_seqlen_host,
            token_input_pos,
            token_cu_num_pages,
            cache_loc_host,
            swa_cache,
            mhc_cache,
            output_flat,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            topk_flat.stride(0),
            topk_flat.stride(1),
            output_flat.stride(0),
            output_flat.stride(1),
            output_flat.stride(2),
            swa_page_stride,
            swa_token_stride,
            mhc_page_stride,
            mhc_token_stride,
            swa_tokens_per_block,
            mhc_tokens_per_block,
            num_seq,
            active_tokens,
            softmax_scale,
            max_compressed_len,
            _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[4],
            _DSV4_TRITON_HEAD_DIM,
            block_dim,
            num_warps=8,
        )
    elif compress_ratio == 128:
        _update_ratio128_decode_caches_kernel[(active_tokens,)](
            kv_flat,
            compressor_kv_flat,
            compressor_gate_flat,
            token_input_pos,
            cache_loc_host,
            token_cu_num_pages,
            swa_cache,
            compressor_kv_cache,
            compressor_gate_cache,
            kv_flat.stride(0),
            kv_flat.stride(1),
            compressor_kv_flat.stride(0),
            compressor_kv_flat.stride(1),
            compressor_gate_flat.stride(0),
            compressor_gate_flat.stride(1),
            swa_page_stride,
            swa_token_stride,
            compressor_page_stride,
            compressor_token_stride,
            swa_tokens_per_block,
            compressor_tokens_per_block,
            _DSV4_TRITON_HEAD_DIM,
            block_dim,
            num_warps=8,
        )
        _emit_ratio128_mhc_rows_kernel[(active_tokens,)](
            compressor_kv_cache,
            compressor_gate_cache,
            compressor_ape,
            compressor_norm_weight,
            freqs_real,
            token_input_pos,
            cache_loc_host,
            token_cu_num_pages,
            mhc_cache,
            compressor_page_stride,
            compressor_token_stride,
            mhc_page_stride,
            mhc_token_stride,
            compressor_tokens_per_block,
            mhc_tokens_per_block,
            compressor_ape.stride(0),
            compressor_ape.stride(1),
            compressor_norm_weight.stride(0),
            freqs_real.stride(0),
            freqs_real.stride(1),
            freqs_real.stride(2),
            max_compressed_len,
            rms_norm_eps,
            _DSV4_TRITON_HEAD_DIM,
            rope_dim,
            block_dim,
            num_warps=8,
        )
        _ratio128_compressed_attention_kernel[(flat_capacity, _DSV4_TRITON_LOCAL_NUM_HEADS)](
            q_flat,
            attn_sink,
            token_input_pos,
            cache_loc_host,
            token_cu_num_pages,
            swa_cache,
            mhc_cache,
            output_flat,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            output_flat.stride(0),
            output_flat.stride(1),
            output_flat.stride(2),
            swa_page_stride,
            swa_token_stride,
            mhc_page_stride,
            mhc_token_stride,
            swa_tokens_per_block,
            mhc_tokens_per_block,
            active_tokens,
            softmax_scale,
            window_size,
            max_compressed_len,
            _DSV4_TRITON_HEAD_DIM,
            block_dim,
            num_warps=8,
        )
    else:
        raise RuntimeError(f"unsupported Triton DSV4 compress_ratio={compress_ratio}")

    if out is not None:
        return out.new_empty(0)
    return output


def _validate_triton_deepseek_v4_sparse_attention_v2_contract(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: Optional[int],
    rms_norm_eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor],
    *,
    validate_metadata_values: bool,
) -> None:
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    _validate_rank("compressor_kv", compressor_kv, 3)
    _validate_rank("compressor_gate", compressor_gate, 3)
    _validate_rank("compressor_ape", compressor_ape, 2)
    _validate_rank("compressor_norm_weight", compressor_norm_weight, 1)
    _validate_rank("freqs_cis_table", freqs_cis_table, 2)
    _validate_rank("position_ids", position_ids, 2)
    _validate_int_metadata("batch_info_host", batch_info_host)
    _validate_int_metadata("seq_len_host", seq_len_host)
    _validate_int_metadata("input_pos_host", input_pos_host)
    _validate_int_metadata("cu_seqlen_host", cu_seqlen_host)
    _validate_int_metadata("cache_loc_host", cache_loc_host)
    _validate_int_metadata("cu_num_pages_host", cu_num_pages_host)

    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be bfloat16 for the Triton DSV4 contract, got {q.dtype}")
    if kv.dtype != torch.bfloat16:
        raise TypeError(f"kv must be bfloat16 for the Triton DSV4 contract, got {kv.dtype}")
    if attn_sink.dtype != torch.float32:
        raise TypeError(f"attn_sink must be float32, got {attn_sink.dtype}")
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_idxs must be int32 or int64, got {topk_idxs.dtype}")
    if position_ids.dtype not in (torch.int32, torch.int64, torch.int):
        raise TypeError(f"position_ids must be int32/int64, got {position_ids.dtype}")
    if not freqs_cis_table.is_complex():
        raise TypeError(f"freqs_cis_table must be complex, got {freqs_cis_table.dtype}")

    batch_size, seq_len, num_heads, head_dim = q.shape
    if num_heads != _DSV4_TRITON_LOCAL_NUM_HEADS:
        raise ValueError(
            f"q local head count must be {_DSV4_TRITON_LOCAL_NUM_HEADS}, got {num_heads}"
        )
    if head_dim != _DSV4_TRITON_HEAD_DIM:
        raise ValueError(f"q head dimension must be {_DSV4_TRITON_HEAD_DIM}, got {head_dim}")
    if kv.shape != (batch_size, seq_len, _DSV4_TRITON_HEAD_DIM):
        raise ValueError(
            f"kv must have shape {(batch_size, seq_len, _DSV4_TRITON_HEAD_DIM)}, "
            f"got {tuple(kv.shape)}"
        )
    if attn_sink.shape != (num_heads,):
        raise ValueError(f"attn_sink must have shape ({num_heads},), got {tuple(attn_sink.shape)}")
    if topk_idxs.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            f"topk_idxs prefix must be {(batch_size, seq_len)}, got {tuple(topk_idxs.shape[:2])}"
        )
    if position_ids.shape != (batch_size, seq_len):
        raise ValueError(
            f"position_ids must have shape {(batch_size, seq_len)}, got {tuple(position_ids.shape)}"
        )

    if compress_ratio not in _DSV4_TRITON_TOPK_WIDTH_BY_RATIO:
        raise ValueError(f"compress_ratio must be one of 0, 4, or 128; got {compress_ratio}")
    expected_topk_width = _DSV4_TRITON_TOPK_WIDTH_BY_RATIO[compress_ratio]
    if topk_idxs.shape[-1] != expected_topk_width:
        raise ValueError(
            f"topk_idxs width must be {expected_topk_width} for compress_ratio={compress_ratio}, "
            f"got {topk_idxs.shape[-1]}"
        )

    expected_compressor_dim = _DSV4_TRITON_COMPRESSOR_DIM_BY_RATIO[compress_ratio]
    expected_compressor_shape = (batch_size, seq_len, expected_compressor_dim)
    if compressor_kv.shape != expected_compressor_shape:
        raise ValueError(
            f"compressor_kv must have shape {expected_compressor_shape}, "
            f"got {tuple(compressor_kv.shape)}"
        )
    if compressor_gate.shape != expected_compressor_shape:
        raise ValueError(
            f"compressor_gate must have shape {expected_compressor_shape}, "
            f"got {tuple(compressor_gate.shape)}"
        )
    expected_ape_shape = (
        (0, 0) if compress_ratio == 0 else (compress_ratio, expected_compressor_dim)
    )
    if compressor_ape.shape != expected_ape_shape:
        raise ValueError(
            f"compressor_ape must have shape {expected_ape_shape}, "
            f"got {tuple(compressor_ape.shape)}"
        )
    expected_norm_shape = (0,) if compress_ratio == 0 else (_DSV4_TRITON_HEAD_DIM,)
    if compressor_norm_weight.shape != expected_norm_shape:
        raise ValueError(
            f"compressor_norm_weight must have shape {expected_norm_shape}, "
            f"got {tuple(compressor_norm_weight.shape)}"
        )

    if window_size != _DSV4_TRITON_WINDOW_SIZE:
        raise ValueError(f"window_size must be {_DSV4_TRITON_WINDOW_SIZE}, got {window_size}")
    if rope_dim != _DSV4_TRITON_ROPE_DIM:
        raise ValueError(f"rope_dim must be {_DSV4_TRITON_ROPE_DIM}, got {rope_dim}")
    if softmax_scale <= 0.0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale}")
    if rms_norm_eps <= 0.0:
        raise ValueError(f"rms_norm_eps must be positive, got {rms_norm_eps}")
    if freqs_cis_table.shape[-1] != rope_dim // 2:
        raise ValueError(
            f"freqs_cis_table last dimension must be {rope_dim // 2}, "
            f"got {freqs_cis_table.shape[-1]}"
        )
    if compress_ratio == 0:
        if max_compressed_len not in (None, 0):
            raise ValueError(
                "max_compressed_len must be None or 0 for ratio-0 DSV4 attention, "
                f"got {max_compressed_len}"
            )
    elif max_compressed_len is None or max_compressed_len <= 0:
        raise ValueError(
            f"max_compressed_len must be positive for compress_ratio={compress_ratio}, "
            f"got {max_compressed_len}"
        )

    _validate_contract_cache("swa_cache", swa_cache, _DSV4_TRITON_HEAD_DIM, torch.bfloat16)
    _validate_contract_cache("mhc_cache", mhc_cache, _DSV4_TRITON_HEAD_DIM, torch.bfloat16)
    _validate_contract_resource_placeholder("compressor_kv_cache", compressor_kv_cache)
    _validate_contract_resource_placeholder("compressor_gate_cache", compressor_gate_cache)

    if out is not None:
        if out.shape != q.shape:
            raise ValueError(f"out shape must be {tuple(q.shape)}, got {tuple(out.shape)}")
        if out.dtype != q.dtype:
            raise TypeError(f"out dtype must be {q.dtype}, got {out.dtype}")
        if out.device != q.device:
            raise ValueError(f"out must be on {q.device}, got {out.device}")

    if not validate_metadata_values:
        return

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode
    max_graph_tokens = int(batch_size * seq_len)
    if active_tokens > max_graph_tokens:
        raise ValueError(
            f"active tokens ({active_tokens}) exceed q capacity ({max_graph_tokens}); "
            "padded graph slots may be present, but active tokens must fit."
        )
    if seq_len_host.numel() < num_seq:
        raise ValueError(
            f"seq_len_host needs at least {num_seq} entries, got {seq_len_host.numel()}"
        )
    if input_pos_host.numel() < num_seq:
        raise ValueError(
            f"input_pos_host needs at least {num_seq} entries, got {input_pos_host.numel()}"
        )
    if cu_seqlen_host.numel() < num_seq + 1:
        raise ValueError(
            f"cu_seqlen_host needs at least {num_seq + 1} entries, got {cu_seqlen_host.numel()}"
        )
    if cu_num_pages_host.numel() < num_seq + 1:
        raise ValueError(
            f"cu_num_pages_host needs at least {num_seq + 1} entries, "
            f"got {cu_num_pages_host.numel()}"
        )


def _deepseek_v4_triton_cached_attention_fallback_reason(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    out: Optional[torch.Tensor],
    *,
    compressor_kv: Optional[torch.Tensor] = None,
    compressor_gate: Optional[torch.Tensor] = None,
    compressor_ape: Optional[torch.Tensor] = None,
    compressor_norm_weight: Optional[torch.Tensor] = None,
    freqs_cis_table: Optional[torch.Tensor] = None,
    mhc_cache: Optional[torch.Tensor] = None,
    compressor_kv_cache: Optional[torch.Tensor] = None,
    compressor_gate_cache: Optional[torch.Tensor] = None,
    max_compressed_len: Optional[int] = None,
    rope_dim: Optional[int] = None,
) -> tuple[Optional[str], int]:
    """Return why the Triton cached kernel is not selected, plus active token count."""
    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode

    if _force_torch_cached_reference():
        return (
            f"{_DSV4_FORCE_TORCH_REFERENCE_ENV} is set; using Torch cached reference.",
            active_tokens,
        )

    if compress_ratio == 0:
        if num_prefill != 0 or num_prefill_tokens != 0:
            return (
                _deepseek_v4_ratio0_prefill_mixed_triton_skip_reason(
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
                    compress_ratio,
                    out,
                    num_seq,
                    active_tokens,
                ),
                active_tokens,
            )
        return (
            deepseek_v4_ratio0_swa_triton_skip_reason(
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
                compress_ratio,
                out,
                num_decode,
            ),
            active_tokens,
        )

    if compress_ratio == 4:
        required_tensors = (
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            freqs_cis_table,
            mhc_cache,
            compressor_kv_cache,
            compressor_gate_cache,
        )
        if any(tensor is None for tensor in required_tensors) or rope_dim is None:
            return (
                "ratio-4 compressed Triton attention requires compressor and cache resources; "
                "using the incremental Torch cached reference.",
                active_tokens,
            )
        if num_prefill != 0 or num_prefill_tokens != 0:
            return (
                _deepseek_v4_compressed_prefill_mixed_triton_skip_reason(
                    q,
                    kv,
                    attn_sink,
                    topk_idxs,
                    compressor_kv,
                    compressor_gate,
                    compressor_ape,
                    compressor_norm_weight,
                    freqs_cis_table,
                    seq_len_host,
                    input_pos_host,
                    cu_seqlen_host,
                    cache_loc_host,
                    cu_num_pages_host,
                    swa_cache,
                    mhc_cache,
                    compressor_kv_cache,
                    compressor_gate_cache,
                    window_size,
                    compress_ratio,
                    max_compressed_len,
                    rope_dim,
                    out,
                    num_seq,
                    active_tokens,
                ),
                active_tokens,
            )
        if num_decode <= 0:
            return (f"active_tokens must be positive, got {active_tokens}", active_tokens)
        return (
            _deepseek_v4_ratio4_triton_skip_reason(
                q,
                kv,
                attn_sink,
                topk_idxs,
                compressor_kv,
                compressor_gate,
                compressor_ape,
                compressor_norm_weight,
                freqs_cis_table,
                seq_len_host,
                input_pos_host,
                cu_seqlen_host,
                cache_loc_host,
                cu_num_pages_host,
                swa_cache,
                mhc_cache,
                compressor_kv_cache,
                compressor_gate_cache,
                window_size,
                compress_ratio,
                max_compressed_len,
                rope_dim,
                out,
                num_decode,
            ),
            active_tokens,
        )

    if compress_ratio == 128:
        required_tensors = (
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            freqs_cis_table,
            mhc_cache,
            compressor_kv_cache,
            compressor_gate_cache,
        )
        if any(tensor is None for tensor in required_tensors) or rope_dim is None:
            return (
                "ratio-128 compressed Triton attention requires compressor and cache resources; "
                "using the incremental Torch cached reference.",
                active_tokens,
            )
        if num_prefill != 0 or num_prefill_tokens != 0:
            return (
                _deepseek_v4_compressed_prefill_mixed_triton_skip_reason(
                    q,
                    kv,
                    attn_sink,
                    topk_idxs,
                    compressor_kv,
                    compressor_gate,
                    compressor_ape,
                    compressor_norm_weight,
                    freqs_cis_table,
                    seq_len_host,
                    input_pos_host,
                    cu_seqlen_host,
                    cache_loc_host,
                    cu_num_pages_host,
                    swa_cache,
                    mhc_cache,
                    compressor_kv_cache,
                    compressor_gate_cache,
                    window_size,
                    compress_ratio,
                    max_compressed_len,
                    rope_dim,
                    out,
                    num_seq,
                    active_tokens,
                ),
                active_tokens,
            )
        if num_decode <= 0:
            return (f"active_tokens must be positive, got {active_tokens}", active_tokens)
        return (
            _deepseek_v4_ratio128_triton_skip_reason(
                q,
                kv,
                attn_sink,
                topk_idxs,
                compressor_kv,
                compressor_gate,
                compressor_ape,
                compressor_norm_weight,
                freqs_cis_table,
                seq_len_host,
                input_pos_host,
                cu_seqlen_host,
                cache_loc_host,
                cu_num_pages_host,
                swa_cache,
                mhc_cache,
                compressor_kv_cache,
                compressor_gate_cache,
                window_size,
                compress_ratio,
                max_compressed_len,
                rope_dim,
                out,
                num_decode,
            ),
            active_tokens,
        )

    return (
        f"compress_ratio must be one of 0, 4, or 128; got {compress_ratio}",
        active_tokens,
    )


@torch.library.custom_op(
    "auto_deploy::triton_deepseek_v4_sparse_attention_v2_with_cache",
    mutates_args=(),
)
def triton_deepseek_v4_sparse_attention_v2_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: Optional[int],
    rms_norm_eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Executable Plan 01 contract for the Triton DSV4 cached attention op.

    The production paths cover guarded ratio-0 SWA prefill/mixed/decode and
    compressed ratio-4 / ratio-128 prefill/mixed/decode with CUDA BF16 tensors
    and device page metadata. Unsupported cases intentionally keep using the
    Torch cached reference so the public op boundary remains debuggable. The
    local-rank graph contract observed for DeepSeek V4 TP=8 attention is:

    - ``q`` is BF16 ``[batch, seq_len, 8, 512]`` and ``kv`` is BF16
      ``[batch, seq_len, 512]``.
    - ``topk_idxs`` is caller-built graph/debug metadata. Its width is 128 for
      ratio-0, 640 for ratio-4, and 192 for ratio-128. Negative entries are
      masked, duplicate indices receive independent probability mass, and sink
      logits normalize softmax without contributing values.
    - ``swa_cache`` and ``mhc_cache`` are caller-owned BF16 paged resources.
      Compressor cache tensors are caller-owned placeholders for later kernel
      plans; FP8 NoPE cache resources are not consumed by this Wave 1 boundary.
    - Passing ``out=`` selects the CUDA-graph replay convention: the caller-owned
      output buffer is filled and the op returns an empty tensor sentinel.
    """
    _validate_triton_deepseek_v4_sparse_attention_v2_contract(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        max_compressed_len,
        rms_norm_eps,
        rope_dim,
        out,
        validate_metadata_values=True,
    )
    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    seq_len_triton = _triton_metadata_on_q_device(seq_len_host, q)
    input_pos_triton = _triton_metadata_on_q_device(input_pos_host, q)
    cu_seqlen_triton = _triton_metadata_on_q_device(cu_seqlen_host, q)
    cache_loc_triton = _triton_metadata_on_q_device(cache_loc_host, q)
    cu_num_pages_triton = _triton_metadata_on_q_device(cu_num_pages_host, q)
    fallback_reason, active_tokens = _deepseek_v4_triton_cached_attention_fallback_reason(
        q,
        kv,
        attn_sink,
        topk_idxs,
        batch_info_host,
        seq_len_triton,
        input_pos_triton,
        cu_seqlen_triton,
        cache_loc_triton,
        cu_num_pages_triton,
        swa_cache,
        window_size,
        compress_ratio,
        out,
        compressor_kv=compressor_kv,
        compressor_gate=compressor_gate,
        compressor_ape=compressor_ape,
        compressor_norm_weight=compressor_norm_weight,
        freqs_cis_table=freqs_cis_table,
        mhc_cache=mhc_cache,
        compressor_kv_cache=compressor_kv_cache,
        compressor_gate_cache=compressor_gate_cache,
        max_compressed_len=max_compressed_len,
        rope_dim=rope_dim,
    )
    if fallback_reason is None:
        if compress_ratio == 0:
            if num_prefill != 0 or num_prefill_tokens != 0:
                return triton_deepseek_v4_ratio0_swa_prefill_mixed_attention_with_cache(
                    q,
                    kv,
                    attn_sink,
                    topk_idxs,
                    batch_info_host,
                    seq_len_triton,
                    input_pos_triton,
                    cu_seqlen_triton,
                    cache_loc_triton,
                    cu_num_pages_triton,
                    swa_cache,
                    softmax_scale,
                    window_size,
                    out=out,
                )
            return triton_deepseek_v4_ratio0_swa_attention_with_cache(
                q,
                kv,
                attn_sink,
                topk_idxs,
                seq_len_triton,
                input_pos_triton,
                cu_seqlen_triton,
                cache_loc_triton,
                cu_num_pages_triton,
                swa_cache,
                softmax_scale,
                window_size,
                active_tokens,
                out=out,
            )
        if compress_ratio == 4:
            if num_prefill != 0 or num_prefill_tokens != 0:
                return triton_deepseek_v4_compressed_prefill_mixed_attention_with_cache(
                    q,
                    kv,
                    attn_sink,
                    topk_idxs,
                    compressor_kv,
                    compressor_gate,
                    compressor_ape,
                    compressor_norm_weight,
                    freqs_cis_table,
                    batch_info_host,
                    seq_len_triton,
                    input_pos_triton,
                    cu_seqlen_triton,
                    cache_loc_triton,
                    cu_num_pages_triton,
                    swa_cache,
                    mhc_cache,
                    compressor_kv_cache,
                    compressor_gate_cache,
                    softmax_scale,
                    window_size,
                    compress_ratio,
                    int(max_compressed_len),
                    rms_norm_eps,
                    rope_dim,
                    out=out,
                )
            return triton_deepseek_v4_ratio4_attention_with_cache(
                q,
                kv,
                attn_sink,
                topk_idxs,
                compressor_kv,
                compressor_gate,
                compressor_ape,
                compressor_norm_weight,
                freqs_cis_table,
                seq_len_triton,
                input_pos_triton,
                cu_seqlen_triton,
                cache_loc_triton,
                cu_num_pages_triton,
                swa_cache,
                mhc_cache,
                compressor_kv_cache,
                compressor_gate_cache,
                softmax_scale,
                window_size,
                int(max_compressed_len),
                rms_norm_eps,
                rope_dim,
                num_decode,
                out=out,
            )
        if compress_ratio == 128:
            if num_prefill != 0 or num_prefill_tokens != 0:
                return triton_deepseek_v4_compressed_prefill_mixed_attention_with_cache(
                    q,
                    kv,
                    attn_sink,
                    topk_idxs,
                    compressor_kv,
                    compressor_gate,
                    compressor_ape,
                    compressor_norm_weight,
                    freqs_cis_table,
                    batch_info_host,
                    seq_len_triton,
                    input_pos_triton,
                    cu_seqlen_triton,
                    cache_loc_triton,
                    cu_num_pages_triton,
                    swa_cache,
                    mhc_cache,
                    compressor_kv_cache,
                    compressor_gate_cache,
                    softmax_scale,
                    window_size,
                    compress_ratio,
                    int(max_compressed_len),
                    rms_norm_eps,
                    rope_dim,
                    out=out,
                )
            return triton_deepseek_v4_ratio128_attention_with_cache(
                q,
                kv,
                attn_sink,
                topk_idxs,
                compressor_kv,
                compressor_gate,
                compressor_ape,
                compressor_norm_weight,
                freqs_cis_table,
                seq_len_triton,
                input_pos_triton,
                cu_seqlen_triton,
                cache_loc_triton,
                cu_num_pages_triton,
                swa_cache,
                mhc_cache,
                compressor_kv_cache,
                compressor_gate_cache,
                softmax_scale,
                window_size,
                int(max_compressed_len),
                rms_norm_eps,
                rope_dim,
                num_decode,
                out=out,
            )
        raise RuntimeError(f"unsupported Triton DSV4 compress_ratio={compress_ratio}")
    _warn_torch_cached_reference_fallback(fallback_reason)
    return torch_deepseek_v4_sparse_attention_v2_with_cache(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        max_compressed_len,
        rms_norm_eps,
        rope_dim,
        out=out,
    )


@triton_deepseek_v4_sparse_attention_v2_with_cache.register_fake
def triton_deepseek_v4_sparse_attention_v2_with_cache_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: Optional[int],
    rms_norm_eps: float,
    rope_dim: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_triton_deepseek_v4_sparse_attention_v2_contract(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        max_compressed_len,
        rms_norm_eps,
        rope_dim,
        out,
        validate_metadata_values=False,
    )
    if out is not None:
        return out.new_empty(0)
    return q.new_empty(q.shape).contiguous()


@AttentionRegistry.register("deepseek_v4_sparse")
class DeepSeekV4SparseAttention(AttentionDescriptor):
    """Cached DeepSeek V4 sparse attention descriptor for the integrated DSV4 path."""

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 10

    @classmethod
    def get_source_attention_op(cls):
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        if _force_torch_cached_reference():
            return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache.default
        return torch.ops.auto_deploy.triton_deepseek_v4_sparse_attention_v2_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return [
            "batch_info_host",
            "seq_len_host",
            "input_pos_host",
            "cu_seqlen_host",
            "cache_loc_host",
            "cu_num_pages_host",
        ]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        kv_fake = source_attn_node.args[1].meta["val"]
        head_dim = int(kv_fake.shape[-1])
        dtype = cls.resolve_cache_dtype(cache_config.dtype, kv_fake.dtype)
        compressor_state_dim = head_dim * 2
        return {
            "swa_cache": KVPagedResourceHandler(
                num_kv_heads=1,
                head_dim=head_dim,
                dtype=dtype,
                kv_factor=1,
                kv_layout="NHD",
            ),
            "mhc_cache": KVPagedResourceHandler(
                num_kv_heads=1,
                head_dim=head_dim,
                dtype=dtype,
                kv_factor=1,
                kv_layout="NHD",
            ),
            "compressor_kv_cache": KVPagedResourceHandler(
                num_kv_heads=1,
                head_dim=compressor_state_dim,
                dtype=torch.float32,
                kv_factor=1,
                kv_layout="NHD",
            ),
            "compressor_gate_cache": KVPagedResourceHandler(
                num_kv_heads=1,
                head_dim=compressor_state_dim,
                dtype=torch.float32,
                kv_factor=1,
                kv_layout="NHD",
            ),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        softmax_scale, window_size, compress_ratio, max_compressed_len, rms_norm_eps, rope_dim = (
            extract_op_args(
                source_attn_node,
                "softmax_scale",
                "window_size",
                "compress_ratio",
                "max_compressed_len",
                "rms_norm_eps",
                "rope_dim",
            )
        )
        if window_size is None:
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node is missing window_size metadata."
            )
        return [
            softmax_scale,
            window_size,
            compress_ratio,
            max_compressed_len,
            rms_norm_eps,
            rope_dim,
        ]
