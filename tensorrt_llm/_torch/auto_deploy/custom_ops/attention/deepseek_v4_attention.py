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

from typing import List, Optional

import torch
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


def _validate_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}, got rank {tensor.dim()}")


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


def _gather_selected_kv(kv: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, k_select = topk_idxs.shape
    head_dim = kv.shape[-1]
    gather_topk_idxs = topk_idxs.to(torch.long).clamp(min=0)
    gather_idx = gather_topk_idxs.unsqueeze(-1).expand(batch_size, seq_len, k_select, head_dim)
    expanded_kv = kv.unsqueeze(1).expand(batch_size, seq_len, kv.shape[1], head_dim)
    return torch.gather(expanded_kv, dim=2, index=gather_idx)


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
    selected_kv = _gather_selected_kv(kv, topk_idxs)
    q_compute = q.to(compute_dtype)
    selected_kv_compute = selected_kv.to(compute_dtype)

    logits = torch.einsum("bshd,bskd->bshk", q_compute, selected_kv_compute)
    logits = logits * softmax_scale
    invalid = topk_idxs < 0
    logits = logits.masked_fill(invalid.unsqueeze(2), float("-inf"))
    sink_logits = attn_sink.to(dtype=compute_dtype).reshape(1, 1, -1, 1)
    sink_logits = sink_logits.expand(q.shape[0], q.shape[1], q.shape[2], 1)
    logits_with_sink = torch.cat([logits, sink_logits], dim=-1)

    weights_with_sink = torch.softmax(logits_with_sink, dim=-1, dtype=torch.float32)
    weights = weights_with_sink[..., :-1].to(compute_dtype)
    output = torch.einsum("bshk,bskd->bshd", weights, selected_kv_compute)
    output = output.to(q.dtype).contiguous()
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
        # KVPagedResourceHandler with NHD layout:
        # [num_pages, tokens_per_block, kv_factor, num_kv_heads, head_dim].
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

    old_completed = input_pos // compress_ratio
    new_completed = (input_pos + compressor_kv_seq.shape[0]) // compress_ratio
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
        compressed_len = (query_pos + 1) // compress_ratio
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


@AttentionRegistry.register("deepseek_v4_sparse")
class DeepSeekV4SparseAttention(AttentionDescriptor):
    """Cached DeepSeek V4 sparse attention descriptor for the SWA reference path."""

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
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache.default

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
