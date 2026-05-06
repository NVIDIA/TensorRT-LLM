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

"""DeepSeek V4 sparse attention source and cached reference ops."""

from typing import Optional

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from ..._compat import KvCacheConfig
from ...distributed import common as dist_common
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BatchInfo,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    UnpagedResourceHandler,
)

__all__ = [
    "DeepSeekV4SparseAttention",
    "torch_deepseek_v4_sparse_attention",
    "torch_deepseek_v4_sparse_attention_v2",
    "torch_deepseek_v4_sparse_attention_with_cache",
    "torch_deepseek_v4_sparse_attention_v2_with_cache",
]

_SPARSE_ATTENTION_CHUNK_TARGET_BYTES = 512 * 1024 * 1024
_SPARSE_ATTENTION_MAX_CHUNK_TOKENS = 64
_SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)


def _validate_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}, got rank {tensor.dim()}")


def _validate_compress_ratio(compress_ratio: int) -> None:
    if compress_ratio not in _SUPPORTED_COMPRESS_RATIOS:
        raise ValueError(
            "DeepSeek V4 cached sparse attention supports "
            f"compress_ratio in {_SUPPORTED_COMPRESS_RATIOS}, got {compress_ratio}"
        )


def _validate_deepseek_v4_sparse_attention_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
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


def _validate_topk_idx_bounds(topk_idxs: torch.Tensor, kv_rows: int) -> None:
    valid_topk_idxs = topk_idxs[topk_idxs >= 0]
    if valid_topk_idxs.numel() == 0:
        return

    max_topk_idx = int(valid_topk_idxs.max().item())
    if max_topk_idx >= kv_rows:
        raise ValueError(
            f"topk_idxs entries must be less than kv rows {kv_rows}, got {max_topk_idx}"
        )


def _validate_swa_cache_inputs(q: torch.Tensor, kv: torch.Tensor, swa_cache: torch.Tensor) -> None:
    _validate_rank("swa_cache", swa_cache, 3)
    if not swa_cache.is_floating_point():
        raise TypeError(f"swa_cache must be floating point, got {swa_cache.dtype}")
    if swa_cache.device != q.device:
        raise ValueError(f"swa_cache must be on {q.device}, got {swa_cache.device}")
    if swa_cache.shape[-1] != kv.shape[-1]:
        raise ValueError(
            f"swa_cache head dimension must be {kv.shape[-1]}, got {swa_cache.shape[-1]}"
        )


def _ceil_pow2_scale(amax: torch.Tensor, max_value: float, min_value: float) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(amax.clamp_min(min_value) / max_value)))


def _fake_fp8_act_quant(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(-1, dim // block_size, block_size)
    scale = _ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 448.0, 1.0e-4)
    quant = torch.clamp(grouped / scale, -448.0, 448.0).to(dtype).float()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _fake_fp4_act_quant(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(-1, dim // block_size, block_size)
    scale = _ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 6.0, 6.0 * 2.0**-126)
    normalized = torch.clamp(grouped / scale, -6.0, 6.0)
    abs_normalized = normalized.abs()
    quant_abs = torch.zeros_like(abs_normalized)
    quant_abs = torch.where(abs_normalized > 0.25, torch.full_like(quant_abs, 0.5), quant_abs)
    quant_abs = torch.where(abs_normalized > 0.75, torch.full_like(quant_abs, 1.0), quant_abs)
    quant_abs = torch.where(abs_normalized > 1.25, torch.full_like(quant_abs, 1.5), quant_abs)
    quant_abs = torch.where(abs_normalized > 1.75, torch.full_like(quant_abs, 2.0), quant_abs)
    quant_abs = torch.where(abs_normalized > 2.5, torch.full_like(quant_abs, 3.0), quant_abs)
    quant_abs = torch.where(abs_normalized > 3.5, torch.full_like(quant_abs, 4.0), quant_abs)
    quant_abs = torch.where(abs_normalized > 5.0, torch.full_like(quant_abs, 6.0), quant_abs)
    quant = quant_abs * normalized.sign()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim <= 1:
        return x
    if dim & (dim - 1):
        raise ValueError(f"Hadamard rotation requires power-of-two dimension, got {dim}.")

    out = x.reshape(-1, dim).float()
    width = 1
    while width < dim:
        out = out.reshape(-1, dim // (2 * width), 2, width)
        left = out[..., 0, :]
        right = out[..., 1, :]
        out = torch.cat((left + right, left - right), dim=-1).flatten(-2)
        width *= 2
    return (out * (dim**-0.5)).reshape_as(x).to(x.dtype)


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    compute = x.to(torch.float32)
    output = compute * torch.rsqrt(compute.square().mean(dim=-1, keepdim=True) + eps)
    if weight.numel() != 0:
        output = output * weight.to(device=x.device, dtype=torch.float32)
    return output.to(x.dtype)


def _apply_interleaved_rope_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    if x.shape[-1] == 0:
        return x.contiguous()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(x.dtype)


def _apply_compressed_rope_and_quantize(
    compressed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    rotate: bool = False,
) -> torch.Tensor:
    if rope_dim < 0 or rope_dim > compressed.shape[-1]:
        raise ValueError(f"rope_dim must be in [0, {compressed.shape[-1]}], got {rope_dim}")
    nope_dim = compressed.shape[-1] - rope_dim
    nope, pe = torch.split(compressed, [nope_dim, rope_dim], dim=-1)
    pe = _apply_interleaved_rope_ref(pe, cos, sin)
    compressed = torch.cat((nope, pe), dim=-1)
    if rotate:
        return _fake_fp4_act_quant(_hadamard_rotate(compressed), block_size=32)
    nope, pe = torch.split(compressed, [nope_dim, rope_dim], dim=-1)
    nope = _fake_fp8_act_quant(nope, block_size=64)
    return torch.cat((nope, pe), dim=-1)


def _overlap_transform_projected(
    tensor: torch.Tensor,
    head_dim: int,
    value: float,
) -> torch.Tensor:
    batch_size, compressed_len, ratio, _ = tensor.shape
    previous = tensor[:, :, :, :head_dim]
    current = tensor[:, :, :, head_dim:]
    prefix = tensor.new_full((batch_size, 1, ratio, head_dim), value)
    previous = torch.cat((prefix, previous[:, :-1]), dim=1)
    return torch.cat((previous, current), dim=2)


def _build_full_compressed_kv(
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    compress_ratio: int,
    max_compressed_len: int,
    rotate: bool = False,
) -> torch.Tensor:
    if compress_ratio == 0:
        return compressor_kv.new_empty(compressor_kv.shape[0], 0, compressor_kv.shape[-1])

    _validate_rank("compressor_kv", compressor_kv, 3)
    _validate_rank("compressor_gate", compressor_gate, 3)
    _validate_rank("compressor_ape", compressor_ape, 2)
    _validate_rank("compressor_norm_weight", compressor_norm_weight, 1)
    _validate_rank("cos_table", cos_table, 2)
    _validate_rank("sin_table", sin_table, 2)
    _validate_rank("position_ids", position_ids, 2)
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
    if seq_len > max_compressed_tokens:
        raise ValueError(f"seq_len {seq_len} exceeds compressed capacity {max_compressed_tokens}")
    if seq_len == 0:
        return compressor_kv.new_empty(batch_size, max_compressed_len, head_dim)

    row_offsets = torch.arange(max_compressed_len, device=compressor_kv.device)
    token_offsets = torch.arange(compress_ratio, device=compressor_kv.device)
    gather_idxs = row_offsets.unsqueeze(1) * compress_ratio + token_offsets
    valid = gather_idxs < seq_len
    gather_idxs = torch.where(valid, gather_idxs, torch.zeros_like(gather_idxs))
    flat_idxs = gather_idxs.reshape(-1)

    kv = compressor_kv[:, flat_idxs].view(batch_size, max_compressed_len, compress_ratio, state_dim)
    gate = compressor_gate[:, flat_idxs].view(
        batch_size, max_compressed_len, compress_ratio, state_dim
    )
    gate = gate + compressor_ape.to(device=gate.device, dtype=gate.dtype)
    gate = torch.where(
        valid.view(1, max_compressed_len, compress_ratio, 1),
        gate,
        gate.new_full((), -1.0e20),
    )
    if overlap:
        kv = _overlap_transform_projected(kv, head_dim, 0.0)
        gate = _overlap_transform_projected(gate, head_dim, -1.0e20)

    compressed = (kv * gate.softmax(dim=2)).sum(dim=2)
    compressed = _rms_norm_ref(compressed, compressor_norm_weight, rms_norm_eps)

    row_start = row_offsets * compress_ratio
    row_start = torch.minimum(row_start, torch.full_like(row_start, seq_len - 1))
    compressed_position_ids = position_ids[:, row_start]
    cos = cos_table[compressed_position_ids]
    sin = sin_table[compressed_position_ids]
    return _apply_compressed_rope_and_quantize(compressed, cos, sin, rope_dim, rotate=rotate)


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


def _to_host_long(name: str, tensor: torch.Tensor, length: int) -> torch.Tensor:
    flat = tensor.detach().cpu().to(torch.long).flatten()
    if flat.numel() < length:
        raise ValueError(f"{name} must have at least {length} elements, got {flat.numel()}")
    return flat[:length]


def _write_swa_cache(
    kv_seq: torch.Tensor,
    swa_cache: torch.Tensor,
    slot_idx: int,
    input_pos: int,
) -> None:
    if input_pos < 0:
        raise ValueError(f"input_pos must be non-negative, got {input_pos}")
    end_pos = input_pos + kv_seq.shape[0]
    if end_pos > swa_cache.shape[1]:
        raise ValueError(
            f"Sequence write [{input_pos}, {end_pos}) exceeds SWA cache length {swa_cache.shape[1]}"
        )
    swa_cache[slot_idx, input_pos:end_pos].copy_(kv_seq.to(swa_cache.dtype))


def _write_mhc_cache(
    kv_seq: torch.Tensor,
    mhc_cache: torch.Tensor,
    slot_idx: int,
    row_idx: int,
) -> None:
    if row_idx < 0:
        raise ValueError(f"row_idx must be non-negative, got {row_idx}")
    end_pos = row_idx + kv_seq.shape[0]
    if end_pos > mhc_cache.shape[1]:
        raise ValueError(
            f"Compressed row write [{row_idx}, {end_pos}) exceeds MHC cache length "
            f"{mhc_cache.shape[1]}"
        )
    mhc_cache[slot_idx, row_idx:end_pos].copy_(kv_seq.to(mhc_cache.dtype))


def _slice_sequence_tokens(
    tensor: torch.Tensor,
    seq_idx: int,
    flat_start: int,
    seq_len: int,
) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor.new_empty(seq_len, *tensor.shape[2:])
    if tensor.shape[0] > seq_idx and tensor.shape[0] != 1:
        return tensor[seq_idx, :seq_len]
    return tensor.reshape(-1, *tensor.shape[2:])[flat_start : flat_start + seq_len]


def _prefill_kv_source(
    kv: torch.Tensor,
    kv_seq: torch.Tensor,
    seq_idx: int,
    num_seq: int,
) -> torch.Tensor:
    if kv.shape[0] > seq_idx and kv.shape[0] != 1:
        return kv[seq_idx : seq_idx + 1]
    if num_seq == 1:
        return kv
    return kv_seq.unsqueeze(0)


def _slice_sequence_positions(
    position_ids: torch.Tensor,
    seq_idx: int,
    flat_start: int,
    seq_len: int,
) -> torch.Tensor:
    if position_ids.shape[0] > seq_idx and position_ids.shape[0] != 1:
        return position_ids[seq_idx : seq_idx + 1, :seq_len]
    return position_ids.reshape(1, -1)[:, flat_start : flat_start + seq_len]


def _slice_sequence_kv_rows(
    kv: torch.Tensor,
    seq_idx: int,
    flat_start: int,
    seq_len: int,
    num_seq: int,
    compress_ratio: int,
) -> torch.Tensor:
    if compress_ratio == 0:
        return _slice_sequence_tokens(kv, seq_idx, flat_start, seq_len)
    if kv.shape[0] > seq_idx and kv.shape[0] != 1:
        return kv[seq_idx]
    if num_seq == 1:
        return kv.reshape(-1, kv.shape[-1])
    raise ValueError(
        "Flattened compressed DeepSeek V4 sparse attention KV rows are not supported; "
        f"pass batched kv for compress_ratio={compress_ratio}."
    )


def _cached_sparse_attention_from_positions(
    q_token: torch.Tensor,
    attn_sink: torch.Tensor,
    swa_cache: torch.Tensor,
    slot_idx: int,
    positions: torch.Tensor,
    softmax_scale: float,
    max_position_exclusive: Optional[int] = None,
) -> torch.Tensor:
    positions_long = positions.to(torch.long)
    valid_positions = positions_long[positions_long >= 0]
    if valid_positions.numel() > 0:
        max_position = int(valid_positions.max().item())
        if max_position >= swa_cache.shape[1]:
            raise ValueError(
                f"topk/cache position {max_position} exceeds SWA cache length {swa_cache.shape[1]}"
            )
        if max_position_exclusive is not None and max_position >= max_position_exclusive:
            raise ValueError(
                f"topk/cache position {max_position} must be less than current sequence "
                f"cache write end {max_position_exclusive}"
            )

    gather_positions = positions_long.clamp(min=0)
    selected_kv = swa_cache[slot_idx, gather_positions].to(q_token.dtype).unsqueeze(0)
    local_topk = torch.arange(positions_long.numel(), dtype=torch.long, device=q_token.device).view(
        1, 1, -1
    )
    local_topk = torch.where(positions_long.view(1, 1, -1) < 0, -1, local_topk)
    return torch_deepseek_v4_sparse_attention(
        q_token.view(1, 1, *q_token.shape),
        selected_kv,
        attn_sink,
        local_topk,
        softmax_scale,
    ).view(*q_token.shape)


def _cached_local_window_attention(
    q_seq: torch.Tensor,
    attn_sink: torch.Tensor,
    swa_cache: torch.Tensor,
    slot_idx: int,
    input_pos: int,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    for token_offset in range(q_seq.shape[0]):
        query_pos = input_pos + token_offset
        start_pos = max(0, query_pos - window_size + 1)
        positions = torch.arange(start_pos, query_pos + 1, device=q_seq.device)
        outputs.append(
            _cached_sparse_attention_from_positions(
                q_seq[token_offset],
                attn_sink,
                swa_cache,
                slot_idx,
                positions,
                softmax_scale,
            )
        )
    if not outputs:
        return q_seq.new_empty(q_seq.shape)
    return torch.stack(outputs, dim=0)


def _gather_unpaged_rows(
    cache: torch.Tensor,
    slot_idx: int,
    start_pos: int,
    end_pos: int,
    dtype: torch.dtype,
    width: Optional[int] = None,
) -> torch.Tensor:
    if start_pos < 0 or end_pos < start_pos:
        raise ValueError(f"Invalid cache slice [{start_pos}, {end_pos})")
    if end_pos > cache.shape[1]:
        raise ValueError(
            f"Cache slice [{start_pos}, {end_pos}) exceeds cache length {cache.shape[1]}"
        )
    rows = cache[slot_idx, start_pos:end_pos]
    if width is not None:
        rows = rows[..., :width]
    return rows.to(dtype)


def _compressed_row_from_unpaged_state(
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    slot_idx: int,
    row_idx: int,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    compress_ratio: int,
    head_dim: int,
    state_dim: int,
    dtype: torch.dtype,
    rotate: bool = False,
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
                        -1.0e20,
                        dtype=dtype,
                        device=compressor_gate_cache.device,
                    )
                )
                continue
            kv_state = compressor_kv_cache[slot_idx, position].to(dtype)
            gate_state = compressor_gate_cache[slot_idx, position].to(dtype)
            kv_rows.append(kv_state[:head_dim])
            gate_rows.append(gate_state[:head_dim] + compressor_ape[offset, :head_dim].to(dtype))

        for offset in range(compress_ratio):
            position = anchor + offset
            kv_state = compressor_kv_cache[slot_idx, position].to(dtype)
            gate_state = compressor_gate_cache[slot_idx, position].to(dtype)
            kv_rows.append(kv_state[head_dim : 2 * head_dim])
            gate_rows.append(
                gate_state[head_dim : 2 * head_dim]
                + compressor_ape[offset, head_dim : 2 * head_dim].to(dtype)
            )
    else:
        for offset in range(compress_ratio):
            position = anchor + offset
            kv_state = compressor_kv_cache[slot_idx, position].to(dtype)
            gate_state = compressor_gate_cache[slot_idx, position].to(dtype)
            kv_rows.append(kv_state[:head_dim])
            gate_rows.append(gate_state[:head_dim] + compressor_ape[offset, :head_dim].to(dtype))

    kv = torch.stack(kv_rows, dim=0)
    gate = torch.stack(gate_rows, dim=0)
    pooled = (kv * gate.softmax(dim=0)).sum(dim=0)
    pooled = _rms_norm_ref(pooled.unsqueeze(0), compressor_norm_weight, rms_norm_eps).squeeze(0)
    del state_dim
    cos = cos_table[anchor].unsqueeze(0)
    sin = sin_table[anchor].unsqueeze(0)
    return _apply_compressed_rope_and_quantize(
        pooled.unsqueeze(0),
        cos,
        sin,
        rope_dim,
        rotate=rotate,
    ).squeeze(0)


def _update_compressed_unpaged_caches(
    compressor_kv_seq: torch.Tensor,
    compressor_gate_seq: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    slot_idx: int,
    input_pos: int,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    compress_ratio: int,
    max_compressed_len: int,
) -> None:
    if compress_ratio == 0 or compressor_kv_seq.numel() == 0:
        return

    if compressor_kv_seq.shape != compressor_gate_seq.shape:
        raise ValueError(
            "compressor_kv and compressor_gate sequence slices must have matching shapes, "
            f"got {tuple(compressor_kv_seq.shape)} and {tuple(compressor_gate_seq.shape)}"
        )
    state_dim = int(compressor_kv_seq.shape[-1])
    head_dim = state_dim // (2 if compress_ratio == 4 else 1)
    _write_swa_cache(compressor_kv_seq, compressor_kv_cache, slot_idx, input_pos)
    _write_swa_cache(compressor_gate_seq, compressor_gate_cache, slot_idx, input_pos)

    old_completed = min(input_pos // compress_ratio, max_compressed_len)
    new_completed = min(
        (input_pos + compressor_kv_seq.shape[0]) // compress_ratio, max_compressed_len
    )
    compressed_rows = []
    for row_idx in range(old_completed, new_completed):
        compressed_rows.append(
            _compressed_row_from_unpaged_state(
                compressor_kv_cache,
                compressor_gate_cache,
                slot_idx,
                row_idx,
                compressor_ape,
                compressor_norm_weight,
                cos_table,
                sin_table,
                rms_norm_eps,
                rope_dim,
                compress_ratio,
                head_dim,
                state_dim,
                compressor_kv_seq.dtype,
            )
        )
    if compressed_rows:
        _write_mhc_cache(
            torch.stack(compressed_rows, dim=0),
            mhc_cache,
            slot_idx,
            old_completed,
        )


def _update_raw_unpaged_caches(
    compressor_kv_seq: torch.Tensor,
    compressor_gate_seq: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    slot_idx: int,
    input_pos: int,
) -> None:
    if compressor_kv_seq.numel() == 0:
        return
    if compressor_kv_seq.shape != compressor_gate_seq.shape:
        raise ValueError(
            "compressor_kv and compressor_gate sequence slices must have matching shapes, "
            f"got {tuple(compressor_kv_seq.shape)} and {tuple(compressor_gate_seq.shape)}"
        )
    _write_swa_cache(compressor_kv_seq, compressor_kv_cache, slot_idx, input_pos)
    _write_swa_cache(compressor_gate_seq, compressor_gate_cache, slot_idx, input_pos)


def _select_ratio4_indexer_rows(
    q_index: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv_cache: torch.Tensor,
    indexer_compressor_gate_cache: torch.Tensor,
    slot_idx: int,
    query_pos: int,
    index_topk: int,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    max_compressed_len: int,
) -> torch.Tensor:
    if index_topk <= 0:
        return torch.empty(0, dtype=torch.int64, device=q_index.device)

    visible_len = min((query_pos + 1) // 4, max_compressed_len)
    if visible_len <= 0:
        return torch.full((index_topk,), -1, dtype=torch.int64, device=q_index.device)

    index_head_dim = int(q_index.shape[-1])
    state_dim = int(indexer_compressor_kv_cache.shape[-1])
    index_k = torch.stack(
        [
            _compressed_row_from_unpaged_state(
                indexer_compressor_kv_cache,
                indexer_compressor_gate_cache,
                slot_idx,
                row_idx,
                indexer_compressor_ape,
                indexer_compressor_norm_weight,
                cos_table,
                sin_table,
                rms_norm_eps,
                rope_dim,
                4,
                index_head_dim,
                state_dim,
                q_index.dtype,
                rotate=True,
            )
            for row_idx in range(visible_len)
        ],
        dim=0,
    )
    index_score = torch.einsum("hd,td->ht", q_index, index_k).float()
    index_score = (index_score.relu() * indexer_weights.float().unsqueeze(-1)).sum(dim=0)
    if dist_common.is_initialized() and dist_common.get_world_size() > 1:
        dist_common.all_reduce(index_score, op=dist_common.ReduceOp.SUM)

    topk_count = min(index_topk, visible_len)
    selected = index_score.topk(topk_count, dim=-1).indices.to(torch.int64)
    if topk_count < index_topk:
        pad = torch.full(
            (index_topk - topk_count,),
            -1,
            dtype=selected.dtype,
            device=selected.device,
        )
        selected = torch.cat((selected, pad), dim=0)
    return selected


def _cached_compressed_attention(
    q_seq: torch.Tensor,
    attn_sink: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    slot_idx: int,
    input_pos: int,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: int,
    softmax_scale: float,
    topk_seq: Optional[torch.Tensor] = None,
    indexer_q_seq: Optional[torch.Tensor] = None,
    indexer_weights_seq: Optional[torch.Tensor] = None,
    indexer_compressor_kv_cache: Optional[torch.Tensor] = None,
    indexer_compressor_gate_cache: Optional[torch.Tensor] = None,
    indexer_compressor_ape: Optional[torch.Tensor] = None,
    indexer_compressor_norm_weight: Optional[torch.Tensor] = None,
    cos_table: Optional[torch.Tensor] = None,
    sin_table: Optional[torch.Tensor] = None,
    rms_norm_eps: float = 1e-6,
    rope_dim: Optional[int] = None,
) -> torch.Tensor:
    outputs = []
    for token_offset in range(q_seq.shape[0]):
        query_pos = input_pos + token_offset
        local_start = max(0, query_pos - window_size + 1)
        local_kv = _gather_unpaged_rows(
            swa_cache,
            slot_idx,
            local_start,
            query_pos + 1,
            q_seq.dtype,
        )
        local_idxs = torch.arange(local_kv.shape[0], dtype=torch.int64, device=q_seq.device)
        if compress_ratio == 4:
            if (
                topk_seq is None
                or indexer_q_seq is None
                or indexer_weights_seq is None
                or indexer_compressor_kv_cache is None
                or indexer_compressor_gate_cache is None
                or indexer_compressor_ape is None
                or indexer_compressor_norm_weight is None
                or cos_table is None
                or sin_table is None
                or rope_dim is None
            ):
                raise ValueError("Ratio-4 cached decode requires indexer tensors and caches.")
            index_topk = max(int(topk_seq.shape[-1]) - int(window_size), 0)
            selected_rows = _select_ratio4_indexer_rows(
                indexer_q_seq[token_offset],
                indexer_weights_seq[token_offset],
                indexer_compressor_kv_cache,
                indexer_compressor_gate_cache,
                slot_idx,
                query_pos,
                index_topk,
                indexer_compressor_ape,
                indexer_compressor_norm_weight,
                cos_table,
                sin_table,
                rms_norm_eps,
                rope_dim,
                max_compressed_len,
            )
            gather_rows = selected_rows.clamp(min=0)
            compressed_kv = mhc_cache[slot_idx, gather_rows].to(q_seq.dtype)
            compressed_idxs = torch.where(
                selected_rows < 0,
                selected_rows,
                torch.arange(selected_rows.numel(), dtype=torch.int64, device=q_seq.device)
                + local_kv.shape[0],
            )
        else:
            compressed_len = min((query_pos + 1) // compress_ratio, max_compressed_len)
            compressed_kv = _gather_unpaged_rows(
                mhc_cache,
                slot_idx,
                0,
                compressed_len,
                q_seq.dtype,
            )
            compressed_idxs = torch.arange(
                compressed_kv.shape[0], dtype=torch.int64, device=q_seq.device
            )
            compressed_idxs = compressed_idxs + local_kv.shape[0]
        topk = torch.cat((local_idxs, compressed_idxs), dim=0).view(1, 1, -1)
        kv = torch.cat((local_kv, compressed_kv), dim=0)
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


def _cached_topk_attention(
    q_seq: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_seq: torch.Tensor,
    swa_cache: torch.Tensor,
    slot_idx: int,
    cache_end_pos: int,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    for token_offset in range(q_seq.shape[0]):
        outputs.append(
            _cached_sparse_attention_from_positions(
                q_seq[token_offset],
                attn_sink,
                swa_cache,
                slot_idx,
                topk_seq[token_offset],
                softmax_scale,
                cache_end_pos,
            )
        )
    if not outputs:
        return q_seq.new_empty(q_seq.shape)
    return torch.stack(outputs, dim=0)


def _cached_decode_topk_positions(
    topk_seq: torch.Tensor,
    input_pos: int,
    window_size: Optional[int],
    compress_ratio: int,
) -> torch.Tensor:
    if compress_ratio == 0 or window_size is None:
        return topk_seq

    local_window_cols = min(window_size, topk_seq.shape[-1])
    if local_window_cols == 0:
        return topk_seq

    token_offsets = torch.arange(topk_seq.shape[0], device=topk_seq.device).unsqueeze(1)
    query_positions = input_pos + token_offsets
    window_offsets = torch.arange(local_window_cols, device=topk_seq.device)
    local_topk = query_positions - local_window_cols + 1 + window_offsets
    local_topk = torch.where(local_topk < 0, -1, local_topk)
    local_topk = local_topk.to(topk_seq.dtype)
    if local_window_cols == topk_seq.shape[-1]:
        return local_topk
    return torch.cat((local_topk, topk_seq[..., local_window_cols:]), dim=-1)


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
    enable_sharding: bool = False,
    layer_type: str = "mha_sparse",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
) -> torch.Tensor:
    """Reference DeepSeek V4 sparse attention source op.

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

    _validate_deepseek_v4_sparse_attention_inputs(q, kv, attn_sink, topk_idxs)
    _validate_topk_idx_bounds(topk_idxs, kv.shape[1])

    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    batch_size, seq_len, num_heads, q_head_dim = q.shape
    _, _, k_select = topk_idxs.shape
    num_tokens = batch_size * seq_len
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if num_tokens == 0:
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

    return output


@torch_deepseek_v4_sparse_attention.register_fake
def torch_deepseek_v4_sparse_attention_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    enable_sharding: bool = False,
    layer_type: str = "mha_sparse",
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
    return q.new_empty(q.shape).contiguous()


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
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv: torch.Tensor,
    indexer_compressor_gate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    softmax_scale: float,
    enable_sharding: bool = False,
    layer_type: str = "mha_sparse",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
) -> torch.Tensor:
    """DeepSeek V4 sparse source op with explicit compressor projections."""
    del (
        indexer_q,
        indexer_weights,
        indexer_compressor_kv,
        indexer_compressor_gate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        enable_sharding,
        layer_type,
        layer_idx,
        window_size,
        head_dim,
    )
    _validate_deepseek_v4_sparse_attention_inputs(q, kv, attn_sink, topk_idxs)
    _validate_compress_ratio(compress_ratio)
    if compress_ratio:
        if max_compressed_len is None:
            raise ValueError("max_compressed_len is required for compressed attention.")
        if rope_dim is None:
            raise ValueError("rope_dim is required for compressed attention.")
        compressed_kv = _build_full_compressed_kv(
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            cos_table,
            sin_table,
            position_ids,
            rms_norm_eps,
            rope_dim,
            compress_ratio,
            max_compressed_len,
        ).to(kv.dtype)
        kv = torch.cat((kv, compressed_kv), dim=1)
    return torch_deepseek_v4_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)


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
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv: torch.Tensor,
    indexer_compressor_gate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    softmax_scale: float,
    enable_sharding: bool = False,
    layer_type: str = "mha_sparse",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
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
        rms_norm_eps,
    )
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    _validate_rank("compressor_kv", compressor_kv, 3)
    _validate_rank("compressor_gate", compressor_gate, 3)
    _validate_rank("compressor_ape", compressor_ape, 2)
    _validate_rank("compressor_norm_weight", compressor_norm_weight, 1)
    _validate_rank("cos_table", cos_table, 2)
    _validate_rank("sin_table", sin_table, 2)
    _validate_rank("position_ids", position_ids, 2)
    _validate_rank("indexer_q", indexer_q, 4)
    _validate_rank("indexer_weights", indexer_weights, 3)
    _validate_rank("indexer_compressor_kv", indexer_compressor_kv, 3)
    _validate_rank("indexer_compressor_gate", indexer_compressor_gate, 3)
    _validate_rank("indexer_compressor_ape", indexer_compressor_ape, 2)
    _validate_rank("indexer_compressor_norm_weight", indexer_compressor_norm_weight, 1)
    return q.new_empty(q.shape).contiguous()


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_sparse_attention_with_cache",
    mutates_args=("swa_cache",),
)
def torch_deepseek_v4_sparse_attention_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    swa_cache: torch.Tensor,
    softmax_scale: float,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
) -> torch.Tensor:
    """Reference cached DeepSeek V4 sparse attention.

    The model forms query, KV, and top-k tensors before this op. The cached op
    writes those formed KV rows to the unpaged cache and evaluates sparse
    attention either directly for prefill or by reading cache-position top-k
    entries for decode.
    """
    _validate_deepseek_v4_sparse_attention_inputs(q, kv, attn_sink, topk_idxs)
    _validate_swa_cache_inputs(q, kv, swa_cache)
    _validate_compress_ratio(compress_ratio)
    if window_size is not None and window_size <= 0:
        raise ValueError(f"window_size must be positive when provided, got {window_size}")

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode
    q_flat = q.reshape(-1, *q.shape[2:])
    if active_tokens > q_flat.shape[0]:
        raise ValueError(
            f"active token count {active_tokens} exceeds flattened q tokens {q_flat.shape[0]}"
        )

    seq_len_host = _to_host_long("seq_len", seq_len, num_seq)
    input_pos_host = _to_host_long("input_pos", input_pos, num_seq)
    slot_idx_host = _to_host_long("slot_idx", slot_idx, num_seq)
    cu_seqlen_host = _to_host_long("cu_seqlen", cu_seqlen, num_seq + 1)

    output_flat = torch.zeros_like(q_flat)

    for seq_idx in range(num_seq):
        seq_len_i = int(seq_len_host[seq_idx].item())
        if seq_len_i == 0:
            continue
        flat_start = int(cu_seqlen_host[seq_idx].item())
        input_pos_i = int(input_pos_host[seq_idx].item())
        slot_idx_i = int(slot_idx_host[seq_idx].item())
        if slot_idx_i < 0 or slot_idx_i >= swa_cache.shape[0]:
            raise ValueError(f"slot_idx must be in [0, {swa_cache.shape[0]}), got {slot_idx_i}")
        q_seq = q_flat[flat_start : flat_start + seq_len_i]
        kv_seq = _slice_sequence_kv_rows(
            kv,
            seq_idx,
            flat_start,
            seq_len_i,
            num_seq,
            compress_ratio,
        )
        topk_seq = _slice_sequence_tokens(topk_idxs, seq_idx, flat_start, seq_len_i)
        if q_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} q slice has length {q_seq.shape[0]}, expected {seq_len_i}"
            )
        if kv_seq.shape[0] < seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} kv slice has length {kv_seq.shape[0]}, "
                f"expected at least {seq_len_i}"
            )
        if topk_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} topk_idxs slice has length {topk_seq.shape[0]}, "
                f"expected {seq_len_i}"
            )
        _write_swa_cache(kv_seq, swa_cache, slot_idx_i, input_pos_i)

        if input_pos_i == 0:
            kv_source = _prefill_kv_source(kv, kv_seq, seq_idx, num_seq)
            _validate_topk_idx_bounds(topk_seq.unsqueeze(0), kv_source.shape[1])
            output_flat[flat_start : flat_start + seq_len_i] = torch_deepseek_v4_sparse_attention(
                q_seq.unsqueeze(0),
                kv_source,
                attn_sink,
                topk_seq.unsqueeze(0),
                softmax_scale,
            ).squeeze(0)
        elif compress_ratio == 0 and window_size is not None:
            output_flat[flat_start : flat_start + seq_len_i] = _cached_local_window_attention(
                q_seq,
                attn_sink,
                swa_cache,
                slot_idx_i,
                input_pos_i,
                window_size,
                softmax_scale,
            )
        else:
            cached_topk_seq = _cached_decode_topk_positions(
                topk_seq,
                input_pos_i,
                window_size,
                compress_ratio,
            )
            output_flat[flat_start : flat_start + seq_len_i] = _cached_topk_attention(
                q_seq,
                attn_sink,
                cached_topk_seq,
                swa_cache,
                slot_idx_i,
                input_pos_i + kv_seq.shape[0],
                softmax_scale,
            )

    if active_tokens < q_flat.shape[0]:
        output_flat[active_tokens:].zero_()
    return output_flat.view_as(q)


@torch_deepseek_v4_sparse_attention_with_cache.register_fake
def torch_deepseek_v4_sparse_attention_with_cache_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    swa_cache: torch.Tensor,
    softmax_scale: float,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
) -> torch.Tensor:
    _validate_compress_ratio(compress_ratio)
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    _validate_rank("swa_cache", swa_cache, 3)
    del (
        kv,
        attn_sink,
        topk_idxs,
        batch_info_host,
        seq_len,
        input_pos,
        slot_idx,
        cu_seqlen,
        swa_cache,
        softmax_scale,
        window_size,
        compress_ratio,
    )
    return q.new_empty(q.shape).contiguous()


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_sparse_attention_v2_with_cache",
    mutates_args=(
        "swa_cache",
        "mhc_cache",
        "compressor_kv_cache",
        "compressor_gate_cache",
        "indexer_compressor_kv_cache",
        "indexer_compressor_gate_cache",
    ),
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
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv: torch.Tensor,
    indexer_compressor_gate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    indexer_compressor_kv_cache: torch.Tensor,
    indexer_compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
    rope_dim: Optional[int] = None,
) -> torch.Tensor:
    """Reference unpaged cached DeepSeek V4 sparse attention with compressor state."""
    _validate_deepseek_v4_sparse_attention_inputs(q, kv, attn_sink, topk_idxs)
    _validate_swa_cache_inputs(q, kv, swa_cache)
    _validate_swa_cache_inputs(q, kv, mhc_cache)
    _validate_rank("compressor_kv_cache", compressor_kv_cache, 3)
    _validate_rank("compressor_gate_cache", compressor_gate_cache, 3)
    _validate_rank("indexer_q", indexer_q, 4)
    _validate_rank("indexer_weights", indexer_weights, 3)
    _validate_rank("indexer_compressor_kv", indexer_compressor_kv, 3)
    _validate_rank("indexer_compressor_gate", indexer_compressor_gate, 3)
    _validate_rank("indexer_compressor_ape", indexer_compressor_ape, 2)
    _validate_rank("indexer_compressor_norm_weight", indexer_compressor_norm_weight, 1)
    _validate_rank("indexer_compressor_kv_cache", indexer_compressor_kv_cache, 3)
    _validate_rank("indexer_compressor_gate_cache", indexer_compressor_gate_cache, 3)
    _validate_compress_ratio(compress_ratio)
    if window_size is not None and window_size <= 0:
        raise ValueError(f"window_size must be positive when provided, got {window_size}")
    if compress_ratio:
        if window_size is None:
            raise ValueError("window_size is required for compressed cached attention.")
        if max_compressed_len is None or max_compressed_len <= 0:
            raise ValueError(
                "max_compressed_len must be positive for compressed cached attention, "
                f"got {max_compressed_len}"
            )
        if rope_dim is None:
            raise ValueError("rope_dim is required for compressed cached attention.")

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode
    q_flat = q.reshape(-1, *q.shape[2:])
    if active_tokens > q_flat.shape[0]:
        raise ValueError(
            f"active token count {active_tokens} exceeds flattened q tokens {q_flat.shape[0]}"
        )

    seq_len_host = _to_host_long("seq_len", seq_len, num_seq)
    input_pos_host = _to_host_long("input_pos", input_pos, num_seq)
    slot_idx_host = _to_host_long("slot_idx", slot_idx, num_seq)
    cu_seqlen_host = _to_host_long("cu_seqlen", cu_seqlen, num_seq + 1)

    output_flat = torch.zeros_like(q_flat)
    compressed_capacity = int(max_compressed_len) if compress_ratio else 0

    for seq_idx in range(num_seq):
        seq_len_i = int(seq_len_host[seq_idx].item())
        if seq_len_i == 0:
            continue
        flat_start = int(cu_seqlen_host[seq_idx].item())
        input_pos_i = int(input_pos_host[seq_idx].item())
        slot_idx_i = int(slot_idx_host[seq_idx].item())
        if slot_idx_i < 0 or slot_idx_i >= swa_cache.shape[0]:
            raise ValueError(f"slot_idx must be in [0, {swa_cache.shape[0]}), got {slot_idx_i}")

        q_seq = q_flat[flat_start : flat_start + seq_len_i]
        kv_seq = _slice_sequence_tokens(kv, seq_idx, flat_start, seq_len_i)
        topk_seq = _slice_sequence_tokens(topk_idxs, seq_idx, flat_start, seq_len_i)
        indexer_q_seq = _slice_sequence_tokens(indexer_q, seq_idx, flat_start, seq_len_i)
        indexer_weights_seq = _slice_sequence_tokens(
            indexer_weights, seq_idx, flat_start, seq_len_i
        )
        if q_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} q slice has length {q_seq.shape[0]}, expected {seq_len_i}"
            )
        if kv_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} kv slice has length {kv_seq.shape[0]}, expected {seq_len_i}"
            )
        if topk_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} topk_idxs slice has length {topk_seq.shape[0]}, "
                f"expected {seq_len_i}"
            )

        _write_swa_cache(kv_seq, swa_cache, slot_idx_i, input_pos_i)

        if compress_ratio:
            compressor_kv_seq = _slice_sequence_tokens(
                compressor_kv, seq_idx, flat_start, seq_len_i
            )
            compressor_gate_seq = _slice_sequence_tokens(
                compressor_gate, seq_idx, flat_start, seq_len_i
            )
            indexer_compressor_kv_seq = _slice_sequence_tokens(
                indexer_compressor_kv, seq_idx, flat_start, seq_len_i
            )
            indexer_compressor_gate_seq = _slice_sequence_tokens(
                indexer_compressor_gate, seq_idx, flat_start, seq_len_i
            )
            _update_compressed_unpaged_caches(
                compressor_kv_seq,
                compressor_gate_seq,
                compressor_ape,
                compressor_norm_weight,
                cos_table,
                sin_table,
                slot_idx_i,
                input_pos_i,
                mhc_cache,
                compressor_kv_cache,
                compressor_gate_cache,
                rms_norm_eps,
                rope_dim,
                compress_ratio,
                compressed_capacity,
            )
            if compress_ratio == 4:
                _update_raw_unpaged_caches(
                    indexer_compressor_kv_seq,
                    indexer_compressor_gate_seq,
                    indexer_compressor_kv_cache,
                    indexer_compressor_gate_cache,
                    slot_idx_i,
                    input_pos_i,
                )

            if input_pos_i == 0:
                position_ids_seq = _slice_sequence_positions(
                    position_ids, seq_idx, flat_start, seq_len_i
                )
                output_flat[flat_start : flat_start + seq_len_i] = (
                    torch_deepseek_v4_sparse_attention_v2(
                        q_seq.unsqueeze(0),
                        kv_seq.unsqueeze(0),
                        attn_sink,
                        topk_seq.unsqueeze(0),
                        compressor_kv_seq.unsqueeze(0),
                        compressor_gate_seq.unsqueeze(0),
                        compressor_ape,
                        compressor_norm_weight,
                        cos_table,
                        sin_table,
                        position_ids_seq,
                        indexer_q_seq.unsqueeze(0),
                        indexer_weights_seq.unsqueeze(0),
                        indexer_compressor_kv_seq.unsqueeze(0),
                        indexer_compressor_gate_seq.unsqueeze(0),
                        indexer_compressor_ape,
                        indexer_compressor_norm_weight,
                        softmax_scale,
                        window_size=window_size,
                        compress_ratio=compress_ratio,
                        max_compressed_len=max_compressed_len,
                        rope_dim=rope_dim,
                        rms_norm_eps=rms_norm_eps,
                    ).squeeze(0)
                )
            else:
                output_flat[flat_start : flat_start + seq_len_i] = _cached_compressed_attention(
                    q_seq,
                    attn_sink,
                    swa_cache,
                    mhc_cache,
                    slot_idx_i,
                    input_pos_i,
                    window_size,
                    compress_ratio,
                    compressed_capacity,
                    softmax_scale,
                    topk_seq=topk_seq,
                    indexer_q_seq=indexer_q_seq,
                    indexer_weights_seq=indexer_weights_seq,
                    indexer_compressor_kv_cache=indexer_compressor_kv_cache,
                    indexer_compressor_gate_cache=indexer_compressor_gate_cache,
                    indexer_compressor_ape=indexer_compressor_ape,
                    indexer_compressor_norm_weight=indexer_compressor_norm_weight,
                    cos_table=cos_table,
                    sin_table=sin_table,
                    rms_norm_eps=rms_norm_eps,
                    rope_dim=rope_dim,
                )
        elif input_pos_i == 0:
            kv_source = _prefill_kv_source(kv, kv_seq, seq_idx, num_seq)
            _validate_topk_idx_bounds(topk_seq.unsqueeze(0), kv_source.shape[1])
            output_flat[flat_start : flat_start + seq_len_i] = torch_deepseek_v4_sparse_attention(
                q_seq.unsqueeze(0),
                kv_source,
                attn_sink,
                topk_seq.unsqueeze(0),
                softmax_scale,
            ).squeeze(0)
        elif window_size is not None:
            output_flat[flat_start : flat_start + seq_len_i] = _cached_local_window_attention(
                q_seq,
                attn_sink,
                swa_cache,
                slot_idx_i,
                input_pos_i,
                window_size,
                softmax_scale,
            )
        else:
            output_flat[flat_start : flat_start + seq_len_i] = _cached_topk_attention(
                q_seq,
                attn_sink,
                topk_seq,
                swa_cache,
                slot_idx_i,
                input_pos_i + kv_seq.shape[0],
                softmax_scale,
            )

    if active_tokens < q_flat.shape[0]:
        output_flat[active_tokens:].zero_()
    return output_flat.view_as(q)


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
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv: torch.Tensor,
    indexer_compressor_gate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    indexer_compressor_kv_cache: torch.Tensor,
    indexer_compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
    rope_dim: Optional[int] = None,
) -> torch.Tensor:
    _validate_compress_ratio(compress_ratio)
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    _validate_rank("compressor_kv", compressor_kv, 3)
    _validate_rank("compressor_gate", compressor_gate, 3)
    _validate_rank("compressor_ape", compressor_ape, 2)
    _validate_rank("compressor_norm_weight", compressor_norm_weight, 1)
    _validate_rank("cos_table", cos_table, 2)
    _validate_rank("sin_table", sin_table, 2)
    _validate_rank("position_ids", position_ids, 2)
    _validate_rank("indexer_q", indexer_q, 4)
    _validate_rank("indexer_weights", indexer_weights, 3)
    _validate_rank("indexer_compressor_kv", indexer_compressor_kv, 3)
    _validate_rank("indexer_compressor_gate", indexer_compressor_gate, 3)
    _validate_rank("indexer_compressor_ape", indexer_compressor_ape, 2)
    _validate_rank("indexer_compressor_norm_weight", indexer_compressor_norm_weight, 1)
    _validate_rank("swa_cache", swa_cache, 3)
    _validate_rank("mhc_cache", mhc_cache, 3)
    _validate_rank("compressor_kv_cache", compressor_kv_cache, 3)
    _validate_rank("compressor_gate_cache", compressor_gate_cache, 3)
    _validate_rank("indexer_compressor_kv_cache", indexer_compressor_kv_cache, 3)
    _validate_rank("indexer_compressor_gate_cache", indexer_compressor_gate_cache, 3)
    del (
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        cos_table,
        sin_table,
        position_ids,
        indexer_q,
        indexer_weights,
        indexer_compressor_kv,
        indexer_compressor_gate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        batch_info_host,
        seq_len,
        input_pos,
        slot_idx,
        cu_seqlen,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        indexer_compressor_kv_cache,
        indexer_compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        max_compressed_len,
        rms_norm_eps,
        rope_dim,
    )
    return q.new_empty(q.shape).contiguous()


@AttentionRegistry.register("deepseek_v4_sparse")
class DeepSeekV4SparseAttention(AttentionDescriptor):
    """Cached DeepSeek V4 sparse attention descriptor for reference validation."""

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 17

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> list[str]:
        return ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        kv_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        head_dim = int(kv_fake.shape[-1])
        compressor_kv_fake: FakeTensor = source_attn_node.args[4].meta["val"]
        compressor_state_dim = int(compressor_kv_fake.shape[-1])
        if compressor_state_dim <= 0:
            compressor_state_dim = head_dim
        indexer_compressor_kv_fake: FakeTensor = source_attn_node.args[13].meta["val"]
        indexer_compressor_state_dim = int(indexer_compressor_kv_fake.shape[-1])
        dtype = cls.resolve_cache_dtype(cache_config.dtype, kv_fake.dtype)
        return {
            "swa_cache": UnpagedResourceHandler(
                head_dim,
                dtype=dtype,
            ),
            "mhc_cache": UnpagedResourceHandler(head_dim, dtype=dtype),
            "compressor_kv_cache": UnpagedResourceHandler(
                compressor_state_dim,
                dtype=torch.float32,
            ),
            "compressor_gate_cache": UnpagedResourceHandler(
                compressor_state_dim,
                dtype=torch.float32,
            ),
            "indexer_compressor_kv_cache": UnpagedResourceHandler(
                indexer_compressor_state_dim,
                dtype=torch.float32,
            ),
            "indexer_compressor_gate_cache": UnpagedResourceHandler(
                indexer_compressor_state_dim,
                dtype=torch.float32,
            ),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> list[Constant]:
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
        if not isinstance(softmax_scale, float):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"float softmax_scale, got {softmax_scale!r}."
            )
        if window_size is not None and not isinstance(window_size, int):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"int window_size or None, got {window_size!r}."
            )
        if not isinstance(compress_ratio, int):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"int compress_ratio, got {compress_ratio!r}."
            )
        if compress_ratio not in _SUPPORTED_COMPRESS_RATIOS:
            raise RuntimeError(
                "DeepSeek V4 sparse attention cache insertion supports "
                f"compress_ratio in {_SUPPORTED_COMPRESS_RATIOS}, got {compress_ratio}."
            )
        if max_compressed_len is not None and not isinstance(max_compressed_len, int):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"int max_compressed_len or None, got {max_compressed_len!r}."
            )
        if not isinstance(rms_norm_eps, float):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"float rms_norm_eps, got {rms_norm_eps!r}."
            )
        if rope_dim is not None and not isinstance(rope_dim, int):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"int rope_dim or None, got {rope_dim!r}."
            )
        return [
            softmax_scale,
            window_size,
            compress_ratio,
            max_compressed_len,
            rms_norm_eps,
            rope_dim,
        ]
