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

"""TP weight-sharding helpers for VisualGen multi-GPU unit tests.

Mirrors ``Linear._calc_shard`` for MLP dims and ``Attention.shard_start`` for
head-aligned Q/K/V shards so reference weights match TP module layouts.
"""

from __future__ import annotations

import torch


def calc_shard(total: int, tp_size: int, rank: int) -> int:
    """Start index for *rank* when splitting *total* elements across *tp_size* ranks."""
    return (total // tp_size) * rank + min(total % tp_size, rank)


def qkv_head_bounds(
    tp_rank: int,
    tp_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    ulysses_size: int = 1,
) -> tuple[int, int, int, int]:
    """Return (q_start, q_end, kv_start, kv_end) feature bounds for one TP rank."""
    gqa_ratio = num_attention_heads // num_key_value_heads
    kv_heads_per_ulysses = num_key_value_heads // ulysses_size
    kv_head_start = calc_shard(kv_heads_per_ulysses, tp_size, tp_rank) * ulysses_size
    kv_head_end = calc_shard(kv_heads_per_ulysses, tp_size, tp_rank + 1) * ulysses_size
    attn_head_start = kv_head_start * gqa_ratio
    attn_head_end = kv_head_end * gqa_ratio
    q_start = attn_head_start * head_dim
    q_end = attn_head_end * head_dim
    kv_start = kv_head_start * head_dim
    kv_end = kv_head_end * head_dim
    return q_start, q_end, kv_start, kv_end


def shard_dim0(tensor: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Shard a tensor along dim 0 (column-parallel output / 1D bias)."""
    start = calc_shard(tensor.shape[0], tp_size, tp_rank)
    end = calc_shard(tensor.shape[0], tp_size, tp_rank + 1)
    return tensor[start:end].contiguous()


def shard_dim1(tensor: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Shard a tensor along dim 1 (row-parallel input)."""
    start = calc_shard(tensor.shape[1], tp_size, tp_rank)
    end = calc_shard(tensor.shape[1], tp_size, tp_rank + 1)
    return tensor[:, start:end].contiguous()


def shard_fused_qkv_by_heads(
    tensor: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    q_dim: int,
    kv_dim: int,
    ulysses_size: int = 1,
) -> torch.Tensor:
    """Shard fused QKV using Attention head boundaries, not flat row splits."""
    q, k, v = tensor.split([q_dim, kv_dim, kv_dim], dim=0)
    q_start, q_end, kv_start, kv_end = qkv_head_bounds(
        tp_rank, tp_size, num_attention_heads, num_key_value_heads, head_dim, ulysses_size
    )
    return torch.cat([q[q_start:q_end], k[kv_start:kv_end], v[kv_start:kv_end]], dim=0).contiguous()


def shard_kv_dim0(
    tensor: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    num_key_value_heads: int,
    head_dim: int,
    ulysses_size: int = 1,
) -> torch.Tensor:
    """Shard column-parallel K/V (or KV-norm) weights along head boundaries."""
    _, _, kv_start, kv_end = qkv_head_bounds(
        tp_rank, tp_size, num_key_value_heads, num_key_value_heads, head_dim, ulysses_size
    )
    return tensor[kv_start:kv_end].contiguous()


def shard_q_dim0(
    tensor: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    num_attention_heads: int,
    head_dim: int,
    num_key_value_heads: int | None = None,
    ulysses_size: int = 1,
) -> torch.Tensor:
    """Shard column-parallel Q weights along head boundaries."""
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    q_start, q_end, _, _ = qkv_head_bounds(
        tp_rank, tp_size, num_attention_heads, num_key_value_heads, head_dim, ulysses_size
    )
    return tensor[q_start:q_end].contiguous()


def shard_q_dim1(
    tensor: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    num_attention_heads: int,
    head_dim: int,
    num_key_value_heads: int | None = None,
    ulysses_size: int = 1,
) -> torch.Tensor:
    """Shard row-parallel output-proj weights along Q head boundaries."""
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    q_start, q_end, _, _ = qkv_head_bounds(
        tp_rank, tp_size, num_attention_heads, num_key_value_heads, head_dim, ulysses_size
    )
    return tensor[:, q_start:q_end].contiguous()


def shard_kv_dim1(
    tensor: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    num_key_value_heads: int,
    head_dim: int,
    ulysses_size: int = 1,
) -> torch.Tensor:
    """Shard row-parallel K/V output weights along head boundaries."""
    _, _, kv_start, kv_end = qkv_head_bounds(
        tp_rank, tp_size, num_key_value_heads, num_key_value_heads, head_dim, ulysses_size
    )
    return tensor[:, kv_start:kv_end].contiguous()


def shard_fused_gate_up(tensor: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Shard fused gate_up weight [2*intermediate, ...] preserving gate/up structure."""
    half = tensor.shape[0] // 2
    gate, up = tensor.split([half, half], dim=0)
    return torch.cat(
        [
            shard_dim0(gate, tp_rank, tp_size),
            shard_dim0(up, tp_rank, tp_size),
        ],
        dim=0,
    )


def copy_tp_parameter(
    tp_name: str,
    ref_param: torch.Tensor,
    tp_param: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    num_attention_heads: int,
    head_dim: int,
    num_key_value_heads: int | None = None,
    ulysses_size: int = 1,
) -> None:
    """Copy one reference parameter into its TP-sharded counterpart."""
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    if tp_param.shape == ref_param.shape:
        tp_param.data.copy_(ref_param.data)
        return

    if tp_param.ndim >= 2 and tp_param.shape[1] == ref_param.shape[1]:
        if "qkv_proj" in tp_name or "add_qkv_proj" in tp_name:
            q_dim = num_attention_heads * head_dim
            kv_dim = num_key_value_heads * head_dim
            tp_param.data.copy_(
                shard_fused_qkv_by_heads(
                    ref_param.data,
                    tp_rank,
                    tp_size,
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                    q_dim,
                    kv_dim,
                    ulysses_size,
                )
            )
        elif "to_q" in tp_name:
            tp_param.data.copy_(
                shard_q_dim0(
                    ref_param.data,
                    tp_rank,
                    tp_size,
                    num_attention_heads,
                    head_dim,
                    num_key_value_heads,
                    ulysses_size,
                )
            )
        elif (
            "to_k" in tp_name
            or "to_v" in tp_name
            or "add_k_proj" in tp_name
            or "add_v_proj" in tp_name
        ):
            tp_param.data.copy_(
                shard_kv_dim0(
                    ref_param.data, tp_rank, tp_size, num_key_value_heads, head_dim, ulysses_size
                )
            )
        elif "gate_up_proj" in tp_name:
            tp_param.data.copy_(shard_fused_gate_up(ref_param.data, tp_rank, tp_size))
        else:
            tp_param.data.copy_(shard_dim0(ref_param.data, tp_rank, tp_size))
    elif tp_param.ndim >= 2 and tp_param.shape[0] == ref_param.shape[0]:
        if "to_add_out" in tp_name:
            tp_param.data.copy_(
                shard_kv_dim1(
                    ref_param.data, tp_rank, tp_size, num_key_value_heads, head_dim, ulysses_size
                )
            )
        elif "to_out" in tp_name:
            tp_param.data.copy_(
                shard_q_dim1(
                    ref_param.data,
                    tp_rank,
                    tp_size,
                    num_attention_heads,
                    head_dim,
                    num_key_value_heads,
                    ulysses_size,
                )
            )
        else:
            tp_param.data.copy_(shard_dim1(ref_param.data, tp_rank, tp_size))
    elif tp_param.ndim == 1 and tp_param.shape[0] < ref_param.shape[0]:
        if "qkv_proj" in tp_name or "add_qkv_proj" in tp_name:
            q_dim = num_attention_heads * head_dim
            kv_dim = num_key_value_heads * head_dim
            tp_param.data.copy_(
                shard_fused_qkv_by_heads(
                    ref_param.data,
                    tp_rank,
                    tp_size,
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                    q_dim,
                    kv_dim,
                    ulysses_size,
                )
            )
        elif "to_q" in tp_name or "norm_q" in tp_name:
            tp_param.data.copy_(
                shard_q_dim0(
                    ref_param.data,
                    tp_rank,
                    tp_size,
                    num_attention_heads,
                    head_dim,
                    num_key_value_heads,
                    ulysses_size,
                )
            )
        elif (
            "to_k" in tp_name
            or "to_v" in tp_name
            or "add_k_proj" in tp_name
            or "add_v_proj" in tp_name
            or "norm_added_k" in tp_name
            or "norm_k" in tp_name
        ):
            tp_param.data.copy_(
                shard_kv_dim0(
                    ref_param.data, tp_rank, tp_size, num_key_value_heads, head_dim, ulysses_size
                )
            )
        elif "gate_up_proj" in tp_name:
            tp_param.data.copy_(shard_fused_gate_up(ref_param.data, tp_rank, tp_size))
        else:
            tp_param.data.copy_(shard_dim0(ref_param.data, tp_rank, tp_size))
    else:
        raise ValueError(f"Cannot shard {tp_name}: ref={ref_param.shape}, tp={tp_param.shape}")
