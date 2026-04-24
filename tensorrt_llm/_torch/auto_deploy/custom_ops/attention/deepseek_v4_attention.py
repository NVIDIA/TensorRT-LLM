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

"""DeepSeek V4 sparse/HMA attention source op."""

from typing import Optional

import torch


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
    del enable_sharding, layer_type

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
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    del softmax_scale, enable_sharding, layer_type
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    if out is not None:
        return out.new_empty(0)
    return q.new_empty(q.shape).contiguous()
