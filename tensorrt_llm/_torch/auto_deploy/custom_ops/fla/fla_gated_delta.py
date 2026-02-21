# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Torch reference implementation of the Gated Delta Rule for linear attention.

The Gated Delta Rule extends the basic Delta Rule with an exponential decay gate `g`:
  Basic:  S = S + k * (v - S*k) * beta
  Gated:  S = S * exp(g) + k * (v - S*k) * beta

This op is used by Qwen3Next's GatedDeltaNet layers.

Reference:
  - HF transformers v4.57.1 `torch_chunk_gated_delta_rule`:
    https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L309-L397
  - Gated Delta Networks paper: https://arxiv.org/abs/2412.06464
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize along the last dimension."""
    x_f32 = x.float()
    s = (x_f32 * x_f32).sum(dim=-1, keepdim=True)
    return (x_f32 * torch.rsqrt(s + eps)).to(x.dtype)


def _torch_chunk_gated_delta_rule_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure-torch chunked gated delta rule computation.

    Adapted from HF transformers v4.57.1 modeling_qwen3_next.py `torch_chunk_gated_delta_rule`.

    Args:
        query: [B, H, S, K] - query states (already l2-normalized and head-expanded)
        key:   [B, H, S, K] - key states (already l2-normalized and head-expanded)
        value: [B, H, S, V] - value states
        g:     [B, H, S]    - gating/decay values (negative log-space)
        beta:  [B, H, S]    - beta scaling values (sigmoid-activated)
        chunk_size: chunk size for chunked processing
        scale: optional scaling factor for queries (defaults to K^-0.5)

    Returns:
        output: [B, H, S, V]
    """
    initial_dtype = query.dtype
    query, key, value = [x.contiguous().to(torch.float32) for x in (query, key, value)]
    beta = beta.contiguous().to(torch.float32)
    g = g.contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    if scale is None:
        scale = 1.0 / (k_head_dim**0.5)
    query = query * scale

    # Pad sequence to be divisible by chunk_size
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape to chunks: [B, H, num_chunks, chunk_size, D]
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # Chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # Process each chunk recurrently
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    # Remove padding and reshape back
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    return core_attn_out.to(initial_dtype)


@torch.library.custom_op("auto_deploy::torch_gated_delta_rule", mutates_args=())
def torch_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Gated Delta Rule custom op for linear attention (torch reference implementation).

    All inputs use the autodeploy [B, S, H, D] (bsnd) layout convention.
    L2 normalization is applied internally to q and k.
    GVA (Grouped Value Attention) is supported: q/k may have fewer heads than v/g/beta,
    in which case q/k are repeat-interleaved to match v's head count.

    Args:
        q:    [B, S, Hg, K] - query states (raw, will be l2-normalized internally)
        k:    [B, S, Hg, K] - key states (raw, will be l2-normalized internally)
        v:    [B, S, H, V]  - value states
        g:    [B, S, H]     - gating/decay values
        beta: [B, S, H]     - beta scaling values
        scale: optional query scaling factor (defaults to K^-0.5)

    Returns:
        output: [B, S, H, V]
    """
    # L2 normalize q and k
    q = _l2_normalize(q)
    k = _l2_normalize(k)

    # Expand q/k heads to match v's head count (GVA)
    num_k_heads = q.shape[2]
    num_v_heads = v.shape[2]
    if num_v_heads > num_k_heads:
        assert num_v_heads % num_k_heads == 0, (
            f"num_v_heads ({num_v_heads}) must be divisible by num_k_heads ({num_k_heads})"
        )
        q = q.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    # Transpose from bsnd -> bhsd for internal computation
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    g_t = g.transpose(1, 2)
    beta_t = beta.transpose(1, 2)

    out = _torch_chunk_gated_delta_rule_impl(q_t, k_t, v_t, g_t, beta_t, scale=scale)

    # Transpose back from bhsd -> bsnd
    return out.transpose(1, 2).contiguous()


@torch_gated_delta_rule.register_fake
def torch_gated_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    # Output shape is [B, S, H, V] matching v (not q/k which may have fewer heads)
    return torch.empty_like(v)
