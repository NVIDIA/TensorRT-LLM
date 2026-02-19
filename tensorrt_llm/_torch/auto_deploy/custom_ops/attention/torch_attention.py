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

"""Torch reference implementations for attention."""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _apply_logit_softcapping(attn_scores: torch.Tensor, logit_cap: Optional[float]) -> torch.Tensor:
    """Apply logit softcapping using the formula: logit_cap * tanh(logits / logit_cap)"""
    if logit_cap is not None and logit_cap > 0.0:
        return logit_cap * torch.tanh(attn_scores / logit_cap)
    return attn_scores


def _convert_boolean_mask_to_float(attn_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert boolean attention mask to floating point mask.
    Args:
        attn_mask: Boolean tensor where True allows attention, False blocks it
        dtype: Target dtype for the output mask
    Returns:
        Floating point mask where True -> 1.0, False -> -inf
    """
    if attn_mask.dtype == torch.bool:
        float_mask = torch.zeros_like(attn_mask, dtype=dtype)
        float_mask = float_mask.masked_fill(attn_mask, 1.0)  # True -> 1.0
        float_mask = float_mask.masked_fill(~attn_mask, float("-inf"))  # False -> -inf
        return float_mask
    return attn_mask


@torch.library.custom_op("auto_deploy::torch_attention_repeat_kv", mutates_args=())
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states.clone()  # Ensure we don't return an alias
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    # Return a contiguous clone to avoid aliasing issues
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim).contiguous()


@repeat_kv.register_fake
def repeat_kv_fake(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    replicated_shape = (batch, num_key_value_heads * n_rep, slen, head_dim)
    return torch.empty(replicated_shape, device=hidden_states.device, dtype=hidden_states.dtype)


@torch.library.custom_op("auto_deploy::torch_attention_sdpa", mutates_args=())
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """A carbon copy of torch.nn.functional.scaled_dot_product_attention as custom op.

    Using this custom op instead of using the functional directly ensures consistent representation
    of the vanilla sdpa in a graph.
    """

    return F.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


@scaled_dot_product_attention.register_fake
def scaled_dot_product_attention_fake(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """Fake implementation of scaled_dot_product_attention."""
    return query.new_empty(*query.shape[:-1], value.shape[-1]).contiguous()


# Unified attention op
@torch.library.custom_op("auto_deploy::torch_attention", mutates_args=())
def torch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    logit_cap: Optional[float] = None,
    layout: str = "bnsd",  # "bnsd" or "bsnd"
) -> torch.Tensor:
    """
    SDPA attention (with optional GQA) that supports two memory layouts via `layout`:
      - "bnsd": [batch, num_heads, seq_len, head_dim]
      - "bsnd": [batch, seq_len, num_heads, head_dim]

    The `attn_mask` is always interpreted as [b, n, s_q, s_k].

    Returns a tensor in the SAME layout as inputs specified by `layout`.
    """
    if layout not in ("bnsd", "bsnd"):
        raise ValueError(f"layout must be 'bnsd' or 'bsnd', got {layout!r}")

    if layout == "bsnd":
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

    b, n_heads, s_q, head_dim = query.shape  # bnsd format: [batch, num_heads, seq_len, head_dim]
    _, n_kv_heads, s_k, _ = key.shape  # bnsd format: [batch, num_kv_heads, seq_len, head_dim]

    # Inputs are already in bnsd format, no need to transpose
    query_t = query  # [b, n_heads, s_q, head_dim]
    key_t = key  # [b, n_kv_heads, s_k, head_dim]
    value_t = value  # [b, n_kv_heads, s_k, v_head_dim]

    # Handle GQA by repeating KV if needed
    if n_heads != n_kv_heads:
        n_rep = n_heads // n_kv_heads
        key_t = repeat_kv(key_t, n_rep)
        value_t = repeat_kv(value_t, n_rep)

    # Set scale
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores: Q @ K^T
    attn_scores = torch.matmul(query_t, key_t.transpose(-2, -1)) * scale  # [b, n_heads, s_q, s_k]

    # Apply attention mask if provided
    if attn_mask is not None:
        # Convert boolean mask to float if needed
        attn_mask = _convert_boolean_mask_to_float(attn_mask, attn_scores.dtype)
        attn_scores = attn_scores + attn_mask

    # Apply causal mask if specified and only during the context phase
    if is_causal and s_q == s_k:  # Only apply causal mask during context processing
        causal_mask = torch.triu(
            torch.ones(s_q, s_k, device=query.device, dtype=torch.bool),
            diagonal=1,  # Use diagonal=1 for standard causal masking
        )
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Apply sliding window mask if specified
    if sliding_window is not None and sliding_window > 0:
        # Handle position calculation for both context and generation phases
        if s_q == s_k:
            # Context phase: standard position calculation
            query_positions = torch.arange(s_q, device=query.device)
            key_positions = torch.arange(s_k, device=query.device)
        else:
            # Generation phase: query is at position s_k (after the cache)
            query_positions = torch.arange(s_k, s_k + s_q, device=query.device)  # [s_k] for s_q=1
            key_positions = torch.arange(s_k, device=query.device)  # [0,1,2,...,s_k-1]

        # Create position difference matrix: query_pos - key_pos
        pos_diff = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)  # [s_q, s_k]

        # Sliding window mask: allow attention only if 0 <= pos_diff < sliding_window_size
        sliding_window_mask = (pos_diff < 0) | (pos_diff >= sliding_window)  # [s_q, s_k]
        attn_scores.masked_fill_(sliding_window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Apply logit softcapping if enabled
    attn_scores = _apply_logit_softcapping(attn_scores, logit_cap)

    # Apply sinks if provided
    if sinks is not None:
        # Concatenate sinks to attention scores following the reference implementation
        # sinks should have n_heads elements, each head gets its own sink value
        # Expand sinks to [b, n_heads, s_q, 1] - one sink column per head
        sinks_expanded = sinks.reshape(1, -1, 1, 1).expand(
            b, n_heads, s_q, 1
        )  # [b, n_heads, s_q, 1]

        # Concatenate along the key dimension (last dimension)
        logits_max = torch.max(attn_scores, dim=-1, keepdim=True).values
        sinks = torch.exp(sinks_expanded - logits_max)
        unnormalized_scores = torch.exp(attn_scores - logits_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
        scores = unnormalized_scores / normalizer
        # Use only the non-sink portion for computing output
        # We added exactly 1 column, so remove exactly 1 column
        attn_out = torch.matmul(scores, value_t)  # [b, n_heads, s_q, v_head_dim]
    else:
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_out = torch.matmul(attn_weights, value_t)  # [b, n_heads, s_q, v_head_dim]

    # Apply dropout if specified
    if dropout_p > 0.0:
        attn_out = F.dropout(attn_out, p=dropout_p, training=False)

    if layout == "bsnd":
        return attn_out.transpose(1, 2).contiguous()
    else:
        return attn_out.contiguous()


@torch_attention.register_fake
def torch_attention_fake(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    sinks=None,
    sliding_window=None,
    logit_cap=None,
    layout: str = "bnsd",
):
    return query.new_empty(*query.shape[:-1], value.shape[-1]).contiguous()
