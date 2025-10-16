"""Torch reference implementations for attention."""

import math
from typing import Optional

import torch
import torch.nn as nn
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


def update_kv_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,  # metadata
    input_pos: torch.Tensor,  # metadata
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation for update kv cache function. Assumes KV cache layout to be [B,S,N,D].
    This function can be used to build reference attention implementations that use KV cache.
    """

    for idx in range(seq_len.shape[0]):
        k_cache[cache_loc[idx], input_pos[idx] : input_pos[idx] + seq_len[idx], :, :] = key_states[
            seq_start[idx] : seq_start[idx] + seq_len[idx], ...
        ]
        v_cache[cache_loc[idx], input_pos[idx] : input_pos[idx] + seq_len[idx], :, :] = (
            value_states[seq_start[idx] : seq_start[idx] + seq_len[idx], ...]
        )


@torch.library.custom_op("auto_deploy::torch_attention_fused_mla_ref", mutates_args=())
def fused_mla_ref(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    seq_len: torch.Tensor,  # metadata
    input_pos: torch.Tensor,  # metadata
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,  # caches
    v_cache: torch.Tensor,  # caches
    freqs_cis: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference implementation for Fused MLA with KV cache support.
    This implementation flattens the inputs and can be used as a reference to debug the triton kernels.
    """
    # Compute parameters
    bs, num_heads, q_len, qk_nope_head_dim = q_nope.shape
    qk_rope_head_dim = q_pe.shape[-1]
    v_head_dim = kv.shape[-1] - qk_nope_head_dim
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Flatten inputs
    bs_view = (bs * q_len,)

    k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

    q_nope = q_nope.transpose(1, 2).view(*bs_view, num_heads, qk_nope_head_dim).contiguous()
    q_pe = q_pe.transpose(1, 2).clone().view(*bs_view, num_heads, qk_rope_head_dim).contiguous()
    k_nope = k_nope.clone().transpose(1, 2).view(*bs_view, num_heads, qk_nope_head_dim).contiguous()
    k_pe = k_pe.clone().transpose(1, 2).view(*bs_view, -1, qk_rope_head_dim).contiguous()
    value_states = value_states.transpose(1, 2).view(*bs_view, -1, v_head_dim).contiguous()

    if freqs_cis is not None:
        cos_base = freqs_cis[0, ...]
        sin_base = freqs_cis[1, ...]
        for i in range(seq_len.shape[0]):
            start = seq_start[i]
            length = seq_len[i]
            if q_len == 1:
                idx = (input_pos[i] + length - 1).item()
                pos_ids = torch.tensor(idx, device=cos_base.device)
            else:
                pos_ids = torch.arange(input_pos[i], input_pos[i] + length, device=cos_base.device)

            cos = cos_base[pos_ids]  # [..., 1, head_dim]
            sin = sin_base[pos_ids]
            q_slice = q_pe[start : start + length]
            k_slice = k_pe[start : start + length]

            q_rot, k_rot = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
                q_slice,
                k_slice,
                cos,
                sin,
                -2,
            )

            q_pe[start : start + length] = q_rot
            k_pe[start : start + length] = k_rot

    query_states = k_pe.new_empty(*bs_view, num_heads, q_head_dim)  # [b*s,n,d]
    query_states[..., :qk_nope_head_dim] = q_nope
    query_states[..., qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(*bs_view, num_heads, q_head_dim)
    key_states[..., :qk_nope_head_dim] = k_nope
    key_states[..., qk_nope_head_dim:] = k_pe

    # Update KV cache
    update_kv_cache(
        key_states, value_states, k_cache, v_cache, seq_len, input_pos, cache_loc, seq_start
    )

    # Compute attention
    attn_outputs = []
    for idx in range(seq_len.shape[0]):
        # Get inputs from KV cache
        k = k_cache[cache_loc[idx], : input_pos[idx] + seq_len[idx], :, :]  # [kv_seq_len, n, d]
        v = v_cache[cache_loc[idx], : input_pos[idx] + seq_len[idx], :, :]  # [kv_seq_len, n, d]
        # Generate attention mask
        if q_len == 1:
            # Generate phase - single token attention mask
            attn_mask = torch.zeros(
                1, input_pos[idx] + 1, device=query_states.device, dtype=query_states.dtype
            )
        else:
            # Context phase - causal attention mask
            temp_mask = torch.ones(
                seq_len[idx],
                input_pos[idx] + seq_len[idx],
                dtype=torch.bool,
                device=query_states.device,
            ).tril(diagonal=0)
            attn_bias = torch.zeros(
                seq_len[idx],
                input_pos[idx] + seq_len[idx],
                device=query_states.device,
                dtype=query_states.dtype,
            )
            attn_mask = attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf")).to(
                query_states.device
            )

        # Compute attention weights
        attn_weights = (
            torch.matmul(
                query_states[seq_start[idx] : seq_start[idx] + seq_len[idx], :, :].transpose(0, 1),
                k.transpose(0, 1).transpose(1, 2),
            )
            * 1
            / math.sqrt(query_states.size(-1))
        )
        attn_weights = attn_weights + attn_mask
        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = torch.nn.functional.dropout(attn_weights, p=0.0, training=False)
        attn_output = torch.matmul(attn_weights, v.transpose(0, 1))
        attn_outputs.append(attn_output)

    if q_len == 1:
        attn_output = torch.stack(attn_outputs)
    else:
        attn_output = torch.cat(attn_outputs, dim=-2).unsqueeze(0)

    return attn_output


@fused_mla_ref.register_fake
def fused_mla_ref_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    seq_len: torch.Tensor,  # metadata
    input_pos: torch.Tensor,  # metadata
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,  # caches
    v_cache: torch.Tensor,  # caches
    freqs_cis: torch.Tensor,
    softmax_scale: Optional[float] = None,
):
    """Fake Fused MLA+Rope with KV cache support."""
    v_head_dim = kv.shape[-1] - q_nope.shape[-1]
    return torch.empty_like(kv[..., -v_head_dim:])


@torch.library.custom_op("auto_deploy::torch_attention_deepseek_fused_mla", mutates_args=())
def fused_mla(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """MultiHeadLatentAttention as implemented in DeepSeekV3Attention. This does not capture KV cache use/update."""
    # Did not implement KV cache logic, since we would be writing our own custom op and inserting KV caches later
    # Compute parameters
    bs, num_heads, q_len, qk_nope_head_dim = q_nope.shape
    qk_rope_head_dim = q_pe.shape[-1]
    v_head_dim = kv.shape[-1] - qk_nope_head_dim
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
    kv_seq_len = value_states.shape[-2]

    cos = cos[position_ids]
    sin = sin[position_ids]
    q_pe, k_pe = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(q_pe, k_pe, cos, sin)

    query_states = k_pe.new_empty(bs, num_heads, q_len, q_head_dim)
    query_states[:, :, :, :qk_nope_head_dim] = q_nope
    query_states[:, :, :, qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(bs, num_heads, q_len, q_head_dim)
    key_states[:, :, :, :qk_nope_head_dim] = k_nope
    key_states[:, :, :, qk_nope_head_dim:] = k_pe

    # Use old logic
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * softmax_scale

    if attn_weights.size() != (bs, num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bs, num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
    assert attention_mask is not None
    if attention_mask is not None:
        if attention_mask.size() != (bs, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bs, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=False)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bs, num_heads, q_len, v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bs, num_heads, q_len, v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    # We do not return attn_weights along with attn_output
    return attn_output


@fused_mla.register_fake
def fused_mla(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    v_head_dim = kv.shape[-1] - q_nope.shape[-1]
    return torch.empty_like(kv[..., -v_head_dim:])


@torch.library.custom_op("auto_deploy::torch_attention_deepseek_mla", mutates_args=())
def mla(
    q_nope: torch.Tensor,  # Down projected q_nope
    q_pe: torch.Tensor,  # q_pe after applying rope
    kv: torch.Tensor,  # compressed kv after passing through layernorm
    pe: torch.Tensor,  # k_pe after applying rope
    attention_mask: torch.Tensor,  # attention mask
    softmax_scale: float,  # softmax scale
) -> torch.Tensor:
    """
    Reference implementation for MLA style attention that handles compressed kv.
    """
    scores = (
        torch.einsum("bhsc,btc->bsht", q_nope, kv) + torch.einsum("bhsr,btr->bsht", q_pe, pe)
    ) * softmax_scale
    if attention_mask is not None:
        scores += attention_mask.unsqueeze(1)
    scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(q_nope)
    attn_output = torch.einsum("bsht,btc->bshc", scores, kv)
    return attn_output


@mla.register_fake
def mla(
    q_nope: torch.Tensor,  # Down projected q_nope
    q_pe: torch.Tensor,  # q_pe after applying rope
    kv: torch.Tensor,  # compressed kv after passing through layernorm
    k_pe: torch.Tensor,  # k_pe after applying rope
    attention_mask: torch.Tensor,  # attention mask
    softmax_scale: float,  # softmax scale
) -> torch.Tensor:
    """MLA style attention that handles compressed kv."""
    return torch.empty_like(q_nope)
