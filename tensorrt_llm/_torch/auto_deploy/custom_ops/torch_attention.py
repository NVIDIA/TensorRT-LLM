"""Torch reference implementations for attention."""

import math
from typing import Optional

import torch
import torch.nn as nn

from ...attention_backend.vanilla import repeat_kv


# Function to apply rotary positional embeddings (RoPE)
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    seq_len: int,
    head_dim: int,
    rope_theta: Optional[float] = None,
    rope_scale: Optional[float] = None,
):
    """
    Apply rotary positional embeddings to query and key tensors.

    Args:
        q: Query tensor of shape [batch, n_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, n_kv_heads, seq_len, head_dim]
        seq_len: Sequence length
        head_dim: Dimension of each head
        rope_theta: Base value for RoPE (default 10000.0)
        rope_scale: Scaling factor for positions (default 1.0)

    Returns:
        Tuple of transformed query and key tensors
    """
    device = q.device
    original_dtype = q.dtype

    # Apply default values if None
    theta = 10000.0 if rope_theta is None else rope_theta
    scale = 1.0 if rope_scale is None else rope_scale

    # Generate position indices
    position = torch.arange(seq_len, device=device).float()
    # Apply scaling factor to positions if provided
    if scale != 1.0:
        position = position / scale

    # Create the frequency matrix - ensure stable computation in float32
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # Compute the product of positions and frequencies
    # Shape: [seq_len, head_dim/2]
    freqs = torch.outer(position, inv_freq)

    # Compute the rotation matrix elements: cos and sin
    # Shape: [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    # Ensure stable computation of sin/cos in float32
    cos = torch.cos(emb).to(dtype=torch.float32)
    sin = torch.sin(emb).to(dtype=torch.float32)

    # Reshape for broadcasting
    # Shape: [1, 1, seq_len, head_dim]
    cos = cos.view(1, 1, seq_len, head_dim)
    sin = sin.view(1, 1, seq_len, head_dim)

    # Always compute in float32 for numerical stability
    q_float = q.to(dtype=torch.float32)
    k_float = k.to(dtype=torch.float32)

    # For the even indices of the dimension
    q_embed_even = q_float[..., 0::2]
    q_embed_odd = q_float[..., 1::2]
    k_embed_even = k_float[..., 0::2]
    k_embed_odd = k_float[..., 1::2]

    # Apply the rotation using the identities:
    # q' = q * cos + rotate(q) * sin
    # k' = k * cos + rotate(k) * sin
    # where rotate(x) swaps the even and odd dimensions and negates the odd dimensions
    q_rotated = torch.cat(
        [
            q_embed_even * cos[..., 0::2] - q_embed_odd * sin[..., 0::2],
            q_embed_odd * cos[..., 1::2] + q_embed_even * sin[..., 1::2],
        ],
        dim=-1,
    )

    k_rotated = torch.cat(
        [
            k_embed_even * cos[..., 0::2] - k_embed_odd * sin[..., 0::2],
            k_embed_odd * cos[..., 1::2] + k_embed_even * sin[..., 1::2],
        ],
        dim=-1,
    )

    # Convert back to the original dtype
    return q_rotated.to(dtype=original_dtype), k_rotated.to(dtype=original_dtype)


@torch.library.custom_op("attention::fused_mha", mutates_args=())
def fused_mha(
    q: torch.Tensor,  # [b, s, n, h_d]
    k: torch.Tensor,  # [b, s, n, h_d]
    v: torch.Tensor,  # [b, s, n, h_d]
    head_dim: int,
    pos_embd_mode: Optional[str] = None,
    rope_theta: Optional[float] = None,
    rope_scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused MHA+Rope that takes raw input from q, k, v GEMMs.

    Rope is performed according to the specified rope configuration. No support for caching.
    """
    # b, s info
    b, s = q.shape[:2]

    # reshapes and transpose to [bnsd] layout
    q = q.view(b, s, -1, head_dim).transpose(1, 2)
    k = k.view(b, s, -1, head_dim).transpose(1, 2)
    v = v.view(b, s, -1, head_dim).transpose(1, 2)

    # some more info
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]

    # rope embedding
    if pos_embd_mode == "rope":
        # Apply rotary positional embeddings
        q, k = apply_rotary_pos_emb(q, k, s, head_dim, rope_theta, rope_scale)
    elif pos_embd_mode is not None:
        raise ValueError(f"Unknown positional embedding mode: {pos_embd_mode}.")

    # repeat kv
    k = repeat_kv(k, num_heads // num_kv_heads)
    v = repeat_kv(v, num_heads // num_kv_heads)

    # Make sure all tensors have the same dtype before attention
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # attention (assumed layout is bnsd)
    y = torch.nn.functional.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        is_causal=True,
    )

    # back to [b, s, n*h_d] layout
    y = y.transpose(1, 2).contiguous().view(b, s, -1)

    return y


@fused_mha.register_fake
def fused_mha_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    pos_embd_mode: Optional[str] = None,
    rope_theta: Optional[float] = None,
    rope_scale: Optional[float] = None,
) -> torch.Tensor:
    """Fake Fused MHA+Rope that takes raw input from q, k, v GEMMs."""
    return torch.empty_like(q)


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


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
@torch.inference_mode()
def apply_rotary_pos_emb_ds(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q = q.unflatten(-1, (-1, 2)).transpose(-1, -2).reshape_as(q)
    k = k.unflatten(-1, (-1, 2)).transpose(-1, -2).reshape_as(k)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.library.custom_op("attention::fused_mla_ref", mutates_args=())
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
        cos = freqs_cis[0, ...]
        sin = freqs_cis[1, ...]
        for idx in range(seq_len.shape[0]):
            (
                q_pe[seq_start[idx] : seq_start[idx] + seq_len[idx], ...],
                k_pe[seq_start[idx] : seq_start[idx] + seq_len[idx], ...],
            ) = apply_rotary_pos_emb_ds(
                q_pe[seq_start[idx] : seq_start[idx] + seq_len[idx], ...],
                k_pe[seq_start[idx] : seq_start[idx] + seq_len[idx], ...],
                cos,
                sin,
                torch.arange(input_pos[idx] + seq_len[idx])[-1]
                if q_len == 1
                else torch.arange(input_pos[idx] + seq_len[idx]),
                -2,
            )

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


@torch.library.custom_op("deepseek::fused_mla", mutates_args=())
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

    q_pe, k_pe = apply_rotary_pos_emb_ds(q_pe, k_pe, cos, sin, position_ids)

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
