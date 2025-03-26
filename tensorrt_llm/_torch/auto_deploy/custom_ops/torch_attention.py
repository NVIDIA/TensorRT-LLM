"""Torch reference implementations for attention."""

from typing import Optional, Tuple

import torch

from ...attention_backend.vanilla import repeat_kv


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    seq_len: int,
    head_dim: int,
    rope_theta: Optional[float] = None,
    rope_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors using Neox-style (non-interleaved)
    splitting of the last dimension.

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

    # Compute cos and sin values and reshape for broadcasting.
    cos = (
        torch.cos(freqs).to(torch.float32).unsqueeze(0).unsqueeze(0)
    )  # shape: [1, 1, seq_len, head_dim/2]
    sin = (
        torch.sin(freqs).to(torch.float32).unsqueeze(0).unsqueeze(0)
    )  # shape: [1, 1, seq_len, head_dim/2]

    # Always compute in float32.
    q_float = q.to(torch.float32)
    k_float = k.to(torch.float32)

    # Split the last dimension into two halves.
    q1, q2 = q_float[..., : head_dim // 2], q_float[..., head_dim // 2 :]
    k1, k2 = k_float[..., : head_dim // 2], k_float[..., head_dim // 2 :]

    # Apply rotation for Neox-style RoPE:
    q_rotated_first = q1 * cos - q2 * sin
    q_rotated_second = q1 * sin + q2 * cos
    k_rotated_first = k1 * cos - k2 * sin
    k_rotated_second = k1 * sin + k2 * cos

    q_rotated = torch.cat([q_rotated_first, q_rotated_second], dim=-1)
    k_rotated = torch.cat([k_rotated_first, k_rotated_second], dim=-1)

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
