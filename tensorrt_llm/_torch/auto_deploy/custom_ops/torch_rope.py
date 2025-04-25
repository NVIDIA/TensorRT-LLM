from typing import Optional, Tuple

import torch


# Function to apply rotary positional embeddings (RoPE)
# Used by torch.ops.attention.fused_mha
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


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.library.custom_op("rope::torch_apply_rope_with_explicit_cos_sin", mutates_args=())
def torch_apply_rope_with_explicit_cos_sin(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference PyTorch implementation of HF-style RoPE:
    - Input layout: non-interleaved, [B, N, S, D] with unsqueeze_dim=1 and
        [B, S, N, D] with unsqueeze_dim=2, default is [B, N, S, D]
    - Frequencies are provided as separate `cos` and `sin` tensors of shape [B, S, head_dim].
    """
    # in HF, cos/sin tensor are passed in as x.dtype, this is to double ensure
    cos = cos.type_as(q)
    sin = sin.type_as(q)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch_apply_rope_with_explicit_cos_sin.register_fake
def torch_apply_rope_with_explicit_cos_sin_fake(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    return q, k


@torch.library.custom_op("rope::torch_apply_rope_with_complex_freqs", mutates_args=())
def torch_apply_rope_with_complex_freqs(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,  # shape [B, S, head_dim//2]
    unsqueeze_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference PyTorch implementation of interleaved (complex) RoPE:
    - Input layout: [B, S, N, D] (interleaved)
    - Frequencies are combined into a single complex-valued tensor `freqs_cis`
        of shape [B, S, head_dim // 2].
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(unsqueeze_dim)
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


@torch_apply_rope_with_complex_freqs.register_fake
def torch_apply_rope_with_complex_freqs_fake(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,  # shape [B, S, head_dim//2]
    unsqueeze_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return xq, xk


@torch.library.custom_op("rope::torch_apply_rope_with_qk_interleaving", mutates_args=())
def torch_apply_rope_with_qk_interleaving(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek-style RoPE: interleaves Q/K channels and returns rotated (q_embed, k_embed).
    - Input layout: [B, S, N, D] or [B*S, N, D] or [B, N, S, D]
    - Frequencies are provided as separate `cos` and `sin` tensors of shape
        [B, S, 1, D] or [B*S, 1, D] or [B, 1, S, D] matching input shape.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # Rewrite below code to accept 3D input:
    # b, h, s, d = q.shape
    # q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    # b, h, s, d = k.shape
    # k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q = q.unflatten(-1, (-1, 2)).transpose(-1, -2).reshape_as(q)
    k = k.unflatten(-1, (-1, 2)).transpose(-1, -2).reshape_as(k)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch_apply_rope_with_qk_interleaving.register_fake
def torch_apply_rope_with_qk_interleaving_fake(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    return q, k
