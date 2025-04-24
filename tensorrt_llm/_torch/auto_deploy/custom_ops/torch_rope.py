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


# Copied from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L339
@torch.inference_mode()
def apply_rotary_pos_emb_ds(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    RoPE implementation by DeepSeek:
    Apply rotary positional embeddings by interleaving Q/K ,
    indexing cos/sin tables with position_ids, and returning rotated q, k.
    cos:  [seq_len, head_dim]
    sin:  [seq_len, head_dim]
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q = q.unflatten(-1, (-1, 2)).transpose(-1, -2).reshape_as(q)
    k = k.unflatten(-1, (-1, 2)).transpose(-1, -2).reshape_as(k)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.library.custom_op("rope::torch_apply_explicit_rope", mutates_args=())
def torch_apply_explicit_rope(
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


@torch_apply_explicit_rope.register_fake
def torch_apply_explicit_rope_fake(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    return q, k


@torch.library.custom_op("rope::torch_apply_complex_rope", mutates_args=())
def torch_apply_complex_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,  # shape [B, S, head_dim//2]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference PyTorch implementation of interleaved (complex) RoPE:
    - Input layout: [B, S, N, D] (interleaved)
    - Frequencies are combined into a single complex-valued tensor `freqs_cis`
        of shape [B, S, head_dim // 2].
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


@torch_apply_complex_rope.register_fake
def torch_apply_complex_rope_fake(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,  # shape [B, S, head_dim//2]
) -> Tuple[torch.Tensor, torch.Tensor]:
    return xq, xk


@torch.library.custom_op("rope::torch_apply_rope_with_qk_interleaving", mutates_args=())
def torch_apply_rope_with_qk_interleaving(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,  # [B, 1, seq_len, head_dim]
    sin: torch.Tensor,  # [B, 1, seq_len, head_dim]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek-style RoPE: interleaves Q/K channels and returns rotated (q_embed, k_embed).
    """
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch_apply_rope_with_qk_interleaving.register_fake
def torch_apply_rope_with_qk_interleaving_fake(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return q, k
