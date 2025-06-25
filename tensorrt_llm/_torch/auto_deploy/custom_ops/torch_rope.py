from typing import Tuple

import torch


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.library.custom_op("auto_deploy::torch_rope_with_explicit_cos_sin", mutates_args=())
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
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op("auto_deploy::torch_rope_with_complex_freqs", mutates_args=())
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
    return torch.empty_like(xq), torch.empty_like(xk)


@torch.library.custom_op("auto_deploy::torch_rope_with_qk_interleaving", mutates_args=())
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
    return torch.empty_like(q), torch.empty_like(k)
