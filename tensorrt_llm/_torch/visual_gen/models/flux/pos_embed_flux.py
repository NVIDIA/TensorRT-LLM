# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX attention utilities: position embeddings and rotary encoding.

Key Components:
- FluxPosEmbed: 2D rotary position embeddings for (txt, h, w) axes
- get_1d_rotary_pos_embed: 1D rotary position embedding computation
- apply_rotary_emb: Apply rotary embeddings to tensors
- prepare_flux_image_ids / prepare_flux_text_ids: Position ID helpers
"""

from typing import List, Tuple

import torch
import torch.nn as nn


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    use_real: bool = True,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float64,
) -> "Tuple[torch.Tensor, torch.Tensor] | torch.Tensor":
    """Compute 1D rotary position embeddings.

    Args:
        dim: Embedding dimension (must be even)
        pos: Position tensor of shape (seq_len,)
        theta: RoPE theta parameter
        use_real: Return (cos, sin) instead of complex exp
        repeat_interleave_real: Interleave or repeat cos/sin
        freqs_dtype: Dtype for frequency computation

    Returns:
        If use_real=True: Tuple of (cos, sin) tensors each of shape (seq_len, dim)
        If use_real=False: Single complex tensor of shape (seq_len, dim/2)
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    # Compute frequency bands
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))

    # Outer product: (seq_len, dim/2)
    freqs = torch.outer(pos.to(freqs_dtype), freqs)

    if use_real:
        if repeat_interleave_real:
            # Repeat each frequency: [f0, f0, f1, f1, ...]
            freqs = freqs.repeat_interleave(2, dim=-1)
        else:
            # Repeat pattern: [f0, f1, ..., f0, f1, ...]
            freqs = freqs.repeat(1, 2)

        cos = freqs.cos().to(pos.dtype)
        sin = freqs.sin().to(pos.dtype)
        return cos, sin
    else:
        # Return complex exponential
        return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, seq_len, heads, dim) or (batch, seq_len, dim)
        freqs_cis: Tuple of (cos, sin) from get_1d_rotary_pos_embed or FluxPosEmbed

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    cos, sin = freqs_cis

    # Upcast to float32 for RoPE multiply-add (matches HF diffusers).
    # BF16 accumulates significant rounding error over 57 blocks x 50 steps.
    x_fp32 = x.float()
    cos = cos.float()
    sin = sin.float()

    # Handle different input shapes
    ndim = x.ndim
    if ndim == 4:
        # (batch, seq, heads, dim)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim)
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif ndim == 3:
        # (batch, seq, dim)
        cos = cos.unsqueeze(0)  # (1, seq, dim)
        sin = sin.unsqueeze(0)

    # Rotate pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    x_rotated = torch.stack([-x_fp32[..., 1::2], x_fp32[..., 0::2]], dim=-1).flatten(-2)

    return (x_fp32 * cos + x_rotated * sin).to(x.dtype)


class FluxPosEmbed(nn.Module):
    """2D Rotary Position Embedding for FLUX.

    FLUX uses 3-axis RoPE encoding with dimensions:
    - txt_dim (16): For text sequence marker
    - h_dim (56): For height positions
    - w_dim (56): For width positions

    Total: 16 + 56 + 56 = 128 = attention_head_dim

    Position IDs format:
    - txt_ids: (seq_len, 3) with zeros (text has no spatial position)
    - img_ids: (seq_len, 3) with [0, h_pos, w_pos] for each patch

    The concatenated IDs (txt_ids + img_ids) are passed to forward().
    """

    def __init__(self, theta: int = 10000, axes_dim: List[int] = None):
        """Initialize FluxPosEmbed.

        Args:
            theta: Base for exponential frequency computation
            axes_dim: Dimensions for each axis [txt_dim, h_dim, w_dim]
                     Default: [16, 56, 56] which sums to 128
        """
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim if axes_dim is not None else [16, 56, 56]

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings from position IDs.

        Args:
            ids: Position IDs tensor of shape (seq_len, 3)
                 Column 0: text marker (0 for text, 0 for image)
                 Column 1: height position
                 Column 2: width position

        Returns:
            Tuple of (freqs_cos, freqs_sin), each of shape (seq_len, head_dim)
        """
        n_axes = ids.shape[-1]
        assert n_axes == len(self.axes_dim), (
            f"ids has {n_axes} axes but axes_dim has {len(self.axes_dim)} entries"
        )
        cos_out = []
        sin_out = []
        pos = ids.float()

        # Determine frequency dtype based on device
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        # Compute RoPE for each axis
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        # Concatenate along dimension axis
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)

        return freqs_cos, freqs_sin


def prepare_flux_image_ids(
    height: int,
    width: int,
    patch_size: int = 2,
    vae_scale_factor: int = 8,
    device: torch.device = None,
) -> torch.Tensor:
    """Prepare position IDs for image latents in FLUX.

    FLUX packs 2x2 patches, so the effective grid is (height/16, width/16).

    Args:
        height: Image height in pixels
        width: Image width in pixels
        patch_size: Packing patch size (default: 2)
        vae_scale_factor: VAE spatial downsampling factor (default: 8)
        device: Target device

    Returns:
        Position IDs tensor of shape (num_patches, 3) with columns [0, h_pos, w_pos]
    """
    # Compute latent dimensions after VAE and packing
    latent_h = height // (vae_scale_factor * patch_size)
    latent_w = width // (vae_scale_factor * patch_size)

    # Create position grid
    h_pos = torch.arange(latent_h, device=device)
    w_pos = torch.arange(latent_w, device=device)

    # Create meshgrid
    h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing="ij")

    # Flatten and stack: (num_patches, 3)
    # Column 0: text marker (0 for images)
    # Column 1: height position
    # Column 2: width position
    img_ids = torch.zeros(latent_h * latent_w, 3, device=device)
    img_ids[:, 1] = h_grid.flatten()
    img_ids[:, 2] = w_grid.flatten()

    return img_ids


def prepare_flux_text_ids(
    seq_len: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Prepare position IDs for text tokens in FLUX.

    Text tokens have no spatial position, so all IDs are zeros.

    Args:
        seq_len: Text sequence length
        device: Target device

    Returns:
        Position IDs tensor of shape (seq_len, 3) with all zeros
    """
    return torch.zeros(seq_len, 3, device=device)
