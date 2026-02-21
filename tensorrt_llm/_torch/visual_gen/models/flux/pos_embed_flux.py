# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX position embedding utilities.

Key Components:
- FluxPosEmbed: Multi-axis rotary position embeddings (FLUX.1: 3-axis, FLUX.2: 4-axis)
- get_1d_rotary_pos_embed: 1D rotary position embedding computation

RoPE application uses the shared apply_rotary_emb from modules/attention.py.
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


class FluxPosEmbed(nn.Module):
    """Multi-axis Rotary Position Embedding for FLUX models.

    Computes RoPE for each axis independently and concatenates the results.
    Parameterized by axes_dim to support different FLUX variants:
    - FLUX.1: axes_dim=[16, 56, 56] (3-axis: txt, h, w), theta=10000
    - FLUX.2: axes_dim=[32, 32, 32, 32] (4-axis: t, h, w, l), theta=2000

    Total dimension = sum(axes_dim) = attention_head_dim (128 for both).
    """

    def __init__(self, theta: int = 10000, axes_dim: List[int] = None):
        """Initialize FluxPosEmbed.

        Args:
            theta: Base for exponential frequency computation
            axes_dim: Dimensions for each axis. Default: [16, 56, 56] (FLUX.1)
        """
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim if axes_dim is not None else [16, 56, 56]

    @torch.compiler.disable
    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings from position IDs.

        Args:
            ids: Position IDs tensor of shape (seq_len, n_axes)

        Returns:
            Tuple of (freqs_cos, freqs_sin), each of shape (1, seq_len, 1, head_dim)
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

        # Concatenate along dimension axis and reshape to [1, S, 1, D]
        # to match WAN's format expected by the shared apply_rotary_emb()
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
        freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

        return freqs_cos, freqs_sin
