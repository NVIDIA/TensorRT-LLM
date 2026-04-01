# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

# LTX-2 specific 3D RoPE with interleaved and split variants,
# fractional position normalization, and middle-indices-grid support.

import functools
import math
from enum import Enum
from typing import Callable, Tuple

import numpy as np
import torch
from einops import rearrange


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> torch.Tensor:
    if rope_type == LTXRopeType.INTERLEAVED:
        return _apply_interleaved_rotary_emb(input_tensor, *freqs_cis)
    elif rope_type == LTXRopeType.SPLIT:
        return _apply_split_rotary_emb(input_tensor, *freqs_cis)
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")


def _apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")
    return input_tensor * cos_freqs + input_tensor_rot * sin_freqs


def _apply_split_rotary_emb(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    needs_reshape = False
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        _, h, t, _ = cos_freqs.shape
        b = input_tensor.shape[0]
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]

    output = split_input * cos_freqs.unsqueeze(-2)
    first_half_output = output[..., :1, :]
    second_half_output = output[..., 1:, :]

    first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)
    second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)

    output = rearrange(output, "... d r -> ... (d r)")
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)

    return output


@functools.lru_cache(maxsize=5)
def _generate_freq_grid_np(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


@functools.lru_cache(maxsize=5)
def _generate_freq_grid_pytorch(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count
    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
        )
    )
    return indices.to(dtype=torch.float32) * (math.pi / 2)



# Cache for device-transferred frequency grids to avoid CPU->GPU during CUDA graph capture
_freq_grid_device_cache: dict = {}

def _get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int]) -> torch.Tensor:
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )
    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        dim=-1,
    )
    return fractional_positions


def _generate_freqs(
    indices: torch.Tensor,
    indices_grid: torch.Tensor,
    max_pos: list[int],
    use_middle_indices_grid: bool,
) -> torch.Tensor:
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start = indices_grid[..., 0]
        indices_grid_end = indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = _get_fractional_positions(indices_grid, max_pos)
    target_device = fractional_positions.device
    if not (indices.device == target_device):
        _cache_key = (indices.data_ptr(), str(target_device))
        if _cache_key not in _freq_grid_device_cache:
            _freq_grid_device_cache[_cache_key] = indices.to(device=target_device)
        indices = _freq_grid_device_cache[_cache_key]
    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def _split_freqs_cis(
    freqs: torch.Tensor, pad_size: int, num_attention_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

    b, t = cos_freq.shape[0], cos_freq.shape[1]
    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1).swapaxes(1, 2)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1).swapaxes(1, 2)
    return cos_freq, sin_freq


def _interleaved_freqs_cis(freqs: torch.Tensor, pad_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    freq_grid_generator: Callable[[float, int, int], torch.Tensor] = _generate_freq_grid_pytorch,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin RoPE embeddings for the given position indices.

    Args:
        indices_grid: Position indices [B, n_dims, T] or [B, n_dims, T, 2].
        dim: Inner dimension (num_heads * head_dim).
        out_dtype: Output dtype for the embeddings.
        theta: RoPE base frequency.
        max_pos: Maximum positions per dimension for fractional normalization.
        use_middle_indices_grid: If True, use midpoint of start/end indices.
        num_attention_heads: Number of attention heads (for split mode reshaping).
        rope_type: INTERLEAVED or SPLIT.
        freq_grid_generator: Function to generate the frequency grid.

    Returns:
        Tuple of (cos_freq, sin_freq) tensors.
    """
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    indices = freq_grid_generator(theta, indices_grid.shape[1], dim)
    freqs = _generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = _split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = _interleaved_freqs_cis(freqs, dim % n_elem)

    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)
