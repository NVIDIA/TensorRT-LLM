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
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
) -> torch.Tensor:
    if rope_type == LTXRopeType.INTERLEAVED:
        # Note: INTERLEAVED rope is a legacy mode. Prefer SPLIT instead.
        return apply_interleaved_rotary_emb(input_tensor, *freqs_cis)
    elif rope_type == LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, *freqs_cis)
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")


def apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

    return out


def apply_split_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    if sin_freqs.shape != cos_freqs.shape:
        raise ValueError(
            f"apply_split_rotary_emb: sin_freqs.shape {tuple(sin_freqs.shape)} must equal "
            f"cos_freqs.shape {tuple(cos_freqs.shape)}."
        )
    needs_reshape = input_tensor.ndim != 4 and cos_freqs.ndim == 4
    if needs_reshape:
        b_freq = cos_freqs.shape[0]
        h = cos_freqs.shape[1]
        b_in = input_tensor.shape[0]
        if b_freq not in (1, b_in):
            raise ValueError(
                f"apply_split_rotary_emb: cos_freqs batch ({b_freq}) must be 1 "
                f"(broadcast) or equal input_tensor batch ({b_in})."
            )
        # `unflatten` only touches the last dim, keeping the batch and seq dims as
        # the input tensor's own symbolic ints under torch.compile. `reshape(b_in,
        # t, h, -1)` would have forced Dynamo to specialise those dims because it
        # cannot prove `b_in == cos_freqs.shape[0]` and `seq == t` across tensors.
        input_tensor = input_tensor.unflatten(-1, (h, -1)).transpose(1, 2)

    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]

    output = split_input * cos_freqs.unsqueeze(-2)
    first_half_output = output[..., :1, :].addcmul(-sin_freqs.unsqueeze(-2), second_half_input)
    second_half_output = output[..., 1:, :].addcmul(sin_freqs.unsqueeze(-2), first_half_input)
    output = torch.cat([first_half_output, second_half_output], dim=-2)

    output = rearrange(output, "... d r -> ... (d r)")
    if needs_reshape:
        # `transpose(1, 2).flatten(-2)` keeps the batch and seq dims symbolic; using
        # `reshape(b_in, t, -1)` would force Dynamo to specialise both axes.
        output = output.transpose(1, 2).flatten(-2)

    return output


@functools.lru_cache(maxsize=5)
def generate_freq_grid_np(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
    device: torch.device | None = None,
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
    # as_tensor (not tensor): under torch.compile the numpy ops above are traced as tensors,
    # so torch.tensor() would copy-construct from a tensor (warns); as_tensor casts in place.
    return torch.as_tensor(pow_indices * math.pi / 2, dtype=torch.float32).to(device=device)


@functools.lru_cache(maxsize=5)
def generate_freq_grid_pytorch(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
    device: torch.device | None = None,
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
    indices = indices.to(dtype=torch.float32)

    indices = indices * math.pi / 2

    return indices.to(device=device)


def get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int]) -> torch.Tensor:
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )
    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        dim=-1,
    )
    return fractional_positions


def generate_freqs(
    indices: torch.Tensor, indices_grid: torch.Tensor, max_pos: list[int], use_middle_indices_grid: bool
) -> torch.Tensor:
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)

    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def split_freqs_cis(freqs: torch.Tensor, pad_size: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape freqs to be compatible with multi-head attention
    b = cos_freq.shape[0]
    t = cos_freq.shape[1]

    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1)

    cos_freq = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)
    sin_freq = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)
    return cos_freq, sin_freq


def interleaved_freqs_cis(freqs: torch.Tensor, pad_size: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
    freq_grid_generator: Callable[[float, int, int, torch.device], torch.Tensor] = generate_freq_grid_pytorch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    # The grids are computed on the CPU and cached per device: a pageable host-to-device copy blocks
    # the host until the device drains, so repeating it every call keeps the host from running ahead.
    indices = freq_grid_generator(theta, indices_grid.shape[1], dim, indices_grid.device)
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        # 2 because of cos and sin by 3 for (t, x, y), 1 for temporal only
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)
