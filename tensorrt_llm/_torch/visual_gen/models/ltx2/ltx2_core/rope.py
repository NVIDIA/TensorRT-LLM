# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Translate upstream RoPE to TRTLLM's token-major fused-kernel layout."""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer import rope as upstream
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.rope import (
    LTXRopeType as LTXRopeType,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.rope import (
    generate_freq_grid_pytorch as _generate_freq_grid_pytorch,
)


def _token_major(freqs):
    return tuple(torch.cat((f.transpose(1, 2), f.transpose(1, 2)), dim=-1) for f in freqs)


def _split_freqs_cis(freqs, pad_size, num_attention_heads):
    return _token_major(upstream.split_freqs_cis(freqs, pad_size, num_attention_heads))


def precompute_freqs_cis(
    indices_grid,
    dim,
    out_dtype,
    theta=10000.0,
    max_pos=None,
    use_middle_indices_grid=False,
    num_attention_heads=32,
    rope_type=LTXRopeType.INTERLEAVED,
    freq_grid_generator=_generate_freq_grid_pytorch,
    freq_grid_cache=None,
):
    # Current upstream caches grids per device; no separate TRTLLM cache is needed.
    freqs = upstream.precompute_freqs_cis(
        indices_grid,
        dim,
        out_dtype,
        theta,
        max_pos,
        use_middle_indices_grid,
        num_attention_heads,
        rope_type,
        freq_grid_generator,
    )
    return _token_major(freqs) if rope_type == LTXRopeType.SPLIT else freqs


def _apply_split_rotary_emb(input_tensor, cos_freqs, sin_freqs):
    original_shape = input_tensor.shape
    heads = cos_freqs.shape[2]
    x = input_tensor.unflatten(-1, (heads, -1)) if input_tensor.ndim != 4 else input_tensor
    half = x.shape[-1] // 2
    result = upstream.apply_split_rotary_emb(
        x.transpose(1, 2),
        cos_freqs[..., :half].transpose(1, 2),
        sin_freqs[..., :half].transpose(1, 2),
    )
    return result.transpose(1, 2).reshape(original_shape)


def apply_rotary_emb(input_tensor, freqs_cis, rope_type=LTXRopeType.INTERLEAVED):
    if rope_type == LTXRopeType.SPLIT:
        return _apply_split_rotary_emb(input_tensor, *freqs_cis)
    return upstream.apply_rotary_emb(input_tensor, freqs_cis, rope_type)


# Explicit compatibility exports also used by external transformer/test imports.
_generate_freq_grid_np = upstream.generate_freq_grid_np
_get_fractional_positions = upstream.get_fractional_positions
_generate_freqs = upstream.generate_freqs
_interleaved_freqs_cis = upstream.interleaved_freqs_cis
_apply_interleaved_rotary_emb = upstream.apply_interleaved_rotary_emb
