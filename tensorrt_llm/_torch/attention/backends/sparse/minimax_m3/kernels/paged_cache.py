# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged-cache writes shared by the MiniMax-M3 kernels and backends."""

from __future__ import annotations

from typing import Literal

import torch

# fp8_e4m3_saturate: E4M3 casts saturated to +-448, the bits the CUDA FP8 writers produce (__NV_SATFINITE /
# cvt.rn.satfinite); NaN stays NaN. Where torch's own E4M3 cast already saturates (c10's float8_e4m3fn conversion,
# shared by its CPU and CUDA kernels) the cast alone is used: an explicit clamp would add a kernel per cast for the same
# bytes. A torch whose cast does not saturate gets the clamp.
# Kill: FP8_SATURATE = False (the plain cast, whatever it does).
# SHARED_FP8_SATURATE: whether the E4M3 cache writes below and the dense-decode query cast use it. Off unless a model
# opts in (RuntimeHooks.fp8_saturating_casts, armed by the executor through arm_fp8_saturation): the plain cast.
FP8_SATURATE = True
SHARED_FP8_SATURATE = False
_E4M3_MAX = 448.0


def _cast_saturates() -> bool:
    """Whether this torch's E4M3 cast maps every out-of-range value, infinities included, to +-448."""
    probe = torch.tensor([449.0, 463.0, 464.0, 1e4, float("inf"), -464.0, -float("inf")])
    want = torch.tensor([448.0, 448.0, 448.0, 448.0, 448.0, -448.0, -448.0])
    return torch.equal(probe.to(torch.float8_e4m3fn).float(), want)


# Probed once at import on the CPU, so no CUDA work runs here (nor later inside a graph capture).
_CAST_SATURATES = _cast_saturates()


def fp8_e4m3_saturate(values: torch.Tensor) -> torch.Tensor:
    """``values`` as float8_e4m3fn, saturated to +-448 (bitwise the plain cast for every finite in-range value).
    Values already in float8_e4m3fn are returned as they are (they are finite by construction or already NaN)."""
    if values.dtype == torch.float8_e4m3fn:
        return values
    if FP8_SATURATE and not _CAST_SATURATES:
        values = values.clamp(-_E4M3_MAX, _E4M3_MAX)
    return values.to(torch.float8_e4m3fn)


def arm_fp8_saturation() -> None:
    """Saturate the E4M3 cache writes and the dense-decode query cast in this process
    (RuntimeHooks.fp8_saturating_casts)."""
    global SHARED_FP8_SATURATE
    SHARED_FP8_SATURATE = True


def to_e4m3(values: torch.Tensor) -> torch.Tensor:
    """``values`` as float8_e4m3fn: saturated when armed (SHARED_FP8_SATURATE), else the plain cast."""
    return fp8_e4m3_saturate(values) if SHARED_FP8_SATURATE else values.to(torch.float8_e4m3fn)


def _to_cache_dtype(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return to_e4m3(values) if dtype == torch.float8_e4m3fn else values.to(dtype)


def write_kv_slots(
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    values: torch.Tensor,
    num_live_tokens: int,
    *,
    layout: Literal["NHD", "HND"] = "NHD",
) -> None:
    """Write per-token values into a K, V, or index-K cache at given slots.

    Handles a 3-D flat-slot cache and a 4-D paged view. `layout` sets the paged
    axis order: "NHD" is [num_pages, tokens_per_block, num_heads, channel],
    "HND" is [num_pages, num_heads, tokens_per_block, channel]. The paged view
    is non-contiguous, so the slot id is split into (page, within) and written
    by multi-dim assignment. `values` is always [num_tokens, num_heads, channel].

    Callers must provide valid slots for every live token. The production M3
    mapping satisfies this contract: ``get_block_ids_per_seq`` canonicalizes
    padded ``BAD_PAGE_INDEX`` entries before ``build_paged_kv_slot_mapping``
    selects only the allocated live-token positions.

    `num_live_tokens` is how many leading rows own a real cache slot; the rest
    are dropped. It carries no default because a caller passing the padded
    token extent of a piecewise CUDA graph corrupts the cache silently: torch
    wraps a negative index, so the -1 sentinel past the live count lands in the
    last page instead of raising.
    """
    # Trimming by count keeps this sync-free, the sentinel tail being contiguous
    # by construction, where masking on the slot values would not.
    if num_live_tokens < 0:
        raise ValueError(f"num_live_tokens must be non-negative, got {num_live_tokens}")
    if out_cache_loc.shape[0] < num_live_tokens or values.shape[0] < num_live_tokens:
        raise ValueError(
            f"num_live_tokens={num_live_tokens} exceeds the rows supplied "
            f"(out_cache_loc={out_cache_loc.shape[0]}, values={values.shape[0]})"
        )
    if num_live_tokens == 0:
        return
    out_cache_loc = out_cache_loc[:num_live_tokens]
    values = values[:num_live_tokens]

    with torch.no_grad():
        if cache.ndim >= 4:
            token_axis = 2 if layout == "HND" else 1
            tokens_per_block = int(cache.shape[token_axis])
            out_long = out_cache_loc.to(torch.long)
            page = out_long // tokens_per_block
            within = out_long % tokens_per_block
            if layout == "HND":
                # Advanced indices on dims 0 and 2 broadcast to [num_tokens] and
                # move front, giving a [num_tokens, num_heads, channel] target.
                cache[page, :, within, :] = _to_cache_dtype(values, cache.dtype)
            else:
                cache[page, within] = _to_cache_dtype(values, cache.dtype)
        else:
            cache.index_copy_(0, out_cache_loc.to(torch.long), _to_cache_dtype(values, cache.dtype))


__all__ = ["write_kv_slots"]
