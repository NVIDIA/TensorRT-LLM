"""Shared absolute-RoPE helpers and knobs (no packaging policy).
Packaging lives with the consumer:
  - det → ``det_attn_rope`` (opaque ``custom_op``)
  - Combined diff → ``combined.attn`` (nested)
  - Chunked diff → ``chunked.attn`` (in-slab)
"""

from __future__ import annotations

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.rope_math import (
    DEFAULT_ABS_ROPE_NUM_TILES,
    default_rope_dim_split,
    h_positions,
    rope_inv_freqs,
    rot_abs_axis_impl,
    t_positions,
)

# Re-export math for existing call sites / tests.
__all__ = [
    "DEFAULT_ABS_ROPE_NUM_TILES",
    "apply_abs_rope",
    "apply_abs_rope_slab",
    "configure_abs_rope",
    "default_rope_dim_split",
    "rope_inv_freqs",
]


def apply_abs_rope_slab(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    w_pos: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rotate this W-extent (full volume or one slab). Returns a new tensor."""
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    t = x.shape[1]
    h = x.shape[2]
    xt = rot_abs_axis_impl(x[..., :d_t], t_positions(t, x.device), inv_t, axis=1, compute_dtype=compute_dtype)
    xh = rot_abs_axis_impl(
        x[..., d_t : d_t + d_h],
        h_positions(h, x.device),
        inv_h,
        axis=2,
        compute_dtype=compute_dtype,
    )
    xw = rot_abs_axis_impl(x[..., d_t + d_h :], w_pos, inv_w, axis=3, compute_dtype=compute_dtype)
    return torch.cat([xt, xh, xw], dim=-1)


def apply_abs_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int = DEFAULT_ABS_ROPE_NUM_TILES,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Eager fixed-``num_tiles`` W-slab abs-RoPE (test / reference helper).
    Production packaging is owned by ``det_attn_rope`` / ``diff_attn`` — not here.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"compute_dtype must be float32 or bfloat16, got {compute_dtype}")

    slabs = torch.chunk(x, num_tiles, dim=3)
    w_off = 0
    parts: list[torch.Tensor] = []
    for slab in slabs:
        w_slab = slab.shape[3]
        w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
        parts.append(
            apply_abs_rope_slab(
                slab,
                rope_split,
                inv_freqs,
                w_pos=w_pos,
                compute_dtype=compute_dtype,
            )
        )
        w_off = w_off + w_slab
    return torch.cat(parts, dim=3)


def configure_abs_rope(
    module_root: nn.Module,
    *,
    num_tiles: int = DEFAULT_ABS_ROPE_NUM_TILES,
    compute_dtype: torch.dtype = torch.float32,
) -> None:
    """Set abs-RoPE tile count / compute dtype on every ``NeighborhoodAttention3D``."""
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D  # noqa: PLC0415

    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"compute_dtype must be float32 or bfloat16, got {compute_dtype}")
    for module in module_root.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.rope_num_tiles = num_tiles
            module.rope_compute_dtype = compute_dtype
