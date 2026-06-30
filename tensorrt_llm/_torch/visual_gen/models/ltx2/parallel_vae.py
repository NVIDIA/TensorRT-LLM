# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

"""Tile-parallel VAE decode for LTX-2.

``VideoDecoder.tiled_decode`` decodes a latent as ~27 overlapping tiles, each an
independent forward, then blends overlaps by a weighted sum
(``out = Σ_tiles decode·mask / Σ_tiles mask``). The tiles are embarrassingly
parallel; only the final blend couples them. ``tile_parallel_decode`` distributes
the tiles across the VAE process group (LPT by volume), accumulates each rank's
subset into a full-size buffer, then ``all_reduce`` the numerator + denominator so
every rank ends up with the full decoded video — numerically equal to single-GPU
``tiled_decode`` up to bf16 summation re-association (overlap regions only).

The decode is deterministic for the production LTX-2 VAE (``timestep_conditioning``
and per-block ``inject_noise`` are both False in the checkpoint), so the threaded
``generator`` is never consumed and per-rank tile subsets cannot diverge.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from .ltx2_core.types import VideoLatentShape

if TYPE_CHECKING:
    from .ltx2_core.video_vae.video_vae import TilingConfig, VideoDecoder


def _tile_in_volume(tile) -> int:
    """Input-latent voxel count (T×H×W) — proxy for this tile's decode cost."""
    c = tile.in_coords
    return (c[2].stop - c[2].start) * (c[3].stop - c[3].start) * (c[4].stop - c[4].start)


def assign_tiles_lpt(tiles: list, world: int, rank: int) -> list:
    """LPT (longest-processing-time) greedy assignment by input volume.

    Deterministic and identical on every rank (same sorted order, same
    least-loaded tie-break), so each tile is owned by exactly one rank with no
    cross-rank coordination — a hard requirement for the blend sum to be correct
    (no tile decoded twice or skipped). Balances by work (volume), not tile
    count, since boundary tiles are smaller than interior ones.
    """
    load = [0] * world
    mine = []
    for t in sorted(tiles, key=lambda t: -_tile_in_volume(t)):
        r = min(range(world), key=lambda i: load[i])
        load[r] += _tile_in_volume(t)
        if r == rank:
            mine.append(t)
    return mine


def tile_parallel_decode(
    video_decoder: "VideoDecoder",
    latent: torch.Tensor,
    tiling_config: "TilingConfig",
    pg: "dist.ProcessGroup",
    timestep: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Distributed equivalent of ``video_decoder.tiled_decode`` over group ``pg``.

    Every rank in ``pg`` must hold the full ``latent`` (replicated) and call this
    collectively. Returns the full decoded video on every rank.
    """
    rank = dist.get_rank(pg)
    world = dist.get_world_size(pg)

    tiles = video_decoder._prepare_tiles(latent, tiling_config)
    mine = assign_tiles_lpt(tiles, world, rank)

    out_shape = (
        VideoLatentShape.from_torch_shape(latent.shape)
        .upscale(video_decoder.video_downscale_factors)
        .to_torch_shape()
    )
    buf = torch.zeros(out_shape, device=latent.device, dtype=latent.dtype)
    wts = torch.zeros(out_shape, device=latent.device, dtype=latent.dtype)

    for tile in mine:
        decoded = video_decoder.forward(latent[tile.in_coords], timestep, generator)
        mask = tile.blend_mask(latent.device, latent.dtype)
        oc = tile.out_coords
        tlen = min(oc[2].stop - oc[2].start, decoded.shape[2], buf.shape[2] - oc[2].start)
        mask_slice = mask[:, :, :tlen, :, :] if mask.shape[2] > 1 else mask
        out_t = slice(oc[2].start, oc[2].start + tlen)
        buf[:, :, out_t, oc[3], oc[4]] += decoded[:, :, :tlen, :, :] * mask_slice
        wts[:, :, out_t, oc[3], oc[4]] += mask_slice

    dist.all_reduce(buf, group=pg)
    dist.all_reduce(wts, group=pg)
    return buf / wts.clamp(min=1e-8)
