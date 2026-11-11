"""Video modality tiling helpers.
Provides :class:`VideoModalityTilingHelper`, which tiles and blends video
:class:`Modality` token sequences by spatial/temporal region.  Tile geometry is represented by the existing
:class:`Tile` NamedTuple from :mod:`ltx_core.tiling`; no distributed
primitives are required.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.devices import upload_from_host
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.modality import Modality
from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import (
    DimensionTilingConfig,
    SplitOperation,
    Tile,
    TileCountConfig,
    TileSizeConfig,
    create_tiles,
    identity_mapping_operation,
    split_at_seams,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.tools import VideoLatentTools
from tensorrt_llm._torch.visual_gen.models.ltx_core.types import VideoLatentShape

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TilePlan:
    """One tile's token layout for one positions tensor; device tables reused at every step.
    The layout is fixed once planned; only the per-``(dtype, device)`` weight caches fill in lazily.
    """

    keep_indices: torch.Tensor
    gen_indices: torch.Tensor
    cond_indices: torch.Tensor | None
    num_tile_gen: int
    cond_blend_weights: torch.Tensor | None
    positions: torch.Tensor
    blend_weights: dict[tuple[torch.dtype, torch.device], torch.Tensor] = field(default_factory=dict)
    cond_weights: dict[tuple[torch.dtype, torch.device], torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class TilingContext:
    """Opaque context produced by :meth:`VideoModalityTilingHelper.tile_modality`.
    Carries the token-level keep indices and per-conditioning-token blend
    weights needed by :meth:`~VideoModalityTilingHelper.blend`.
    """

    keep_indices: torch.Tensor
    """``(num_kept,)`` int64 — sorted indices of tokens the tile processes."""
    num_total_tokens: int
    """Total number of tokens in the full (untiled) sequence."""
    cond_blend_weights: torch.Tensor | None
    """``(num_kept_cond,)`` — weight for each kept conditioning token,
    equal to ``1 / num_tiles_that_keep_this_token``.  ``None`` when
    there are no conditioning tokens."""
    plan: _TilePlan


def seam_split(
    seams: Sequence[int],
    latent_frames: int,
    frames: DimensionTilingConfig,
) -> SplitOperation | None:
    """A temporal split cut on ``seams``, or ``None`` when they cannot carry one.
    ``seams`` are interior latent-frame indices supplied by the caller. A boundary there needs no
    blending: the overlap is denoised for context and dropped (:func:`~ltx_core.tiling.split_at_seams`).
    Leftover segments go to the leading tiles. Missing or non-interior seams fall back to the
    requested overlap split, which blends.
    """
    if frames.num_tiles < 2:
        return None
    interior = sorted({cell for cell in seams if 0 < cell < latent_frames - 1})
    if not interior:
        if seams:
            logger.info(
                "Temporal tiling: seams %s are not interior to %d latent frames; keeping blended tiles",
                list(seams),
                latent_frames,
            )
        return None
    boundaries = [0, *interior, latent_frames - 1]
    logger.info("Temporal tiling: %d tiles cut on seams %s", frames.num_tiles, boundaries)
    return split_at_seams(boundaries, frames.num_tiles, overlap=frames.overlap)


class VideoModalityTilingHelper:
    """Tiles and blends video :class:`Modality` sequences.
    Constructed once with a :class:`TileCountConfig` and
    :class:`VideoLatentTools`.  Tiles are computed at construction and
    available via the :attr:`tiles` property.  Use :meth:`tile_modality`
    and :meth:`blend` with any tile from that list.
    Every tile's token layout depends only on ``modality.positions``. The first
    :meth:`tile_modality` call for a positions tensor plans all tiles at once (its
    host syncs drain the GPU queue a single time, before the sequence's first
    forward) and later calls with the same tensor -- a denoising loop carries it
    unchanged across steps -- read those plans. The helper holds the tensor while
    its plans are live, so the identity check cannot see a recycled object.
    Passing ``seams`` (interior latent-frame indices) lets a temporal split land on those
    cells instead of blended overlaps (:func:`seam_split`); leftover segments go to the
    leading tiles. Missing or non-interior seams keep the blended split. A regular keyframe
    belongs in that list only at strength 0; generated keyframe slots never do.
    Usage::
        helper = VideoModalityTilingHelper(tiling, video_tools)
        for tile in helper.tiles:
            tiled_mod, ctx = helper.tile_modality(modality, tile)
            result = run_model(tiled_mod)
            helper.blend(result, tile, ctx, output=output)
    """

    def __init__(
        self,
        tiling: TileCountConfig | TileSizeConfig,
        video_tools: VideoLatentTools,
        seams: Sequence[int] = (),
    ) -> None:
        self._patchifier = video_tools.patchifier
        self._latent_shape = video_tools.target_shape
        self._num_generated_tokens = self._patchifier.get_token_count(self._latent_shape)
        frames, height, width = tiling.to_splitters(video_tools.scale_factors, causal_temporal=False)
        frames_mapper = identity_mapping_operation
        if seams and not isinstance(tiling, TileCountConfig):
            raise ValueError("Known temporal seams require count-based tiling")
        seam_op = (
            seam_split(seams, self._latent_shape.frames, tiling.frames) if isinstance(tiling, TileCountConfig) else None
        )
        if seam_op is not None:
            frames, frames_mapper = seam_op, partial(identity_mapping_operation, rectangular=True)
        self._tiles = create_tiles(
            torch.Size([self._latent_shape.frames, self._latent_shape.height, self._latent_shape.width]),
            splitters=[frames, height, width],
            mappers=[frames_mapper, identity_mapping_operation, identity_mapping_operation],
        )
        self._plan_positions: torch.Tensor | None = None
        self._plan_num_total = -1
        self._plans: dict[tuple[int, bool], _TilePlan] = {}

    @property
    def tiles(self) -> list[Tile]:
        """All tiles for the configured tiling layout."""
        return self._tiles

    # -- tile modality -----------------------------------------------------

    def tile_modality(
        self, modality: Modality, tile: Tile, *, normalize_positions: bool = True
    ) -> tuple[Modality, TilingContext]:
        """Slice *modality* to the tokens covered by *tile*.
        Selects generated tokens belonging to the tile's spatial region
        and conditioning tokens that overlap with the tile (or have
        negative time coordinates).
        Args:
            normalize_positions: When True, shift all positions so the
                tile's generated tokens start at zero in every dimension.
        Returns:
            A ``(tiled_modality, context)`` tuple.  Pass *context* to
            :meth:`blend` together with the model output.
        """
        plan = self._plan(modality, tile, normalize_positions=normalize_positions)
        keep_indices = plan.keep_indices

        tile_attention_mask = None
        if modality.attention_mask is not None:
            tile_attention_mask = modality.attention_mask[:, keep_indices, :][:, :, keep_indices]

        # Sliced, not recomputed: the marker must survive `normalize_positions`, which shifts every
        # tile's generated tokens to start at zero and so destroys the "temporal start == 0" and
        # "temporal extent == 1" signals the mask could otherwise be derived from.
        tile_keyframes_mask = None
        if modality.keyframes_mask is not None:
            tile_keyframes_mask = modality.keyframes_mask[:, keep_indices]

        tiled = replace(
            modality,
            latent=modality.latent[:, keep_indices, :],
            timesteps=modality.timesteps.select_tokens(keep_indices),
            positions=plan.positions,
            attention_mask=tile_attention_mask,
            keyframes_mask=tile_keyframes_mask,
        )

        return tiled, TilingContext(
            keep_indices=keep_indices,
            num_total_tokens=modality.latent.shape[1],
            cond_blend_weights=plan.cond_blend_weights,
            plan=plan,
        )

    # -- blend -------------------------------------------------------------

    def blend(
        self,
        tile_to_blend: torch.Tensor,
        tile: Tile,
        context: TilingContext,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Blend-weight tile results and accumulate into the full token space.
        Premultiplied (blend-weighted) data is **added** to *output*,
        allowing multiple tiles to be accumulated into the same buffer.
        Args:
            tile_to_blend: Denoised tile tensor ``(B, num_tile_tokens, D)``,
                where the first ``_tile_generated_token_count(tile)``
                entries are generated tokens and the remainder are
                conditioning tokens.
            tile: The :class:`Tile` that was used in :meth:`tile_modality`.
            context: The :class:`TilingContext` returned by :meth:`tile_modality`.
            output: Optional pre-allocated output tensor.  When provided
                its shape must be ``(B, num_total_tokens, D)`` and the
                blended tile is **added** into it.  When ``None`` a new
                zero-filled tensor is created.
        Returns:
            The output tensor with the blended tile added at the correct
            positions.
        """
        plan = context.plan
        batch, _, dim = tile_to_blend.shape
        num_total_tokens = context.num_total_tokens
        expected_shape = (batch, num_total_tokens, dim)

        if output is not None:
            if output.shape != expected_shape:
                raise RuntimeError(f"Expected output shape {expected_shape}, got {output.shape}")
            result = output
        else:
            result = torch.zeros(*expected_shape, device=tile_to_blend.device, dtype=tile_to_blend.dtype)

        # Blend mask is (tile_F, tile_H, tile_W) — one weight per token in row-major order. The dense mask is
        # built on the host only when the plan has no copy on this device and dtype yet.
        blend_weights = _device_weights(plan.blend_weights, lambda: tile.blend_mask.reshape(-1), tile_to_blend)
        result[:, plan.gen_indices, :] += tile_to_blend[:, : plan.num_tile_gen, :] * blend_weights[None, :, None]

        # Scatter kept conditioning tokens, weighted by 1/N where N is
        # the number of tiles that keep each token (so they sum to 1).
        if num_total_tokens > self._num_generated_tokens and plan.cond_blend_weights is not None:
            cond_blend_weights = plan.cond_blend_weights
            weights = _device_weights(plan.cond_weights, lambda: cond_blend_weights, tile_to_blend)
            result[:, plan.cond_indices, :] += tile_to_blend[:, plan.num_tile_gen :, :] * weights[None, :, None]

        return result

    # -- private -----------------------------------------------------------

    def _plan(self, modality: Modality, tile: Tile, *, normalize_positions: bool) -> _TilePlan:
        """*tile*'s plan for this sequence; a new positions tensor or token count plans every tile again."""
        num_total = modality.latent.shape[1]
        if modality.positions is not self._plan_positions or num_total != self._plan_num_total:
            self._plan_positions = modality.positions
            self._plan_num_total = num_total
            self._plans = {}
        tile_idx = next((i for i, t in enumerate(self._tiles) if t.in_coords == tile.in_coords), None)
        if tile_idx is None:
            raise RuntimeError(
                f"Tile with in_coords={tile.in_coords} is not in this helper's tile set; "
                f"pass a tile obtained from `helper.tiles`."
            )
        key = (tile_idx, normalize_positions)
        if key not in self._plans:
            # The one host read of a plan build: how many conditioning tokens each tile keeps is data dependent.
            keep_per_tile_cond = None
            cond_keepers = None
            if num_total > self._num_generated_tokens:
                keep_per_tile_cond = self._all_tiles_cond_keep(modality).cpu()
                cond_keepers = keep_per_tile_cond.sum(dim=0).float()  # (num_cond,)
            for idx, t in enumerate(self._tiles):
                cond_keep = keep_per_tile_cond[idx] if keep_per_tile_cond is not None else None
                self._plans[(idx, normalize_positions)] = self._build_plan(
                    modality, t, normalize_positions, cond_keep, cond_keepers
                )
        return self._plans[key]

    def _build_plan(
        self,
        modality: Modality,
        tile: Tile,
        normalize_positions: bool,
        cond_keep: torch.Tensor | None,
        cond_keepers: torch.Tensor | None,
    ) -> _TilePlan:
        """*tile*'s plan. ``cond_keep`` is the tile's ``(num_cond,)`` keep row, ``cond_keepers`` how many tiles
        keep each conditioning token; both host tensors, ``None`` without conditioning tokens."""
        device = modality.positions.device
        gen_indices = self._generated_token_indices(tile, device=device)
        cond_indices = None
        cond_blend_weights = None
        if cond_keep is not None and cond_keepers is not None:
            # The keep row is on the host: the indices and weights are derived there and uploaded
            # through pinned memory, so no step of the plan waits on the device.
            host_cond_indices = self._num_generated_tokens + cond_keep.nonzero(as_tuple=False).squeeze(1)
            cond_indices = upload_from_host(host_cond_indices, device)
            keep_indices = torch.cat([gen_indices, cond_indices])
            cond_blend_weights = 1.0 / cond_keepers[cond_keep]
        else:
            keep_indices = gen_indices

        num_tile_gen = self._tile_generated_token_count(tile)
        positions = modality.positions[:, :, keep_indices, :]
        if normalize_positions:
            gen_pos = positions[:, :, :num_tile_gen, :]  # (B, 3, num_tile_gen, 2)
            offset = gen_pos[..., 0].amin(dim=2, keepdim=True).unsqueeze(-1)  # (B, 3, 1, 1)
            positions = positions - offset
        return _TilePlan(
            keep_indices=keep_indices,
            gen_indices=gen_indices,
            cond_indices=cond_indices,
            num_tile_gen=num_tile_gen,
            cond_blend_weights=cond_blend_weights,
            positions=positions,
        )

    def _tile_generated_token_count(self, tile: Tile) -> int:
        """Number of generated tokens in *tile*."""
        frame_slice, height_slice, width_slice = tile.in_coords
        tile_shape = VideoLatentShape(
            batch=self._latent_shape.batch,
            channels=self._latent_shape.channels,
            frames=frame_slice.stop - frame_slice.start,
            height=height_slice.stop - height_slice.start,
            width=width_slice.stop - width_slice.start,
        )
        return self._patchifier.get_token_count(tile_shape)

    def _generated_token_indices(self, tile: Tile, device: torch.device | None = None) -> torch.Tensor:
        """Flat token indices of *tile*'s generated tokens in the full sequence."""
        frame_slice, height_slice, width_slice = tile.in_coords
        f = torch.arange(frame_slice.start, frame_slice.stop, device=device)
        h = torch.arange(height_slice.start, height_slice.stop, device=device)
        w = torch.arange(width_slice.start, width_slice.stop, device=device)
        return (
            f[:, None, None] * self._latent_shape.height * self._latent_shape.width
            + h[None, :, None] * self._latent_shape.width
            + w[None, None, :]
        ).reshape(-1)

    def _all_tiles_cond_keep(self, modality: Modality) -> torch.Tensor:
        """Vectorized (num_tiles, num_cond) bool: which tiles keep each conditioning token.
        A conditioning token is kept by a tile when its ``[start, end)`` interval
        overlaps the tile in all three dimensions, or when it has a negative time
        coordinate (reference token).
        """
        cond_positions = modality.positions[:, :, self._num_generated_tokens :, :]  # (B, 3, num_cond, 2)
        device = cond_positions.device

        # Per-tile (start, end) bounds along each axis; small Python loop (num_tiles <= ~16).
        starts_list: list[torch.Tensor] = []
        ends_list: list[torch.Tensor] = []
        for t in self._tiles:
            gen_idx = self._generated_token_indices(t, device=device)
            gen_positions = modality.positions[:, :, gen_idx, :]  # (B, 3, num_tile_gen, 2)
            starts_list.append(gen_positions[..., 0].amin(dim=2))  # (B, 3)
            ends_list.append(gen_positions[..., 1].amax(dim=2))  # (B, 3)
        tile_starts = torch.stack(starts_list, dim=0)  # (num_tiles, B, 3)
        tile_ends = torch.stack(ends_list, dim=0)  # (num_tiles, B, 3)

        cond_starts = cond_positions[..., 0]  # (B, 3, num_cond)
        cond_ends = cond_positions[..., 1]  # (B, 3, num_cond)
        # Broadcast: (1, B, 3, num_cond) vs (num_tiles, B, 3, 1) -> (num_tiles, B, 3, num_cond).
        overlaps = (cond_starts[None] < tile_ends[..., None]) & (cond_ends[None] > tile_starts[..., None])
        overlaps_all_dims = overlaps.all(dim=2)  # (num_tiles, B, num_cond)
        has_negative_time = (cond_positions[:, 0, :, 0] < 0)[None]  # (1, B, num_cond)
        return (overlaps_all_dims | has_negative_time).any(dim=1)  # (num_tiles, num_cond)


def _device_weights(
    cache: dict[tuple[torch.dtype, torch.device], torch.Tensor],
    weights: Callable[[], torch.Tensor],
    like: torch.Tensor,
) -> torch.Tensor:
    """``weights()`` on ``like``'s device and dtype, built and uploaded once per plan through pinned memory."""
    key = (like.dtype, like.device)
    if key not in cache:
        host = weights().to(dtype=like.dtype)
        if like.device.type == "cuda" and host.device.type == "cpu":
            host = host.pin_memory()
        cache[key] = host.to(device=like.device, non_blocking=True)
    return cache[key]
