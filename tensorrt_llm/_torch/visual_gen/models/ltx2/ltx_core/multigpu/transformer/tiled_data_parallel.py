"""Tiled Data Parallel model wrapper for the LTX transformer.
Each GPU processes one or more tiles of the patchified
``(frames, height, width)`` latent.  Tiles are assigned to ranks via
round-robin, so the number of tiles may exceed the number of GPUs.
Tiles may overlap; overlapping regions are blended with trapezoidal
masks so that seam artefacts are suppressed.  Each rank accumulates
its assigned tiles locally, then a single ``all_reduce`` synchronises
the blended output across all ranks.
Conditioning tokens (appended after the generated tokens) are filtered
per tile: only tokens whose positions overlap with the tile's spatial
extent (or that have negative time coordinates) are included.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.modality_tiling import VideoModalityTilingHelper
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.modality import Modality
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling import TileCountConfig
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tools import VideoLatentTools

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.guidance.perturbations import BatchedPerturbationConfig


class TiledDataParallelModelWrapper(torch.nn.Module):
    """Wraps an ``X0Model`` for tiled data parallelism.
    Tiles are distributed across ranks via round-robin, allowing more
    tiles than GPUs (e.g. 16 tiles on 4 GPUs = 4 tiles per rank).
    Each rank processes its assigned tiles sequentially, blending each
    into a full-size accumulator.  A single ``all_reduce(SUM)`` after
    all local tiles produces the final result (blend masks sum to 1
    globally across all tiles).
    Audio is processed untiled on every tile forward; the outputs are
    summed via ``all_reduce`` and divided by the total tile count so
    that all ranks stay in sync.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        video_tools: VideoLatentTools,
        tiling: TileCountConfig,
        group: dist.ProcessGroup,
        normalize_positions: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.group = group
        self.world_size = dist.get_world_size(group)
        self._normalize_positions = normalize_positions
        self._helper = VideoModalityTilingHelper(tiling, video_tools)
        all_tiles = self._helper.tiles
        rank = dist.get_rank(group)
        self._tiles = [t for i, t in enumerate(all_tiles) if i % self.world_size == rank]

    @property
    def num_blocks(self) -> int:
        return self.model.num_blocks

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if video is None:
            return self.model(video, audio, perturbations)

        # Each rank processes its assigned tiles and accumulates locally.
        denoised_video: torch.Tensor | None = None
        denoised_audio: torch.Tensor | None = None
        for tile in self._tiles:
            tiled_video, ctx = self._helper.tile_modality(video, tile, normalize_positions=self._normalize_positions)
            tile_out, audio_out = self.model(tiled_video, audio, perturbations)
            blended = self._helper.blend(tile_out, tile, ctx)
            denoised_video = blended if denoised_video is None else denoised_video + blended
            if audio_out is not None:
                denoised_audio = audio_out if denoised_audio is None else denoised_audio + audio_out

        assert denoised_video is not None

        # All-reduce: sum blended tiles across ranks (masks sum to 1 globally).
        denoised_video = denoised_video.contiguous()
        dist.all_reduce(denoised_video, op=dist.ReduceOp.SUM, group=self.group)

        # Average audio across all tile forwards (each saw different video context).
        if denoised_audio is not None:
            total_tiles = len(self._helper.tiles)
            denoised_audio = denoised_audio.contiguous()
            dist.all_reduce(denoised_audio, op=dist.ReduceOp.SUM, group=self.group)
            denoised_audio = denoised_audio / total_tiles

        return denoised_video, denoised_audio
