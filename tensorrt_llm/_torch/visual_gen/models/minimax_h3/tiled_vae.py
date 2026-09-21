# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MiniMax-H3 spatial tiling with lossless distribution of decoder tiles.

Temporal chunking, spatial overlap geometry, and blending remain owned by
Diffusers. Only independent spatial decoder calls are distributed.
"""

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLMiniMaxH3

MINIMAX_H3_VAE_DEFAULTS = {
    "vae_use_tiling": True,
    "vae_tile_parallel": False,
    "vae_tile_size": 256,
    "vae_tile_overlap": 64,
}


def validate_vae_tiling_config(options: dict) -> None:
    """Reject invalid tile geometry before loading checkpoint weights."""
    for name in ("vae_use_tiling", "vae_tile_parallel"):
        if type(options[name]) is not bool:
            raise ValueError(f"{name} must be a boolean.")
    for name in ("vae_tile_size", "vae_tile_overlap"):
        if type(options[name]) is not int or options[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if options["vae_tile_overlap"] >= options["vae_tile_size"]:
        raise ValueError("vae_tile_overlap must be smaller than vae_tile_size.")
    if options["vae_tile_parallel"] and not options["vae_use_tiling"]:
        raise ValueError("vae_tile_parallel requires vae_use_tiling.")


class TiledAutoencoderKLMiniMaxH3(AutoencoderKLMiniMaxH3):
    """Distribute spatial decode tiles over an explicitly supplied process group.

    Every group member must decode the same latents. Each rank retains the
    complete output, as required by the H3 pipeline. Small canvases with fewer
    tiles than ranks fall back to Diffusers without entering any collectives.
    Encoding retains the reference implementation and RNG behavior.
    """

    tile_parallel_group: dist.ProcessGroup | None = None

    def configure_tiling(self, options: dict, group: dist.ProcessGroup | None = None) -> None:
        """Apply validated pipeline options using checkpoint compression geometry."""
        validate_vae_tiling_config(options)
        for name in ("vae_tile_size", "vae_tile_overlap"):
            if options[name] % self.spatial_compression_ratio:
                raise ValueError(
                    f"{name} must be divisible by the VAE spatial compression ratio "
                    f"({self.spatial_compression_ratio})."
                )
        self.use_tiling = options["vae_use_tiling"]
        self.tile_sample_min_height = self.tile_sample_min_width = options["vae_tile_size"]
        self.tile_sample_min_overlap_height = self.tile_sample_min_overlap_width = options[
            "vae_tile_overlap"
        ]
        self.tile_parallel_group = group if options["vae_tile_parallel"] else None

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        group = self.tile_parallel_group
        if not self.use_tiling or group is None or dist.get_world_size(group) == 1:
            return super()._decode_clip(z)

        ratio = self.spatial_compression_ratio
        height, width = z.shape[-2] * ratio, z.shape[-1] * ratio
        y_starts, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_starts, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        tiles = [
            (y // ratio, h // ratio, x // ratio, w // ratio)
            for y, h in zip(y_starts, y_lengths)
            for x, w in zip(x_starts, x_lengths)
        ]
        world_size = dist.get_world_size(group)
        if len(tiles) < world_size:
            return super()._decode_clip(z)

        rank = dist.get_rank(group)
        decoded_tiles = []
        local = None
        # _split_tiles gives equal-sized tiles, including the final row/column.
        # Gather one wave at a time to bound temporary communication storage.
        for start in range(0, len(tiles), world_size):
            index = start + rank
            if index < len(tiles):
                y, h, x, w = tiles[index]
                local = self.decoder(self.post_quant_conv(z[..., y : y + h, x : x + w]))
                local = local.contiguous()
            else:
                # Every rank decoded a tile in the first wave. Idle tail ranks
                # still participate, so non-divisible tile counts cannot hang.
                local = torch.zeros_like(local)
            gathered = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local, group=group)
            decoded_tiles.extend(gathered[: min(world_size, len(tiles) - start)])

        columns = len(x_starts)
        rows = [decoded_tiles[i : i + columns] for i in range(0, len(tiles), columns)]
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)
