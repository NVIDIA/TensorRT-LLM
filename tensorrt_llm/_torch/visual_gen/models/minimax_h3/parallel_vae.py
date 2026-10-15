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

"""MiniMax-H3 spatial tiling with lossless batching and distribution of decoder tiles.

Temporal chunking, spatial overlap geometry, and blending remain owned by
Diffusers. Only independent spatial decoder calls are batched or distributed.
"""

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLMiniMaxH3

from tensorrt_llm.logger import logger

# Share of the memory available at decode time that the tiles of one decoder call may occupy.
TILE_DECODE_MEMORY_FRACTION = 0.75


def available_device_bytes(device: torch.device) -> int:
    """Memory a decoder call can grow into: free on the device plus idle blocks the allocator holds."""
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


class TiledAutoencoderKLMiniMaxH3(AutoencoderKLMiniMaxH3):
    """Decode spatial tiles in batches on one rank, or distributed over a process group.

    Single rank (no group, a group of one, or fewer tiles than ranks): equal-size tiles are
    decoded several per decoder call. The ViT decoder treats batch entries independently
    (per-sample attention, per-token norms and projections), so the values match the
    reference loop over tiles. The tiles per call are bounded by
    ``max_tiles_per_decoder_call`` and, once ``calibrate_tile_decode_memory`` has measured one
    tile, by the memory available when the call is made.

    Distributed: every group member must decode the same latents and each rank decodes every
    ``world_size``-th tile, batched the same way, then the group gathers the batches. Each
    rank retains the complete output. The pipeline calls decode only on group ranks. Encoding
    retains the reference implementation and RNG behavior.
    """

    tile_parallel_group: dist.ProcessGroup | None = None
    # Decoder activations grow linearly with the tiles per call; this bounds them
    # independently of the free memory.
    max_tiles_per_decoder_call: int = 16
    # Peak decoder bytes per latent element of a tile, from calibrate_tile_decode_memory.
    _decode_bytes_per_latent: float | None = None

    def configure_parallel(self, group: dist.ProcessGroup | None = None) -> None:
        """Enable tile parallelism using the loaded VAE's spatial geometry."""
        if group is None:
            self.tile_parallel_group = None
            return
        for size, overlap in (
            (self.tile_sample_min_height, self.tile_sample_min_overlap_height),
            (self.tile_sample_min_width, self.tile_sample_min_overlap_width),
        ):
            if size <= 0 or overlap <= 0 or overlap >= size:
                raise ValueError("VAE tile overlap must be positive and smaller than tile size.")
            if size % self.spatial_compression_ratio or overlap % self.spatial_compression_ratio:
                raise ValueError("VAE tile geometry must align to the spatial compression ratio.")
        self.use_tiling = True
        self.tile_parallel_group = group

    @torch.inference_mode()
    def calibrate_tile_decode_memory(self) -> None:
        """Measure the decoder's peak memory for one tile of this VAE's tile geometry.

        Decodes one zero tile of ``tile_sample_min_height x tile_sample_min_width`` pixels over
        one temporal clip under the fp16 autocast the pipeline decodes with, and keeps the peak
        bytes per latent element, input included. The decoder runs eagerly here, before any
        ``torch.compile``, so the figure is an upper bound for the compiled decoder. Resets the
        device's peak-memory statistics, so call it once after loading rather than while a
        request is being accounted. Without calibration only ``max_tiles_per_decoder_call``
        bounds a batch. Does nothing off the GPU.
        """
        device = next(self.decoder.parameters()).device
        if device.type != "cuda":
            return
        ratio = self.spatial_compression_ratio
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        tile = torch.zeros(
            (
                1,
                self.config.latent_channels,
                self.tokens_chunk_size + self.token_overlap,
                self.tile_sample_min_height // ratio,
                self.tile_sample_min_width // ratio,
            ),
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            self.decoder(self.post_quant_conv(tile))
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - allocated
        self._decode_bytes_per_latent = peak / tile.numel()
        logger.info(
            f"MiniMax-H3 VAE decode: one {tuple(tile.shape[2:])} latent tile peaks at "
            f"{peak / 2**20:.0f} MiB, {available_device_bytes(device) / 2**30:.1f} GiB available "
            f"at load; up to {self.max_tiles_per_decoder_call} tiles per call."
        )

    def _tiles_per_decoder_call(self, tile: torch.Tensor, remaining: int) -> int:
        """Tiles for the next decoder call: the upper bound, cut to what the available memory holds.

        The count varies with the available memory and on the last chunk, so the decoder's
        batch dimension is dynamic; the compiled decoder blocks must not pin it. The fraction
        of the memory left unused covers the gathered output tiles of the distributed path.
        """
        count = min(self.max_tiles_per_decoder_call, remaining)
        if self._decode_bytes_per_latent and tile.device.type == "cuda":
            per_tile = self._decode_bytes_per_latent * tile.numel()
            budget = available_device_bytes(tile.device) * TILE_DECODE_MEMORY_FRACTION
            count = min(count, int(budget // per_tile))
        return max(1, count)

    def _group_tiles_per_decoder_call(
        self, tile: torch.Tensor, remaining: int, group: dist.ProcessGroup
    ) -> int:
        """The smallest ``_tiles_per_decoder_call`` over the group, so every rank gathers equal batches."""
        count = torch.tensor(self._tiles_per_decoder_call(tile, remaining), device=tile.device)
        dist.all_reduce(count, op=dist.ReduceOp.MIN, group=group)
        return int(count.item())

    def _tile_geometry(
        self, z: torch.Tensor
    ) -> tuple[list[tuple[int, int, int, int]], int, list[int], list[int]]:
        """Latent tile boxes (y, h, x, w) in row-major order, the column count and the overlaps."""
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
        return tiles, len(x_starts), y_overlaps, x_overlaps

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return super()._decode_clip(z)
        tiles, columns, y_overlaps, x_overlaps = self._tile_geometry(z)
        group = self.tile_parallel_group
        world_size = 1 if group is None else dist.get_world_size(group)
        if world_size == 1 or len(tiles) < world_size:
            decoded = self._decode_tiles_batched(z, tiles)
        else:
            decoded = self._decode_tiles_distributed(z, tiles, group)
        rows = [decoded[i : i + columns] for i in range(0, len(tiles), columns)]
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def _decode_tiles_batched(
        self, z: torch.Tensor, tiles: list[tuple[int, int, int, int]]
    ) -> list[torch.Tensor]:
        """Decode every tile on this rank, equal-size tiles several per decoder call.

        Unequal tiles, which the splitter does not produce, get one call each.
        """
        latents = [z[..., y : y + h, x : x + w] for y, h, x, w in tiles]
        if len({tuple(t.shape) for t in latents}) != 1:
            return [self.decoder(self.post_quant_conv(t)) for t in latents]
        batch = z.shape[0]
        decoded: list[torch.Tensor] = []
        start = 0
        while start < len(latents):
            count = self._tiles_per_decoder_call(latents[0], len(latents) - start)
            chunk = torch.cat(latents[start : start + count], dim=0)
            decoded.extend(self.decoder(self.post_quant_conv(chunk)).split(batch, dim=0))
            start += count
        return decoded

    def _decode_tiles_distributed(
        self,
        z: torch.Tensor,
        tiles: list[tuple[int, int, int, int]],
        group: dist.ProcessGroup,
    ) -> list[torch.Tensor]:
        """Decode every ``world_size``-th tile on this rank in batches and gather each batch.

        Tile ``i`` belongs to rank ``i % world_size``. Ranks decode their tiles ``count`` at a
        time, with ``count`` agreed over the group once per clip, and gather after every batch,
        so the temporary communication storage is ``world_size * count`` tiles. A rank whose
        tiles run out before the others sends zeros and keeps joining the gathers, so
        non-divisible tile counts cannot hang.
        """
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        # _split_tiles gives equal-sized tiles, including the final row/column.
        own = [z[..., y : y + h, x : x + w] for y, h, x, w in tiles[rank::world_size]]
        waves = (len(tiles) + world_size - 1) // world_size
        count = self._group_tiles_per_decoder_call(own[0], waves, group)
        batch = z.shape[0]
        decoded_tiles: list[torch.Tensor] = [z.new_empty(0)] * len(tiles)
        for start in range(0, waves, count):
            chunk = own[start : start + count]
            if chunk:
                local = self.decoder(self.post_quant_conv(torch.cat(chunk, dim=0))).contiguous()
            else:
                # Every rank owns at least waves - 1 tiles, so only a last batch can be empty
                # and the previous batch has set local's tile geometry.
                local = local[:0]
            pad = (min(count, waves - start) - len(chunk)) * batch
            if pad:
                local = torch.cat([local, local.new_zeros((pad, *local.shape[1:]))], dim=0)
            gathered = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local, group=group)
            for offset, received in enumerate(gathered):
                for wave, tile in enumerate(received.split(batch, dim=0), start=start):
                    index = wave * world_size + offset
                    if index < len(tiles):
                        decoded_tiles[index] = tile
        return decoded_tiles
