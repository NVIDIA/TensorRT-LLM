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

"""Shape, policy, and forward-context metadata for Video Sparse Attention."""

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from typing import Iterator, Optional, Tuple, TypedDict

import torch

# A 4x4x4 cube is one 64-token sparse block for every VSA fine-stage backend.
VSA_TILE_SIZE: Tuple[int, int, int] = (4, 4, 4)
VSA_BLOCK_SIZE = VSA_TILE_SIZE[0] * VSA_TILE_SIZE[1] * VSA_TILE_SIZE[2]
_DEFAULT_MAX_CACHED_SHAPES = 16
_BITS_PER_WORD = 32


def _get_tile_partition_indices(
    dit_seq_shape: Tuple[int, int, int],
    tile_size: Tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    time, height, width = dit_seq_shape
    tile_time, tile_height, tile_width = tile_size
    num_time = ceil(time / tile_time)
    num_height = ceil(height / tile_height)
    num_width = ceil(width / tile_width)

    block_time = torch.arange(num_time, device=device).view(num_time, 1, 1, 1, 1, 1)
    block_height = torch.arange(num_height, device=device).view(1, num_height, 1, 1, 1, 1)
    block_width = torch.arange(num_width, device=device).view(1, 1, num_width, 1, 1, 1)
    local_time = torch.arange(tile_time, device=device).view(1, 1, 1, tile_time, 1, 1)
    local_height = torch.arange(tile_height, device=device).view(1, 1, 1, 1, tile_height, 1)
    local_width = torch.arange(tile_width, device=device).view(1, 1, 1, 1, 1, tile_width)

    global_time = block_time * tile_time + local_time
    global_height = block_height * tile_height + local_height
    global_width = block_width * tile_width + local_width
    valid = (global_time < time) & (global_height < height) & (global_width < width)
    flat = global_time * (height * width) + global_height * width + global_width
    indices = torch.where(valid, flat, torch.full_like(flat, -1))
    return indices.reshape(-1).to(torch.long)


def _construct_variable_block_sizes(
    dit_seq_shape: Tuple[int, int, int],
    num_tiles: Tuple[int, int, int],
    tile_size: Tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    time, height, width = dit_seq_shape
    tile_time, tile_height, tile_width = tile_size
    num_time, num_height, num_width = num_tiles

    block_time = torch.arange(num_time, device=device)
    block_height = torch.arange(num_height, device=device)
    block_width = torch.arange(num_width, device=device)
    valid_time = (time - block_time * tile_time).clamp(max=tile_time)
    valid_height = (height - block_height * tile_height).clamp(max=tile_height)
    valid_width = (width - block_width * tile_width).clamp(max=tile_width)
    sizes = (
        valid_time.view(num_time, 1, 1)
        * valid_height.view(1, num_height, 1)
        * valid_width.view(1, 1, num_width)
    )
    return sizes.reshape(-1).to(torch.long)


def _pack_valid_slots(valid_slots: torch.BoolTensor) -> torch.Tensor:
    """Pack a padded-slot validity mask into little-endian 32-bit words."""
    bit_weights = 1 << torch.arange(_BITS_PER_WORD, dtype=torch.int64, device=valid_slots.device)
    words = (valid_slots.view(-1, _BITS_PER_WORD).to(torch.int64) * bit_weights).sum(dim=-1)
    return words.to(torch.uint32)


@dataclass(frozen=True, slots=True)
class VSAMetadata:
    """Per-step policy and shape metadata required by the VSA sparse path.

    ``tile_source_index`` maps every padded slot to its compact token (``-1`` for padding),
    ``untile_idx`` maps every compact token back to its padded slot,
    ``variable_block_sizes`` counts the valid tokens per cube, and ``kv_valid_words`` packs
    the padded valid-token mask into 32-bit words for the block-sparse kernels.
    """

    current_timestep: int
    vsa_sparsity: float
    num_cubes: int
    padded_seq_length: int
    variable_block_sizes: torch.LongTensor
    tile_source_index: torch.LongTensor
    untile_idx: torch.LongTensor
    kv_valid_words: torch.Tensor


class _VSAShapeMetadata(TypedDict):
    num_cubes: int
    padded_seq_length: int
    variable_block_sizes: torch.LongTensor
    tile_source_index: torch.LongTensor
    untile_idx: torch.LongTensor
    kv_valid_words: torch.Tensor


class VSAMetadataBuilder:
    """Build VSA metadata while caching shape-dependent index tensors."""

    def __init__(self, max_cached_shapes: int = _DEFAULT_MAX_CACHED_SHAPES) -> None:
        if max_cached_shapes <= 0:
            raise ValueError("max_cached_shapes must be positive")
        self._max_cached_shapes = max_cached_shapes
        self._cache: dict[Tuple[Tuple[int, int, int], torch.device], _VSAShapeMetadata] = {}

    def _build_metadata(
        self,
        dit_seq_shape: Tuple[int, int, int],
        device: torch.device,
    ) -> _VSAShapeMetadata:
        time, height, width = dit_seq_shape
        tile_time, tile_height, tile_width = VSA_TILE_SIZE
        num_tiles = (
            ceil(time / tile_time),
            ceil(height / tile_height),
            ceil(width / tile_width),
        )
        total_seq_length = time * height * width
        padded_seq_length = (
            num_tiles[0] * num_tiles[1] * num_tiles[2] * tile_time * tile_height * tile_width
        )
        num_cubes = num_tiles[0] * num_tiles[1] * num_tiles[2]
        tokens_per_cube = VSA_BLOCK_SIZE

        tile_partition_indices = _get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
        gather_idx = tile_partition_indices[tile_partition_indices >= 0]

        variable_block_sizes = _construct_variable_block_sizes(
            dit_seq_shape, num_tiles, VSA_TILE_SIZE, device
        )
        local_offsets = torch.arange(tokens_per_cube, device=device).expand(
            num_cubes, tokens_per_cube
        )
        cube_offsets = torch.arange(num_cubes, device=device).unsqueeze(1) * tokens_per_cube
        non_pad_index = (cube_offsets + local_offsets)[
            local_offsets < variable_block_sizes.unsqueeze(1)
        ]

        tile_source_index = torch.full((padded_seq_length,), -1, dtype=torch.long, device=device)
        tile_source_index[non_pad_index] = gather_idx
        untile_idx = torch.empty(total_seq_length, dtype=torch.long, device=device)
        untile_idx[gather_idx] = non_pad_index

        return _VSAShapeMetadata(
            num_cubes=num_cubes,
            padded_seq_length=padded_seq_length,
            variable_block_sizes=variable_block_sizes,
            tile_source_index=tile_source_index,
            untile_idx=untile_idx,
            kv_valid_words=_pack_valid_slots(tile_source_index >= 0),
        )

    def build(
        self,
        current_timestep: int,
        raw_latent_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        vsa_sparsity: float,
        device: torch.device,
    ) -> VSAMetadata:
        dit_seq_shape = (
            raw_latent_shape[0] // patch_size[0],
            raw_latent_shape[1] // patch_size[1],
            raw_latent_shape[2] // patch_size[2],
        )
        cache_key = (dit_seq_shape, device)
        shape_metadata = self._cache.get(cache_key)
        if shape_metadata is None:
            if len(self._cache) >= self._max_cached_shapes:
                raise RuntimeError(
                    "VSA metadata cache reached its "
                    f"{self._max_cached_shapes}-shape limit; restart the pipeline or "
                    "reuse a configured resolution/frame profile"
                )
            shape_metadata = self._build_metadata(dit_seq_shape, device)
            self._cache[cache_key] = shape_metadata

        return VSAMetadata(
            current_timestep=current_timestep,
            vsa_sparsity=vsa_sparsity,
            **shape_metadata,
        )

    def clear(self) -> None:
        """Release cached tensors after CUDA Graphs that reference them are cleared."""

        self._cache.clear()


_vsa_forward_context_var: contextvars.ContextVar[Optional[VSAMetadata]] = contextvars.ContextVar(
    "_vsa_forward_context", default=None
)


@contextmanager
def set_vsa_forward_context(metadata: VSAMetadata) -> Iterator[None]:
    """Make VSA metadata visible to attention layers for one model forward."""

    token = _vsa_forward_context_var.set(metadata)
    try:
        yield
    finally:
        _vsa_forward_context_var.reset(token)


def get_vsa_forward_context() -> Optional[VSAMetadata]:
    """Return the metadata for the active VSA model forward, if any."""

    return _vsa_forward_context_var.get(None)


__all__ = [
    "VSA_TILE_SIZE",
    "VSAMetadata",
    "VSAMetadataBuilder",
    "get_vsa_forward_context",
    "set_vsa_forward_context",
]
