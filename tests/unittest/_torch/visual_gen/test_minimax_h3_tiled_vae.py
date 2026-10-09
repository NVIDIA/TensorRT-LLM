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

"""Spatial tile scheduling, reference blending, and configuration regressions."""

from unittest.mock import patch

import pytest
import torch
from diffusers import AutoencoderKLMiniMaxH3

from tensorrt_llm._torch.visual_gen.models.minimax_h3.parallel_vae import (
    TiledAutoencoderKLMiniMaxH3,
)

pytestmark = pytest.mark.cpu_only


class _TileDecoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tile-dependent values expose wrong ordering and overlap blending.
        return (x + x.mean()).repeat_interleave(2, -2).repeat_interleave(2, -1)


def _vae() -> TiledAutoencoderKLMiniMaxH3:
    vae = TiledAutoencoderKLMiniMaxH3.__new__(TiledAutoencoderKLMiniMaxH3)
    torch.nn.Module.__init__(vae)
    vae.spatial_compression_ratio = 2
    vae.use_tiling = True
    vae.tile_sample_min_height = vae.tile_sample_min_width = 8
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 2
    vae.post_quant_conv = torch.nn.Conv3d(1, 1, 1)
    vae.decoder = _TileDecoder()
    return vae


@pytest.mark.parametrize("shape", [(2, 2), (3, 9), (7, 7), (7, 10), (10, 7)])
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_parallel_tiles_match_reference(shape: tuple[int, int], world_size: int) -> None:
    vae = _vae()
    z = torch.arange(2 * 3 * shape[0] * shape[1], dtype=torch.float32).reshape(2, 1, 3, *shape)
    expected = AutoencoderKLMiniMaxH3._decode_clip(vae, z)
    ys, hs, _ = vae._split_tiles(shape[0] * 2, 8, 2)
    xs, ws, _ = vae._split_tiles(shape[1] * 2, 8, 2)
    tiles = [
        vae.decoder(vae.post_quant_conv(z[..., y // 2 : (y + h) // 2, x // 2 : (x + w) // 2]))
        for y, h in zip(ys, hs)
        for x, w in zip(xs, ws)
    ]
    group = object()
    vae.tile_parallel_group = group
    for rank in range(world_size):
        wave = 0

        def all_gather(outputs: list[torch.Tensor], local: torch.Tensor, *, group: object) -> None:
            nonlocal wave
            index = wave * world_size + rank
            if index < len(tiles):
                torch.testing.assert_close(local, tiles[index], rtol=0, atol=0)
            else:
                assert torch.count_nonzero(local) == 0
            for offset, output in enumerate(outputs):
                index = wave * world_size + offset
                output.copy_(tiles[index] if index < len(tiles) else torch.zeros_like(output))
            wave += 1

        with (
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.get_rank", return_value=rank),
            patch("torch.distributed.all_gather", side_effect=all_gather) as gather,
        ):
            actual = vae._decode_clip(z)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if len(tiles) < world_size or world_size == 1:
            gather.assert_not_called()
        else:
            assert wave == (len(tiles) + world_size - 1) // world_size


def test_no_tile_group_and_disabled_tiling_do_not_use_collectives() -> None:
    vae = _vae()
    z = torch.randn(1, 1, 2, 7, 10)
    with patch("torch.distributed.all_gather") as gather:
        torch.testing.assert_close(vae._decode_clip(z), AutoencoderKLMiniMaxH3._decode_clip(vae, z))
        vae.disable_tiling()
        vae.tile_parallel_group = object()
        torch.testing.assert_close(vae._decode_clip(z), vae.decoder(vae.post_quant_conv(z)))
    gather.assert_not_called()


@pytest.mark.parametrize(
    "name,value",
    [("tile_sample_min_height", 65), ("tile_sample_min_overlap_width", 3)],
)
def test_parallel_tile_geometry_must_align_to_checkpoint(name: str, value: int) -> None:
    vae = _vae()
    setattr(vae, name, value)
    with pytest.raises(ValueError, match="compression ratio"):
        vae.configure_parallel(group=object())


@pytest.mark.parametrize("use_tiling", [True, False])
def test_no_parallel_group_preserves_loaded_vae_defaults(use_tiling: bool) -> None:
    vae = _vae()
    vae.use_tiling = use_tiling
    vae.tile_sample_min_height = 16
    vae.tile_sample_min_width = 20
    vae.configure_parallel()
    assert vae.use_tiling is use_tiling
    assert (vae.tile_sample_min_height, vae.tile_sample_min_width) == (16, 20)
    assert vae.tile_sample_min_overlap_height == 2
    assert vae.tile_parallel_group is None


@pytest.mark.parametrize("use_tiling", [True, False])
def test_parallel_group_enables_tiling_without_changing_geometry(use_tiling: bool) -> None:
    vae = _vae()
    vae.use_tiling = use_tiling
    vae.tile_sample_min_height = 16
    vae.tile_sample_min_width = 20
    vae.tile_sample_min_overlap_width = 4
    group = object()
    vae.configure_parallel(group=group)
    assert vae.use_tiling
    assert vae.tile_parallel_group is group
    assert (vae.tile_sample_min_height, vae.tile_sample_min_width) == (16, 20)
    assert (vae.tile_sample_min_overlap_height, vae.tile_sample_min_overlap_width) == (2, 4)


@pytest.mark.parametrize("size,overlap", [(0, 2), (8, 0), (8, 8), (8, 10)])
def test_parallel_group_rejects_invalid_loaded_geometry(size: int, overlap: int) -> None:
    vae = _vae()
    vae.tile_sample_min_height = size
    vae.tile_sample_min_overlap_height = overlap
    with pytest.raises(ValueError, match="overlap"):
        vae.configure_parallel(group=object())
