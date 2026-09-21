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

from tensorrt_llm._torch.visual_gen.models.minimax_h3.tiled_vae import (
    MINIMAX_H3_VAE_DEFAULTS,
    TiledAutoencoderKLMiniMaxH3,
    validate_vae_tiling_config,
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
    vae.post_quant_conv = torch.nn.Identity()
    vae.decoder = _TileDecoder()
    vae.configure_tiling({**MINIMAX_H3_VAE_DEFAULTS, "vae_tile_size": 8, "vae_tile_overlap": 2})
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
        vae.decoder(z[..., y // 2 : (y + h) // 2, x // 2 : (x + w) // 2])
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


def test_disabled_tiling_and_single_rank_do_not_use_collectives() -> None:
    vae = _vae()
    z = torch.randn(1, 1, 2, 7, 10)
    with patch("torch.distributed.all_gather") as gather:
        torch.testing.assert_close(vae._decode_clip(z), AutoencoderKLMiniMaxH3._decode_clip(vae, z))
        vae.disable_tiling()
        vae.tile_parallel_group = object()
        torch.testing.assert_close(vae._decode_clip(z), vae.decoder(z))
    gather.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"vae_use_tiling": "false"},
        {"vae_tile_parallel": 1},
        {"vae_tile_size": 0},
        {"vae_tile_size": True},
        {"vae_tile_overlap": 0},
        {"vae_tile_overlap": -1},
        {"vae_tile_overlap": 256},
        {"vae_tile_size": 32.5},
        {"vae_use_tiling": False, "vae_tile_parallel": True},
    ],
)
def test_invalid_tiling_options(overrides: dict) -> None:
    with pytest.raises(ValueError):
        validate_vae_tiling_config({**MINIMAX_H3_VAE_DEFAULTS, **overrides})


@pytest.mark.parametrize("name", ["vae_tile_size", "vae_tile_overlap"])
def test_tile_geometry_must_align_to_checkpoint(name: str) -> None:
    with pytest.raises(ValueError, match="compression ratio"):
        _vae().configure_tiling({**MINIMAX_H3_VAE_DEFAULTS, name: 65})
