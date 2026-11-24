# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moonvit_patch_embed entry: each certified cell against its reference within the band
stated below, and bit for bit against the engine MoonViT3d code it replaces."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.vision.moonvit_patch_embed import (
    moonvit_patch_embed,
)
from tensorrt_llm._torch.models import modeling_kimi_k3_vl as engine_k3

#: Kimi K3.1's MoonViT: 14-px patches, a 64 x 64 position table at width 1024, 4 frames of temporal table.
PATCH, TABLE_SIDE, WIDTH, FRAMES = 14, 64, 1024, 4

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 (sm_107 driven separately)",
)

#: bf16 results against an fp32 / fp64 reference rounded once: the entries round intermediate bf16 values, each half
#: an ulp; four ulps of the row's scale covers them.
_BF16_BAND = dict(rtol=4 * 2.0**-8, atol=4 * 2.0**-8)


def _gen(cell) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)


def _ids(op):
    return [c.why[:48] for c in op.CELLS]


@pytest.mark.parametrize("cell", moonvit_patch_embed.CELLS, ids=_ids(moonvit_patch_embed))
def test_patch_embed_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(moonvit_patch_embed)
    gen = _gen(cell)
    grids = cell.spec["grids"]
    patches = sum(t * h * w for t, h, w in grids)
    pixels = torch.randn(patches, 3, PATCH, PATCH, generator=gen, device="cuda").bfloat16()
    embed = engine_k3.K3PatchEmbed3d(WIDTH, PATCH, TABLE_SIDE, TABLE_SIDE, FRAMES).to(
        device="cuda", dtype=torch.bfloat16
    )
    grid_thws = torch.tensor(grids, dtype=torch.int64, device="cuda")

    out = moonvit_patch_embed(
        pixels, embed.proj.weight, embed.pos_emb.weight, embed.pos_emb.time_weight, grids
    )
    with torch.no_grad():
        assert torch.equal(out, embed(pixels, grid_thws))
    ref = moonvit_patch_embed.reference(
        pixels, embed.proj.weight, embed.pos_emb.weight, embed.pos_emb.time_weight, grids
    )
    scale = ref.float().abs().amax(dim=-1, keepdim=True)
    torch.testing.assert_close(out.float() / scale, ref.float() / scale, **_BF16_BAND)


def test_patch_embed_rejects_a_grid_past_the_temporal_table() -> None:
    pixels = torch.zeros(5 * 4, 3, PATCH, PATCH, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(8, 3, PATCH, PATCH, device="cuda", dtype=torch.bfloat16)
    table = torch.zeros(4, 4, 8, device="cuda", dtype=torch.bfloat16)
    time_weight = torch.zeros(FRAMES, 1, 8, device="cuda", dtype=torch.bfloat16)
    with validating(moonvit_patch_embed), pytest.raises(ValueError, match="more frames"):
        moonvit_patch_embed(pixels, weight, table, time_weight, [(5, 2, 2)])
