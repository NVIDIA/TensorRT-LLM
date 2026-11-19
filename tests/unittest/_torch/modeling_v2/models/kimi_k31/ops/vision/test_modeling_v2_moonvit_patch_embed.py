# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moonvit_patch_embed entry: a sample of its certified cells
against its reference within the bf16 band (``_moonvit_cells.BF16_BAND``)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.vision.moonvit_patch_embed import (
    moonvit_patch_embed,
)
from tensorrt_llm._torch.models import modeling_kimi_k3_vl as engine_k3

#: Kimi K3.1's MoonViT: 14-px patches, a 64 x 64 position table at width 1024, 4 frames of temporal table.
PATCH, TABLE_SIDE, WIDTH, FRAMES = 14, 64, 1024, 4

__extra_import_path__ = [".", "../../../..", "../.."]
from _k31_utils import on_sm, sample_cells  # noqa: E402 — path above
from _moonvit_cells import BF16_BAND, cell_generator, cell_ids  # noqa: E402 — path above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(103, 107)


CELLS = sample_cells(moonvit_patch_embed.CELLS)[:2]


@pytest.mark.parametrize("cell", CELLS, ids=cell_ids(CELLS))
def test_patch_embed_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(moonvit_patch_embed)
    gen = cell_generator(cell)
    grids = cell.spec["grids"]
    patches = sum(t * h * w for t, h, w in grids)
    pixels = torch.randn(patches, 3, PATCH, PATCH, generator=gen, device="cuda").bfloat16()
    embed = engine_k3.K3PatchEmbed3d(WIDTH, PATCH, TABLE_SIDE, TABLE_SIDE, FRAMES).to(
        device="cuda", dtype=torch.bfloat16
    )
    with torch.no_grad():
        for param in embed.parameters():
            param.normal_(0.0, 0.02, generator=gen)
    out = moonvit_patch_embed(
        pixels, embed.proj.weight, embed.pos_emb.weight, embed.pos_emb.time_weight, grids
    )
    ref = moonvit_patch_embed.reference(
        pixels, embed.proj.weight, embed.pos_emb.weight, embed.pos_emb.time_weight, grids
    )
    scale = ref.float().abs().amax(dim=-1, keepdim=True)
    torch.testing.assert_close(out.float() / scale, ref.float() / scale, **BF16_BAND)
