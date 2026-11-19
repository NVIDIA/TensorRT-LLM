# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moonvit_rope_2d entry: a sample of its certified cells
against its reference within the bf16 band (``_moonvit_cells.BF16_BAND``)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.vision.moonvit_rope_2d import (
    TABLE_SIDE,
    THETA,
    moonvit_rope_2d,
    moonvit_rope_angles,
    moonvit_rope_table,
)

__extra_import_path__ = [".", "../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _moonvit_cells import BF16_BAND, cell_generator, cell_ids  # noqa: E402 — path above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(103, 107)


CELLS = sample_cells(moonvit_rope_2d.CELLS)[:2]


@pytest.mark.parametrize("cell", CELLS, ids=cell_ids(CELLS))
def test_rope_2d_cells(cell, certifying) -> None:  # noqa: F811
    certifying(moonvit_rope_2d)
    gen = cell_generator(cell)
    grids, heads, head_dim = cell.spec["grids"], cell.spec["heads"], cell.spec["head_dim"]
    rows = sum(t * h * w for t, h, w in grids)
    qkv = torch.randn(rows, 3 * heads * head_dim, generator=gen, device="cuda").bfloat16()
    q, k, _ = qkv.split(heads * head_dim, dim=-1)
    q, k = q.view(rows, heads, head_dim), k.view(rows, heads, head_dim)

    table = moonvit_rope_table(head_dim, TABLE_SIDE, TABLE_SIDE, THETA, torch.device("cuda"))
    freqs_cis = moonvit_rope_angles(table, grids)
    q_out, k_out = moonvit_rope_2d(q, k, freqs_cis)
    q_ref, k_ref = moonvit_rope_2d.reference(q, k, freqs_cis)
    torch.testing.assert_close(q_out, q_ref, **BF16_BAND)
    torch.testing.assert_close(k_out, k_ref, **BF16_BAND)
