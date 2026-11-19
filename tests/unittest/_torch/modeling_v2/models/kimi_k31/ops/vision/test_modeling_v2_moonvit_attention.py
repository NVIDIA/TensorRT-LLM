# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moonvit_attention entry: a sample of its certified cells against its
reference within the bf16 band (``_moonvit_cells.BF16_BAND``)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.vision.moonvit_attention import (
    moonvit_attention,
)

__extra_import_path__ = [".", "../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _moonvit_cells import BF16_BAND, cell_generator, cell_ids  # noqa: E402 — path above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(103, 107)


CELLS = sample_cells(moonvit_attention.CELLS)[:2]


@pytest.mark.parametrize("cell", CELLS, ids=cell_ids(CELLS))
def test_attention_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(moonvit_attention)
    gen = cell_generator(cell)
    seg_lens, heads, head_dim = cell.spec["seg_lens"], cell.spec["heads"], cell.spec["head_dim"]
    capacity = cell.spec["capacity"]
    qkv = torch.randn(sum(seg_lens), 3 * heads * head_dim, generator=gen, device="cuda").bfloat16()
    out = torch.empty(sum(seg_lens), heads * head_dim, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(0, dtype=torch.int8, device="cuda")

    moonvit_attention(qkv, out, seg_lens, capacity, capacity // 4, workspace, heads, head_dim)
    ref = moonvit_attention.reference(
        qkv, out, seg_lens, capacity, capacity // 4, workspace, heads, head_dim
    )
    torch.testing.assert_close(out, ref, **BF16_BAND)
