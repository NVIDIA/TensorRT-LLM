# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moonvit_rms_norm entry: a sample of its certified cells
against its reference within the bf16 band (``_moonvit_cells.BF16_BAND``)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.vision.moonvit_rms_norm import (
    moonvit_rms_norm,
)

__extra_import_path__ = [".", "../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _moonvit_cells import BF16_BAND, cell_generator, cell_ids  # noqa: E402 — path above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(103, 107)


CELLS = sample_cells(moonvit_rms_norm.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=cell_ids(CELLS))
def test_rms_norm_cells(cell, certifying) -> None:  # noqa: F811
    certifying(moonvit_rms_norm)
    gen = cell_generator(cell)
    rows, width, eps = cell.spec["rows"], cell.spec["width"], cell.spec["eps"]
    x = torch.randn(rows, width, generator=gen, device="cuda").bfloat16()
    weight = (1.0 + 0.1 * torch.randn(width, generator=gen, device="cuda")).bfloat16()
    out = moonvit_rms_norm(x, weight, eps)
    torch.testing.assert_close(out, moonvit_rms_norm.reference(x, weight, eps), **BF16_BAND)
