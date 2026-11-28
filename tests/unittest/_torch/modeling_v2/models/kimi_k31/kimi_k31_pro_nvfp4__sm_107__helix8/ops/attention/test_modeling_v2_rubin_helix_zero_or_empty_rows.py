# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_helix_zero_or_empty_rows, the Helix combine mask of the rows without keys.

GPU (sm_107): a sample of the cells bit for bit against the reference.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.ops.attention.rubin_helix_zero_or_empty_rows import (  # noqa: E501
    rubin_helix_zero_or_empty_rows as mask_op,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

#: A typical cell of ``mask_op.CELLS``.
_CELLS = sample_cells(mask_op.CELLS, size_key="rows")[1:2]


on_sm107 = on_sm(107)


@on_sm107
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_mask_certified_cells(cell, certifying):  # noqa: F811
    certifying(mask_op)
    s = cell.spec
    gen = torch.Generator().manual_seed(s["rows"])
    bounds = torch.randint(1, 100, (s["rows"],), generator=gen, dtype=torch.int32)
    counts = torch.randint(1, 100, (s["rows"],), generator=gen, dtype=torch.int32)
    bounds[torch.rand(s["rows"], generator=gen) < s["zero_frac"]] = 0
    counts[torch.rand(s["rows"], generator=gen) < s["empty_frac"]] = 0
    out = mask_op(bounds.cuda(), counts.cuda())
    assert torch.equal(out.cpu(), mask_op.reference(bounds, counts))
