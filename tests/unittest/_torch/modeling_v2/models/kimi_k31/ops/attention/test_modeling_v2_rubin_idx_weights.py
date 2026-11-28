# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_idx_weights, the DSA indexer's logits weights ``(w * qs) * scale`` in one kernel.

GPU (sm_107): a sample of the cells bit for bit against the reference.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_weights import (  # noqa: E501
    rubin_idx_weights as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


on_sm107 = on_sm(107)

# The indexer's constant: index_head_dim^-0.5 x index_num_heads^-0.5.
_SCALE = 128**-0.5 * 32**-0.5


def _inputs(rows, heads, row_stride, seed, device="cpu", tiny=False):
    """``w``: a column slice of a ``[rows, row_stride]`` FP32 projection; ``qs``: UE8M0 scales."""
    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(rows, row_stride, generator=gen) * 0.3
    w = proj.to(device)[:, row_stride - heads :]
    exp = torch.randint(-12, -2, (rows, heads), generator=gen)
    if tiny:
        exp = exp - 120  # products near and below FP32's normal range
    qs = torch.ldexp(torch.ones(rows, heads), exp).to(device)
    return w, qs


@on_sm107
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    w, qs = _inputs(s["rows"], s["heads"], s["row_stride"], s["rows"], "cuda")
    out = op(w, qs, _SCALE)
    assert torch.equal(out.view(torch.int32), op.reference(w, qs, _SCALE).view(torch.int32))
