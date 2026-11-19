# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_merge_helix_topk, the Helix fallback merge of the ranks' Top-K.

GPU (sm_107): a sample of the cells bit for bit against the reference (compares).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_merge_helix_topk import (  # noqa: E501
    rubin_merge_helix_topk as merge_op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

#: A typical cell of ``merge_op.CELLS``.
_CELLS = sample_cells(merge_op.CELLS, size_key="rows")[1:2]


on_sm107 = on_sm(107)


def _merge_inputs(spec, gen):
    cp, rows, k = spec["cp"], spec["rows"], spec["k"]
    # Distinct scores across every rank: a permutation scaled into (0, 1).
    scores = (torch.randperm(cp * rows * k, generator=gen).float() + 1) / (cp * rows * k + 1)
    scores = scores.view(cp, rows, k)
    local = torch.stack([torch.randperm(10 * k, generator=gen)[:k] for _ in range(rows)]).to(
        torch.int32
    )
    drop = torch.rand(cp, rows, k, generator=gen) > spec["fill"]
    return scores.masked_fill(drop, float("-inf")), local


@on_sm107
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_merge_certified_cells(cell, certifying):  # noqa: F811
    certifying(merge_op)
    spec = cell.spec
    scores, local = _merge_inputs(spec, torch.Generator().manual_seed(spec["rows"]))
    rows = local.masked_fill(~torch.isfinite(scores[0]), -1)
    ref = merge_op.reference(scores, rows, 0)
    assert torch.equal(merge_op(scores.cuda(), rows.cuda(), 0).cpu(), ref)
