# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_gvr_topk_prefill, the self-sampling GVR Top-K of the prefill windows (a
deterministic kernel).

GPU (sm_107): a sample of the cells against the reference as sets (the kernel's output order is its own).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_gvr_topk_prefill import (  # noqa: E501
    rubin_gvr_topk_prefill as prefill_op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

#: A boundary and a typical cell of ``prefill_op.CELLS``.
_CELLS = sample_cells(prefill_op.CELLS, size_key="width")[:2]


on_sm107 = on_sm(107)


def _sets(rows: torch.Tensor):
    return [set(r[r >= 0].tolist()) for r in rows.cpu()]


def _logits(rows, width, gen, ties=False, align=4):
    stride = -(-width // align) * align
    base = torch.empty(rows, stride)
    if ties:
        base.copy_(torch.randint(-3, 4, (rows, stride), generator=gen).float())
    else:
        base.copy_(torch.randn(rows, stride, generator=gen))
    return base[:, :width]


@on_sm107
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_prefill_certified_cells(cell, certifying):  # noqa: F811
    certifying(prefill_op)
    s = cell.spec
    gen = torch.Generator().manual_seed(len(s["windows"]))
    logits = _logits(len(s["windows"]), s["width"], gen, s.get("data") == "ties").cuda()
    starts = torch.tensor([a for a, _ in s["windows"]], dtype=torch.int32).cuda()
    ends = torch.tensor([b for _, b in s["windows"]], dtype=torch.int32).cuda()
    out = torch.full((len(s["windows"]), s["top_k"]), -7, dtype=torch.int32, device="cuda")
    prefill_op(logits, starts, ends, out)
    ref = prefill_op.reference(logits, starts, ends, out)
    assert _sets(out) == _sets(ref)
