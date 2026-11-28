# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_topk_decode_cute, the radix Top-K of the Helix key shard.

GPU (sm_107): a sample of the cells against the reference by the multiset of selected values (its boundary
ties may go either way).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.ops.attention.rubin_topk_decode_cute import (  # noqa: E501
    rubin_topk_decode_cute as cute_op,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

#: A boundary and a typical cell of ``cute_op.CELLS``.
_CELLS = sample_cells(cute_op.CELLS, size_key="rows")[:2]


on_sm107 = on_sm(107)


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
def test_cute_certified_cells(cell, certifying):  # noqa: F811
    certifying(cute_op)
    s = cell.spec
    gen = torch.Generator().manual_seed(s["rows"])
    lo, hi = s["row_lens"]
    lens = torch.randint(lo, hi + 1, (s["rows"],), generator=gen, dtype=torch.int32)
    logits = _logits(s["rows"], s["width"], gen).cuda()
    out = torch.full((s["rows"], s["top_k"]), -7, dtype=torch.int32, device="cuda")
    cute_op(logits, lens.cuda(), out, s["top_k"], s["next_n"])
    ref = cute_op.reference(logits, lens, out, s["top_k"], s["next_n"])
    for r in range(s["rows"]):
        got, want = out[r][out[r] >= 0].long().cpu(), ref[r][ref[r] >= 0].long().cpu()
        assert got.numel() == want.numel()
        row = logits[r].cpu()
        assert torch.equal(torch.sort(row[got]).values, torch.sort(row[want]).values)
