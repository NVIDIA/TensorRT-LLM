# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_pad_heads, the copy of the query heads into the pad128 buffers.

GPU (sm_107): a sample of the cells bit for bit against the reference (copies and zeros).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_pad_heads import (  # noqa: E501
    rubin_pad_heads as pad_op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

#: A typical cell of ``pad_op.CELLS``.
_CELLS = sample_cells(pad_op.CELLS, size_key="n")[1:2]


on_sm107 = on_sm(107)


@on_sm107
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_pad_certified_cells(cell, certifying):  # noqa: F811
    certifying(pad_op)
    s = cell.spec
    gen = torch.Generator().manual_seed(s["n"])
    dst = torch.randn(s["n"], s["pad"], s["d"], generator=gen).to(s["dtype"]).cuda()
    src = None
    if s["src"]:
        src = torch.randn(s["n"], s["heads"], s["d"], generator=gen).to(s["dtype"]).cuda()
    ref = pad_op.reference(dst, src, s["heads"])
    assert torch.equal(pad_op(dst, src, s["heads"]), ref)
