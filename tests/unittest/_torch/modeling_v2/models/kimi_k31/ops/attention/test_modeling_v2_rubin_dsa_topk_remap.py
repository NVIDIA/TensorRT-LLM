# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_dsa_topk_remap, the Triton remap of the Top-K token indices to KV pool rows.

GPU (sm_107): a sample of the cells bit for bit against the reference (integers).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_topk_remap import (  # noqa: E501
    rubin_dsa_topk_remap as remap_op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

#: A typical cell of ``remap_op.CELLS``.
_CELLS = sample_cells(remap_op.CELLS, size_key="rows")[1:2]


on_sm107 = on_sm(107)


def _remap_inputs(spec, gen, device="cpu"):
    rows, reqs, pages, bs = spec["rows"], spec["requests"], spec["pages"], spec["block_size"]
    table = torch.randperm(reqs * pages * 2, generator=gen)[: reqs * pages].view(reqs, pages)
    table = table.to(torch.int32)
    if spec.get("holes"):
        table[:, ::5] = -1
    req = torch.randint(0, reqs, (rows,), generator=gen, dtype=torch.int32)
    hi = pages * bs + (bs if spec.get("holes") else 0)
    tok = torch.randint(-1, hi, (rows, spec["topk"]), generator=gen, dtype=torch.int32)
    return [t.to(device) for t in (req, table, tok)]


@on_sm107
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_remap_certified_cells(cell, certifying):  # noqa: F811
    certifying(remap_op)
    s = cell.spec
    req, table, tok = _remap_inputs(s, torch.Generator().manual_seed(s["rows"]), "cuda")
    stride = s["layers"] * s["block_size"]
    args = (req, table, tok, s["block_size"], s["topk"], stride, s["layers"] - 1)
    out = remap_op(*args)
    assert torch.equal(out, remap_op.reference(*args))
