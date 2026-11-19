# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 indexer_topk_prefill_gvr entry: each row's set equals the reference's
(ties to the smaller column)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_prefill_gvr import (  # noqa: E501
    indexer_topk_prefill_gvr as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_topk_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key=None)[:2]


def _build(spec, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ks, ke, start = [], [], 0
    for cached, new in spec["requests"]:
        for i in range(new):
            ks.append(start)
            ke.append(start + cached + i + 1)
        start += cached + new
    rows, n = len(ks), start
    stride = n + spec["logits_pad"]
    base = torch.full((rows, stride), float("nan"), dtype=torch.float32, device="cuda")
    if spec["data"] == "ties":
        base[:, :n] = torch.randint(0, 64, (rows, n), generator=gen, device="cuda").float()
    else:
        # Distinct values per row: a random permutation of a fixed grid (no tie anywhere).
        base[:, :n] = (
            torch.rand(rows, n, generator=gen, device="cuda").argsort(dim=1).float() - n / 2
        ) * 0.25
    return dict(
        logits=base[:, :n],
        row_starts=torch.tensor(ks, dtype=torch.int32, device="cuda"),
        row_ends=torch.tensor(ke, dtype=torch.int32, device="cuda"),
        indices=torch.full((rows, spec["index_topk"]), -5, dtype=torch.int32, device="cuda"),
        index_topk=spec["index_topk"],
    )


def _sorted(t):
    return torch.where(t < 0, 2**31 - 1, t).sort(dim=1).values


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    op(**args)
    out = args["indices"].clone()
    ref = torch.empty_like(out)
    op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    assert torch.equal(_sorted(out), _sorted(ref))
