# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 convert_req_index_to_global entry: integer output, compared exactly.

The top-k rows are built as the target hands them over: distinct request-local key positions,
each row sorted ascending with its -1 slots last.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.convert_req_index_to_global import (  # noqa: E501
    convert_req_index_to_global as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_topk_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="rows")[1:2]


#: Context requests ahead of the generation rows in the "gen_rows" table layout.
_NUM_CONTEXTS = 3


def _sorted_rows(rows: int, topk: int, kv_len: int, invalid: int, gen) -> torch.Tensor:
    picks = topk - invalid
    keys = torch.rand(rows, kv_len, generator=gen).argsort(dim=1)[:, :picks].int()
    keys = keys.sort(dim=1).values
    pad = torch.full((rows, invalid), -1, dtype=torch.int32)
    return torch.cat([keys, pad], dim=1)


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    rows, requests, tpb = spec["rows"], spec["requests"], spec["tokens_per_block"]
    lead = _NUM_CONTEXTS if spec["table_layout"] == "gen_rows" else 0
    table_rows = lead + requests
    shape = (table_rows, spec["table_width"])
    table = torch.randint(0, spec["pool_pages"], shape, generator=gen, dtype=torch.int32).cuda()
    req_id = (torch.arange(rows) * requests // rows).int().cuda()
    token_indices = _sorted_rows(rows, spec["topk"], spec["kv_len"], spec["invalid"], gen).cuda()
    return dict(
        req_id=req_id,
        block_table=table[lead:],
        token_indices=token_indices,
        block_size=tpb,
        num_topk_tokens=spec["topk"],
        stride_factor=spec["page_index_scale"] * tpb,
        layer_id=spec["layer"],
    )


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = op(**args)
    ref = op.reference(**args)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
