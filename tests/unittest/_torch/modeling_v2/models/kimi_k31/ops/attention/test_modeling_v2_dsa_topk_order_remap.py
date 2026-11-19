# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 dsa_topk_order_remap entry: integer output, compared exactly.

The top-k rows are built as a top-k kernel hands them over: distinct request-local key positions
in any order with the unused (-1) slots anywhere in the row. The entry must equal its reference
(a torch sort, then the pool rows) bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_topk_order_remap import (  # noqa: E501
    dsa_topk_order_remap as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_topk_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


#: Context requests ahead of the generation rows in the "gen_rows" table layout.
_NUM_CONTEXTS = 3


def _rows(spec, gen) -> torch.Tensor:
    rows, topk, kv_len, invalid = spec["rows"], spec["topk"], spec["kv_len"], spec["invalid"]
    picks = topk - invalid
    if spec["order"] == "clustered":
        # A few dense runs of consecutive positions.
        starts = torch.randint(0, kv_len - picks, (rows, 1), generator=gen, device="cuda")
        keys = (starts + torch.arange(picks, device="cuda")).int()
    else:
        keys = torch.rand(rows, kv_len, generator=gen, device="cuda").argsort(dim=1)[:, :picks]
        keys = keys.int()
    pad = torch.full((rows, invalid), -1, dtype=torch.int32, device="cuda")
    full = torch.cat([keys, pad], dim=1)
    if spec["order"] == "ascending":
        keyed = torch.where(full < 0, torch.iinfo(torch.int32).max, full).sort(dim=1).values
        return torch.where(keyed == torch.iinfo(torch.int32).max, -1, keyed)
    perm = torch.rand(rows, topk, generator=gen, device="cuda").argsort(dim=1)
    return full.gather(1, perm).contiguous()


def _build(spec, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, requests, tpb = spec["rows"], spec["requests"], spec["tokens_per_block"]
    lead = _NUM_CONTEXTS if spec["table_layout"] == "gen_rows" else 0
    shape = (lead + requests, spec["table_width"])
    table = torch.randint(
        0, spec["pool_pages"], shape, generator=gen, dtype=torch.int32, device="cuda"
    )
    if spec.get("holes"):
        table.masked_fill_(torch.rand(shape, generator=gen, device="cuda") < 0.05, -1)
    req_id = (torch.arange(rows) * requests // rows).int().cuda()
    return dict(
        req_id=req_id,
        block_table=table[lead:],
        token_indices=_rows(spec, gen),
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
