# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 dsa_topk_order_remap entry: integer output, compared exactly.

The top-k rows are built as a top-k kernel hands them over: distinct request-local key positions
in any order with the unused (-1) slots anywhere in the row. The entry must equal the torch sort
followed by convert_req_index_to_global bit for bit, and the engine op it replaces.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.dsa_topk_order_remap import (  # noqa: E501
    dsa_topk_order_remap_into,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.convert_req_index_to_global import (  # noqa: E501
    convert_req_index_to_global,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_topk_order_remap import (  # noqa: E501
    dsa_topk_order_remap as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_topk_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103

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


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = op(**args)
    ref = op.reference(**args)
    torch.cuda.synchronize()
    assert out.dtype == torch.int32 and out.shape == args["token_indices"].shape
    assert torch.equal(out, ref)
    # The engine op it replaces, on the torch-sorted rows.
    keyed = torch.where(args["token_indices"] < 0, 2**31 - 1, args["token_indices"])
    ordered = keyed.sort(dim=1).values
    ordered = torch.where(ordered == 2**31 - 1, -1, ordered).contiguous()
    engine = convert_req_index_to_global(**dict(args, token_indices=ordered))
    assert torch.equal(out, engine)


@pytest.mark.parametrize("rows,kv_len", [(8, 190_000), (16, 700_000), (3, 3_000)])
def test_order_only_matches_the_torch_sort(rows, kv_len):
    """Without a block table the kernel only orders the rows (``-1`` slots last)."""
    spec = dict(op.CELLS[0].spec, rows=rows, kv_len=kv_len, invalid=37, order="shuffled")
    gen = torch.Generator(device="cuda").manual_seed(rows)
    topk = _rows(spec, gen)
    out = torch.empty_like(topk)
    dsa_topk_order_remap_into(topk, out)
    keyed = torch.where(topk < 0, 2**31 - 1, topk).sort(dim=1).values
    assert torch.equal(out, torch.where(keyed == 2**31 - 1, -1, keyed))


def test_strided_rows_and_graph_replay():
    """Rows of a wider buffer (the indexer's top-k buffer slice) under CUDA-graph replay."""
    spec = op.CELLS[2].spec
    args = _build(spec, seed=11)
    wide = torch.full((spec["rows"] + 2, spec["topk"] + 128), -7, dtype=torch.int32, device="cuda")
    wide[1 : 1 + spec["rows"], : spec["topk"]] = args["token_indices"]
    view = wide[1 : 1 + spec["rows"], : spec["topk"]]
    ref = op.reference(**dict(args, token_indices=view.contiguous()))
    out = torch.empty(spec["rows"], spec["topk"], dtype=torch.int32, device="cuda")
    dsa_topk_order_remap_into(
        view,
        out,
        args["req_id"],
        args["block_table"],
        spec["tokens_per_block"],
        args["stride_factor"],
        spec["layer"],
    )
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    out.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsa_topk_order_remap_into(
            view,
            out,
            args["req_id"],
            args["block_table"],
            spec["tokens_per_block"],
            args["stride_factor"],
            spec["layer"],
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_guard_refuses_a_repeated_position():
    args = _build(op.CELLS[0].spec, seed=1)
    topk = args["token_indices"].clone()
    valid = (topk[0] >= 0).nonzero().flatten()
    topk[0, valid[1]] = topk[0, valid[0]]
    with pytest.raises(AssertionError, match="repeats"):
        with validating(op):
            op(**dict(args, token_indices=topk))


def test_guard_refuses_a_request_past_the_table():
    args = _build(op.CELLS[6].spec, seed=2)
    req_id = args["req_id"].clone()
    req_id[-1] = args["block_table"].shape[0]
    with pytest.raises(AssertionError, match="past the block table"):
        with validating(op):
            op(**dict(args, req_id=req_id))


def test_guard_refuses_int32_overflow():
    args = _build(op.CELLS[0].spec, seed=3)
    with pytest.raises(AssertionError, match="overflow"):
        with validating(op):
            op(**dict(args, stride_factor=2**30))
