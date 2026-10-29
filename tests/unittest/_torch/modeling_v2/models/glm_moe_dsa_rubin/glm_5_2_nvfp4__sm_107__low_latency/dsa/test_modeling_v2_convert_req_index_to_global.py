# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM convert_req_index_to_global entry: integer pool rows, compared exactly.

The block table has more rows than requests (the context rows of the engine's table come first
when ``table_layout`` is ``gen_rows``) and -1 entries past each request's pages; the TopK rows hold
request-local tokens below each request's KV length, with ``invalid`` -1 slots at the end.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.dsa.convert_req_index_to_global import (  # noqa: E501
    convert_req_index_to_global as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's entries are certified on sm_107"
)

assert torch.cuda.is_available(), "convert_req_index_to_global requires a CUDA device"


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    rows, topk, tpb = spec["rows"], spec["topk"], spec["tokens_per_block"]
    width, pool = spec["table_width"], spec["pool_pages"]
    pages = -(-spec["kv_len"] // tpb)
    table = torch.full((rows + 2, width), -1, dtype=torch.int32)
    for r in range(rows + 2):
        table[r, :pages] = torch.randint(0, pool, (pages,), generator=gen, dtype=torch.int32)
    if spec["table_layout"] == "gen_rows":
        # The engine's table holds the context requests' rows first; the op gets the generation
        # requests' rows, a view starting past them.
        table = table.cuda()[2:]
    else:
        table = table[:rows].cuda()
    # A request shorter than the TopK fills its first kv_len slots; the rest are -1.
    valid = topk - spec["invalid"]
    tokens = torch.full((rows, topk), -1, dtype=torch.int32)
    for r in range(rows):
        tokens[r, :valid] = torch.randperm(spec["kv_len"], generator=gen)[:valid].to(torch.int32)
    return dict(
        req_id=torch.arange(rows, dtype=torch.int32, device="cuda"),
        block_table=table,
        token_indices=tokens.cuda(),
        block_size=tpb,
        num_topk_tokens=topk,
        stride_factor=spec["page_index_scale"] * tpb,
        layer_id=spec["layer"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref = op.reference(**args)
    out = op(**args)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    if cell.spec["invalid"]:
        assert (out[:, -cell.spec["invalid"] :] == -1).all()


def test_guard_refuses_fewer_topk_columns_than_the_rows_hold():
    args = _build(op.CELLS[0].spec, seed=81)
    with pytest.raises(AssertionError, match="unwritten"):
        with validating(op):
            op(**dict(args, num_topk_tokens=1024))


def test_guard_refuses_a_request_past_the_table():
    args = _build(op.CELLS[0].spec, seed=82)
    req = torch.full_like(args["req_id"], args["block_table"].shape[0])
    with pytest.raises(AssertionError, match="past the block table"):
        with validating(op):
            op(**dict(args, req_id=req))
