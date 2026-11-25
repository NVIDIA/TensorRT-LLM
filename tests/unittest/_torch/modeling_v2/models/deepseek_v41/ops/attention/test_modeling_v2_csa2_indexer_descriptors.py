# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_indexer_descriptors entry: block tables, lengths and page validity of the paged indexer.

Integer index arithmetic only: every cell compares the kernel with the plain-torch reference bit for bit.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_descriptors import (  # noqa: E501
    csa2_indexer_descriptors,
)


def _ids(c):
    return [x.why[:48] for x in c]


def _inputs(spec, gen):
    nreq, rows, pages, ratio = spec["nreq"], spec["rows"], spec["pages"], spec["ratio"]
    table = (
        torch.randperm(nreq * pages + 5, generator=gen, device=DEV)[: nreq * pages]
        .view(nreq, pages)
        .int()
    )
    table = torch.where(
        torch.rand(nreq, pages, generator=gen, device=DEV) < spec["unmapped"], -1, table
    ).int()
    pos = torch.randint(
        0, pages * (128 // ratio) * ratio, (nreq * rows,), generator=gen, device=DEV
    )
    rid = torch.arange(nreq, device=DEV).repeat_interleave(rows)
    if spec.get("stray"):
        rid[-3:] = nreq + 3  # rows whose request is outside the table
    return table.contiguous(), rid, ((pos + 1) // ratio).long(), ratio, spec.get("masked", False)


@pytest.mark.parametrize(
    "cell", csa2_indexer_descriptors.CELLS, ids=_ids(csa2_indexer_descriptors.CELLS)
)
def test_indexer_descriptors_cells(cell, certifying):  # noqa: F811
    certifying(csa2_indexer_descriptors)
    args = _inputs(cell.spec, gen_for(cell))
    with validating(csa2_indexer_descriptors):
        got = csa2_indexer_descriptors(*args)
    ref = csa2_indexer_descriptors.reference(*args)
    assert len(got) == len(ref) == (5 if cell.spec.get("masked") else 4)
    for name, g, r in zip(("blocks", "context", "visible", "valid", "masked_blocks"), got, ref):
        assert g.shape == r.shape and g.dtype == r.dtype, (name, g.shape, r.shape, g.dtype, r.dtype)
        assert torch.equal(g, r), name
    assert not got[3].all() or cell.spec["unmapped"] == 0, "the cell must exercise unmapped pages"


def test_indexer_descriptors_worked_example():
    table = torch.tensor([[4, -1, 2]], dtype=torch.int32, device=DEV)
    rid = torch.zeros(2, dtype=torch.int64, device=DEV)
    visible = torch.tensor([70, 0], dtype=torch.int64, device=DEV)
    with validating(csa2_indexer_descriptors):
        blocks, context, vis, valid = csa2_indexer_descriptors(table, rid, visible, 1)
    # ratio 1: a GLOBAL page is two native pages
    assert blocks[0].tolist() == [8, 9, 0, 0, 4, 5]
    assert valid[0].tolist() == [True, True, False, False, True, True]
    assert context.flatten().tolist() == [70, 1] and vis.tolist() == [70, 0]
    with validating(csa2_indexer_descriptors):
        masked_blocks = csa2_indexer_descriptors(table, rid, visible, 1, masked=True)[4]
    flag = 1 << 30  # DEEPGEMM_MASKED_PAGE: the unmapped GLOBAL page's two native pages
    assert masked_blocks[0].tolist() == [8, 9, flag, flag, 4, 5]


def test_cuda_graph_replay_follows_device_data():
    """Visible lengths are device data: a replay with new lengths must give what an eager call gives."""
    spec = dict(ratio=2, nreq=8, rows=4, pages=60, unmapped=0.2)
    table, rid, visible, ratio, _ = _inputs(spec, torch.Generator(device=DEV).manual_seed(2))
    run = lambda v: csa2_indexer_descriptors(table, rid, v, ratio)  # noqa: E731
    run(visible)  # eager warm-up (kernel JIT)
    torch.cuda.synchronize()
    static = visible.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run(static)
    static.copy_(visible // 2)
    g.replay()
    torch.cuda.synchronize()
    ref = run(visible // 2)
    assert all(torch.equal(a, b) for a, b in zip(out, ref))
