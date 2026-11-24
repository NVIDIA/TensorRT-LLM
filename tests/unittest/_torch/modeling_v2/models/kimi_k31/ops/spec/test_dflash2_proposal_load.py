# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k31_dflash2_proposal_load entry: every cell bitwise against the store's load."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_proposal_load import (  # noqa: E501
    k31_dflash2_proposal_load as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: F401 — needs the path declared above

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (103, 107),
    reason="the K3.1 DFlash2 proposal entry is certified on sm_103 and sm_107",
)

K_MAX = 7


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, slots, rows = spec["n"], spec["slots"], spec["rows"]
    store_cand = torch.randint(0, 163840, (rows, K_MAX, 32), generator=g, device="cuda")
    store_q = torch.rand(rows, K_MAX, 32, generator=g, device="cuda")
    store_tok = torch.randint(0, 163840, (rows, K_MAX), generator=g, device="cuda")
    store_tok[-1] = -1  # an invalidated row
    slot_ids = torch.randperm(rows, generator=g, device="cuda")[:n]
    drafts = store_tok[slot_ids, :slots].clone()
    if spec["walked"] != "all":
        changed = torch.rand(n, generator=g, device="cuda") < 0.5
        if spec["walked"] == "none":
            changed[:] = True
        pos = torch.randint(0, slots, (n,), generator=g, device="cuda")
        bumped = drafts.clone()
        bumped[torch.arange(n, device="cuda"), pos] += 1
        drafts = torch.where(changed[:, None], bumped, drafts)
    return [store_cand, store_q, store_tok, slot_ids.to(spec["ids"]), drafts.to(spec["ids"])]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    cand, q = op(*args)
    cand_ref, q_ref = op.reference(*args)
    assert torch.equal(cand, cand_ref)
    assert torch.equal(q, q_ref)


def test_cuda_graph_replay() -> None:
    args = _build(op.CELLS[2].spec, seed=3)
    op(*args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            cand, q = op(*args)
    torch.cuda.current_stream().wait_stream(stream)
    fresh = _build(op.CELLS[2].spec, seed=4)
    for static, new in zip(args, fresh):
        static.copy_(new)
    graph.replay()
    cand_ref, q_ref = op.reference(*args)
    torch.cuda.synchronize()
    assert torch.equal(cand, cand_ref) and torch.equal(q, q_ref)


def test_guard_refuses_a_longer_block_than_the_store() -> None:
    args = _build(op.CELLS[2].spec, seed=5)
    args[4] = torch.cat([args[4], args[4][:, :1]], dim=1)
    with pytest.raises(AssertionError, match="K exceeds the store"):
        with validating(op):
            op(*args)
