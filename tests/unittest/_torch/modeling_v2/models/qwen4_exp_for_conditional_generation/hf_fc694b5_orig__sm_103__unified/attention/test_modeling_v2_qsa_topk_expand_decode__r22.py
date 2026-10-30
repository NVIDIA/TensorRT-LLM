# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_topk_expand_decode entry.

Cells against the entry's reference (a stable descending sort, ties to the higher block,
then the expansion entry's reference): the block list and the token selection exactly.
Columns past each row's visible count carry garbage (huge values, NaN) that must never
be read. Beyond the cells: a stale-read check under graph replay.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.attention.qsa_topk_expand_decode__r22 import (  # noqa: E501
    qsa_topk_expand_decode as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qsa_topk_expand_decode requires a CUDA device"

DEV = "cuda"
W, K, CR, TOKEN_TOPK = 2560, 512, 4, 2048


def _build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    if "chunk" in spec:
        # One request's consecutive context rows: every row's KV length is the chunk's end.
        start, count = spec["chunk"]
        pos = torch.arange(start, start + count, dtype=torch.int64, device=DEV)
        seqlen = torch.full((count,), start + count, dtype=torch.int32, device=DEV)
    else:
        pos = torch.tensor(spec["pos"], dtype=torch.int64, device=DEV)
        seqlen = (pos + 1).to(torch.int32)
    rows = pos.numel()
    scores = torch.randn(rows, W, generator=g, device=DEV).abs() * 3.0
    if spec.get("relu_sparse"):
        # Fewer positive scores than the budget, so the zero class straddles it.
        scores = torch.relu(torch.randn(rows, W, generator=g, device=DEV) - 2.2)
        scores[:, ::7] = -0.0  # negative zero ties with zero
    if spec.get("narrow"):
        scores = 1.0 + 0.05 * torch.rand(rows, W, generator=g, device=DEV)
    if spec.get("lognormal"):
        scores = torch.exp(1.74 + 0.2 * torch.randn(rows, W, generator=g, device=DEV))
        # bf16-like resolution, so equal scores tie; a zero in every other row.
        scores = scores.to(torch.bfloat16).float()
        scores[::2, 5] = 0.0
    ends = ((pos + 1) // CR).to(torch.int32)
    for r in range(rows):  # garbage past the visible blocks
        n = int(ends[r])
        scores[r, n:] = 1.0e30
        if n + 3 < W:
            scores[r, n + 3] = float("nan")
    topk = torch.full((rows, K), -7, dtype=torch.int32, device=DEV)
    return [scores, ends, pos, seqlen, topk, CR, TOKEN_TOPK]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    want_topk, want_sel = op.reference(*args)
    margin = cell.spec.get("out_margin")
    if margin is None:
        sel = op(*args)
    else:
        # Rows [margin, margin + rows) of a step-sized selection; the rows around stay put.
        rows = args[0].shape[0]
        whole = torch.full(
            (rows + 2 * margin, TOKEN_TOPK + CR - 1), -9, dtype=torch.int32, device=DEV
        )
        sel = op(*args, out=whole[margin : margin + rows])
        assert sel.data_ptr() == whole[margin].data_ptr()
        assert (whole[:margin] == -9).all() and (whole[margin + rows :] == -9).all(), (
            "rows outside out"
        )
    assert sel.dtype == torch.int32 and sel.shape == want_sel.shape
    assert torch.equal(args[4], want_topk), "block list"
    assert torch.equal(sel, want_sel.to(torch.int32)), "token selection"


def test_reads_after_the_wait_under_graph_replay() -> None:
    args = _build(dict(pos=(8700,)), seed=21)
    src = args[0].clone()

    def step():
        args[0].copy_(src * 1.0)
        return op(*args)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_sel = step()
    for trial in range(3):
        src.copy_(_build(dict(pos=(8700,)), seed=100 + trial)[0])
        graph.replay()
        torch.cuda.synchronize()
        got_topk = args[4].clone()
        sel = op(src.clone(), *args[1:])
        assert torch.equal(graph_sel, sel) and torch.equal(got_topk, args[4]), f"replay {trial}"
