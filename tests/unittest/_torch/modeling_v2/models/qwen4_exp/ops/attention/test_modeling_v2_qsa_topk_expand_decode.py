# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_topk_expand_decode entry.

Cells against the entry's reference (a stable descending sort, ties to the higher block,
then the expansion entry's reference): the block list and the token selection exactly.
Columns past each row's visible count carry garbage (huge values, NaN) that must never
be read. Beyond the cells: the wide rows against the canonical composition, a
stale-read check under graph replay at each decode-path cell's shape, and the
`token_topk` guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_canonical_selection import (  # noqa: E501
    qsa_canonical_selection,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_expand_block_indices import (  # noqa: E501
    qsa_expand_block_indices,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_decode import (  # noqa: E501
    qsa_topk_expand_decode as op,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_select import (  # noqa: E501
    qsa_topk_select,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qsa_topk_expand_decode requires a CUDA device"

DEV = "cuda"
K, CR, TOKEN_TOPK = 512, 4, 2048


def _build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    W = spec["width"]
    if "chunk" in spec or "requests" in spec:
        # Each request's consecutive rows (a context chunk, or a verify step's 1 + draft
        # rows): every row's KV length is the request's last row + 1.
        requests = spec["requests"] if "requests" in spec else (spec["chunk"],)
        pos = torch.cat(
            [torch.arange(s, s + c, dtype=torch.int64, device=DEV) for s, c in requests]
        )
        seqlen = torch.cat(
            [torch.full((c,), s + c, dtype=torch.int32, device=DEV) for s, c in requests]
        )
    else:
        pos = torch.tensor(spec["pos"], dtype=torch.int64, device=DEV)
        seqlen = (pos + 1).to(torch.int32)
    rows = pos.numel()
    assert rows == spec["rows"], (rows, spec["rows"])
    scores = torch.randn(rows, W, generator=g, device=DEV).abs() * 3.0
    if spec.get("relu_sparse"):
        # Fewer positive scores than the budget, so the zero class straddles it.
        scores = torch.relu(torch.randn(rows, W, generator=g, device=DEV) - 2.2)
        scores[:, ::7] = -0.0  # negative zero ties with zero
    if spec.get("relu_sum"):
        # The indexer's form: four heads' ReLUs summed, times a scale (zero in 1 of 16).
        scores = torch.zeros(rows, W, device=DEV)
        for _ in range(4):
            scores += torch.relu(torch.randn(rows, W, generator=g, device=DEV))
        scores *= 0.0884
    if spec.get("equal"):
        # Every score tied: the budget is the highest-index blocks.
        scores = torch.full((rows, W), 0.5, device=DEV)
    if spec.get("narrow"):
        scores = 1.0 + 0.05 * torch.rand(rows, W, generator=g, device=DEV)
    if spec.get("lognormal"):
        scores = torch.exp(1.74 + 0.2 * torch.randn(rows, W, generator=g, device=DEV))
        # bf16-like resolution, so equal scores tie; a zero in every other row.
        scores = scores.to(torch.bfloat16).float()
        scores[::2, 5] = 0.0
    ends = ((pos + 1) // CR).to(torch.int32)
    assert int(ends.max()) <= W, "a window wider than the score table"
    # Garbage past the visible blocks: huge values, and a NaN three columns on.
    cols = torch.arange(W, device=DEV)
    scores = torch.where(cols[None, :] >= ends[:, None], 1.0e30, scores)
    scores = torch.where(cols[None, :] == ends[:, None] + 3, float("nan"), scores)
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


@pytest.mark.parametrize("width", [65536, 65568, 65600])
@pytest.mark.parametrize("distribution", ["relu_sum", "relu_sparse", "equal"])
def test_wide_rows_equal_the_canonical_composition(width: int, distribution: str) -> None:
    """Match the fused block order and expansion to exact top-k plus canonical rewrite.

    The TP1 table includes eight pages of headroom; the default table includes four.
    Ragged windows cross the block budget and slab boundaries and carry every causal
    tail width. Both paths must ignore the poisoned score columns past each window.
    """
    positions = (0, 1, 2, 3, 2043, 2047, 2051, 16383, 84000, 99999, 262143)
    spec = dict(rows=len(positions), pos=positions, width=width, **{distribution: True})
    args = _build(spec, seed=width)
    scores, ends, pos, lengths, fused_topk, cr, token_topk = args
    lengths[8] += 3  # A verify row whose request continues past this query position.
    starts = torch.zeros_like(ends)
    composed_topk = torch.full_like(fused_topk, -1)
    qsa_topk_select(scores, starts, ends, composed_topk, index_topk=K)
    qsa_canonical_selection(scores, starts, ends, composed_topk)
    composed = qsa_expand_block_indices(
        composed_topk, pos, lengths, compress_ratio=cr, token_topk=token_topk
    )
    fused = op(*args)
    expected_topk, expected = op.reference(*args)
    assert torch.equal(composed_topk, expected_topk), "canonical block list"
    assert torch.equal(composed, expected), "canonical token expansion"
    assert torch.equal(fused_topk, composed_topk), "fused block order"
    assert torch.equal(fused, composed), "fused token expansion"


# The decode-path shapes, captured in the decode CUDA graph: tp1's one row at 2560 columns,
# the default target's relu-sparse decode and verify cells (1 and 128 rows at core's width, 1024
# at MTP's), and the wide tables' decode cells (`graph` in their spec).
_GRAPH_SPECS = [dict(rows=1, pos=(8700,), width=2560)] + [
    c.spec
    for c in op.CELLS
    if c.spec.get("graph") or (c.spec.get("relu_sparse") and "out_margin" not in c.spec)
]


@pytest.mark.parametrize(
    "spec", _GRAPH_SPECS, ids=[f"{s['rows']}x{s['width']}" for s in _GRAPH_SPECS]
)
def test_reads_after_the_wait_under_graph_replay(spec) -> None:
    args = _build(spec, seed=21)
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
        src.copy_(_build(spec, seed=100 + trial)[0])
        graph.replay()
        torch.cuda.synchronize()
        got_topk = args[4].clone()
        sel = op(src.clone(), *args[1:])
        assert torch.equal(graph_sel, sel) and torch.equal(got_topk, args[4]), f"replay {trial}"
        want_topk, want_sel = op.reference(src, *args[1:])
        assert torch.equal(got_topk, want_topk), f"replay {trial}: block list"
        assert torch.equal(graph_sel, want_sel), f"replay {trial}: token selection"


def test_guard_refuses_a_token_budget_that_is_not_k_blocks() -> None:
    """`token_topk` sizes the selection's rows while `k * compress_ratio` places the selected
    tokens: a smaller budget writes past each row, into the next row and, for the last one,
    past `out` -- here into the margin rows of a larger buffer, where it shows."""
    spec = dict(rows=2, pos=(9215, 9211), relu_sparse=True, width=2304)
    args = _build(spec, seed=5)
    budget = TOKEN_TOPK // 2  # 256 blocks' worth, while 512 are selected
    margin = 2
    whole = torch.full((2 + 2 * margin, budget + CR - 1), -9, dtype=torch.int32, device=DEV)
    args[6] = budget
    op(*args, out=whole[margin : margin + 2])
    torch.cuda.synchronize()
    assert (whole[margin + 2 :] != -9).any(), "the hazard this guard covers is gone"

    with validating(op), pytest.raises(AssertionError, match="token_topk = k \\* compress_ratio"):
        op(*args, out=whole[margin : margin + 2])
