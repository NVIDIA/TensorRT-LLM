# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_topk_expand_decode entry.

Cells against the entry's reference (a stable descending sort, ties to the higher block,
then the expansion entry's reference): the block list and the token selection exactly.
Columns past each row's visible count carry garbage (huge values, NaN) that must never
be read. Beyond the cells: bitwise the composition it replaces (`qsa_topk_select`, then
`qsa_canonical_selection`, then `qsa_expand_block_indices`, each certified on its own) on
one representative decode batch and one context chunk; a stale-read check under graph
replay; one compiled kernel for every row count.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_canonical_selection import (  # noqa: E501
    qsa_canonical_selection,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_expand_block_indices import (  # noqa: E501
    qsa_expand_block_indices,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_topk_expand_decode import (  # noqa: E501
    _COMPILED,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_topk_expand_decode import (  # noqa: E501
    qsa_topk_expand_decode as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_topk_select import (  # noqa: E501
    qsa_topk_select,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qsa_topk_expand_decode requires a CUDA device"

DEV = "cuda"
W, K, CR, TOKEN_TOPK = 2560, 512, 4, 2048


def _build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    pos = torch.tensor(spec["pos"], dtype=torch.int64, device=DEV)
    rows = pos.numel()
    scores = torch.randn(rows, W, generator=g, device=DEV).abs() * 3.0
    if spec.get("ties"):
        scores = torch.round(scores * 2.0) / 2.0  # few distinct values: ties at the threshold
    if spec.get("narrow"):
        scores = 1.0 + 0.05 * torch.rand(rows, W, generator=g, device=DEV)
    ends = ((pos + 1) // CR).to(torch.int32)
    for r in range(rows):  # garbage past the visible blocks
        n = int(ends[r])
        scores[r, n:] = 1.0e30
        if n + 3 < W:
            scores[r, n + 3] = float("nan")
    seqlen = (pos + 1).to(torch.int32)
    topk = torch.full((rows, K), -7, dtype=torch.int32, device=DEV)
    return [scores, ends, pos, seqlen, topk, CR, TOKEN_TOPK]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    want_topk, want_sel = op.reference(*args)
    sel = op(*args)
    assert sel.dtype == torch.int32 and sel.shape == want_sel.shape
    assert torch.equal(args[4], want_topk), "block list"
    assert torch.equal(sel, want_sel.to(torch.int32)), "token selection"


def _relu_sparse_rows(rows: int, seed: int) -> torch.Tensor:
    """Score rows the canonical rewrite exists for: relu-sparse, so fewer positive scores
    than the budget and the zero class (negative zeros among it) straddles it."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.relu(torch.randn(rows, W, generator=g, device=DEV) - 2.2)
    scores[:, ::7] = -0.0  # negative zero ties with zero
    return scores


def test_bitwise_the_canonical_composition() -> None:
    """Windows across the budget (below it, at it, past it, the whole table); the block
    list equals `qsa_topk_select` + `qsa_canonical_selection` and the token selection that
    list's expansion, bitwise."""
    rows = 64
    scores = _relu_sparse_rows(rows, seed=zlib.crc32(b"relu_sparse") % 2**31)
    g = torch.Generator(device=DEV).manual_seed(5)
    pos = torch.randint(0, CR * W, (rows,), generator=g, device=DEV, dtype=torch.int64)
    pos[:6] = torch.tensor([0, 4 * K - 1, 4 * K, 4 * K + 3, 8700, CR * W - 1], device=DEV)
    ends = ((pos + 1) // CR).to(torch.int32)
    seqlen = (pos + 1).to(torch.int32)
    topk = torch.full((rows, K), -7, dtype=torch.int32, device=DEV)
    sel = op(scores, ends, pos, seqlen, topk, CR, TOKEN_TOPK)
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    want = torch.full_like(topk, -7)
    qsa_topk_select(scores, starts, ends, want, index_topk=K)
    qsa_canonical_selection(scores, starts, ends, want)
    want_sel = qsa_expand_block_indices(want, pos, seqlen, compress_ratio=CR, token_topk=TOKEN_TOPK)
    assert torch.equal(topk, want), "block list"
    assert torch.equal(sel, want_sel.to(torch.int32)), "token selection"


def test_a_context_chunk_into_a_row_range_of_the_selection() -> None:
    """A context chunk's 2048 consecutive rows (one request, positions 6144..8191), written
    through `out` into rows [64, 2112) of a step-sized selection: the block list and those
    rows equal the canonical composition bitwise; the rows around them are untouched."""
    rows = 2048
    scores = _relu_sparse_rows(rows, seed=31)
    pos = torch.arange(6144, 6144 + rows, device=DEV, dtype=torch.int64)
    ends = ((pos + 1) // CR).to(torch.int32)
    seqlen = torch.full((rows,), 6144 + rows, dtype=torch.int32, device=DEV)
    topk = torch.full((rows, K), -7, dtype=torch.int32, device=DEV)
    whole = torch.full((rows + 128, TOKEN_TOPK + CR - 1), -9, dtype=torch.int32, device=DEV)
    sel = op(scores, ends, pos, seqlen, topk, CR, TOKEN_TOPK, out=whole[64 : 64 + rows])
    assert sel.data_ptr() == whole[64].data_ptr()
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    want = torch.full_like(topk, -7)
    qsa_topk_select(scores, starts, ends, want, index_topk=K)
    qsa_canonical_selection(scores, starts, ends, want)
    want_sel = qsa_expand_block_indices(want, pos, seqlen, compress_ratio=CR, token_topk=TOKEN_TOPK)
    assert torch.equal(topk, want), "block list"
    assert torch.equal(whole[64 : 64 + rows], want_sel.to(torch.int32)), "token selection"
    assert (whole[:64] == -9).all() and (whole[64 + rows :] == -9).all(), "rows outside out"


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


def test_one_kernel_serves_every_row_count() -> None:
    """The row count is a runtime argument: an eager step's context rows come in any count,
    and every count here runs the one kernel compiled for the table, each matching the
    reference."""
    compiled = None
    for rows in (1, 2, 7, 33, 300):
        g = torch.Generator(device=DEV).manual_seed(rows)
        pos = torch.randint(0, CR * W, (rows,), generator=g, device=DEV, dtype=torch.int64)
        args = _build(dict(pos=tuple(pos.tolist())), seed=rows)
        want_topk, want_sel = op.reference(*args)
        sel = op(*args)
        assert torch.equal(args[4], want_topk), f"block list, {rows} rows"
        assert torch.equal(sel, want_sel.to(torch.int32)), f"token selection, {rows} rows"
        compiled = len(_COMPILED) if compiled is None else compiled
        assert len(_COMPILED) == compiled, f"{rows} rows compiled another kernel"
