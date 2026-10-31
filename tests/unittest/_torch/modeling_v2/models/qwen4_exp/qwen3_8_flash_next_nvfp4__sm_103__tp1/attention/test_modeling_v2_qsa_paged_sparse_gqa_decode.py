# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_sparse_gqa_decode entry: cells against the composition
entry's fp32 reference in that entry's band (4 ulp of each head row's magnitude, 2 ulp
relative RMS), the selection's read-after-wait under graph replay with PDL, and the row
guard."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_topk_expand_decode import (  # noqa: E501
    qsa_topk_expand_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_sparse_gqa_decode import (  # noqa: E501
    qsa_paged_sparse_gqa_decode as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa_decode requires a CUDA device"

DEV = "cuda"
Q_HEADS, KV_HEADS, D, TPB, CR, TOKEN_TOPK, PAGES, SLOTS = 24, 2, 256, 32, 4, 2048, 320, 6


def _build(spec, seed):
    """Paged K/V for one decode row per position, the selection QSA's expansion emits
    (512 random complete groups, then the incomplete group, then -1), q and the gate as
    column views of the target's 55-head fused projection row."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    pos = torch.tensor(spec["pos"], dtype=torch.int64, device=DEV)
    rows = pos.numel()
    pages_per = [int(p) // TPB + 1 for p in spec["pos"]]
    pool_pages = sum(pages_per) + 5
    page_ids = torch.randperm(pool_pages, generator=g, device=DEV).to(torch.int32)
    block_table = torch.full((SLOTS, PAGES), -1, dtype=torch.int32, device=DEV)
    slot_of = list(range(1, rows + 1))
    cursor = 0
    for slot, count in zip(slot_of, pages_per):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    pool = torch.randn(pool_pages, 2, KV_HEADS, TPB, D, generator=g, device=DEV).to(torch.bfloat16)
    k_cache, v_cache = pool[:, 0], pool[:, 1]
    fused = torch.randn(rows, 55 * D, generator=g, device=DEV).to(torch.bfloat16)
    q = fused.view(rows, 55, D)[:, :Q_HEADS]
    gate = fused[:, 28 * D : (28 + Q_HEADS) * D] if spec["gated"] else None
    block_topk = TOKEN_TOPK // CR
    selected = torch.full((rows, TOKEN_TOPK + CR - 1), -1, dtype=torch.int32, device=DEV)
    for r, p in enumerate(spec["pos"]):
        visible = p + 1
        groups = visible // CR
        chosen = torch.randperm(groups, generator=g, device=DEV)[: min(groups, block_topk)]
        toks = (chosen[:, None] * CR + torch.arange(CR, device=DEV)).reshape(-1)
        tail = torch.arange(groups * CR, visible, device=DEV)
        row = torch.cat([toks, tail]).to(torch.int32)
        selected[r, : row.numel()] = row
        if spec.get("holes"):
            selected[r, 1 : row.numel() : 2] = -1
    request_indices = torch.tensor(slot_of, dtype=torch.int32, device=DEV)
    return [q, k_cache, v_cache, block_table, selected, request_indices, TPB, D**-0.5, gate]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    out = op(*args)
    ref = op.reference(*args)
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape and out.is_contiguous()
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _replays_like_eager(step, refresh) -> None:
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    for trial in range(3):
        refresh()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_out, step()), f"replay {trial}"


def test_reads_the_selection_after_the_wait_under_graph_replay() -> None:
    """The target's top-k runs inside the graph right before the op: a PDL producer that
    releases the op right after its own wait, before it writes the selection. A read of the
    selection before the op's wait would see the previous replay's."""
    args = _build(dict(pos=(8700,), gated=True), seed=23)
    pos = torch.tensor([8700], dtype=torch.int64, device=DEV)
    ends = ((pos + 1) // CR).to(torch.int32)
    seqlen = (pos + 1).to(torch.int32)
    scores = torch.rand(1, 2304, device=DEV)
    topk = torch.empty(1, TOKEN_TOPK // CR, dtype=torch.int32, device=DEV)

    def step():
        sel = qsa_topk_expand_decode(scores, ends, pos, seqlen, topk, CR, TOKEN_TOPK)
        return op(*args[:4], sel, *args[5:])

    _replays_like_eager(step, lambda: scores.copy_(torch.rand(scores.shape, device=DEV)))


def test_guard_refuses_rows_past_max_rows() -> None:
    args = _build(dict(pos=(8700,), gated=True), seed=5)
    many = [a.expand(300, *a.shape[1:]) if i in (0, 4, 5) else a for i, a in enumerate(args[:6])]
    with validating(op), pytest.raises(AssertionError):
        op(*many, *args[6:8], None)  # 300 rows: past MAX_ROWS
