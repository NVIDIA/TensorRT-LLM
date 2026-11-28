# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_sparse_gqa_e4m3_decode entry (SM103): cells against the
composition entry's fp32 reference in the bf16 decode entry's band (4 ulp of each head row's
magnitude, 2 ulp relative RMS), over an e4m3 pool at the tep4 shard (6 query heads on one KV
head); the composed top-k + op step under graph replay with PDL; and the envelope."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention import (
    qsa_paged_sparse_gqa_e4m3_decode as entry,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_decode import (  # noqa: E501
    qsa_topk_expand_decode,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

op = entry.qsa_paged_sparse_gqa_e4m3_decode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the e4m3 decode GQA entry is certified on SM103",
)

DEV = "cuda"
#: The tep4 shard: 6 q heads on one KV head; the fused attention input row at a tep4 rank is
#: 4352 wide (17 heads of 256): q, its gate, k, v, the index rows and a pad head.
Q_HEADS, KV_HEADS, D, TPB, CR, TOKEN_TOPK, SLOTS, FUSED_HEADS = 6, 1, 256, 32, 4, 2048, 12, 17


def _positions(spec) -> list[int]:
    if isinstance(spec["pos"], tuple):
        return list(spec["pos"])
    rpr = spec.get("rows_per_request", 1)
    return [spec["pos"] + 997 * r + i for r in range(spec.get("requests", 1)) for i in range(rpr)]


def _build(spec, seed):
    """Paged e4m3 K/V (unit scale) for each request, the selection QSA's expansion emits (512
    random complete groups, the incomplete group, then -1), q and the gate as column views of
    the tep4 rank's fused attention input row."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    pos = _positions(spec)
    rows = len(pos)
    rpr = spec.get("rows_per_request", rows if isinstance(spec["pos"], tuple) else 1)
    requests = -(-rows // rpr)
    last = [max(pos[r * rpr : (r + 1) * rpr]) for r in range(requests)]
    pages_per = [p // TPB + 1 for p in last]
    pool_pages = sum(pages_per) + 5
    width = max(pages_per) + 3
    page_ids = torch.randperm(pool_pages, generator=g, device=DEV).to(torch.int32)
    block_table = torch.full((SLOTS, width), -1, dtype=torch.int32, device=DEV)
    slot_of = list(range(1, requests + 1))
    cursor = 0
    for slot, count in zip(slot_of, pages_per):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    pool = torch.randn(pool_pages, 2, KV_HEADS, TPB, D, generator=g, device=DEV)
    pool = pool.to(torch.float8_e4m3fn)
    k_cache, v_cache = pool[:, 0], pool[:, 1]
    fused = torch.randn(rows, FUSED_HEADS * D, generator=g, device=DEV).to(torch.bfloat16)
    q = fused.view(rows, FUSED_HEADS, D)[:, :Q_HEADS]
    gate = fused[:, Q_HEADS * D : 2 * Q_HEADS * D] if spec["gated"] else None
    block_topk = TOKEN_TOPK // CR
    selected = torch.full((rows, TOKEN_TOPK + CR - 1), -1, dtype=torch.int32, device=DEV)
    for r, p in enumerate(pos):
        visible = p + 1
        groups = visible // CR
        chosen = torch.randperm(groups, generator=g, device=DEV)[: min(groups, block_topk)]
        toks = (chosen[:, None] * CR + torch.arange(CR, device=DEV)).reshape(-1)
        tail = torch.arange(groups * CR, visible, device=DEV)
        row = torch.cat([toks, tail]).to(torch.int32)
        selected[r, : row.numel()] = row
        if spec.get("holes"):
            selected[r, 1 : row.numel() : 2] = -1
    request_indices = torch.tensor(
        [slot_of[r // rpr] for r in range(rows)], dtype=torch.int32, device=DEV
    )
    return [q, k_cache, v_cache, block_table, selected, request_indices, TPB, D**-0.5, gate]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    assert entry.supports(*args[:7])
    out = op(*args)
    ref = op.reference(*args)
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape and out.is_contiguous()
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


@pytest.mark.parametrize(
    "rows, splits, tps", [(1, 33, 1), (4, 33, 1), (8, 33, 1), (12, 17, 2), (16, 17, 2), (32, 9, 4)]
)
def test_plan_at_the_selection_width(rows, splits, tps) -> None:
    assert entry.plan(rows, KV_HEADS, TOKEN_TOPK + CR - 1) == (splits, tps)


def test_reads_the_selection_after_the_wait_under_graph_replay() -> None:
    """The target's top-k runs inside the graph right before the op: a PDL producer that
    releases the op right after its own wait, before it writes the selection. A read of the
    selection before the op's wait would see the previous replay's -- where the two launches
    overlap. CUDA may also run them one after the other, and then this replay comparison passes
    whatever the op's read order: it is a graph-replay regression of the composed step, and the
    read-after-wait itself is not shown here."""
    args = _build(dict(pos=(8700,), gated=True), seed=23)
    pos = torch.tensor([8700], dtype=torch.int64, device=DEV)
    ends = ((pos + 1) // CR).to(torch.int32)
    seqlen = (pos + 1).to(torch.int32)
    scores = torch.rand(1, 2304, device=DEV)
    topk = torch.empty(1, TOKEN_TOPK // CR, dtype=torch.int32, device=DEV)

    def step():
        sel = qsa_topk_expand_decode(scores, ends, pos, seqlen, topk, CR, TOKEN_TOPK)
        return op(*args[:4], sel, *args[5:])

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    for trial in range(3):
        scores.copy_(torch.rand(scores.shape, device=DEV))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_out, step()), f"replay {trial}"


def test_envelope() -> None:
    """More than eight query heads a KV head, a bf16 pool, too few selected tokens or too many
    rows are outside the entry."""
    args = _build(dict(pos=(8700,), gated=True), seed=5)
    assert entry.supports(*args[:7])
    q, k, v, bt, sel, req, tpb = args[:7]
    wide = torch.zeros(1, 12, D, dtype=torch.bfloat16, device=DEV)
    assert not entry.supports(wide, k, v, bt, sel, req, tpb)
    assert not entry.supports(q, k.to(torch.bfloat16), v.to(torch.bfloat16), bt, sel, req, tpb)
    assert not entry.supports(q, k, v, bt, sel[:, :192].contiguous(), req, tpb)
    assert not entry.supports(q, k, v, bt, sel, req, tpb, max_rows=0)
    # The hard row ceiling (`MAX_ROWS`, 32) holds whatever the caller's cap: 32 otherwise valid
    # rows are taken and 33 are not, both under `max_rows=33`.
    for rows, taken in ((entry.MAX_ROWS, True), (entry.MAX_ROWS + 1, False)):
        q_n = q[:1].expand(rows, -1, -1).contiguous()
        sel_n = sel[:1].expand(rows, -1).contiguous()
        req_n = req[:1].expand(rows).contiguous()
        assert entry.supports(q_n, k, v, bt, sel_n, req_n, tpb, max_rows=33) is taken, rows
