# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for `csa2_indexer_topk_packed` (stock's packed context indexer: prefix gather, non-paged FP4 MQA logits,
prefill top-k per row) and its host plan `csa2_indexer_plan`, against the dense per-request oracle.

The logits kernel accumulates in its own order, so the gate is the indexer entries' one: the unreachable (-1) picks
agree exactly and more than 97% of the reference's picks are in the op's set. The candidate source is gated on its
position set against the tiled `csa2_indexer_topk_prefill` publication (same 97%), and a consumer over the packed
candidates on the score band of `check_consumer_picks`.
"""

import pytest
import torch
from _v41_util import DEV, IS_SM103, check_consumer_picks, overlap, request_index_pools

__extra_import_path__ = ["../../../.."]
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_decode import (  # noqa: E501
    csa2_indexer_topk_prefill,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_packed import (  # noqa: E501
    csa2_indexer_plan,
    csa2_indexer_topk_packed,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_pack_rows import (  # noqa: E501
    csa2_pack_rows,
    csa2_unpack_rows,
)

pytestmark = pytest.mark.skipif(
    not IS_SM103, reason="the FP4 MQA-logits kernels are certified on sm_103 only"
)
HEADS, TOPK, MAX_SEQ = 32, 512, 1 << 17  # 32 heads: the candidate consumer's sparse logits kernel
BUDGET = 1 << 30


def _qdq(q):
    return csa2_unpack_rows(csa2_pack_rows(q.reshape(-1, 128), "index"), "index", q.dtype).view_as(
        q
    )


def _step(ratio, reqs, seed=0):
    """A context step of requests `(history, rows)`: pool, table, row positions / requests, segments, queries."""
    lens = [h + n for h, n in reqs]
    pool, table = request_index_pools(ratio, lens, seed)
    pos = torch.cat([torch.arange(h, h + n, dtype=torch.int32) for h, n in reqs]).to(DEV)
    rid = torch.cat([torch.full((n,), r, dtype=torch.long) for r, (h, n) in enumerate(reqs)]).to(
        DEV
    )
    segs, t = [], 0
    for r, (h, n) in enumerate(reqs):
        segs.append((t, t + n, r, h + n))
        t += n
    q = _qdq(torch.randn(len(pos), HEADS, 128, device=DEV).bfloat16())
    w = torch.rand(len(pos), HEADS, device=DEV) * 0.1
    vis = ((pos.long() + 1) // ratio).contiguous()
    return pool, table, pos, rid, segs, q, w, vis


def test_plan_follows_stock_grouping():
    """Pieces of `budget // columns(width)` rows; consecutive requests merge up to the chunk and budget."""
    segs = [(0, 300, 0, 3300), (300, 420, 1, 320), (420, 470, 2, 50)]
    groups, desc = csa2_indexer_plan(segs, 2, MAX_SEQ // 2, BUDGET, DEV)
    assert [g[:4] for g in groups] == [(0, 470, 1650, 1650 + 160 + 25, 0, 3)[:4]]
    assert desc.cpu().tolist() == [
        [0, 1650, 0, 0, 0],
        [1, 160, 1650, 300, 300],
        [2, 25, 1810, 420, 420],
    ]
    budget = 1 << 20
    groups, desc = csa2_indexer_plan([(0, 4000, 0, 6000)], 1, MAX_SEQ, budget, DEV)
    step = budget // 6144  # rows per piece at the widest piece's 256-padded columns
    d = desc.cpu().tolist()
    assert [p[3] for p in d] == list(range(0, 4000, step))  # pieces of `step` rows
    assert [p[1] for p in d] == [
        2000 + min(s + step, 4000) for s in range(0, 4000, step)
    ]  # each its last row's view
    assert (
        groups[0][0] == 0
        and groups[-1][1] == 4000
        and all(a[1] == b[0] for a, b in zip(groups, groups[1:]))
    )
    assert len(groups) > 1 and all(
        (g[1] - g[0]) * (-(-g[2] // 256) * 256) <= budget for g in groups
    )


@pytest.mark.parametrize(
    "cell", csa2_indexer_topk_packed.CELLS, ids=[c.why[:44] for c in csa2_indexer_topk_packed.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_indexer_topk_packed)
    s = cell.spec
    ratio = s["ratio"]
    pool, table, pos, rid, segs, q, w, vis = _step(ratio, s["reqs"], seed=7)
    mp = MAX_SEQ // ratio
    plan = csa2_indexer_plan(segs, ratio, mp, s.get("budget", BUDGET), DEV)
    want = (8, 50) if s.get("cand") else None
    args = (q, w, vis, pool, table, ratio, TOPK, plan, mp, want)
    with validating(csa2_indexer_topk_packed):
        got, cand = csa2_indexer_topk_packed(*args)
    ref = csa2_indexer_topk_packed.reference(*args)
    assert got.shape == ref.shape
    assert torch.equal(got < 0, ref < 0)
    assert overlap(got, ref) > 0.97
    if want is None:
        return
    # the candidates as the tiled path publishes them, and a consumer over the packed ones
    span = -(-max(sg[3] for sg in segs) // ratio)
    width = -(-span // 8) * 8
    _, tiled = csa2_indexer_topk_prefill(
        q, w, pos, rid, pool, table, ratio, TOPK, width, None, want, 1024
    )
    assert cand[0].shape == tiled[0].shape
    assert overlap(cand[0], tiled[0]) > 0.97
    q2 = _qdq(torch.randn_like(q.float()).bfloat16())
    picks, _ = csa2_indexer_topk_prefill(
        q2, w, pos, rid, pool, table, ratio, TOPK, width, cand, None, 1024
    )
    with true_fp32_matmul():
        check_consumer_picks(picks, q2, w, rid, pool, table, ratio, TOPK, cand[0], vis, width)


def test_matches_the_tiled_path():
    """The packed and the tiled (paged-logits) context indexer pick the same positions up to ties."""
    ratio = 1
    pool, table, pos, rid, segs, q, w, vis = _step(ratio, [(5000, 700), (100, 300)], seed=4)
    mp = MAX_SEQ // ratio
    got, _ = csa2_indexer_topk_packed(
        q, w, vis, pool, table, ratio, TOPK, csa2_indexer_plan(segs, ratio, mp, BUDGET, DEV), mp
    )
    width = -(-5700 // 8) * 8
    want, _ = csa2_indexer_topk_prefill(
        q, w, pos, rid, pool, table, ratio, TOPK, width, None, None, 1024
    )
    assert got.shape == want.shape
    assert torch.equal(got < 0, want < 0)
    assert overlap(got, want) > 0.99


def test_unmapped_pages_are_never_picked():
    ratio = 1
    pool, table, pos, rid, segs, q, w, vis = _step(ratio, [(1000, 400)], seed=6)
    table = table.clone()
    table[0, 3] = -1  # positions 384..511 have no page
    mp = MAX_SEQ // ratio
    plan = csa2_indexer_plan(segs, ratio, mp, BUDGET, DEV)
    got, _ = csa2_indexer_topk_packed(q, w, vis, pool, table, ratio, TOPK, plan, mp)
    assert not bool(((got >= 384) & (got < 512)).any())
    ref = csa2_indexer_topk_packed.reference(q, w, vis, pool, table, ratio, TOPK, plan, mp)
    assert overlap(got, ref) > 0.97


def test_rejections():
    pool, table, pos, rid, segs, q, w, vis = _step(1, [(200, 50)])
    groups, desc = plan = csa2_indexer_plan(segs, 1, MAX_SEQ, BUDGET, DEV)
    csa2_indexer_topk_packed.is_valid(q, w, vis, pool, table, 1, TOPK, plan, MAX_SEQ)
    with pytest.raises(AssertionError, match="int32"):
        csa2_indexer_topk_packed.is_valid(
            q, w, vis, pool, table, 1, TOPK, (groups, desc.long()), MAX_SEQ
        )
    with pytest.raises(AssertionError, match="covers"):
        csa2_indexer_topk_packed.is_valid(
            q[:10], w[:10], vis[:10], pool, table, 1, TOPK, plan, MAX_SEQ
        )
    with pytest.raises(AssertionError, match="ratio"):
        csa2_indexer_topk_packed.is_valid(q, w, vis, pool, table, 3, TOPK, plan, MAX_SEQ)


def test_one_group_alive_at_a_time():
    """As stock's eager prefill (and the KV sizing that reserves one group's logits): a step of several groups peaks
    no higher than its largest group alone, so a group's logits are freed before the next group allocates its own."""
    ratio = 1
    pool, table, pos, rid, segs, q, w, vis = _step(ratio, [(2000, 2500)], seed=8)
    mp = MAX_SEQ // ratio
    groups, desc = csa2_indexer_plan(segs, ratio, mp, 1 << 21, DEV)
    assert len(groups) > 2
    largest = max(groups, key=lambda g: (g[1] - g[0]) * g[2])

    def peak(plan):
        csa2_indexer_topk_packed(
            q, w, vis, pool, table, ratio, TOPK, plan, mp
        )  # warm (selector JIT, pool)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        csa2_indexer_topk_packed(q, w, vis, pool, table, ratio, TOPK, plan, mp)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    logits = (largest[1] - largest[0]) * (-(-largest[2] // 8) * 8) * 4
    assert peak((groups, desc)) <= peak(([largest], desc)) + logits // 4
