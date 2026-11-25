# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for csa2_shared_plan, csa2_local_window and csa2_shared_stage (stock's shared-bank context staging).

Plan and window copy: index arithmetic and row copies, compared exactly with the references. Stage: every query's
compacted bank rows, dequantized with the stage's scale, against its valid window and selected rows in slot order: the
window rows to E4M3's subnormals (E4M3 rows under power-of-two group scales, rescaled by a power of two: 2e-3 relative
RMS, from the few values the shift moves below 2^-6), the compressed rows within
4% relative RMS (one E4M3 rounding of an E2M1 x E4M3 product: 1/16 per element at most); `lens` exactly; the scales
consistent (unit-scale Q: `q_scale == 1`, BMM1 = kv_dequant * softmax scale, BMM2 = kv_dequant, a power of two).
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_chunk_window import (  # noqa: E501
    csa2_chunk_window,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_pack_rows import (  # noqa: E501
    csa2_pack_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_shared_stage import (  # noqa: E501
    csa2_local_window,
    csa2_shared_plan,
    csa2_shared_stage,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="SM103 kernels",
)
W, BLOCKS = 128, 48


def _ids(c):
    return [x.why[:48] for x in c]


def _rel(a, b):
    return (
        (a.float() - b.float()).square().mean().sqrt()
        / b.float().square().mean().sqrt().clamp_min(1e-30)
    ).item()


def _step(reqs, floor=None):
    """(segs, pos, rid, cs, req_cs, floor) of a context step of requests `(history, rows)`."""
    segs, t0 = [], 0
    for r, (h, k) in enumerate(reqs):
        if k:
            segs.append((t0, t0 + k, r, h + k))
        t0 += k
    pos = torch.cat([torch.arange(h, h + k) for h, k in reqs]).int().to(DEV)
    rid = torch.cat([torch.full((k,), r) for r, (_, k) in enumerate(reqs)]).long().to(DEV)
    cs = torch.cat([torch.full((k,), h) for h, k in reqs]).int().to(DEV)
    req_cs = torch.tensor([h for h, _ in reqs], dtype=torch.int64, device=DEV)
    fl = None
    if floor is not None:
        fl = torch.cat([torch.full((k,), f) for f, (_, k) in zip(floor, reqs)]).int().to(DEV)
    return segs, pos, rid, cs, req_cs, fl


@pytest.mark.parametrize("cell", csa2_shared_plan.CELLS, ids=_ids(csa2_shared_plan.CELLS))
def test_plan_cells(cell, certifying):  # noqa: F811
    certifying(csa2_shared_plan)
    s = cell.spec
    segs, pos, rid, _, _, fl = _step(s["reqs"], s.get("floor"))
    args = (segs, len(s["reqs"]), pos, rid, fl, W, s["ratio"])
    with validating(csa2_shared_plan):
        got = csa2_shared_plan(*args)
    ref = csa2_shared_plan.reference(*args)
    for f in (
        "swa_offsets",
        "main_offsets",
        "swa_pages",
        "swa_first",
        "swa_floors",
        "local_slot",
        "hist",
        "keep",
    ):
        g, r = getattr(got, f), getattr(ref, f)
        assert g.shape == r.shape and torch.equal(g.cpu().long(), r.cpu().long()), f
    for f in (
        "swa_rows",
        "main_rows",
        "local_rows",
        "swa_page_size",
        "main_page_size",
        "num_requests",
        "window",
    ):
        assert getattr(got, f) == getattr(ref, f), f
    assert (
        got.local_slot.max() < got.local_rows and len(set(got.local_slot.tolist())) == pos.numel()
    )


@pytest.mark.parametrize("cell", csa2_shared_plan.CELLS, ids=_ids(csa2_shared_plan.CELLS))
def test_plan_writes_every_chunk_row_to_the_pool(cell):
    """The rows written back to the SWA pool (`keep`) are every row of each request's chunk, as stock writes each row's
    SWA slot: prefix reuse publishes whole 128-row pages, so a chunk's rows before the last window (the start of its
    last page included) must be written too. Before, only the last `window` rows were, and a request reusing such a
    page read stale bytes as its SWA history."""
    s = cell.spec
    segs, pos, rid, _, _, fl = _step(s["reqs"], s.get("floor"))
    plan = csa2_shared_plan(segs, len(s["reqs"]), pos, rid, fl, W, s["ratio"])
    want = sorted((r, p) for r, (h, k) in enumerate(s["reqs"]) for p in range(h, h + k))
    assert sorted(zip(plan.keep[0].tolist(), plan.keep[1].tolist())) == want


def _swa_pool(nreq, gen, unmapped=False):
    pages = nreq * BLOCKS + 4
    table = (
        torch.randperm(pages, generator=gen, device=DEV)[: nreq * BLOCKS].view(nreq, BLOCKS).int()
    )
    if unmapped:  # the page holding the history of the last request is gone
        table[-1, :] = torch.where(torch.arange(BLOCKS, device=DEV) % 3 == 1, -1, table[-1, :])
    rows = torch.randn(pages * 128, 512, generator=gen, device=DEV).bfloat16() * 4
    pool = csa2_pack_rows(rows, "swa").view(pages, 128, 528)
    return pool, table


@pytest.mark.parametrize("cell", csa2_local_window.CELLS, ids=_ids(csa2_local_window.CELLS))
def test_local_window_cells(cell, certifying):  # noqa: F811
    certifying(csa2_local_window)
    s, gen = cell.spec, gen_for(cell)
    segs, pos, rid, *_ = _step(s["reqs"])
    plan = csa2_shared_plan(segs, len(s["reqs"]), pos, rid, None, W, 1)
    pool, table = _swa_pool(len(s["reqs"]), gen, s.get("unmapped", False))
    local = torch.randint(
        0, 256, (plan.local_rows, 528), generator=gen, device=DEV, dtype=torch.uint8
    )
    want_local, want_pool = csa2_local_window.reference(local, pool, table, plan.hist, plan.keep)
    with validating(csa2_local_window):
        csa2_local_window(local, pool, table, plan.hist, plan.keep)
    assert torch.equal(local, want_local) and torch.equal(pool, want_pool)
    assert plan.hist.shape[1] or plan.keep.shape[1]


def _stage_inputs(s, gen):
    reqs, ratio, k = s["reqs"], s["ratio"], s["topk"]
    nreq = len(reqs)
    segs, pos, rid, cs, req_cs, fl = _step(reqs, s.get("floor"))
    n = pos.numel()
    plan = csa2_shared_plan(segs, nreq, pos, rid, fl, W, ratio)
    pool, table = _swa_pool(nreq, gen, s.get("unmapped", False))
    local = torch.zeros(plan.local_rows, 528, dtype=torch.uint8, device=DEV)
    step_rows = csa2_pack_rows(torch.randn(n, 512, generator=gen, device=DEV).bfloat16() * 4, "swa")
    local[plan.local_slot] = step_rows  # what csa2_store_rows writes
    csa2_local_window(local, pool, table, plan.hist, plan.keep)
    valid = csa2_chunk_window(pos, rid, cs, req_cs, pool, table, W, fl)[1]
    q = torch.randn(n, 64, 512, generator=gen, device=DEV).bfloat16()
    if not ratio:
        return (plan, local, None, None, None, None, 0, valid, q, 512**-0.5, True, 0)
    rpp = 128 // ratio
    maxp = max(h + r for h, r in reqs) // ratio + 1
    pages_per = -(-maxp // rpp)
    pool_pages = nreq * pages_per + 2
    main = csa2_pack_rows(
        torch.randn(pool_pages * rpp, 512, generator=gen, device=DEV).bfloat16() * 3, "main"
    )
    mtable = (
        torch.randperm(pool_pages, generator=gen, device=DEV)[: nreq * pages_per]
        .view(nreq, pages_per)
        .int()
    )
    visible = ((pos.long() + 1) // ratio).contiguous()
    sel = torch.randint(-1, maxp, (n, k), generator=gen, device=DEV)
    sel = torch.where(
        sel < visible[:, None], sel, -1
    ).int()  # the indexer selects visible rows only
    return (plan, local, main, mtable, sel, visible, 1 << 20, valid, q, 512**-0.5, True, 0)


def _check_stage(args, got):
    bank, idx, lens, q_scale, bmm1, bmm2 = got
    rows, cnt = csa2_shared_stage.reference(*args)
    assert bank.dtype == torch.float8_e4m3fn and idx.dtype == lens.dtype == torch.int32
    assert torch.equal(lens, cnt), "per-query valid counts"
    kv = float(bmm2)
    assert float(q_scale) == 1.0 and kv == 2.0 ** round(torch.log2(bmm2).item()), (
        float(q_scale),
        kv,
    )
    assert abs(float(bmm1[0]) - kv * args[9]) <= 1e-6 * float(
        bmm1[0]
    )  # unit-scale Q: BMM1 = kv_dequant * softmax
    got_rows = bank.float()[idx.long().clamp_min(0)] * kv  # [n, cap, 512]
    keep = torch.arange(idx.shape[1], device=DEV)[None, :] < cnt[:, None]
    nswa = (args[7] >= 0).sum(1)  # window rows a query may read at most
    is_swa = idx.long() < 1 + args[0].swa_rows
    win = keep & is_swa
    sel = keep & ~is_swa
    assert win.any()
    print(
        f"window rows {_rel(got_rows[win], rows[win]):.2e}, selected rows "
        f"{_rel(got_rows[sel], rows[sel]) if sel.any() else 0:.4f}, kv_dequant {kv}"
    )
    assert _rel(got_rows[win], rows[win]) < 2e-3, _rel(got_rows[win], rows[win])
    if sel.any():
        assert _rel(got_rows[sel], rows[sel]) < 0.04, _rel(got_rows[sel], rows[sel])
    assert (nswa >= win.sum(1)).all()


@pytest.mark.parametrize("cell", csa2_shared_stage.CELLS, ids=_ids(csa2_shared_stage.CELLS))
def test_stage_cells(cell, certifying):  # noqa: F811
    certifying(csa2_shared_stage)
    args = _stage_inputs(cell.spec, gen_for(cell))
    with validating(csa2_shared_stage):
        got = csa2_shared_stage(*args)
    if cell.spec.get("second"):
        # the next layer of the staging group: its own window rows, no compaction, same selections
        first = [t.clone() for t in got[:3]]
        args = list(args)
        valid = args[7]
        local = args[1].clone()
        n = args[8].shape[0]
        local[args[0].local_slot] = csa2_pack_rows(
            torch.randn(n, 512, device=DEV).bfloat16(), "swa"
        )
        args[1], args[7], args[10], args[11] = local, None, False, 1
        args[8] = (args[8].float() * 0.5).bfloat16()
        got = csa2_shared_stage(*args)
        assert torch.equal(got[1], first[1]) and torch.equal(got[2], first[2]), (
            "indices / lens are the group's"
        )
        args[7] = valid  # the reference reads the validity
    _check_stage(tuple(args), got)


def test_stage_worked_example():
    """One request, history 2, three rows at positions 2..4, window rows only: bank rows 1..130 hold positions 0..129
    of the request's window region; query t reads positions 0..2+t, i.e. bank rows 1..3+t, in order."""
    segs, pos, rid, cs, req_cs, _ = _step([(2, 3)])
    plan = csa2_shared_plan(segs, 1, pos, rid, None, W, 0)
    assert plan.swa_rows == 3 + W - 1 and plan.main_rows == 0
    gen = torch.Generator(device=DEV).manual_seed(1)
    pool, table = _swa_pool(1, gen)
    local = torch.zeros(plan.local_rows, 528, dtype=torch.uint8, device=DEV)
    local[plan.local_slot] = csa2_pack_rows(torch.randn(3, 512, device=DEV).bfloat16(), "swa")
    csa2_local_window(local, pool, table, plan.hist, plan.keep)
    valid = csa2_chunk_window(pos, rid, cs, req_cs, pool, table, W, None)[1]
    q = torch.randn(3, 64, 512, device=DEV).bfloat16()
    _, idx, lens, *_ = csa2_shared_stage(
        plan, local, None, None, None, None, 0, valid, q, 512**-0.5, True, 0
    )
    assert lens.tolist() == [3, 4, 5]
    for t in range(3):
        assert idx[t, : 3 + t].tolist() == list(range(1, 4 + t)) and (idx[t, 3 + t :] == -1).all()


def test_rejections():
    segs, pos, rid, *_ = _step([(0, 10)])
    with pytest.raises(AssertionError):
        csa2_shared_plan.is_valid(segs, 1, pos.long(), rid, None, W, 1)  # positions are int32
    with pytest.raises(AssertionError):
        csa2_shared_plan.is_valid(segs, 1, pos, rid, None, 64, 1)  # the window is the page size
