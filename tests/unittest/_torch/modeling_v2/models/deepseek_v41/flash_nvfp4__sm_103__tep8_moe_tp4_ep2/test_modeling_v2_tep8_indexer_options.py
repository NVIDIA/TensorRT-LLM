# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The options the tep8 target passes to the family's `csa2_indexer_topk_decode`: `publish_stream` / `publish_done`
(the candidate source publishes on a side stream) and `cand_sched` (the consumers' schedule computed once per step by
`csa2_consumer_schedule`), `logits_sched` (the paged-logits schedule built ahead by `csa2_logits_schedule`). Each must
give bit for bit the picks and candidates of the default call. `mask_unmapped` (the engine's mask of unallocated pages
before the top-k) is checked on a table with holes."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # _v41_util
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "ops" / "attention")
)  # the family indexer test
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # _validating
from _v41_util import overlap  # noqa: E402
from test_modeling_v2_csa2_indexer_decode import TOPK, _decode_cell  # noqa: E402

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.ops.attention.csa2_logits_schedule import (  # noqa: E402,E501
    csa2_logits_schedule,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_consumer_schedule import (  # noqa: E402
    csa2_consumer_schedule,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_decode import (  # noqa: E402
    csa2_indexer_topk_decode,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_descriptors import (  # noqa: E402,E501
    csa2_indexer_descriptors,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)


@pytest.mark.parametrize(
    "cell", [c for c in csa2_indexer_topk_decode.CELLS if c.spec["cand"]], ids=lambda c: c.why[:44]
)
def test_side_publication_and_shared_schedule(cell):
    s = cell.spec
    (pool, table, pos, rid, q, w), mp = _decode_cell(s)
    ratio = s["ratio"]
    desc = csa2_indexer_descriptors(
        table, rid.long().contiguous(), ((pos.long() + 1) // ratio).contiguous(), ratio
    )
    args = (q, w, pos, rid, pool, table, ratio, TOPK, mp)
    plain, pcand = csa2_indexer_topk_decode(*args, None, (8, 40), desc)
    side, done = torch.cuda.Stream(), torch.cuda.Event()
    got, cand = csa2_indexer_topk_decode(
        *args, None, (8, 40), desc, publish_stream=side, publish_done=done
    )
    torch.cuda.current_stream().wait_event(done)
    assert torch.equal(got, plain) and all(torch.equal(a, b) for a, b in zip(cand, pcand))
    q2 = q.flip(0).contiguous()
    cargs = (q2, w, pos, rid, pool, table, ratio, TOPK, mp)
    ref, _ = csa2_indexer_topk_decode(*cargs, cand, None, desc)
    sched = csa2_consumer_schedule(cand[2].reshape(-1), desc[0], cand[1])
    for _ in range(2):  # two consumer layers sharing the step's schedule
        picks, _ = csa2_indexer_topk_decode(*cargs, cand, None, desc, cand_sched=sched)
        assert torch.equal(picks, ref)


@pytest.mark.parametrize("cell", csa2_indexer_topk_decode.CELLS, ids=lambda c: c.why[:44])
def test_logits_schedule_built_ahead(cell):
    s = cell.spec
    (pool, table, pos, rid, q, w), mp = _decode_cell(s)
    ratio = s["ratio"]
    desc = csa2_indexer_descriptors(
        table, rid.long().contiguous(), ((pos.long() + 1) // ratio).contiguous(), ratio
    )
    want = (8, 40) if s["cand"] else None
    args = (q, w, pos, rid, pool, table, ratio, TOPK, mp)
    plain, pcand = csa2_indexer_topk_decode(*args, None, want, desc)
    got, cand = csa2_indexer_topk_decode(
        *args, None, want, desc, logits_sched=csa2_logits_schedule(desc[1])
    )
    assert torch.equal(got, plain)
    if want is not None:
        assert all(torch.equal(a, b) for a, b in zip(cand, pcand))


@pytest.mark.parametrize("cell", csa2_logits_schedule.CELLS, ids=lambda c: c.why[:44])
def test_logits_schedule_cells(cell):
    s = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(s["n"])
    context = torch.randint(
        1, s["longest"], (s["n"],), generator=gen, device="cuda", dtype=torch.int32
    )
    L = s.get("rpr", 1)
    csa2_logits_schedule.is_valid(context, L)
    got, ref = csa2_logits_schedule(context, L), csa2_logits_schedule.reference(context, L)
    assert got.shape == ref.shape and got.dtype == ref.dtype


@pytest.mark.parametrize("cell", csa2_indexer_topk_decode.CELLS, ids=lambda c: c.why[:44])
def test_rows_per_request_shares_the_block_table(cell):
    """`rows_per_request = L`: a request's L rows read its keys once (engine's next_n grouping) -- the same logits, so
    the exact radix top-k picks bit for bit as per-row; the self-sampling GVR top-k grouped per request (next_n = L, the
    engine's call) picks the per-row GVR set up to ties. With the schedule built ahead for the grouping as well."""
    s = cell.spec
    (pool, table, pos, rid, q, w), mp = _decode_cell(s)
    ratio, L = s["ratio"], s["rows"]
    desc = csa2_indexer_descriptors(
        table, rid.long().contiguous(), ((pos.long() + 1) // ratio).contiguous(), ratio
    )
    want = (8, 40) if s["cand"] else None
    args = (q, w, pos, rid, pool, table, ratio, TOPK, mp)
    plain, pcand = csa2_indexer_topk_decode(*args, None, want, desc)
    got, cand = csa2_indexer_topk_decode(*args, None, want, desc, rows_per_request=L)
    assert torch.equal(got, plain)
    if want is not None:
        assert all(torch.equal(a, b) for a, b in zip(cand, pcand))
    ahead = csa2_logits_schedule(desc[1], L)
    got2, _ = csa2_indexer_topk_decode(
        *args, None, want, desc, logits_sched=ahead, rows_per_request=L
    )
    assert torch.equal(got2, plain)
    if TOPK in (512, 1024, 2048):
        h1, _ = csa2_indexer_topk_decode(*args, None, want, desc, heuristic=True)
        hl, _ = csa2_indexer_topk_decode(
            *args, None, want, desc, heuristic=True, rows_per_request=L
        )
        assert torch.equal(hl < 0, h1 < 0) and overlap(hl, h1) > 0.99


def test_rows_per_request_gvr_at_k512():
    """The model's K = 512 with the self-sampling GVR top-k, grouped per request vs per row (up to ties)."""
    import test_modeling_v2_csa2_indexer_decode as fam

    spec = dict(
        ratio=1, nreq=3, rows=6, cand=False
    )  # the grouped engine call serves ratios 1 and 4 (ratio 2: per row)
    (pool, table, pos, rid, q, w), mp = _decode_cell(spec)
    desc = csa2_indexer_descriptors(
        table, rid.long().contiguous(), ((pos.long() + 1) // 1).contiguous(), 1
    )
    args = (q, w, pos, rid, pool, table, 1, 512, mp)
    h1, _ = csa2_indexer_topk_decode(*args, None, None, desc, heuristic=True)
    hl, _ = csa2_indexer_topk_decode(*args, None, None, desc, heuristic=True, rows_per_request=6)
    assert torch.equal(hl < 0, h1 < 0) and overlap(hl, h1) > 0.99
    r1, _ = csa2_indexer_topk_decode(*args, None, None, desc)
    assert overlap(hl, r1) > 0.97 and fam is not None


@pytest.mark.parametrize("heuristic", [False, True])
def test_mask_unmapped_pages_before_the_topk(heuristic):
    """`mask_unmapped`: with unallocated pages inside the visible span (scored on page 0 by the paged kernel), the top-k
    runs over the mapped columns only, as the engine's `mask_logits_` before its selection: no pick lands on a hole and
    every row keeps its full `k` picks (the exact radix and the grouped self-sampling GVR agree up to ties)."""
    spec = dict(ratio=1, nreq=4, rows=6, cand=False)
    (pool, table, pos, rid, q, w), mp = _decode_cell(spec)
    table = table.clone()
    table[:, [3, 7, 11]] = -1  # holes well inside every request's ~3000 visible rows
    vis = (pos.long() + 1).contiguous()
    desc = csa2_indexer_descriptors(table, rid.long().contiguous(), vis, 1)
    args = (q, w, pos, rid, pool, table, 1, 512, mp)
    got, _ = csa2_indexer_topk_decode(
        *args, None, None, desc, heuristic=heuristic, rows_per_request=6, mask_unmapped=True
    )
    page = got.long().clamp(min=0) // 128
    on_hole = (table[rid.long()].gather(1, page) < 0) & (got >= 0)
    assert not bool(on_hole.any()), "a pick landed on an unallocated page"
    assert bool(((got >= 0).sum(1) == 512).all()), (
        "every row keeps k picks among its mapped columns"
    )
    exact, _ = csa2_indexer_topk_decode(*args, None, None, desc, mask_unmapped=True)
    assert overlap(got, exact) > 0.99


@pytest.mark.parametrize(
    "cell", [c for c in csa2_indexer_topk_decode.CELLS if c.spec["cand"]], ids=lambda c: c.why[:44]
)
def test_candidate_decode_topk(cell):
    """`cand_decode_topk`: the candidate source's block top-k is the engine's decode selector (multi-CTA radix) instead
    of the prefill radix: the same candidate blocks up to ties at the k-th block maximum, the same counts, and the
    same picks of the layer itself."""
    s = cell.spec
    (pool, table, pos, rid, q, w), mp = _decode_cell(s)
    ratio = s["ratio"]
    desc = csa2_indexer_descriptors(
        table, rid.long().contiguous(), ((pos.long() + 1) // ratio).contiguous(), ratio
    )
    args = (q, w, pos, rid, pool, table, ratio, TOPK, mp)
    plain, (p0, b0, c0) = csa2_indexer_topk_decode(*args, None, (8, 40), desc)
    got, (p1, b1, c1) = csa2_indexer_topk_decode(*args, None, (8, 40), desc, cand_decode_topk=True)
    assert torch.equal(got, plain) and torch.equal(c1, c0)
    agree = total = 0
    for a, b in zip(p1.tolist(), p0.tolist()):
        sa, sb = {x for x in a if x >= 0}, {x for x in b if x >= 0}
        agree += len(sa & sb)
        total += len(sb)
    assert agree / total > 0.99
    # and a consumer reading the decode selector's candidates (its block ids `b1` included) picks as from the default's
    q2 = q.flip(0).contiguous()
    ref, _ = csa2_indexer_topk_decode(
        q2, w, pos, rid, pool, table, ratio, TOPK, mp, (p0, b0, c0), None, desc
    )
    via, _ = csa2_indexer_topk_decode(
        q2, w, pos, rid, pool, table, ratio, TOPK, mp, (p1, b1, c1), None, desc
    )
    assert overlap(via, ref) > 0.97


@pytest.mark.parametrize(
    "cell", [c for c in csa2_indexer_topk_decode.CELLS if c.spec["cand"]], ids=lambda c: c.why[:44]
)
def test_consumer_takes_bf16_weights(cell):
    """A candidate consumer handed its head weights already in bf16 (the tep8 target projects them into bf16 on the side
    stream, as the engine's consumers) picks bit for bit as with the fp32 weights (the kernel casts them to bf16)."""
    s = cell.spec
    (pool, table, pos, rid, q, w), mp = _decode_cell(s)
    ratio = s["ratio"]
    desc = csa2_indexer_descriptors(
        table, rid.long().contiguous(), ((pos.long() + 1) // ratio).contiguous(), ratio
    )
    _, cand = csa2_indexer_topk_decode(
        q, w, pos, rid, pool, table, ratio, TOPK, mp, None, (8, 40), desc
    )
    q2 = q.flip(0).contiguous()
    ref, _ = csa2_indexer_topk_decode(
        q2, w, pos, rid, pool, table, ratio, TOPK, mp, cand, None, desc
    )
    w16 = w.to(torch.bfloat16)
    csa2_indexer_topk_decode.is_valid(
        q2, w16, pos, rid, pool, table, ratio, TOPK, mp, cand, None, desc
    )
    got, _ = csa2_indexer_topk_decode(
        q2, w16, pos, rid, pool, table, ratio, TOPK, mp, cand, None, desc
    )
    assert torch.equal(got, ref)
    with pytest.raises(AssertionError):  # only a consumer: the full-width logits take fp32 weights
        csa2_indexer_topk_decode.is_valid(
            q, w16, pos, rid, pool, table, ratio, TOPK, mp, None, (8, 40), desc
        )
