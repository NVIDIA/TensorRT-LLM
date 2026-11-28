# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the dsv4_route_nvfp4_quant entry: the fused routing + NVFP4 launch against torch routing and `fp4_quantize`.

Gate: each row picks the same expert set (random fp32 scores do not tie); the picked experts' weights agree to one bf16
step; the quantized bytes and scales match except where the kernel's approximate reciprocal puts a value on the other
side of a rounding tie (at most 0.2% of the packed bytes, and
every scale byte of a block whose data differs is the reference's).
"""

import pytest
import torch
from _megamoe_reference import megamoe_slots, mxfp8_rows  # noqa: E402
from _v41_util import DEV, gen_for

__extra_import_path__ = [
    "../../../..",
    "../..",
]  # the modeling_v2 test helpers, the family test directory
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (  # noqa: E501
    dsv4_route_nvfp4_quant,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="CuTe-DSL kernel, SM100 family",
)


def _inputs(s, gen):
    x = (torch.randn(s["n"], s["hidden"], generator=gen, device=DEV) * 2).bfloat16()
    logits = torch.randn(s["n"], s["experts"], generator=gen, device=DEV) * 3
    bias = torch.randn(s["experts"], generator=gen, device=DEV) * 0.5
    g = (448.0 * 6.0 / x.float().abs().max()).reshape(1).to(DEV)
    return logits, bias, x, g, s["top_k"], s["scale"]


@pytest.mark.parametrize(
    "cell", dsv4_route_nvfp4_quant.CELLS, ids=[c.why[:44] for c in dsv4_route_nvfp4_quant.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(dsv4_route_nvfp4_quant)
    s = cell.spec
    args = _inputs(s, gen_for(cell))
    ids, w, xq, sf = dsv4_route_nvfp4_quant(*args)
    rids, rw, rxq, rsf = dsv4_route_nvfp4_quant.reference(*args)
    assert (
        ids.dtype == torch.int32
        and w.dtype == torch.bfloat16
        and xq.dtype == torch.uint8
        and sf.dtype == torch.uint8
    )
    assert (
        ids.shape == (s["n"], s["top_k"])
        and xq.shape == (s["n"], s["hidden"] // 2)
        and sf.shape == (s["n"], s["hidden"] // 16)
    )
    order, rorder = ids.long().argsort(-1), rids.long().argsort(-1)
    assert torch.equal(ids.long().gather(1, order), rids.long().gather(1, rorder)), (
        "different experts picked"
    )
    # one bf16 step of each reference weight (a step is at most 2**-7 of the value)
    aligned, raligned = w.float().gather(1, order), rw.float().gather(1, rorder)
    assert ((aligned - raligned).abs() <= 2.0**-7 * raligned.abs()).all(), (
        "a weight differs by more than one bf16 step"
    )
    assert torch.equal(sf, rsf.view_as(sf)) or (sf != rsf.view_as(sf)).float().mean() < 2e-3
    assert (xq != rxq).float().mean() < 2e-3


def test_rejections():
    cell = dsv4_route_nvfp4_quant.CELLS[0]
    args = list(_inputs(cell.spec, gen_for(cell)))
    dsv4_route_nvfp4_quant.is_valid(*args)
    bad = list(args)
    bad[0] = args[0].bfloat16()
    with pytest.raises(AssertionError):
        dsv4_route_nvfp4_quant.is_valid(*bad)
    bad = list(args)
    bad[2] = args[2].float()
    with pytest.raises(AssertionError):
        dsv4_route_nvfp4_quant.is_valid(*bad)


@pytest.mark.parametrize(
    "rows,max_t,copies", [(17, 64, False), (150, 150, True)], ids=["slots-256", "replicas-150"]
)
def test_staged_into_bucket(rows, max_t, copies):
    """The staged form writes the bucket exactly as the unstaged launch
    plus a slot lookup and the stage-inputs copy would."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_expert_slots import (
        megamoe_slot_table,
    )

    experts, hidden, top_k = 384, 5120, 6
    spec = dict(n=rows, hidden=hidden, experts=experts, top_k=top_k, scale=1.5)
    gen = gen_for(dsv4_route_nvfp4_quant.CELLS[1])
    logits, bias, x, g, k, scale = _inputs(spec, gen)
    assign = list(range(experts)) + (list(range(32)) if copies else [])
    table = megamoe_slot_table(assign, experts, 0, DEV)
    idx = torch.full((max_t, top_k), 7, dtype=torch.int64, device=DEV)
    w = torch.full((max_t, top_k), 3.0, dtype=torch.float32, device=DEV)
    act = torch.zeros((max_t, hidden // 2), dtype=torch.uint8, device=DEV)
    asf = torch.zeros((max_t, hidden // 16), dtype=torch.uint8, device=DEV)
    x8 = torch.zeros((max_t, hidden), dtype=torch.float8_e4m3fn, device=DEV)
    x8sf = torch.zeros(
        ((max_t + 127) // 128 * 128 * (hidden // 32) + 511) // 512 * 512,
        dtype=torch.uint8,
        device=DEV,
    )
    ids, _, xq, sf = dsv4_route_nvfp4_quant(
        logits, bias, x, g, k, scale, torch.float32,
        staged=(idx, w, act, asf), slot_of_expert=table, replicas=copies, mxfp8=(x8, x8sf),
    )  # fmt: skip
    uids, uw, uxq, usf = dsv4_route_nvfp4_quant(logits, bias, x, g, k, scale, torch.float32)
    assert torch.equal(ids, uids) and torch.equal(act[:rows], uxq) and torch.equal(asf[:rows], usf)
    assert torch.equal(w[:rows], uw) and (idx[rows:] == -1).all() and (w[rows:] == 0).all()
    live = idx[:rows]
    owner = torch.tensor(assign, dtype=torch.int64, device=DEV)
    assert (owner[live] == uids.long()).all(), "a pick must land on a slot holding its expert"
    # and on the copy the table names for the row (copy h(row) % copies), not, e.g., always the first one
    assert torch.equal(live, megamoe_slots(table, uids, experts).long()), "replica choice differs"
    if copies:
        replicated = uids.long() < 32
        spread = [live[replicated & (uids.long() == e)].unique().numel() for e in range(32)]
        assert max(spread) == 2, "a replicated expert's picks should reach both of its copies"
    # the MXFP8 rows of the fused shared expert: the live rows' E4M3 bytes and UE8M0 scales, bit for bit
    want_q, want_e = mxfp8_rows(x)
    assert torch.equal(x8[:rows].view(torch.uint8), want_q.view(torch.uint8))
    pad = (max_t + 127) // 128 * 128

    def unblock(flat):  # [pad128(max_t), H/32] per-row scale bytes from the 128x4 atom layout
        cols = hidden // 32
        atoms = flat[: pad * cols].view(pad // 128, cols // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
        return atoms.reshape(pad, cols)

    assert torch.equal(unblock(x8sf)[:rows], want_e)


@pytest.mark.parametrize("staged", [False, True], ids=["unstaged", "staged-replicas"])
def test_counters_add_the_counted_rows_picks(staged):
    """`counters` (the engine's warmup placement `RouteCounters`): the launch adds the picks of the rows below the
    live-row bound whose mask byte is set and whose logits are not all
    zero, while the flag is set; the routing itself is unchanged."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_expert_slots import (
        megamoe_slot_table,
    )

    experts, hidden, top_k, rows, max_t = 384, 5120, 6, 150, 150
    spec = dict(n=rows, hidden=hidden, experts=experts, top_k=top_k, scale=1.5)
    logits, bias, x, g, k, scale = _inputs(spec, gen_for(dsv4_route_nvfp4_quant.CELLS[1]))
    logits[4] = (
        0.0  # an all-zero row (an attention-DP dummy) never counts: forward row 14, below the bound, mask set
    )
    hist = torch.zeros(experts, dtype=torch.int32, device=DEV)
    flag = torch.ones(1, dtype=torch.int32, device=DEV)
    bound = torch.full((1,), 120, dtype=torch.int32, device=DEV)
    mask = (torch.arange(4096, device=DEV) % 3 != 0).to(torch.int8)
    extra = {}
    if staged:
        assign = list(range(experts)) + list(range(32))
        extra = dict(
            staged=(
                torch.empty((max_t, top_k), dtype=torch.int64, device=DEV),
                torch.empty((max_t, top_k), dtype=torch.float32, device=DEV),
                torch.empty((max_t, hidden // 2), dtype=torch.uint8, device=DEV),
                torch.empty((max_t, hidden // 16), dtype=torch.uint8, device=DEV),
            ),
            slot_of_expert=megamoe_slot_table(assign, experts, 0, DEV),
            replicas=True,
        )
    counters = (hist, flag, bound, mask, 10)  # this launch's rows are rows 10.. of the forward
    dsv4_route_nvfp4_quant.is_valid(
        logits, bias, x, g, k, scale, torch.float32, counters=counters, **extra
    )
    ids, w, _, _ = dsv4_route_nvfp4_quant(
        logits, bias, x, g, k, scale, torch.float32, counters=counters, **extra
    )
    uids, uw, _, _ = dsv4_route_nvfp4_quant(logits, bias, x, g, k, scale, torch.float32)
    assert torch.equal(ids, uids) and torch.equal(w, uw)
    r = torch.arange(rows, device=DEV)
    live = (r < 120 - 10) & (mask[r + 10] != 0) & (logits.abs().amax(-1) > 0)
    want = torch.bincount(ids[live].long().reshape(-1), minlength=experts).to(torch.int32)
    assert torch.equal(hist, want)
    flag.zero_()  # frozen: nothing more counts
    dsv4_route_nvfp4_quant(logits, bias, x, g, k, scale, torch.float32, counters=counters, **extra)
    assert torch.equal(hist, want)
