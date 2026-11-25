# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a tep4 MoE block takes the fused finalize all-reduce (`TEP_FINAR_ENV`).

* The switch: unset / 0 off, `1` and `early` on (the two release points), anything else raises.
* The build: off without the switch; set without the tep4 route front it warns and builds
  nothing. The ranks agree on the row counts (`_tep_finar_rows`): only those every rank's route
  front serves.
* `_tep_finar_routed`: a captured block of a served row count with the site's codes and the
  decode router runs the route front without the runner's finalize, then the fused launch on
  slot `layer mod 3` over the route front's weights and slot -> row table; anything else returns
  None (the caller keeps the finalized path and `core.tp_sum`). Layers out of order raise.

* The shared-expert quarter (`TEP_FINAR_SHARED_ENV`): off by default; its build takes only the
  measured rows the fused launch serves; a layer's quarter is its rows of the gate and up halves,
  the gate-logit row and its down columns; a block of such a row count runs the quarter on the
  side stream, hands the next site a zero shared expert and gate, and passes the quarter to the
  fused launch after joining the side stream.

The ops are recorders; no kernel runs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

TOPK, HIDDEN = 10, 8


@pytest.mark.parametrize(
    "value, expected", [(None, ""), ("0", ""), ("1", "publish"), (" EARLY ", "early")]
)
def test_switch_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(M.TEP_FINAR_ENV, raising=False)
    else:
        monkeypatch.setenv(M.TEP_FINAR_ENV, value)
    assert M._tep_finar_mode() == expected


def test_switch_refuses_another_value(monkeypatch):
    monkeypatch.setenv(M.TEP_FINAR_ENV, "2")
    with pytest.raises(ValueError, match="expected 0, 1 or early"):
        M._tep_finar_mode()


def test_build_is_off_without_the_switch(monkeypatch):
    monkeypatch.delenv(M.TEP_FINAR_ENV, raising=False)
    assert M._tep_finar_build(SimpleNamespace(), torch.device("cuda")) is None


def _agree_core(route_front, peers_rows=None, layers=48):
    """A tep4 core whose `tp_sum` adds three peers' flags to this rank's: each peer serves
    `peers_rows` (default: what this rank's route front serves), or with a list of three, peer
    i serves `peers_rows[i]`."""
    st = None
    mine = ()
    if route_front is not None:
        st = M._RouteFront(None)
        st.tactics = {r: (8, 44) for r in route_front}
        st.layers = {i: i for i in range(layers)}
        mine = route_front

    each = [mine if peers_rows is None else peers_rows] * 3
    if isinstance(peers_rows, list):
        each = peers_rows

    def tp_sum(flags):
        total = flags.clone()
        for rows in each:
            for r in rows:
                total[0, r - 1] += 1.0
        return total

    return SimpleNamespace(
        tep_fast=True,
        _route_front=st,
        mapping=SimpleNamespace(is_multi_node=lambda: False, tp_rank=0),
        hidden=M._TEP_FINAR.HIDDEN,
        topk=M._TEP_FINAR.TOP_K,
        tp_sum=tp_sum,
    )


def test_build_needs_the_tep4_route_front(monkeypatch):
    warned = []
    monkeypatch.setenv(M.TEP_FINAR_ENV, "1")
    monkeypatch.setattr(M.logger, "warning", warned.append)
    core = _agree_core(None, peers_rows=())
    assert M._tep_finar_build(core, torch.device("cpu")) is None
    assert len(warned) == 1 and "route front" in warned[0]


@pytest.mark.parametrize(
    "mine, peers, layers, agreed",
    [
        ((4, 8, 12, 16), None, 48, (4, 8, 12, 16)),
        ((4, 8, 12, 16, 32), None, 48, (4, 8, 12, 16)),  # past the launch's rows
        ((4, 8, 12, 16), (4, 16), 48, (4, 16)),  # a peer's route front lacks 8 and 12
        (None, (4, 8), 48, ()),  # this rank has no route front, its peers do
        ((4, 8), (), 48, ()),  # the peers have none
        ((4, 8), None, 47, ()),  # MoE layers not a multiple of the slots
        # This rank and two peers serve 8, the third does not: 8 is out (all four, not most).
        ((4, 8, 16), [(4, 8, 16), (4, 8, 16), (4, 16)], 48, (4, 16)),
        ((4, 8), [(4, 8), (8,), (4,)], 48, ()),  # every count misses on some peer
    ],
)
def test_ranks_agree_on_the_rows(mine, peers, layers, agreed):
    """The fused launch serves only the row counts every rank's route front serves: a block that
    took it on some ranks and `core.tp_sum` on another would leave both waiting."""
    core = _agree_core(mine, peers, layers)
    assert M._tep_finar_rows(core, torch.device("cpu")) == agreed


def _setup(monkeypatch, capturing: bool, layers=6):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(M, "_decode_router", lambda x, router: "logits")

    def front(core, st, x, L, layer, logits, fp4=None, finalize=True):
        calls.append(("front", layer, fp4 is not None, finalize))
        return "fc2", None, None, None, "wts"

    monkeypatch.setattr(M, "_route_front_run", front)

    def fused(fc2, wts, perm, rows, slot, op):
        calls.append(("fused", fc2, wts, tuple(perm.shape), rows, slot, perm.data_ptr()))
        return "summed"

    monkeypatch.setattr(M, "tep_finalize_allreduce", fused)
    st = M._RouteFront(None)
    st.tactics = {4: (8, 44), 16: (8, 44)}
    weights = [SimpleNamespace(fc1_w=torch.zeros(1), router=None) for _ in range(layers)]
    st.layers = {L.fc1_w.data_ptr(): i for i, L in enumerate(weights)}
    st.views = {
        r: [[torch.zeros(r * TOPK, dtype=torch.int32)] for _ in range(layers)] for r in (4, 16)
    }
    core = SimpleNamespace(_route_front=st, hidden=HIDDEN, topk=TOPK)
    fin = M._TepFinar("op", (4, 16), layers)
    return core, fin, weights, calls


def _run(core, fin, L, rows=4, fp4=True, router=None):
    x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
    act = (torch.zeros(rows, 4), torch.zeros(rows)) if fp4 else None
    return M._tep_finar_routed(core, fin, x, L, act, router or M._decode_router)


def test_a_captured_step_folds_every_layer_on_rotating_slots(monkeypatch):
    core, fin, weights, calls = _setup(monkeypatch, capturing=True)
    for step in range(2):
        for L in weights:
            assert _run(core, fin, L) == "summed"
    fused = [c for c in calls if c[0] == "fused"]
    assert [c[5] for c in fused] == [0, 1, 2, 0, 1, 2] * 2
    assert all(c[1:5] == ("fc2", "wts", (4, TOPK), 4) for c in fused)
    # Each layer's launch reads that layer's own routing table at the step's row count.
    tables = core._route_front.views[4]
    assert [c[6] for c in fused] == [tables[i][0].data_ptr() for i in range(len(weights))] * 2
    assert all(c[2:] == (True, False) for c in calls if c[0] == "front")


@pytest.mark.parametrize(
    "capturing, rows, fp4, router, why",
    [
        (False, 4, True, None, "an eager step"),
        (True, 8, True, None, "a row count the fold does not serve"),
        (True, 4, False, None, "no site codes"),
        (True, 4, True, "prefill", "another router"),
    ],
)
def test_other_blocks_keep_the_finalized_path(monkeypatch, capturing, rows, fp4, router, why):
    core, fin, weights, calls = _setup(monkeypatch, capturing)
    core._route_front.tactics[8] = (8, 0)
    other = (lambda x, w: None) if router == "prefill" else None
    assert _run(core, fin, weights[0], rows, fp4, other) is None, why
    assert calls == []


def test_layers_out_of_order_raise(monkeypatch):
    core, fin, weights, _ = _setup(monkeypatch, capturing=True)
    _run(core, fin, weights[0])
    with pytest.raises(RuntimeError, match="three consecutive launches"):
        _run(core, fin, weights[2])


def test_moe_block_sums_once(monkeypatch):
    """With the fold the block skips `core.expert_sum`; without it the block keeps it."""
    side = SimpleNamespace(wait_stream=lambda s: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: side)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: __import__("contextlib").nullcontext())
    monkeypatch.setattr(M, "_shared_expert", lambda *a: ("shared", "gate"))
    for folded in ("summed", None):
        monkeypatch.setattr(M, "_tep_finar_routed", lambda *a, _r=folded: _r)
        sums = []
        core = SimpleNamespace(
            _side_stream=side,
            _tep_finar=object(),
            expert_sum=lambda run, x, pad: sums.append(1) or "stock",
        )
        routed, gate, shared = M._moe_block(core, torch.zeros(4, 8), None, 0)
        assert routed == ("summed" if folded else "stock") and len(sums) == (0 if folded else 1)


def test_shared_quarter_needs_the_fused_launch(monkeypatch):
    warned = []
    monkeypatch.delenv(M.TEP_FINAR_ENV, raising=False)
    monkeypatch.setenv(M.TEP_FINAR_SHARED_ENV, "1")
    monkeypatch.setattr(M.logger, "warning", warned.append)
    assert M._tep_finar_build(SimpleNamespace(), torch.device("cuda")) is None
    assert len(warned) == 1 and M.TEP_FINAR_ENV in warned[0]


def test_shared_quarter_is_off_by_default(monkeypatch):
    monkeypatch.delenv(M.TEP_FINAR_SHARED_ENV, raising=False)
    assert M._tep_finar_shared_build(SimpleNamespace(), (4, 8, 12, 16), None) is None


def test_shared_quarter_rows_are_the_measured_ones():
    assert M._TEP_FINAR_SHARED_ROWS == (12, 16)


def _quarter_core(inter=32, hidden=8, layers=2, rank=1):
    """Layers whose `[gate | up | gate logit | pad]` rows hold their own row index (and the down
    projection its column index), so a quarter's rows name where they came from."""
    weights = []
    for i in range(layers):
        gu = torch.arange(2 * inter + 8, dtype=torch.float32)[:, None].expand(-1, hidden)
        dn = torch.arange(inter, dtype=torch.float32)[None, :].expand(hidden, -1) + 64 * i
        weights.append(
            SimpleNamespace(
                sh_gug=(gu + 100 * i).to(torch.bfloat16).contiguous().t(),
                sh_dn=dn.to(torch.bfloat16).contiguous().t(),
            )
        )
    return SimpleNamespace(
        shared_inter=inter,
        hidden=hidden,
        mapping=SimpleNamespace(tp_rank=rank),
        _decode=SimpleNamespace(_layers=weights),
    )


@pytest.mark.parametrize(
    "served, inter, built, warns",
    [
        ((4, 8, 12, 16), 32, (12, 16), False),
        ((4, 16), 32, (16,), False),
        ((4, 8), 32, None, True),  # the fused launch serves no measured row
        ((12, 16), 36, None, True),  # an intermediate that does not split into 4 x 8k
    ],
)
def test_shared_quarter_build(monkeypatch, served, inter, built, warns):
    warned = []
    monkeypatch.setenv(M.TEP_FINAR_SHARED_ENV, "1")
    monkeypatch.setattr(M.logger, "warning", warned.append)
    core = _quarter_core(inter=inter)
    got = M._tep_finar_shared_build(core, served, torch.device("cpu"))
    assert (got.rows if got is not None else None) == built
    assert bool(warned) is warns
    if got is not None:
        q = inter // 4
        assert got.inter_q == q and len(got.quarters) == 2
        assert tuple(got.zero_shared.shape) == (max(built), core.hidden)
        assert tuple(got.zero_gate.shape) == (max(built), 1) and not got.zero_gate.any()


def test_shared_quarter_weights_take_this_ranks_rows():
    core = _quarter_core(inter=32, rank=1)
    L = core._decode._layers[1]
    gu_t, dn_t = M._tep_finar_quarter_weights(core, L, 1)
    gu, dn = gu_t.t(), dn_t.t()
    q = 8
    rows = gu[:, 0].float().tolist()
    want = [100 + r for r in range(q, 2 * q)] + [100 + 32 + r for r in range(q, 2 * q)] + [164]
    assert rows[: 2 * q + 1] == want
    assert gu.shape[0] == 24 and not gu[2 * q + 1 :].any()  # padded to a multiple of 8 rows
    assert dn.shape == (core.hidden, q) and dn[0].float().tolist() == [
        64 + c for c in range(q, 2 * q)
    ]
    assert gu_t.stride(0) == 1 and dn_t.stride(0) == 1  # column-major operands


def test_moe_block_runs_the_quarter_and_hands_zeros(monkeypatch):
    """At a quarter row count the side stream runs the quarter, the next site gets the zero
    shared expert and gate, and the fused launch gets the quarter."""
    side = SimpleNamespace(wait_stream=lambda s: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: side)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: __import__("contextlib").nullcontext())
    monkeypatch.setattr(M, "_shared_expert", lambda *a: ("whole", "whole gate"))
    monkeypatch.setattr(M, "_tep_finar_quarter", lambda core, x, L: ("quarter", "quarter gate"))
    monkeypatch.setattr(M, "_tep_finar_serves", lambda *a: True)
    seen = []
    monkeypatch.setattr(M, "_tep_finar_routed", lambda *a: seen.append(a[-1]) or "summed")
    zero = torch.zeros(16, 4)
    shared = M._TepFinarShared((12, 16), {}, 8, zero, zero[:, 1:2])
    fin = M._TepFinar("op", (4, 8, 12, 16), 6, shared)
    core = SimpleNamespace(_side_stream=side, _tep_finar=fin, expert_sum=None)
    for rows, quarter in ((12, True), (8, False)):
        seen.clear()
        routed, gate, sh = M._moe_block(core, torch.zeros(rows, 4), None, 0)
        assert routed == "summed"
        if quarter:
            assert seen == [("quarter", "quarter gate")]
            assert tuple(sh.shape) == (rows, 4) and tuple(gate.shape) == (rows, 1)
        else:
            assert seen == [None] and (sh, gate) == ("whole", "whole gate")


def test_routed_joins_the_side_stream_and_passes_the_quarter(monkeypatch):
    core, fin, weights, calls = _setup(monkeypatch, capturing=True)
    joined = []
    main = SimpleNamespace(wait_stream=lambda s: joined.append(s))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    core._side_stream = "side"

    def fused(fc2, wts, perm, rows, slot, op, *quarter):
        calls.append(("fused", slot, quarter))
        return "summed"

    monkeypatch.setattr(M, "tep_finalize_allreduce", fused)
    x = torch.zeros(16, HIDDEN, dtype=torch.bfloat16)
    act = (torch.zeros(16, 4), torch.zeros(16))
    out = M._tep_finar_routed(core, fin, x, weights[0], act, M._decode_router, ("q", "g"))
    assert out == "summed" and joined == ["side"]
    assert [c for c in calls if c[0] == "fused"] == [("fused", 0, ("q", "g"))]
