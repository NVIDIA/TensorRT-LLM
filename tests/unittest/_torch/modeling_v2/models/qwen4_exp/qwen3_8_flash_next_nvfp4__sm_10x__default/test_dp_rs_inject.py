# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP fused reduce-scatter (`DP_RS_INJECT_ENV`), its finalize form
(`DP_MOE_FIN_RS_ENV`) and the draft MoE's parts on it (`DP_DRAFTPACK_ENV`), on CPU recorders.

* which trunk blocks hand the next site an injected residual (`_dp_rs_inject_next`), and the round
  trip's tail in place of its reduce-scatter (the plain one where the tail refuses);
* `_moe_block` with the fused tail: the side stream joined before the fused launch, and the window
  entering the runner without its finalize at the finalize form's rows;
* the build: off, ignored layouts, a peer's compile failure, row counts a check drops on any
  rank, the finalize form only where its grid is one resident wave, and draft row counts whose
  fused launch is not the draft path's bits;
* the draft: the combine in place of `mtp_moe_combine_inject`, the head taking the combine's norm
  only for that very tensor, the runner's own routing, and the parts' spellings.
No kernel runs; the launches are recorders.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits {other.name}")


def _streams(monkeypatch, log):
    main, side = _Stream("main", log), _Stream("side", log)
    current = [main]

    @contextlib.contextmanager
    def stream(s):
        current.append(s)
        yield
        current.pop()

    monkeypatch.setattr(M.torch.cuda, "current_stream", lambda *a: current[-1])
    monkeypatch.setattr(M.torch.cuda, "stream", stream)
    monkeypatch.setattr(M.torch.cuda, "is_current_stream_capturing", lambda: True)
    return main, side, current


class _Rs:
    """A stand-in `RsInject`: records its launches, refuses what `refuse` names."""

    def __init__(self, log, refuse=None, refuse_fin=None, pdl=False):
        self.log, self.refuse, self.refuse_fin, self.pdl = log, refuse, refuse_fin, pdl

    def refuses(self, partial, hc, inject, norm, shared, gate):
        return self.refuse

    def refuses_fin(self, hc, inject, norm, shared, gate):
        return self.refuse_fin

    def fin_operands_refuse(self, rows, g2, w, e2p):
        return None

    def run(self, partial, hc, inject, norm, eps, shared, gate):
        self.log.append(("fused", partial, shared, gate))
        return "hc_out", "normed"

    def run_fin(self, g2, w, e2p, hc, inject, norm, eps, shared, gate):
        self.log.append(("fused fin", g2, w, e2p, shared, gate))
        return "hc_out", "normed"


def _rsi(log, rows=(8, 12), fin_rows=(), draft_rows=(2, 3), **kw):
    return M._DpRsInject(
        _Rs(log, **kw), frozenset(rows), frozenset(fin_rows), frozenset(draft_rows)
    )


# ------------------------------------------------------------------- which blocks fuse


@pytest.mark.parametrize(
    "built, captured, pad, rows, i, expected",
    [
        (True, True, 0, 8, 0, "next site"),
        (True, True, 0, 8, 2, "final"),
        (False, True, 0, 8, 0, None),
        (True, False, 0, 8, 0, None),  # eager
        (True, True, 1, 8, 0, None),  # padded
        (True, True, 0, 9, 0, None),  # a row count the build did not keep
        (True, True, 0, 8, 1, None),  # the layer before the PLE layer
    ],
)
def test_next_site(built, captured, pad, rows, i, expected):
    core = SimpleNamespace(_rs_inject=_rsi([]) if built else None, ple_layer=2)
    layers = [SimpleNamespace(ahc_norm=f"norm{k}") for k in range(3)]
    got = M._dp_rs_inject_next(core, captured, pad, rows, i, layers, "final norm", "hc", "inj")
    if expected is None:
        assert got is None
    else:
        assert got == ("hc", "inj", "norm1" if expected == "next site" else "final norm")


# ------------------------------------------------------------------- the round trip's tail


class _Trip:
    def takes(self, rows, x):
        return 1 <= rows <= 16


def _expert_sum(monkeypatch, calls):
    monkeypatch.setenv(M.DP_MOE_COMM_ENV, "lamport_step")

    def lamport(x, trip, reduce, release=False):
        calls.append("lamport reduce" if reduce else "lamport gather")
        return x[: x.shape[0] // 4] if reduce else x.repeat(4, 1)

    monkeypatch.setattr(M, "dep_lamport_round_trip", lamport)
    core = SimpleNamespace(
        dp_size=4,
        dp_group=[0, 1, 2, 3],
        mapping=SimpleNamespace(tp_group_pg=None),
        _dp_lamport=lambda group: _Trip(),
    )
    return M.Qwen4ExpDefaultModelingV2._expert_sum_op(core)


@pytest.mark.parametrize("fused, pad", [(True, 0), (False, 0), (True, 1)])
def test_round_trip_tail(monkeypatch, fused, pad):
    calls = []
    expert_sum = _expert_sum(monkeypatch, calls)
    x = torch.zeros(3, 8)
    seen = []

    def tail(partial):
        seen.append(partial.shape[0])
        return "injected" if fused else None

    out = expert_sum(lambda g: g, x, pad, tail)
    if fused and not pad:
        assert out == "injected" and calls == ["lamport gather"] and seen == [12]
    else:
        # A refusing tail or a padded round trip: the plain reduce-scatter.
        assert calls == ["lamport gather", "lamport reduce"] and out.shape == x.shape
        assert seen == ([] if pad else [12])


def test_tail_joins_the_side_stream_before_the_fused_launch(monkeypatch):
    log = []
    main, side, _ = _streams(monkeypatch, log)
    core = SimpleNamespace(eps=1e-6)
    rsi = _rsi(log)
    site = (torch.zeros(4, 16), "inj", "norm")
    tail = M._rs_inject_tail(core, rsi, site, lambda: ("sh", "gl"), main, side)
    assert tail("partial") == M._RsInjected("hc_out", "normed")
    assert log == ["main waits side", ("fused", "partial", "sh", "gl")]
    log.clear()
    unfin = M._Unfinalized("g2", "w", "e2p")
    assert tail(unfin) == M._RsInjected("hc_out", "normed")
    assert log == ["main waits side", ("fused fin", "g2", "w", "e2p", "sh", "gl")]


def test_tail_refusal_keeps_the_plain_reduce_scatter(monkeypatch):
    log = []
    main, side, _ = _streams(monkeypatch, log)
    rsi = _rsi(log, refuse="hc strides")
    tail = M._rs_inject_tail(
        SimpleNamespace(eps=1e-6),
        rsi,
        (torch.zeros(4, 16), "i", "n"),
        lambda: ("s", "g"),
        main,
        side,
    )
    assert tail("partial") is None and log == []


# ------------------------------------------------------------------- the trunk block


@pytest.mark.parametrize("fin", [False, True])
def test_moe_block_hands_the_injected_residual(monkeypatch, fin):
    log = []
    main, side, current = _streams(monkeypatch, log)
    monkeypatch.setattr(
        M,
        "_shared_expert",
        lambda core, x, L, g: log.append(f"shared on {current[-1].name}") or ("sh", "gl"),
    )
    monkeypatch.setattr(M, "_routed_experts", lambda *a: log.append("routed") or "partial")
    monkeypatch.setattr(
        M,
        "_route_front_unfinalized",
        lambda core, rows, L, mm: log.append("routed unfinalized") or M._Unfinalized(1, 2, 3),
    )

    def expert_sum(run, x, pad_rows, tail=None):
        log.append("gather")
        out = tail(run(x))
        return out

    rsi = _rsi(log, fin_rows=(4,) if fin else ())
    core = SimpleNamespace(
        _side_stream=side,
        dp_size=8,
        dp_launch_order=True,
        expert_sum=expert_sum,
        _rs_inject=rsi,
        _tep_finar=None,
        eps=1e-6,
    )
    out = M._moe_block(
        core, torch.zeros(4, 8), None, 0, rs_next=(torch.zeros(4, 16), "inj", "norm")
    )
    assert out == (M._RsInjected("hc_out", "normed"), None, None)
    launch = ("fused fin", 1, 2, 3, "sh", "gl") if fin else ("fused", "partial", "sh", "gl")
    assert log == [
        "side waits main",
        "gather",
        "shared on side",
        "routed unfinalized" if fin else "routed",
        "main waits side",
        launch,
    ]


def test_moe_block_finalize_form_refused_keeps_the_finalize(monkeypatch):
    log = []
    _, side, _ = _streams(monkeypatch, log)
    monkeypatch.setattr(M, "_shared_expert", lambda *a: ("sh", "gl"))
    monkeypatch.setattr(M, "_routed_experts", lambda *a: log.append("routed") or "partial")
    monkeypatch.setattr(M, "_route_front_unfinalized", lambda *a: pytest.fail("not this one"))
    rsi = _rsi(log, fin_rows=(4,), refuse_fin="inject strides")
    core = SimpleNamespace(
        _side_stream=side,
        dp_size=8,
        dp_launch_order=True,
        expert_sum=lambda run, x, pad, tail=None: tail(run(x)),
        _rs_inject=rsi,
        _tep_finar=None,
        eps=1e-6,
    )
    out = M._moe_block(
        core, torch.zeros(4, 8), None, 0, rs_next=(torch.zeros(4, 16), "inj", "norm")
    )
    assert out[0] == M._RsInjected("hc_out", "normed")
    assert "routed" in log and ("fused", "partial", "sh", "gl") in log


# ------------------------------------------------------------------- the build


def _build_core(monkeypatch, world=8, trip=True, sizes=(1, 2, 3)):
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: True)
    monkeypatch.setattr(M.torch.cuda, "device", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(M.torch.cuda, "synchronize", lambda *a: None)
    return SimpleNamespace(
        _lamport_trip=SimpleNamespace(world=world, max_rows=512) if trip else None,
        dp_size=world,
        dp_rank=0,
        hidden=2560,
        hc_count=4,
        topk=10,
        mtp_enabled=True,
        max_verify_rows=4,
        model_config=SimpleNamespace(cuda_graph_batch_sizes=list(sizes)),
        _route_front=None,
        mapping=None,
    )


class _Dist:
    def __init__(self, peers):
        self.peers = peers  # what the other ranks contribute to each all-gather, in order

    def tp_allgather(self, mine):
        return [mine, *self.peers.pop(0)]


def _patch_build(monkeypatch, dist, checks=None, compile_error=None):
    import tensorrt_llm._torch.distributed.communicator as comm

    monkeypatch.setattr(
        comm, "Distributed", SimpleNamespace(get=lambda mapping: dist), raising=False
    )
    op = SimpleNamespace(
        supported=lambda h, c: True,
        LAYOUTS={4: ((1, 192), (2, 608)), 8: ((1, 304), (2, 608))},
        MAX_ROWS=511,
        parse_layouts=lambda raw: ((1, 100),),
        fin_resident=lambda rows, hc, layouts, sms: rows <= 40,
    )

    def build(trip, hc, layouts, fin, topk, pdl):
        if compile_error:
            raise RuntimeError(compile_error)
        return SimpleNamespace(layouts=layouts, pdl=pdl, fin=fin)

    op.build = build
    monkeypatch.setattr(M, "_rs_inject_op", op)
    checked = []

    def check(core, rs, rows, fin, gen, device, draft=False):
        checked.append((rows, fin, draft))
        return (checks or {}).get(rows, ("", "", ""))

    monkeypatch.setattr(M, "_dp_rs_inject_check", check)
    monkeypatch.setattr(
        M.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=152),
        raising=False,
    )
    monkeypatch.setattr(
        M.torch, "Generator", lambda device=None: SimpleNamespace(manual_seed=lambda s: None)
    )
    return checked


def test_build_off_and_ignored(monkeypatch):
    monkeypatch.setenv(M.DP_RS_INJECT_ENV, "0")
    assert M._dp_rs_inject_build(_build_core(monkeypatch), "cuda") is None
    monkeypatch.delenv(M.DP_RS_INJECT_ENV)
    # No Lamport transport: nothing is built and nothing is exchanged.
    _patch_build(monkeypatch, _Dist([]))
    assert M._dp_rs_inject_build(_build_core(monkeypatch, trip=False), "cuda") is None


def test_build_keeps_rows_every_rank_holds(monkeypatch):
    monkeypatch.delenv(M.DP_RS_INJECT_ENV, raising=False)
    monkeypatch.delenv(M.DP_RS_INJECT_LAYOUTS_ENV, raising=False)
    # Trunk rows 4 / 8 / 12, draft rows 2..3 below the floor of 4: checked 4, 8, 12.
    peer_bad = [(8, "", "finalize: normed differ", ""), (12, "fused: hc_out differ", "", "")]
    dist = _Dist([[""] * 7, [[(4, "", "", ""), *peer_bad]] + [[]] * 6])
    _patch_build(monkeypatch, dist)
    built = M._dp_rs_inject_build(_build_core(monkeypatch), "cuda")
    assert built.rows == frozenset({4, 8}) and built.draft_rows == frozenset()
    assert built.fin_rows == frozenset()  # no route front: no finalize form
    assert built.rs.layouts == ((1, 304), (2, 608))


def test_build_compile_failure_on_a_peer_declines_everywhere(monkeypatch):
    monkeypatch.delenv(M.DP_RS_INJECT_ENV, raising=False)
    dist = _Dist([["", "RuntimeError: ptxas", "", "", "", "", ""]])
    _patch_build(monkeypatch, dist)
    assert M._dp_rs_inject_build(_build_core(monkeypatch), "cuda") is None
    assert dist.peers == []  # no check ran: nobody launched anything


def test_build_finalize_only_on_a_resident_grid(monkeypatch):
    """The finalize form is built, checked and served only at row counts the route front serves
    and whose grid is one resident wave (`fin_resident`; here up to 40 rows): above, the check runs
    the fused launch twice and the block keeps the runner's finalize."""
    monkeypatch.delenv(M.DP_RS_INJECT_ENV, raising=False)
    monkeypatch.delenv(M.DP_MOE_FIN_RS_ENV, raising=False)
    monkeypatch.delenv(M.DP_RS_INJECT_LAYOUTS_ENV, raising=False)
    dist = _Dist([[""] * 7, [[]] * 7])
    checked = _patch_build(monkeypatch, dist)
    core = _build_core(monkeypatch, sizes=(1, 10, 11))  # trunk rows 4, 40, 44
    core.mtp_enabled = False  # no draft row counts
    core._route_front = SimpleNamespace(tactics={8 * r: (16, 8) for r in (4, 40, 44)})
    built = M._dp_rs_inject_build(core, "cuda")
    assert built.rows == frozenset({4, 40, 44})
    assert built.fin_rows == frozenset({4, 40})
    assert [(r, fin) for r, fin, _ in checked] == [(4, True), (40, True), (44, False)]
    assert built.rs.fin
    # No row count in the resident range: no finalize launch is compiled at all.
    dist = _Dist([[""] * 7, [[]] * 7])
    checked = _patch_build(monkeypatch, dist)
    core._route_front = SimpleNamespace(tactics={8 * 44: (16, 8)})
    built = M._dp_rs_inject_build(core, "cuda")
    assert built.fin_rows == frozenset() and not built.rs.fin and not built.rs.pdl


def test_build_drops_draft_rows_whose_fused_launch_is_not_the_draft_paths(monkeypatch):
    """A draft row count whose fused launch differs from `mtp_moe_combine_inject` + the head's
    norm on any rank keeps the draft combine on every rank; the trunk keeps the fused launch."""
    monkeypatch.delenv(M.DP_RS_INJECT_ENV, raising=False)
    monkeypatch.delenv(M.DP_RS_INJECT_LAYOUTS_ENV, raising=False)
    peer = [(4, "", "", "draft: head norm differ"), (8, "", "", "")]
    dist = _Dist([[""] * 7, [peer] + [[]] * 6])
    checked = _patch_build(monkeypatch, dist)
    core = _build_core(monkeypatch, sizes=(1, 2, 4, 8))  # trunk 4, 8, 16, 32; draft 4, 8
    built = M._dp_rs_inject_build(core, "cuda")
    assert built.rows == frozenset({4, 8, 16, 32})
    assert built.draft_rows == frozenset({8})
    assert [(r, d) for r, _, d in checked] == [(4, True), (8, True), (16, False), (32, False)]


def test_build_layouts_env(monkeypatch):
    monkeypatch.delenv(M.DP_RS_INJECT_ENV, raising=False)
    monkeypatch.setenv(M.DP_RS_INJECT_LAYOUTS_ENV, "1:100")
    dist = _Dist([[""] * 7, [[]] * 7])
    _patch_build(monkeypatch, dist)
    built = M._dp_rs_inject_build(_build_core(monkeypatch, world=4), "cuda")
    assert built.rs.layouts == ((1, 100),)


# ------------------------------------------------------------------- the draft


def test_draftpack_parts(monkeypatch):
    core = SimpleNamespace(
        mtp_enabled=True, dp_launch_order=True, _mtp=SimpleNamespace(moe_gscale=None)
    )
    rsi = object()
    for raw, expected in (
        (None, {"combine", "headnorm"}),
        ("0", set()),
        ("1", {"combine", "headnorm", "route"}),
        ("all", {"combine", "headnorm", "route"}),
        ("route", {"route"}),
        ("combine, route", {"combine", "route"}),
    ):
        if raw is None:
            monkeypatch.delenv(M.DP_DRAFTPACK_ENV, raising=False)
        else:
            monkeypatch.setenv(M.DP_DRAFTPACK_ENV, raw)
        assert M._dp_draftpack_parts(core, rsi, "cuda") == expected
    # Without the fused reduce-scatter only the routing part remains.
    monkeypatch.setenv(M.DP_DRAFTPACK_ENV, "all")
    assert M._dp_draftpack_parts(core, None, "cuda") == {"route"}
    for bad in ("headnorm", "pack"):
        monkeypatch.setenv(M.DP_DRAFTPACK_ENV, bad)
        with pytest.raises(ValueError):
            M._dp_draftpack_parts(core, rsi, "cuda")
    # Not the side-stream draft MoE, or native NVFP4 draft experts: nothing.
    monkeypatch.delenv(M.DP_DRAFTPACK_ENV)
    core.dp_launch_order = False
    assert M._dp_draftpack_parts(core, rsi, "cuda") == frozenset()


def _draft_core(log, side, parts, rows=(3,)):
    def expert_sum(run, x, pad_rows, tail=None):
        log.append("gather")
        partial = run(x)
        if tail is not None:
            fused = tail(partial)
            if fused is not None:
                return fused
        log.append("reduce-scatter")
        return partial

    return SimpleNamespace(
        _side_stream=side,
        dp_size=8,
        dp_launch_order=True,
        expert_sum=expert_sum,
        eps=1e-6,
        hc_count=4,
        dp_draftpack=frozenset(parts),
        _rs_inject=_rsi(log, draft_rows=rows),
        tp_sum=lambda t: t,
    )


@pytest.mark.parametrize("captured", [True, False])
def test_draft_combine(monkeypatch, captured):
    log = []
    main, side, current = _streams(monkeypatch, log)
    monkeypatch.setattr(M.torch.cuda, "is_current_stream_capturing", lambda: captured)
    monkeypatch.setattr(M, "qwen4_exp_hc_inject_norm", lambda *a: ("x", "normed"))
    monkeypatch.setattr(M, "_hc_mixer", lambda *a: (torch.zeros(3, 8), "inject"))
    monkeypatch.setattr(M, "_draft_o", lambda core, attn, w: "o")
    monkeypatch.setattr(
        M,
        "_mtp_shared_expert",
        lambda core, W, x: log.append(f"shared on {current[-1].name}") or ("gl", "sh"),
    )
    routes = []

    def moe_block(core, W, x, pad_rows, routed_only, before_window, tail=None, route=False):
        routes.append(route)
        return core.expert_sum(lambda g: before_window() or "partial", x, pad_rows, tail)

    monkeypatch.setattr(M, "_mtp_moe_block", moe_block)
    monkeypatch.setattr(M, "_combine_inject_fits", lambda *a: True)
    monkeypatch.setattr(
        M, "mtp_moe_combine_inject", lambda *a: log.append("combine inject") or "out"
    )
    core = _draft_core(log, side, {"combine", "headnorm", "route"})
    W = SimpleNamespace(o=None, mhc_norm=None, mhc_first=None, mhc_up=None, fhc_norm="fnorm")
    out = M._mtp_tail(core, W, "x0", "attn", "inj0", 0)
    assert routes == [captured]  # `route` rides on captured passes only
    if captured:
        assert out == "hc_out" and "combine inject" not in log
        assert ("fused", "partial", "sh", "gl") in log
        assert core._rs_inject.stash == M._RsInjected("hc_out", "normed")
    else:
        assert out == "out" and log[-1] == "combine inject"
        assert core._rs_inject.stash is None


def test_draft_head_takes_the_combines_norm_only_for_its_tensor(monkeypatch):
    launched = []
    monkeypatch.setattr(
        M, "qwen4_exp_hc_inject_norm", lambda *a: launched.append(a[0]) or (a[0], "launched")
    )
    monkeypatch.setattr(M, "_hc_mixer", lambda core, normed, *a: (normed, None))
    hc_out = torch.zeros(3, 16)
    core = SimpleNamespace(
        _mtp=SimpleNamespace(fhc_norm=None, fhc_first=None, fhc_up=None),
        eps=1e-6,
        hc_count=4,
        _rs_inject=_rsi([]),
    )
    core._rs_inject.stash = M._RsInjected(hc_out, "fused norm")
    assert M._mtp_head_final(core, hc_out) == "fused norm" and launched == []
    # Taken once: the next head launches its norm.
    assert M._mtp_head_final(core, hc_out) == "launched"
    # Another tensor (a gather of the rows) never takes the stash, which is dropped.
    core._rs_inject.stash = M._RsInjected(hc_out, "fused norm")
    other = hc_out[:2]
    assert M._mtp_head_final(core, other) == "launched"
    assert core._rs_inject.stash is None


def test_draft_runner_routes_from_the_logits(monkeypatch):
    calls = []
    monkeypatch.setattr(M, "cublas_mm", lambda x, w: "logits")
    monkeypatch.setattr(
        M, "renorm_moe_routing_op", lambda *a, **k: calls.append("renorm") or ("ids", "w")
    )
    monkeypatch.setattr(M, "fp8_quantize_1x128", lambda x, use_ue8m0: ("xq", "xsf"))
    monkeypatch.setattr(M, "_draft_moe_tactic", lambda core, rows: (16, 8))
    monkeypatch.setattr(
        M,
        "fp8_block_scale_moe_runner",
        lambda **k: calls.append((k["routing_logits"], k["topk_ids"], k["tactic"])) or "out",
    )
    core = SimpleNamespace(
        dp_size=8,
        expert_sum=lambda run, x, pad: run(x),
        topk=10,
        num_experts=512,
        moe_inter=640,
        mtp_expert_offset=0,
        mtp_local_experts=64,
    )
    W = SimpleNamespace(moe_gscale=None, router=None, fc1_w=0, fc1_s=0, fc2_w=0, fc2_s=0)
    x = torch.zeros(4, 8)
    assert M._mtp_moe_block(core, W, x, 0, True, route=True) == "out"
    assert calls == [("logits", None, (16, 8))]
    calls.clear()
    M._mtp_moe_block(core, W, x, 0, True)
    assert calls == ["renorm", (None, "ids", (16, 8))]


# ------------------------------------------------------------------- the build's check


@pytest.mark.parametrize("raises", [None, "fused", "finalize"])
def test_check_launches_three_reduce_scatters_whatever_happens(monkeypatch, raises):
    """Every rank of the group launches the same three reduce-scatters a row count, so a launch
    that raises on this rank is replaced by the plain one (its peers poll for its bytes)."""
    launches = []
    monkeypatch.setattr(
        M, "dep_lamport_round_trip", lambda p, trip, reduce: launches.append("plain") or "routed"
    )
    monkeypatch.setattr(M, "qwen4_exp_hc_inject_norm", lambda *a: ("hc", "n"))
    monkeypatch.setattr(M, "_bitwise", lambda a, b: a == b)
    monkeypatch.setattr(M.torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(
        M, "_dp_rs_inject_operands", lambda core, rows, gen, device: ("h", "i", "n", "s", "g")
    )
    monkeypatch.setattr(M.torch, "randn", lambda *a, **k: torch.zeros(8, 8))

    class Rs:
        def fin_operands_refuse(self, rows, *unfin):
            return None

        def run(self, *a):
            if raises == "fused":
                raise RuntimeError("boom")
            launches.append("fused")
            return "hc", "n"

        def run_fin(self, *a):
            if raises == "finalize":
                raise RuntimeError("boom")
            launches.append("fin")
            return "hc", "x"

    st = SimpleNamespace(views={8: [[torch.zeros(80, dtype=torch.int32)]]})
    monkeypatch.setattr(M, "_decode_router", lambda x, r: "logits")
    monkeypatch.setattr(
        M,
        "_route_front_run",
        lambda core, st, x, L, layer, lg, fp4=None, finalize=True: (
            ("partial",) if finalize else ("fc2", None, None, None, "wts")
        ),
    )
    core = SimpleNamespace(
        _lamport_trip=SimpleNamespace(world=4),
        hidden=8,
        eps=1e-6,
        hc_count=4,
        topk=10,
        _route_front=st,
        _decode=SimpleNamespace(_layers=[SimpleNamespace(router=None)]),
    )
    fused, third, drafted = M._dp_rs_inject_check(core, Rs(), 2, True, None, "cpu")
    expected = {
        None: ["plain", "fused", "fin"],
        "fused": ["plain", "plain", "fin"],
        "finalize": ["plain", "fused", "plain"],
    }[raises]
    assert launches == expected
    assert ("boom" in fused) == (raises == "fused")
    # The finalize form's normed differs from the reference here: reported, never raised.
    assert ("boom" in third) if raises == "finalize" else ("normed" in third)


@pytest.mark.parametrize("combine", ["hc", "other"])
def test_check_holds_draft_rows_to_the_draft_path(monkeypatch, combine):
    """At a draft row count the fused launch's output is also compared, locally and after the
    three launches, with `mtp_moe_combine_inject` + the head's norm without injection."""
    launches = []
    monkeypatch.setattr(
        M, "dep_lamport_round_trip", lambda p, trip, reduce: launches.append("plain") or "routed"
    )
    norms = []
    monkeypatch.setattr(
        M, "qwen4_exp_hc_inject_norm", lambda *a: norms.append(a[1:3]) or ("hc", "n")
    )
    monkeypatch.setattr(
        M, "mtp_moe_combine_inject", lambda *a: launches.append("combine") or combine
    )
    monkeypatch.setattr(M, "_bitwise", lambda a, b: a == b)
    monkeypatch.setattr(M.torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(
        M, "_dp_rs_inject_operands", lambda core, rows, gen, device: ("h", "i", "n", "s", "g")
    )
    monkeypatch.setattr(M.torch, "randn", lambda *a, **k: torch.zeros(8, 8))
    rs = SimpleNamespace(run=lambda *a: launches.append("fused") or ("hc", "n"))
    core = SimpleNamespace(_lamport_trip=SimpleNamespace(world=4), hidden=8, eps=1e-6, hc_count=4)
    fused, third, drafted = M._dp_rs_inject_check(core, rs, 2, False, None, "cpu", True)
    assert launches == ["plain", "fused", "fused", "combine"]
    assert norms[-1] == (None, None)  # the head's norm: no block, no injection
    assert fused == third == ""
    assert drafted == ("" if combine == "hc" else "draft: hc differ")


def test_check_runner_output_failure_still_launches_three(monkeypatch):
    """Forming the runner's output raises on this rank: no launch is skipped (the finalize slot
    runs the fused launch over random partials) and the finalize form is reported off."""
    launches = []
    monkeypatch.setattr(
        M, "dep_lamport_round_trip", lambda p, trip, reduce: launches.append("plain") or "routed"
    )
    monkeypatch.setattr(M, "qwen4_exp_hc_inject_norm", lambda *a: ("hc", "n"))
    monkeypatch.setattr(M, "_bitwise", lambda a, b: a == b)
    monkeypatch.setattr(M.torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(
        M, "_dp_rs_inject_operands", lambda core, rows, gen, device: ("h", "i", "n", "s", "g")
    )
    monkeypatch.setattr(M.torch, "randn", lambda *a, **k: torch.zeros(8, 8))
    monkeypatch.setattr(M, "_decode_router", lambda x, r: "logits")

    def front(*a, **k):
        raise RuntimeError("no tactic")

    monkeypatch.setattr(M, "_route_front_run", front)
    rs = SimpleNamespace(
        run=lambda *a: launches.append("fused") or ("hc", "n"),
        run_fin=lambda *a: pytest.fail("no runner output to finalize"),
    )
    core = SimpleNamespace(
        _lamport_trip=SimpleNamespace(world=4),
        hidden=8,
        eps=1e-6,
        hc_count=4,
        topk=10,
        _route_front=SimpleNamespace(),
        _decode=SimpleNamespace(_layers=[SimpleNamespace(router=None)]),
    )
    fused, third, drafted = M._dp_rs_inject_check(core, rs, 2, True, None, "cpu")
    assert launches == ["plain", "fused", "fused"]
    assert fused == "" and "no tactic" in third and drafted == ""
