# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP packed MoE gather (`DP_MOE_PACK_GATHER_ENV`, opt-in) on CPU recorders: which
blocks take it, the launch order of a block that does (the owner's router, the packed gather, the
shared expert, the routing-only front, the runner, the reduce-scatter or the fused one), the owner
router on `mid_rows_mm` (`DP_MOE_PACK_ROUTER_ENV=mid`), the owner top-k (`DP_MOE_OWNK_ENV`: which
rows take it, its value, its check's logits) and the builds' group agreement."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


class _Logits(str):
    """A recorded router output (`.contiguous()` is itself)."""

    def contiguous(self):
        return self


class _Pack:
    def __init__(self, log, name="pack", out="lg"):
        self.log, self.name, self.out = log, name, out

    def takes(self, rows, x):
        return True

    def __call__(self, x, logits, gs):
        self.log.append((self.name, logits))
        return "xq", "xsf", self.out


def _setup(monkeypatch, log, mid_rows=(), fin_rows=()):
    monkeypatch.setattr(
        M, "_decode_router", lambda x, r: log.append("decode router") or _Logits("own logits")
    )
    monkeypatch.setattr(
        M, "mid_rows_mm_op", lambda x, mw: log.append(("mid router", mw)) or _Logits("mid logits")
    )
    monkeypatch.setattr(
        M,
        "_route_front_only",
        lambda lg, tab, tile, off, local, hidden, fast2, ids_in=False, early=False: log.append(
            ("route only", lg, tab, tile)
            + (("ids_in",) if ids_in else ())
            + (("early",) if early else ())
        )
        or ("ids", "wts"),
    )

    def run(core, st, x, L, layer, logits, fp4, finalize, routed):
        log.append(("runner", finalize, routed))
        return ("out" if finalize else "fc2",)

    monkeypatch.setattr(M, "_route_front_run", run)
    monkeypatch.setattr(
        M, "dep_lamport_round_trip", lambda p, trip, reduce: log.append(("reduce", p)) or "mine"
    )
    st = SimpleNamespace(
        layers={7: 3},
        tactics={32: (16, 0)},
        flats={32: {3: "flat3"}},
        fast2={},
        views={32: {3: [torch.zeros(320, dtype=torch.int32)]}},
    )
    pk = M._DpPack(_Pack(log), frozenset({32}), {9: "mw9"}, frozenset(mid_rows))
    core = SimpleNamespace(
        _route_front=st,
        dp_size=8,
        expert_offset=0,
        local_experts=64,
        hidden=8,
        topk=10,
        eps=1e-6,
        _lamport_trip="trip",
        _rs_inject=None,
    )
    L = SimpleNamespace(
        fc1_w=SimpleNamespace(data_ptr=lambda: 7),
        router=SimpleNamespace(data_ptr=lambda: 9),
        moe_gscale="gs",
    )
    return core, pk, L


@pytest.mark.parametrize("mid", [False, True])
def test_block_launch_order(monkeypatch, mid):
    log = []
    core, pk, L = _setup(monkeypatch, log, mid_rows=(4,) if mid else ())
    forked = []

    def shared_expert():
        log.append("shared")
        forked.append(("sh", "gl"))

    out = M._dp_pack_block(core, pk, torch.zeros(4, 8), L, None, shared_expert, forked, None, None)
    assert out == ("mine", False)
    router = ("mid router", "mw9") if mid else "decode router"
    assert log == [
        router,
        ("pack", "mid logits" if mid else "own logits"),
        "shared",
        ("route only", "lg", "flat3", 16),
        ("runner", True, ("xq", "xsf", "ids", "wts")),
        ("reduce", "out"),
    ]


@pytest.mark.parametrize("ownk", [False, True])
def test_block_releases_the_runner_early_at_the_fronts_early_rows(monkeypatch, ownk):
    """`DP_ROUTE_FRONT_EARLY_ENV`: at a gathered row count the route front releases early, the
    routing-only front (packed or owner-routed rows) takes its early launch too."""
    log = []
    core, pk, L = _setup(monkeypatch, log)
    core._route_front.early = {32}
    if ownk:
        pk.ownk, pk.ownk_rows = _Pack(log, "ownk", "idw"), frozenset({32})
    M._dp_pack_block(core, pk, torch.zeros(4, 8), L, None, lambda: None, [], None, None)
    routed = [e for e in log if e[0] == "route only"]
    assert routed == [
        ("route only", "idw" if ownk else "lg", "flat3", 16)
        + (("ids_in",) if ownk else ())
        + ("early",)
    ]


def test_block_takes_the_owner_top_k_at_its_rows(monkeypatch):
    """At a gathered row count of `ownk_rows` the owner top-k gather runs in place of the packed
    gather and the routing-only front reads its routing rows (`ids_in`); elsewhere nothing
    changes."""
    for ownk_rows, gather, routed in (
        (frozenset({32}), ("ownk", "own logits"), ("route only", "idw", "flat3", 16, "ids_in")),
        (frozenset({64}), ("pack", "own logits"), ("route only", "lg", "flat3", 16)),
    ):
        log = []
        core, pk, L = _setup(monkeypatch, log)
        pk.ownk, pk.ownk_rows = _Pack(log, "ownk", "idw"), ownk_rows
        out = M._dp_pack_block(
            core, pk, torch.zeros(4, 8), L, None, lambda: log.append("shared"), [], None, None
        )
        assert out == ("mine", False)
        assert log == [
            "decode router",
            gather,
            "shared",
            routed,
            ("runner", True, ("xq", "xsf", "ids", "wts")),
            ("reduce", "out"),
        ]


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, M._DP_MOE_OWNK_ROWS),
        ("", M._DP_MOE_OWNK_ROWS),
        ("1", M._DP_MOE_OWNK_ROWS),
        ("all", (1, 1024)),
        ("0", None),
        ("off", None),
    ],
)
def test_ownk_value(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(M.DP_MOE_OWNK_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_MOE_OWNK_ENV, value)
    assert M._dp_ownk_wanted() == expected


def test_ownk_value_rejects_a_range(monkeypatch):
    monkeypatch.setenv(M.DP_MOE_OWNK_ENV, "18:24")
    with pytest.raises(ValueError, match="DP_MOE_OWNK"):
        M._dp_ownk_wanted()


def test_ownk_default_rows_are_the_two_token_front():
    """The default starts where the routing front routes two tokens a warp (17 requests a rank at
    dep8 MTP-3, 544 rows) and ends at the largest measured cap (28, 896 rows): the measured caps
    18 / 20 / 24 / 26 / 28 inside, cap 12 (384) and the unmeasured caps 29..32 outside."""
    from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import (
        moe_route_front__decode as front,
    )

    lo, hi = M._DP_MOE_OWNK_ROWS
    assert lo == front.FAST2_MIN_ROWS and front.MAX_ROWS == M._DP_PACK_MAX_GATHERED
    assert all(lo <= 32 * cap <= hi for cap in (17, 18, 20, 24, 26, 28))
    assert not lo <= 32 * 12 and not any(lo <= 32 * cap <= hi for cap in (29, 32))


def test_ownk_check_logits_cover_every_kind():
    """The owner top-k check draws each kind on some row: plain, the window lifted, coarse ties,
    one lane's experts on top (the full-sort fallback), and +-0.0 ties."""
    gen = torch.Generator().manual_seed(5)
    lg = M._dp_ownk_logits(10, 512, 64, 64, gen, "cpu", 3).float()
    assert lg.dtype is torch.float32 and lg.shape == (10, 512)
    kinds = [(r + 3) % 5 for r in range(10)]
    for r, kind in enumerate(kinds):
        row = lg[r]
        top = row.topk(10).indices
        if kind == 1:
            assert bool(((top >= 64) & (top < 128)).all())
        elif kind == 2:
            assert torch.equal(row * 2, (row * 2).round())
        elif kind == 3:
            assert bool((top // 16 == r % 32).all())
        elif kind == 4:
            assert set(row.unique().tolist()) <= {0.0, 1.0}
            assert bool(torch.signbit(row[::3][1:]).any())


def test_block_with_the_fused_finalize_form(monkeypatch):
    log = []
    core, pk, L = _setup(monkeypatch, log)
    rs = SimpleNamespace(refuses_fin=lambda *a: None)
    core._rs_inject = SimpleNamespace(rs=rs, fin_rows=frozenset({4}))
    seen = []
    monkeypatch.setattr(
        M,
        "_rs_inject_tail",
        lambda core, rsi, site, operands, main, side: lambda p: seen.append(p) or "injected",
    )
    forked = [("sh", "gl")]
    out = M._dp_pack_block(
        core, pk, torch.zeros(4, 8), L, (torch.zeros(4, 32), "i", "n"), None, forked, None, None
    )
    assert out == ("injected", True)
    assert ("runner", False, ("xq", "xsf", "ids", "wts")) in log
    assert isinstance(seen[0], M._Unfinalized) and seen[0].g2 == "fc2" and seen[0].w == "wts"


@pytest.mark.parametrize("rows, layer_ptr, expected", [(4, 7, True), (5, 7, False), (4, 8, False)])
def test_block_declines(monkeypatch, rows, layer_ptr, expected):
    log = []
    core, pk, L = _setup(monkeypatch, log)
    L.fc1_w = SimpleNamespace(data_ptr=lambda: layer_ptr)
    out = M._dp_pack_block(
        core, pk, torch.zeros(rows, 8), L, None, lambda: None, [("s", "g")], None, None
    )
    assert (out is not None) == expected


@pytest.mark.parametrize(
    "capturing, pad, taken", [(True, 0, True), (False, 0, False), (True, 1, False)]
)
def test_moe_block_takes_it_only_in_captured_unpadded_steps(monkeypatch, capturing, pad, taken):
    log = []
    monkeypatch.setattr(M.torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(
        M.torch.cuda, "current_stream", lambda *a: SimpleNamespace(wait_stream=lambda s: None)
    )
    monkeypatch.setattr(M.torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(M, "_shared_expert", lambda *a: ("sh", "gl"))
    monkeypatch.setattr(M, "_dp_pack_block", lambda *a: log.append("pack") or ("packed", False))
    core = SimpleNamespace(
        _side_stream=SimpleNamespace(wait_stream=lambda s: None),
        dp_size=8,
        dp_launch_order=True,
        _tep_finar=None,
        _dp_pack=object(),
        expert_sum=lambda run, x, pad_rows: log.append("bf16 gather") or "plain",
    )
    routed, gate, shared = M._moe_block(core, torch.zeros(4, 8), None, pad)
    assert routed == ("packed" if taken else "plain") and (gate, shared) == ("gl", "sh")
    assert log == (["pack"] if taken else ["bf16 gather"])


class _Dist:
    def __init__(self, peers):
        self.peers = peers

    def tp_allgather(self, mine):
        return [mine, *self.peers.pop(0)]


def test_build_off_ignored_and_agreed(monkeypatch):
    import tensorrt_llm._torch.distributed.communicator as comm

    monkeypatch.delenv(M.DP_MOE_PACK_GATHER_ENV, raising=False)
    assert M._dp_pack_build(SimpleNamespace(), "cuda") is None
    monkeypatch.setenv(M.DP_MOE_PACK_GATHER_ENV, "1")
    monkeypatch.setenv(M.DP_MOE_OWNK_ENV, "0")
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: True)
    monkeypatch.setattr(M.torch.cuda, "device", lambda d: contextlib.nullcontext())
    st = SimpleNamespace(tactics={24: (8, 0), 32: (16, 0), 96: (16, 0)}, fast2={})
    core = SimpleNamespace(
        dp_size=4, _lamport_trip=SimpleNamespace(steps=object(), world=8), _route_front=st
    )
    assert M._dp_pack_build(core, "cuda") is None  # dep4: ignored, nothing exchanged
    core = SimpleNamespace(
        dp_size=8,
        dp_rank=0,
        _lamport_trip=SimpleNamespace(steps=object(), world=8),
        _route_front=st,
        hidden=2560,
        num_experts=512,
        local_experts=64,
        mapping=None,
    )
    monkeypatch.setattr(M, "_PackGather", lambda trip, h, e: "gather")
    monkeypatch.setattr(M, "_route_front_compile", lambda *a: True)
    checks = {24: "", 32: "codes", 96: ""}
    monkeypatch.setattr(M, "_dp_pack_check", lambda core, pk, st, rows, device: checks[rows * 8])
    dist = _Dist(
        [
            [""] * 7,  # the build
            [[24, 32, 96]] * 6 + [[32, 96]],  # one peer lacks 24 (its route front dropped it)
            [[]] * 7,  # the checks
            [[]] * 7,  # the mid rows
        ]
    )
    monkeypatch.setattr(comm, "Distributed", SimpleNamespace(get=lambda m: dist), raising=False)
    built = M._dp_pack_build(core, "cuda")
    assert built.rows == frozenset({96}) and built.mid_rows == frozenset()
    assert built.ownk is None and built.ownk_rows == frozenset()


def _ownk_core(rows):
    st = SimpleNamespace(tactics={g: (16, 0) for g in rows}, fast2={})
    core = SimpleNamespace(
        dp_rank=0,
        _lamport_trip=SimpleNamespace(world=8),
        hidden=2560,
        num_experts=512,
        local_experts=64,
    )
    pk = M._DpPack("gather", frozenset(rows), {}, frozenset())
    return core, pk, st


@pytest.mark.parametrize("value", ["", "all"])
def test_ownk_build_agreed_and_checked(monkeypatch, value):
    """The owner top-k is built at the packed gather's agreed rows in the value's range, where the
    `ids_in` front compiled (it declines the lean rows) on every rank, and kept where every rank's
    check passed."""
    monkeypatch.setenv(M.DP_MOE_OWNK_ENV, value)
    monkeypatch.setattr(M.torch.cuda, "device", lambda d: contextlib.nullcontext())
    built_with = []
    monkeypatch.setattr(
        M, "_PackGather", lambda trip, h, e, ownk: built_with.append(ownk) or "ownk gather"
    )
    compiled = []

    def compile_front(g, hidden, stride, local, tile, fast2, route_only, ids_in, early=False):
        compiled.append((g, route_only, ids_in))
        return g > 256

    monkeypatch.setattr(M, "_route_front_compile", compile_front)
    checked = []

    def check(core, pk, st, rows, device, ownk):
        checked.append((rows * 8, ownk))
        return "routing" if rows * 8 == 768 else ""

    monkeypatch.setattr(M, "_dp_pack_check", check)
    core, pk, st = _ownk_core([256, 384, 544, 768, 896])
    peers_built = [[384, 544, 768, 896]] * 6 + [[544, 768, 896]]
    dist = _Dist([[""] * 7, peers_built, [[]] * 6 + [[(896, "tables")]]])
    M._dp_ownk_build(core, pk, st, "cuda", dist)
    want = [256, 384, 544, 768, 896] if value == "all" else [544, 768, 896]
    assert built_with == [True] and compiled == [(g, True, True) for g in want]
    agreed = [544, 768, 896]
    assert checked == [(g, "ownk gather") for g in agreed]
    # 768 failed here, 896 on a peer: both keep the packed gather of the logits.
    assert pk.ownk_rows == frozenset({544}) and pk.ownk == "ownk gather"


def test_ownk_build_off_and_failed(monkeypatch):
    monkeypatch.setenv(M.DP_MOE_OWNK_ENV, "0")
    core, pk, st = _ownk_core([544])
    M._dp_ownk_build(core, pk, st, "cuda", None)  # off: nothing exchanged
    assert pk.ownk is None and pk.ownk_rows == frozenset()
    monkeypatch.setenv(M.DP_MOE_OWNK_ENV, "1")
    monkeypatch.setattr(M.torch.cuda, "device", lambda d: contextlib.nullcontext())

    def raising(*a, **k):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(M, "_PackGather", raising)
    dist = _Dist([[""] * 7])
    M._dp_ownk_build(core, pk, st, "cuda", dist)  # this rank failed: no rank engages
    assert pk.ownk is None and pk.ownk_rows == frozenset() and dist.peers == []
    monkeypatch.setattr(M, "_PackGather", lambda *a, **k: "ownk gather")
    monkeypatch.setattr(M, "_route_front_compile", lambda *a: True)
    dist = _Dist([["", "", "RuntimeError: no kernel", "", "", "", ""]])
    M._dp_ownk_build(core, pk, st, "cuda", dist)  # a peer failed: no rank engages
    assert pk.ownk is None and pk.ownk_rows == frozenset() and dist.peers == []


@pytest.mark.parametrize("ownk", [False, True])
def test_check_launches_the_gather_when_its_inputs_raise(monkeypatch, ownk):
    """A rank whose check inputs fail to build still launches the packed gather (the check's
    only collective, which its peers poll for), over zeros, and reports the failure; the owner
    top-k check launches the owner top-k gather, not the packed gather of the logits."""
    launched = []

    class _Gather:
        def __init__(self, name="pack"):
            self.name = name

        def __call__(self, x, logits, gs):
            launched.append((self.name, tuple(x.shape), tuple(logits.shape), bool(x.any()), gs))
            return "xq", "xsf", "lg"

    def raising_randn(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(M.torch, "randn", raising_randn)
    monkeypatch.setattr(M.torch.cuda, "synchronize", lambda device=None: None)
    core = SimpleNamespace(
        _lamport_trip=SimpleNamespace(world=8),
        _decode=SimpleNamespace(_layers=[SimpleNamespace(moe_gscale="gs")]),
        dp_rank=3,
        hidden=16,
        num_experts=32,
    )
    pk = M._DpPack(_Gather(), frozenset({16}), {}, frozenset())
    what = M._dp_pack_check(core, pk, None, 2, "cpu", ownk=_Gather("ownk") if ownk else None)
    assert launched == [("ownk" if ownk else "pack", (2, 16), (2, 32), False, "gs")]
    assert what.startswith("inputs raised RuntimeError") and "out of memory" in what


def test_ownk_check_draws_the_same_group_on_every_rank(monkeypatch):
    """Every rank draws every rank's check rows and logits itself, so the draws may not depend on
    the drawing rank (its own `expert_offset` included): the reference would then differ from
    the routing rows the owners sent, and every row count would fail the check on every rank."""

    class _Sent(Exception):
        pass

    def gather(x, logits, gs):
        raise _Sent

    real = M._dp_ownk_logits

    def draws(rank):
        drawn = []
        monkeypatch.setattr(M, "_dp_ownk_logits", lambda *a: drawn.append(real(*a)) or drawn[-1])
        core = SimpleNamespace(
            _lamport_trip=SimpleNamespace(world=8),
            _decode=SimpleNamespace(_layers=[SimpleNamespace(moe_gscale="gs")]),
            dp_rank=rank,
            hidden=16,
            num_experts=512,
            local_experts=64,
            expert_offset=64 * rank,
        )
        pk = M._DpPack("gather", frozenset({80}), {}, frozenset())
        with pytest.raises(_Sent):
            M._dp_pack_check(core, pk, None, 10, "cpu", ownk=gather)
        return drawn

    first, other = draws(0), draws(5)
    assert len(first) == len(other) == 8
    for p, (a, b) in enumerate(zip(first, other)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"rank {p}'s logits differ"


@pytest.mark.parametrize("ownk", [False, True])
def test_check_holds_the_codes_to_what_the_transport_delivers(monkeypatch, ownk):
    """Both gathers, of the logits and the owner top-k one, send the NVFP4 codes through the same
    transport, which delivers a 16-bit half equal to its marker 0x8000 as 0x0000. In both modes the
    check compares the gathered codes to `dep_lamport_transported_codes` of the reference, so a
    row count whose check data holds such a half is not declined (the logits gather declined 15
    of 27 dep8 row counts on raw codes before this fix)."""
    raw = torch.tensor([[0x00, 0x80, 0x12, 0x34]], dtype=torch.uint8)
    delivered = torch.tensor([[0x00, 0x00, 0x12, 0x34]], dtype=torch.uint8)
    scales = torch.zeros([1, 2], dtype=torch.uint8)
    routed = torch.zeros([1, 24], dtype=torch.int16)
    ids, wts = torch.zeros([1, 10], dtype=torch.int32), torch.zeros([1, 10], dtype=torch.bfloat16)
    transported = []

    def transport(codes):
        transported.append(codes)
        out = codes.clone()
        out[:, 1] = 0
        return out

    def gather(x, logits, gs):
        return delivered, scales, routed.view(torch.bfloat16) if ownk else logits

    monkeypatch.setattr(M, "dep_lamport_transported_codes", transport)
    monkeypatch.setattr(M, "dep_lamport_sanitized", lambda t: t)
    monkeypatch.setattr(M, "fp4_quantize", lambda *a: (raw.clone(), scales.clone()))
    monkeypatch.setattr(M, "_route_front_only", lambda *a: (ids, wts))
    monkeypatch.setattr(M, "_route_front_run", lambda *a: (None, None, None, ids, wts))
    monkeypatch.setattr(M, "_routed_rows", lambda *a: routed)
    monkeypatch.setattr(M.torch.cuda, "synchronize", lambda device=None: None)
    core = SimpleNamespace(
        _lamport_trip=SimpleNamespace(world=1),
        _decode=SimpleNamespace(_layers=[SimpleNamespace(moe_gscale="gs")]),
        dp_rank=0,
        hidden=16,
        num_experts=512,
        local_experts=64,
        expert_offset=0,
    )
    st = SimpleNamespace(tactics={1: [0]}, flats={1: [torch.zeros(4)]}, fast2={})
    pk = M._DpPack(gather, frozenset({1}), {}, frozenset())
    what = M._dp_pack_check(core, pk, st, 1, "cpu", ownk=gather if ownk else None)
    assert what == ""
    assert len(transported) == 1 and torch.equal(transported[0], raw)
