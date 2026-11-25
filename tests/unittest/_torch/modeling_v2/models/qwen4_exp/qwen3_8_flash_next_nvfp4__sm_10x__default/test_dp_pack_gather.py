# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP packed MoE gather (`DP_MOE_PACK_GATHER_ENV`, opt-in) on CPU recorders: which
blocks take it, the launch order of a block that does (the owner's router, the packed gather, the
shared expert, the routing-only front, the runner, the reduce-scatter or the fused one), the owner
router on `mid_rows_mm` (`DP_MOE_PACK_ROUTER_ENV=mid`), and the build's group agreement."""

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
    def __init__(self, log):
        self.log = log

    def takes(self, rows, x):
        return True

    def __call__(self, x, logits, gs):
        self.log.append(("pack", logits))
        return "xq", "xsf", "lg"


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
        lambda lg, tab, tile, off, local, hidden, fast2: log.append(("route only", lg, tab, tile))
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
        fast2=set(),
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
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: True)
    monkeypatch.setattr(M.torch.cuda, "device", lambda d: contextlib.nullcontext())
    st = SimpleNamespace(tactics={24: (8, 0), 32: (16, 0), 96: (16, 0)}, fast2=set())
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
