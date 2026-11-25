# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP router front (`DP_ROUTER_FRONT_ENV`).

* A captured decode step's router is the front (`_decode_router_mid`): `mid_rows_mm` over the
  layer's bf16 router at its gathered row counts, `_decode_router` at every other; an eager step
  keeps the prefill router. The route front takes the front as a decode router and calls it.
* The round trip's Lamport gather releases its dependents at the front's release row counts.
* The build is off with the switch unset, declines away from dep4 / dep8 on the Lamport
  transport and SM103, and engages on every rank or none.

The ops are recorders; no kernel runs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


def _front(rows=(64,), release=(256,)):
    router = torch.zeros(4, 2)
    return M._DpRouterFront(frozenset(rows), frozenset(release), {router.data_ptr(): "mw"}), router


def test_front_runs_the_bf16_router_at_its_rows(monkeypatch):
    calls = []
    monkeypatch.setattr(M, "mid_rows_mm_op", lambda x, mw: calls.append(("mid", mw)) or "mid")
    monkeypatch.setattr(M, "_decode_router", lambda x, w: calls.append(("decode",)) or "decode")
    front, router = _front()
    assert front(torch.zeros(64, 4), router) == "mid"
    assert front(torch.zeros(32, 4), router) == "decode"
    assert calls == [("mid", "mw"), ("decode",)]


@pytest.mark.parametrize("captured", [False, True])
def test_decode_router_of_a_captured_dp_step_is_the_front(captured):
    front, _ = _front()
    core = SimpleNamespace(dp_size=8, one_rank=False, decode_mm=None, _router_front=front)
    want = front if captured else M.cublas_mm
    assert M._decode_router_mid(core, SimpleNamespace(router_m=None), captured) is want
    core._router_front = None
    want = M._decode_router if captured else M.cublas_mm
    assert M._decode_router_mid(core, SimpleNamespace(router_m=None), captured) is want


def test_route_front_takes_the_front_as_its_router(monkeypatch):
    front, router = _front()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    seen = []
    monkeypatch.setattr(M, "mid_rows_mm_op", lambda x, mw: "front logits")
    monkeypatch.setattr(
        M,
        "_route_front_run",
        lambda core, st, x, L, layer, logits, fp4: seen.append(logits) or (0,),
    )
    st = M._RouteFront(None)
    st.tactics = {64: (8, 53)}
    L = SimpleNamespace(fc1_w=torch.zeros(1), router=router)
    st.layers = {L.fc1_w.data_ptr(): 0}
    core = SimpleNamespace(_route_front=st, dp_size=8, hidden=4)
    M._routed_experts(core, torch.zeros(64, 4, dtype=torch.bfloat16), L, None, front)
    assert seen == ["front logits"]


def test_round_trip_releases_at_the_fronts_rows(monkeypatch):
    calls = []
    monkeypatch.setenv(M.DP_MOE_COMM_ENV, "lamport_step")

    def lamport(x, trip, reduce, release=False):
        calls.append(("reduce" if reduce else "gather", x.shape[0], release))
        return x[: x.shape[0] // 8] if reduce else x.repeat(8, 1)

    monkeypatch.setattr(M, "dep_lamport_round_trip", lamport)
    trip = SimpleNamespace(takes=lambda rows, x: True)
    front, _ = _front(release=(32 * 8,))
    core = SimpleNamespace(
        dp_size=8,
        dp_group=list(range(8)),
        mapping=SimpleNamespace(tp_group_pg=None),
        _dp_lamport=lambda group: trip,
        _router_front=front,
    )
    expert_sum = M.Qwen4ExpDefaultModelingV2._expert_sum_op(core)
    for rows in (32, 12):
        expert_sum(lambda g: g, torch.zeros(rows, 4), 0)
    assert calls == [
        ("gather", 32, True),
        ("reduce", 256, False),
        ("gather", 12, False),
        ("reduce", 96, False),
    ]


def _core(dp_size=8, trip=True, sizes=(2, 3, 8)):
    return SimpleNamespace(
        dp_size=dp_size,
        dp_rank=0,
        max_verify_rows=4,
        mapping=None,
        model_config=SimpleNamespace(cuda_graph_batch_sizes=list(sizes)),
        _lamport_trip=SimpleNamespace(releasing_gather_fn=lambda: None) if trip else None,
    )


def test_build_is_off_with_the_switch(monkeypatch):
    monkeypatch.delenv(M.DP_ROUTER_FRONT_ENV, raising=False)
    assert M._dp_router_front_build(_core(), [], torch.device("cuda")) is None


@pytest.mark.parametrize(
    "dp_size, trip, capability", [(2, True, (10, 3)), (8, False, (10, 3)), (8, True, (10, 7))]
)
def test_build_declines(monkeypatch, dp_size, trip, capability):
    monkeypatch.setenv(M.DP_ROUTER_FRONT_ENV, "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    assert M._dp_router_front_build(_core(dp_size, trip), [], torch.device("cuda")) is None


class _Dist:
    def __init__(self, others):
        self.others = others

    def tp_allgather(self, value):
        return [value] + list(self.others)


@pytest.mark.parametrize(
    "others, off, engaged", [((), 0, True), (("bad",), 0, False), ((), 3, False)]
)
def test_build_engages_on_every_rank_or_none(monkeypatch, others, off, engaged):
    from tensorrt_llm._torch.distributed import communicator

    monkeypatch.setenv(M.DP_ROUTER_FRONT_ENV, "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 3))
    monkeypatch.setattr(torch.cuda, "device", lambda device: _Nothing())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(
        M.torch, "Generator", lambda device=None: SimpleNamespace(manual_seed=lambda s: None)
    )
    monkeypatch.setattr(M, "MidWeight", lambda w: ("mw", w.data_ptr()))
    monkeypatch.setattr(M, "mid_rows_precompile", lambda mw, lo, hi: None)
    checked = []
    monkeypatch.setattr(M, "_dp_router_front_check", lambda mw, r, gen, d: checked.append(r) or off)
    monkeypatch.setattr(communicator.Distributed, "get", staticmethod(lambda m: _Dist(others)))
    layers = [SimpleNamespace(router=torch.zeros(4, 2)) for _ in range(2)]
    front = M._dp_router_front_build(_core(), layers, torch.device("cuda"))
    # dep8, four rows a request: graph sizes 2 / 3 / 8 -> 64 / 96 / 256 gathered rows.
    assert checked == ([64, 96] if not off else [64])
    if not engaged:
        assert front is None
        return
    assert front.rows == {64, 96} and front.release_rows == {64, 96, 256}
    assert set(front.mids) == {L.router.data_ptr() for L in layers}


class _Nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
