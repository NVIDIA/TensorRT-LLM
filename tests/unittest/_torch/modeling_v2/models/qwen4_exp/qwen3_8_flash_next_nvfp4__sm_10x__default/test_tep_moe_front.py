# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tep4 rank's fused-router decode front (`TEP_MOE_FRONT_ENV`), CPU only.

* **Dispatch** (`_tep_front_logits`, `_route_front_run`): a row count the decode front serves
  enters the front with no logits and runs `moe_decode_front` over the layer's staging buffer;
  every other row count keeps `_decode_router` and `moe_route_front`.
* **Build** (`_tep_decode_front_build`): off by default; set away from a tep4 rank with its fast
  forms it warns without a collective; otherwise every rank compiles and checks the route
  front's row counts up to `MAX_ROWS` and serves only those that pass on every rank.

The op itself is covered by `ops/moe/test_modeling_v2_moe_decode_front__decode.py` (GPU), its
numerics in the engine by the `tep4_fp8_kv_opt_in` accuracy case.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


def _st(decode=()):
    st = M._RouteFront(SimpleNamespace(run_moe_prerouted=lambda *a: ("out",)))
    st.tactics = {4: (8, 44), 8: (8, 0), 16: (8, 44)}
    st.flats = {r: ["flat0"] for r in st.tactics}
    st.views = {r: [("e2p", "p2t", "a", "b", "c", "d")] for r in st.tactics}
    st.decode = frozenset(decode)
    st.stages = ["stage0"]
    return st


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows, served", [(4, True), (8, False), (16, True)])
def test_front_logits(monkeypatch, rows, served):
    calls = []
    monkeypatch.setattr(M, "_decode_router", lambda x, w: calls.append("router") or "logits")
    got = M._tep_front_logits(_st(decode=(4, 16)), torch.zeros(rows, 8), SimpleNamespace(router=0))
    assert (got is None) is served and calls == ([] if served else ["router"])


@pytest.mark.cpu_only
@pytest.mark.parametrize("decode", [True, False])
def test_route_front_run_picks_the_launch(monkeypatch, decode):
    calls = []
    monkeypatch.setattr(
        M._decode_front,
        "moe_decode_front",
        lambda *a: calls.append(("decode", a[4], a[5])) or ("logits", "xq", "xsf", "ids", "wts"),
    )
    monkeypatch.setattr(
        M,
        "moe_route_front",
        lambda *a: calls.append(("route", a[4])) or ("xq", "xsf", "ids", "wts"),
    )
    core = SimpleNamespace(
        expert_offset=128, local_experts=128, topk=10, expert_inter=640, hidden=8
    )
    L = SimpleNamespace(
        router=torch.zeros(8, 512),
        moe_gscale=None,
        fc1_w=None,
        fc1_s=None,
        fc2_w=None,
        fc2_s=None,
        out1_scale=None,
        out1_gate_scale=None,
        out2_scale=None,
    )
    out = M._route_front_run(
        core, _st(), torch.zeros(4, 8), L, 0, None if decode else "logits", ("q", torch.zeros(4))
    )
    assert calls == ([("decode", "stage0", 8)] if decode else [("route", 8)])
    assert out[3:] == ("ids", "wts")


def _build_core(monkeypatch, tep_fast=True, others_serve=(4, 8, 16), checks=None):
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    monkeypatch.setattr(M._decode_front, "compile_front", lambda rows, *a: rows != 8)
    monkeypatch.setattr(M._decode_front, "staging_buffer", lambda device: "stage")
    checks = checks or {}
    monkeypatch.setattr(
        M, "_tep_decode_front_check", lambda core, st, L, rows, gen: (checks.get(rows, ""), 3)
    )
    monkeypatch.setattr(torch.cuda, "device", lambda d: __import__("contextlib").nullcontext())

    class _Gen:
        def __init__(self, device):
            pass

        def manual_seed(self, s):
            return self

    monkeypatch.setattr(torch, "Generator", _Gen)
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, device=None, **k: zeros(*a, **k))
    sums = []

    def tp_sum(flags):
        sums.append(1)
        # The three other ranks serve `others_serve`.
        out = flags.clone()
        for r in others_serve:
            out[0, r - 1] += 3.0
        return out

    core = SimpleNamespace(
        tep_fast=tep_fast,
        hidden=32,
        local_experts=128,
        tp_sum=tp_sum,
        mapping=SimpleNamespace(tp_rank=0),
    )
    return core, warned, sums


@pytest.mark.cpu_only
def test_build_off_by_default(monkeypatch):
    monkeypatch.delenv(M.TEP_MOE_FRONT_ENV, raising=False)
    core, warned, sums = _build_core(monkeypatch)
    st = _st()
    M._tep_decode_front_build(core, st, ["L0"], torch.device("cuda"))
    assert st.decode == frozenset() and not sums and not warned


@pytest.mark.cpu_only
def test_build_away_from_tep4_warns_without_a_collective(monkeypatch):
    monkeypatch.setenv(M.TEP_MOE_FRONT_ENV, "1")
    core, warned, sums = _build_core(monkeypatch, tep_fast=False)
    st = _st()
    M._tep_decode_front_build(core, st, ["L0"], torch.device("cuda"))
    assert st.decode == frozenset() and not sums and len(warned) == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "others, checks, expected",
    [
        ((4, 8, 16), {}, {4, 16}),  # 8 does not compile here
        ((4,), {}, {4}),  # another rank lacks 16
        ((4, 16), {16: "tables (...)"}, {4}),  # 16 fails its check here
        ((), {}, set()),  # no rank in agreement: a warning, the stock router
    ],
)
def test_build_serves_the_rows_every_rank_passes(monkeypatch, others, checks, expected):
    monkeypatch.setenv(M.TEP_MOE_FRONT_ENV, "1")
    core, warned, sums = _build_core(monkeypatch, others_serve=others, checks=checks)
    st = _st()
    M._tep_decode_front_build(core, st, ["L0", "L1"], torch.device("cuda"))
    assert set(st.decode) == expected and len(sums) == 1
    assert st.stages == (["stage", "stage"] if expected else ["stage0"])
    assert bool(warned) is (not expected)


@pytest.mark.cpu_only
def test_build_without_a_route_front_still_joins_the_collective(monkeypatch):
    monkeypatch.setenv(M.TEP_MOE_FRONT_ENV, "1")
    core, warned, sums = _build_core(monkeypatch)
    M._tep_decode_front_build(core, None, ["L0"], torch.device("cuda"))
    assert len(sums) == 1 and len(warned) == 1


@pytest.mark.cpu_only
def test_build_checks_every_layer(monkeypatch):
    monkeypatch.setenv(M.TEP_MOE_FRONT_ENV, "1")
    core, warned, sums = _build_core(monkeypatch)
    seen = []
    monkeypatch.setattr(
        M,
        "_tep_decode_front_check",
        lambda core, st, L, rows, gen: seen.append((L, rows)) or ("bad" if L == "L2" else "", 1),
    )
    st = _st()
    M._tep_decode_front_build(core, st, ["L0", "L1", "L2"], torch.device("cuda"))
    # Rows 4 and 16 compile; the check rotates over them and over every layer.
    assert seen == [("L0", 4), ("L1", 16), ("L2", 4)]
    assert set(st.decode) == {16} and len(sums) == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize("where", ["compile", "check"])
def test_build_declines_a_raising_row_count_and_still_joins_the_collective(monkeypatch, where):
    monkeypatch.setenv(M.TEP_MOE_FRONT_ENV, "1")
    core, warned, sums = _build_core(monkeypatch)

    def boom(rows):
        if rows == 16:
            raise RuntimeError("boom")

    if where == "compile":
        monkeypatch.setattr(
            M._decode_front, "compile_front", lambda rows, *a: boom(rows) or rows != 8
        )
    else:
        monkeypatch.setattr(
            M, "_tep_decode_front_check", lambda core, st, L, rows, gen: boom(rows) or ("", 0)
        )
    st = _st()
    M._tep_decode_front_build(core, st, ["L0", "L1"], torch.device("cuda"))
    assert set(st.decode) == {4} and len(sums) == 1
