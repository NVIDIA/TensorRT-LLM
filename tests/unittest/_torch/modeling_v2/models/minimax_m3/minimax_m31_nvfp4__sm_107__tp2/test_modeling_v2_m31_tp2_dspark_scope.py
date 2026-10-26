# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the tp2 target's DSpark worker sets up at prebuild (its defaults), on a stub worker whose set-up calls
are recorded: S9b ON with an installed fused AR + norm op (S9B-MC's multicast form inside it, ON), S5 and S2 OFF, the
S10c hooks installed wrapped with their work counters, EX11 installed, MXDRAFT asked for the decode rows; each kill
switch turns its own item off. CPU only.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    dspark,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
    m3_report,
)

pytestmark = pytest.mark.cpu_only


def _prebuild(monkeypatch, disabled=(), lm_head=None, a1=None, s4=None, s4_logits=False):
    """Run the real M3DSparkWorker.prebuild on a stub; returns what it asked for. With ``lm_head`` the N3 A1 / MCAG
    tail runs too (A1's set-up stubbed to ``a1``; the S4 set-up stubbed to leave ``s4`` and ``s4_logits``)."""
    asked = {"lines": []}
    monkeypatch.setattr(
        dspark,
        "logger",
        SimpleNamespace(info=asked["lines"].append, warning=asked["lines"].append),
    )
    monkeypatch.setattr(dspark, "N3A_DISABLED", True)
    monkeypatch.setattr(dspark, "N3A_KV_DISABLED", True)
    monkeypatch.setattr(dspark, "MCAG", "MCAG" not in disabled)
    monkeypatch.setattr(m3_report, "_rank", lambda: 0)
    monkeypatch.setattr(m3_report, "_info", lambda message: None)
    monkeypatch.setattr(dspark, "m31_disabled", lambda i: i in disabled)
    monkeypatch.setattr(dspark, "m31_enabled", lambda i: False)
    monkeypatch.setattr(dspark, "MXDRAFT", "MXDRAFT" not in disabled)
    monkeypatch.setattr(dspark, "S9B_MC", "S9B_MC" not in disabled)
    monkeypatch.setattr(
        dspark,
        "route_mx_linears",
        lambda model, rows: asked.setdefault("MXDRAFT", list(rows)) and 0,
    )
    worker = object.__new__(dspark.M3DSparkWorker)
    worker.mapping = SimpleNamespace(tp_rank=0, tp_size=2, enable_attention_dp=False)
    worker.spec_config = SimpleNamespace(max_draft_len=7)

    def prebuild_s4(*a, **k):
        worker._s4 = s4
        worker._s4_logits_on = s4_logits

    worker.prebuild_s4 = prebuild_s4
    worker.enable_lean_glue = lambda model, on: asked.__setitem__("S10", on)
    worker.enable_small_m_linears = lambda *a, **k: False
    worker.enable_window_capped_decode = lambda model, max_batch: asked.__setitem__("S5", max_batch)
    worker.enable_fc_tp_split = lambda model, on=True: asked.__setitem__("S2", on) or on
    worker.enable_fused_ar_norm = lambda model, on=True: asked.__setitem__("S9B", on) or on
    worker.enable_fused_ar_norm_op = lambda model, op: (
        asked.__setitem__("S9B_OP", op) or (op is not None)
    )
    for name, key in (
        ("enable_fused_ctx_store", "S10C_A"),
        ("enable_fused_drafter_inputs", "S10C_C"),
        ("enable_fused_ctx_advance", "S10C_D"),
        ("enable_fused_block_refresh", "S10C_B"),
    ):
        setattr(worker, name, lambda fn, _k=key: asked.__setitem__(_k, fn) or fn is not None)
    worker.enable_fused_forward_indices = lambda model, fn: (
        asked.__setitem__("S10C_E", fn) or (fn is not None)
    )
    worker.enable_fused_qk_norm_rope = lambda model, fn: (
        asked.__setitem__("EX11", fn) or (fn is not None)
    )
    worker._prebuild_a1 = lambda lm_head: a1
    worker._prebuild_draft_a1 = lambda lm_head, draft_model: asked.__setitem__("DRAFT_A1", True)
    worker._prebuild_mcag = lambda lm_head: asked.__setitem__("MCAG_BUILD", True) or 100032
    worker.enable_draft_logits_gather_op = lambda op: asked.__setitem__("MCAG_OP", op) or (
        op is not None
    )
    dspark.M3DSparkWorker.prebuild(worker, lm_head, SimpleNamespace(_dflash_fc_split=None))
    return asked


def test_the_tp2_defaults(monkeypatch):
    asked = _prebuild(monkeypatch)
    assert asked["S9B"] is True and asked["S5"] == 0 and asked["S2"] is False
    assert asked["S10"] is True
    assert callable(asked["S9B_OP"])
    for key in ("S10C_A", "S10C_B", "S10C_C", "S10C_D", "S10C_E", "EX11"):
        assert callable(asked[key]), key
        assert getattr(asked[key], "__wrapped__", None) is not None, key  # the work-counted wrapper
    assert asked["MXDRAFT"] == [7 * b for b in range(1, dspark.MAX_FAST_BATCH + 1)]


def test_s9b_kill_installs_no_op(monkeypatch):
    asked = _prebuild(monkeypatch, disabled={"S9B"})
    assert asked["S9B"] is False and asked["S9B_OP"] is None


def test_s9b_mc_kill_keeps_the_counting_op_without_the_multicast_form(monkeypatch):
    """With S9B-MC killed the installed op never takes the operands (the stock fused all-reduce runs)."""
    asked = _prebuild(monkeypatch, disabled={"S9B_MC"})
    op = asked["S9B_OP"]
    assert op(torch.zeros(4, 8), torch.zeros(4, 8), torch.ones(8), 1e-6) is None


@pytest.mark.parametrize("item", ["S10C_A", "S10C_B", "S10C_C", "S10C_D", "S10C_E", "EX11"])
def test_each_hook_kill_switch(monkeypatch, item):
    asked = _prebuild(monkeypatch, disabled={item})
    assert asked[item] is None


def test_s10c_kills_every_s10c_hook(monkeypatch):
    asked = _prebuild(monkeypatch, disabled={"S10C"})
    assert all(asked[k] is None for k in ("S10C_A", "S10C_B", "S10C_C", "S10C_D", "S10C_E"))


def test_mxdraft_kill(monkeypatch):
    asked = _prebuild(monkeypatch, disabled={"MXDRAFT"})
    assert "MXDRAFT" not in asked


def _mcag_line(asked):
    return [s for s in asked["lines"] if s.startswith("M3.1 MCAG:")]


def test_c6_mcag_declines_with_its_reason_when_s4_is_not_set_up(monkeypatch):
    """C6 (TP2: the vocab shard is past the S4 chain's capacity, so S4 is never set up): no vocab-sharded draft logits
    reach the gather, MCAG is not compiled, no op is installed, and the line says why."""
    asked = _prebuild(monkeypatch, lm_head=SimpleNamespace(), a1={"world": 2})
    assert "MCAG_BUILD" not in asked and asked["MCAG_OP"] is None
    (line,) = _mcag_line(asked)
    assert line.startswith(
        "M3.1 MCAG: draft-logits all-gather on the multicast kernel off on rank 0 ("
    )
    assert "S4 is not set up" in line


def test_c6_mcag_declines_when_s4_logits_is_off(monkeypatch):
    asked = _prebuild(monkeypatch, lm_head=SimpleNamespace(), a1={"world": 2}, s4=object())
    assert "MCAG_BUILD" not in asked and asked["MCAG_OP"] is None
    assert "S4-logits is off" in _mcag_line(asked)[0]


def test_c6_mcag_builds_when_sharded_draft_logits_reach_it(monkeypatch):
    asked = _prebuild(
        monkeypatch, lm_head=SimpleNamespace(), a1={"world": 2}, s4=object(), s4_logits=True
    )
    assert asked["MCAG_BUILD"] is True and callable(asked["MCAG_OP"])
    assert _mcag_line(asked) == [
        "M3.1 MCAG: draft-logits all-gather on the multicast kernel on on rank 0"
    ]


def test_c6_mcag_kill_switch_still_wins(monkeypatch):
    asked = _prebuild(
        monkeypatch,
        disabled={"MCAG"},
        lm_head=SimpleNamespace(),
        a1={"world": 2},
        s4=object(),
        s4_logits=True,
    )
    assert "MCAG_BUILD" not in asked and asked["MCAG_OP"] is None
    assert _mcag_line(asked) == [
        "M3.1 MCAG: draft-logits all-gather on the multicast kernel off on rank 0"
    ]
