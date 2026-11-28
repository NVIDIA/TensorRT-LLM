# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP MoE launch order (`DP_LAUNCH_ORDER_ENV`, default on at a dep4 / dep8 rank on
SM103; CPU recorders): the trunk block's and the draft layer's shared expert are launched on the
side stream from inside the round trip, right after its gather, and joined before the result is
read; elsewhere (and switched off) the order before."""

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
    return main, side, current


def _core(log, side, dp_size=8, order=True):
    def expert_sum(run, x, pad_rows):
        log.append("gather")
        out = run(x)
        log.append("reduce-scatter")
        return out

    return SimpleNamespace(
        _side_stream=side, dp_size=dp_size, dp_launch_order=order, expert_sum=expert_sum
    )


@pytest.mark.parametrize("order, dp_size", [(True, 8), (False, 8), (True, 1)])
def test_trunk_block(monkeypatch, order, dp_size):
    log = []
    main, side, current = _streams(monkeypatch, log)
    monkeypatch.setattr(
        M,
        "_shared_expert",
        lambda core, x, L, g: log.append(f"shared on {current[-1].name}") or ("sh", "gl"),
    )
    monkeypatch.setattr(M, "_routed_experts", lambda *a: log.append("routed") or "routed")
    core = _core(log, side, dp_size, order)
    if dp_size == 1:
        core._tep_finar = None
    routed, gate, shared = M._moe_block(core, torch.zeros(4, 8), None, 0)
    assert (routed, gate, shared) == ("routed", "gl", "sh")
    if order and dp_size > 1:
        assert log == [
            "side waits main",
            "gather",
            "shared on side",
            "routed",
            "reduce-scatter",
            "main waits side",
        ]
    else:
        assert log[:3] == ["side waits main", "shared on side", "gather"]
        assert log[-1] == "main waits side"


@pytest.mark.parametrize("order", [True, False])
def test_draft_tail(monkeypatch, order):
    log = []
    main, side, current = _streams(monkeypatch, log)
    core = _core(log, side, 8, order)
    core.eps, core.hc_count = 1e-6, 4
    monkeypatch.setattr(M, "qwen4_exp_hc_inject_norm", lambda *a: ("x", "normed"))
    monkeypatch.setattr(M, "_hc_mixer", lambda *a: ("mixed", "inject"))
    monkeypatch.setattr(M, "_draft_o", lambda core, attn, w: "o")

    def moe_block(core, W, x, pad_rows, routed_only=False, before_window=None):
        log.append("gather")
        if before_window is not None:
            before_window()
        log.append("routed")
        return "routed"

    monkeypatch.setattr(M, "_mtp_moe_block", moe_block)
    monkeypatch.setattr(
        M,
        "_mtp_shared_expert",
        lambda core, W, x: log.append(f"shared on {current[-1].name}") or ("gl", "sh"),
    )
    monkeypatch.setattr(M, "_combine_inject_fits", lambda *a: True)
    combined = []
    monkeypatch.setattr(M, "mtp_moe_combine_inject", lambda *a: combined.append(a) or "out")
    core.tp_sum = lambda t: t
    W = SimpleNamespace(o=None, mhc_norm=None, mhc_first=None, mhc_up=None)
    assert M._mtp_tail(core, W, "x0", "attn", "inj0", 0) == "out"
    assert combined == [("x", "routed", "gl", "sh", "inject")]
    if order:
        assert log == [
            "side waits main",
            "gather",
            "shared on side",
            "routed",
            "main waits side",
        ]
    else:
        assert log == ["gather", "routed", "shared on main"]
