# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The process-wide features a model opts into through RuntimeHooks and the executor arms
(``_arm_runtime_opt_ins``): the SM107 MXFP8 GEMM tactic table and the late-compile detector. A model
that declares neither leaves both off, even at a table shape on SM107; the MiniMax-M3 sm_107 targets
declare both."""

import importlib

import pytest
import torch

from tensorrt_llm._torch import late_compile
from tensorrt_llm._torch.custom_ops import mxfp8_cute_pinned_tactics as pin
from tensorrt_llm._torch.pyexecutor.py_executor import _arm_runtime_opt_ins
from tensorrt_llm._torch.pyexecutor.runtime_hooks import runtime_hooks_of

pytestmark = pytest.mark.cpu_only

M3_SHAPE = (64, 6144, 6144)  # (m, n, k): a table entry
M3_TARGETS = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__{}.modeling"


class _PlainModel(torch.nn.Module):
    """A model that declares no hooks."""


@pytest.fixture(autouse=True)
def _sm107_unarmed(monkeypatch):
    monkeypatch.setattr(pin, "_ARMED", [False])
    monkeypatch.setattr(pin, "ENABLED", True)
    monkeypatch.setattr(pin, "get_sm_version", lambda: 107)
    monkeypatch.setattr(late_compile, "_ready", [False])


def test_a_model_without_the_opt_in_arms_nothing_at_an_m3_shape():
    hooks = runtime_hooks_of(_PlainModel())

    assert _arm_runtime_opt_ins(hooks) is False
    assert pin.tactic_for(*M3_SHAPE) is None
    assert not late_compile._ready[0]


@pytest.mark.parametrize("tp", ["tp2", "tp4", "tp8"])
def test_the_m3_sm107_targets_arm_both(tp):
    hooks = importlib.import_module(M3_TARGETS.format(tp))._runtime_hooks(None)

    assert hooks.mxfp8_pinned_tactics and hooks.late_compile_detect
    assert _arm_runtime_opt_ins(hooks) is True
    m, n, k = M3_SHAPE
    assert pin.tactic_for(m, n, k) == pin.TACTICS[(n, k, pin._bucket(m))]


def test_the_kill_switch_still_wins_once_armed(monkeypatch):
    pin.arm()
    monkeypatch.setattr(pin, "ENABLED", False)

    assert pin.tactic_for(*M3_SHAPE) is None
