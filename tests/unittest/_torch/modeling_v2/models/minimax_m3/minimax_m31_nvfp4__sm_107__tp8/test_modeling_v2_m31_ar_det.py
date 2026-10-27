# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AR-DET: the target defaults the AllReduce autotune off for its process (the C++ heuristic), keeps an explicit
TLLM_DISABLE_ALLREDUCE_AUTOTUNE as given, and leaves the autotune on with TRTLLM_M31_ENABLE_AR_AUTOTUNE=1."""

import os

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling,
)

KEY = "TLLM_DISABLE_ALLREDUCE_AUTOTUNE"


def test_autotune_defaults_off(monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(modeling, "AR_AUTOTUNE_ENABLED", False)
    assert modeling._pin_allreduce_heuristic(0) is True
    assert os.environ[KEY] == "1"


def test_escape_hatch_keeps_autotune_on(monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(modeling, "AR_AUTOTUNE_ENABLED", True)
    assert modeling._pin_allreduce_heuristic(0) is False
    assert KEY not in os.environ


def test_explicit_setting_is_kept(monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", {KEY: "0"})
    monkeypatch.setattr(modeling, "AR_AUTOTUNE_ENABLED", False)
    assert modeling._pin_allreduce_heuristic(0) is False
    assert os.environ[KEY] == "0"


def test_escape_hatch_follows_the_switch_contract() -> None:
    """TRTLLM_M31_ENABLE_AR_AUTOTUNE is parsed by m31_switches (1/true/on/yes set; ""/0/false/off/no unset), not by
    a separate loose read."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
        m31_switches,
    )

    assert modeling.AR_AUTOTUNE_ENABLED == m31_switches.m31_enabled("AR_AUTOTUNE")


def test_engagement_line_only_when_off(monkeypatch) -> None:
    lines = []
    monkeypatch.setattr(modeling, "logger", type("L", (), {"info": staticmethod(lines.append)}))
    for env, enabled, engaged in (({}, False, True), ({}, True, False), ({KEY: "0"}, False, False)):
        lines.clear()
        monkeypatch.setattr(os, "environ", dict(env))
        monkeypatch.setattr(modeling, "AR_AUTOTUNE_ENABLED", enabled)
        modeling._pin_allreduce_heuristic(3)
        assert any("rank 3: AR_DET engaged" in line for line in lines) == engaged, (env, enabled)
