# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AR-DET: the target defaults the AllReduce autotune off for its process (the C++ size / world heuristic) and keeps
an explicit TLLM_DISABLE_ALLREDUCE_AUTOTUNE as given. CPU only."""

import os

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        modeling,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

KEY = "TLLM_DISABLE_ALLREDUCE_AUTOTUNE"


def test_autotune_defaults_off(monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", {})
    assert modeling._pin_allreduce_heuristic(0) is True
    assert os.environ[KEY] == "1"


def test_explicit_setting_is_kept(monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", {KEY: "0"})
    assert modeling._pin_allreduce_heuristic(0) is False
    assert os.environ[KEY] == "0"


def test_engagement_line_only_when_off(monkeypatch) -> None:
    lines = []
    monkeypatch.setattr(modeling, "logger", type("L", (), {"info": staticmethod(lines.append)}))
    for env, engaged in (({}, True), ({KEY: "0"}, False)):
        lines.clear()
        monkeypatch.setattr(os, "environ", dict(env))
        modeling._pin_allreduce_heuristic(3)
        assert any("rank 3: AR_DET engaged" in line for line in lines) == engaged, env
