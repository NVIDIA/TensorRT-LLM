# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AR-DET: the target defaults the AllReduce autotune off for its process (the C++ heuristic), keeps an explicit
TLLM_DISABLE_ALLREDUCE_AUTOTUNE as given, and leaves the autotune on with TRTLLM_M31_ENABLE_AR_AUTOTUNE=1."""

import os

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling,
)

KEY = "TLLM_DISABLE_ALLREDUCE_AUTOTUNE"


def test_autotune_defaults_off(monkeypatch) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
        m3_report,
    )

    m3_report.reset_work()
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(modeling, "AR_AUTOTUNE_ENABLED", False)
    assert modeling._pin_allreduce_heuristic(0) is True
    assert os.environ[KEY] == "1"
    assert m3_report.work_count("AR_DET") == 1  # gate e: the setting counts once per process


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
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
        m31_switches,
    )

    assert modeling.AR_AUTOTUNE_ENABLED == m31_switches.m31_enabled("AR_AUTOTUNE")
