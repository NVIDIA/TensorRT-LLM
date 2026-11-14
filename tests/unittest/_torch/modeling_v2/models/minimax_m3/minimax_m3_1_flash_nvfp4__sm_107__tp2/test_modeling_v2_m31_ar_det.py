# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AR-DET: the target defaults the AllReduce autotune off for its process (the C++ heuristic) and keeps an explicit
TLLM_DISABLE_ALLREDUCE_AUTOTUNE as given. CPU only."""

import os

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        modeling,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
        m3_report,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

KEY = "TLLM_DISABLE_ALLREDUCE_AUTOTUNE"


def test_autotune_defaults_off(monkeypatch) -> None:
    m3_report.reset_work()
    monkeypatch.setattr(os, "environ", {})
    assert modeling._pin_allreduce_heuristic(0) is True
    assert os.environ[KEY] == "1"
    assert m3_report.work_count("AR_DET") == 1  # the setting counts once per process


def test_explicit_setting_is_kept(monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", {KEY: "0"})
    assert modeling._pin_allreduce_heuristic(0) is False
    assert os.environ[KEY] == "0"
