# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R6 on the sm_107 TP4 target: the dense-MLP layer boundary hands the next layer's QKV its MXFP8 activation.
OFF by default (not bitwise on sm_107): TRTLLM_M31_ENABLE_R6=1 turns it on, TRTLLM_M31_DISABLE_R6=1 forces it off.
The engagement itself ("[M3 dispatch] r6_dense_prequant taken" /
"off") is checked in-model."""

import importlib
import os

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
    m31_switches,
    modeling,
)

pytestmark = pytest.mark.cpu_only


def test_r6_is_off_by_default_on_vr() -> None:
    assert modeling.R6_DENSE_PREQUANT is False


@pytest.mark.parametrize(
    ("enable", "disable", "expected"),
    [("1", "", True), ("1", "1", False), ("0", "", False), ("", "1", False)],
)
def test_r6_switches(
    monkeypatch: pytest.MonkeyPatch, enable: str, disable: str, expected: bool
) -> None:
    names = ("TRTLLM_M31_ENABLE_R6", "TRTLLM_M31_DISABLE_R6")
    original = {name: os.environ.get(name) for name in names}
    monkeypatch.setenv("TRTLLM_M31_ENABLE_R6", enable)
    monkeypatch.setenv("TRTLLM_M31_DISABLE_R6", disable)
    try:
        importlib.reload(m31_switches)  # the switches are read once, at its import
        assert importlib.reload(modeling).R6_DENSE_PREQUANT is expected
    finally:
        for name, value in original.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        importlib.reload(m31_switches)
        importlib.reload(modeling)
