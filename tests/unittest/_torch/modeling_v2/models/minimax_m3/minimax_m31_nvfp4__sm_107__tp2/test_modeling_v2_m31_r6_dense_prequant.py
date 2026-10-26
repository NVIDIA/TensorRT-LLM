# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R6 on the sm_107 TP2 target: the dense-MLP layer boundary hands the next layer's QKV its MXFP8 activation by
VR: OFF by default (not bitwise on sm_107): TRTLLM_M31_ENABLE_R6=1
turns it on, TRTLLM_M31_DISABLE_R6=1 forces it off. The engagement itself ("[M3 dispatch] r6_dense_prequant taken" /
"off") is checked in-model."""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling,
)


def test_r6_is_off_by_default_on_vr() -> None:
    assert modeling.R6_DENSE_PREQUANT is False


@pytest.mark.parametrize(
    ("enable", "disable", "expected"),
    [("1", "", True), ("1", "1", False), ("0", "", False), ("", "1", False)],
)
def test_r6_switches(
    monkeypatch: pytest.MonkeyPatch, enable: str, disable: str, expected: bool
) -> None:
    monkeypatch.setenv("TRTLLM_M31_ENABLE_R6", enable)
    monkeypatch.setenv("TRTLLM_M31_DISABLE_R6", disable)
    try:
        assert importlib.reload(modeling).R6_DENSE_PREQUANT is expected
    finally:
        monkeypatch.delenv("TRTLLM_M31_ENABLE_R6", raising=False)
        monkeypatch.delenv("TRTLLM_M31_DISABLE_R6", raising=False)
        importlib.reload(modeling)
