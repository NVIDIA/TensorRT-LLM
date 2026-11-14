# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R6_DENSE_PREQUANT on the sm_107 TP2 target: the dense-MLP layer boundary hands the next layer's QKV its MXFP8
activation. Off by default (not bitwise on sm_107); ``TRTLLM_M31_ENABLE_R6_DENSE_PREQUANT=1`` turns it on. The
engagement itself ("[M3 dispatch] r6_dense_prequant taken" / "off") is logged at the first dense boundary."""

import importlib
import os

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        m31_switches,
        modeling,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

ENV = "TRTLLM_M31_ENABLE_R6_DENSE_PREQUANT"


def test_off_by_default() -> None:
    assert modeling.R6_DENSE_PREQUANT is m31_switches.m31_enabled("R6_DENSE_PREQUANT")
    if os.environ.get(ENV) is None:
        assert modeling.R6_DENSE_PREQUANT is False


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False)])
def test_the_opt_in(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    original = os.environ.get(ENV)
    monkeypatch.setenv(ENV, value)
    try:
        importlib.reload(m31_switches)
        assert importlib.reload(modeling).R6_DENSE_PREQUANT is expected
    finally:
        if original is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, original)
        importlib.reload(m31_switches)
        importlib.reload(modeling)
