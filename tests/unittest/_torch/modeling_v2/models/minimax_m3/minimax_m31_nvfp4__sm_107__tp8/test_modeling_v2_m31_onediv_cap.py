# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VR TP8 ONE_DIV merge: default off (TRTLLM_M31_ENABLE_LC8_ONEDIV=1 opts in, TRTLLM_M31_DISABLE_LC8_ONEDIV=1 still
wins); when on, the LC8 merge divides once up to 8 rows and TRTLLM_M31_DISABLE_ONEDIV_CAP=1 keeps it on at every row."""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    m31_switches,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
    m3_attn_lc8,
)


def test_onediv_default_off() -> None:
    assert m3_attn_lc8.ONE_DIV_MERGE is False and m3_attn_lc8.ONE_DIV_MAX_ROWS == 8
    assert not any(m3_attn_lc8.one_div_applies(rows) for rows in (1, 4, 8, 9, 32))


@pytest.mark.parametrize(
    "enable, disable, on", [("1", "", True), ("1", "1", False), ("", "", False)]
)
def test_onediv_switches(
    monkeypatch: pytest.MonkeyPatch, enable: str, disable: str, on: bool
) -> None:
    monkeypatch.setenv("TRTLLM_M31_ENABLE_LC8_ONEDIV", enable)
    monkeypatch.setenv("TRTLLM_M31_DISABLE_LC8_ONEDIV", disable)
    try:
        importlib.reload(m31_switches)
        lc8 = importlib.reload(m3_attn_lc8)
        assert lc8.ONE_DIV_MERGE is on
        assert lc8.one_div_applies(8) is on and not lc8.one_div_applies(9)
    finally:
        monkeypatch.delenv("TRTLLM_M31_ENABLE_LC8_ONEDIV", raising=False)
        monkeypatch.delenv("TRTLLM_M31_DISABLE_LC8_ONEDIV", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(m3_attn_lc8)


@pytest.mark.parametrize("disable, applies_at_32", [("", False), ("0", False), ("1", True)])
def test_onediv_cap_switch(
    monkeypatch: pytest.MonkeyPatch, disable: str, applies_at_32: bool
) -> None:
    monkeypatch.setenv("TRTLLM_M31_ENABLE_LC8_ONEDIV", "1")
    monkeypatch.setenv("TRTLLM_M31_DISABLE_ONEDIV_CAP", disable)
    try:
        importlib.reload(m31_switches)
        lc8 = importlib.reload(m3_attn_lc8)
        assert lc8.one_div_applies(1) and lc8.one_div_applies(8)
        assert lc8.one_div_applies(32) is applies_at_32
    finally:
        monkeypatch.delenv("TRTLLM_M31_ENABLE_LC8_ONEDIV", raising=False)
        monkeypatch.delenv("TRTLLM_M31_DISABLE_ONEDIV_CAP", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(m3_attn_lc8)
