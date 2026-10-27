# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ONE_DIV row cap the TP2 target carries from the TP4 base: the LC merge divides once up to 16 rows;
TRTLLM_M31_DISABLE_ONEDIV_CAP=1 keeps it on at every row count, TRTLLM_M31_DISABLE_LC_ONEDIV=1 turns it off. Inert at
TP2 in the baseline (LC_TP2 off: the long-context rows run the Triton sparse decode); the switch contract is kept."""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    m31_switches,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
    m3_attn_lc,
)


def test_onediv_cap_default() -> None:
    assert m3_attn_lc.ONE_DIV_MAX_ROWS == 16
    assert m3_attn_lc.one_div_applies(4) and m3_attn_lc.one_div_applies(16)
    assert not m3_attn_lc.one_div_applies(17) and not m3_attn_lc.one_div_applies(48)


@pytest.mark.parametrize("disable, applies_at_32", [("", False), ("0", False), ("1", True)])
def test_onediv_cap_switch(
    monkeypatch: pytest.MonkeyPatch, disable: str, applies_at_32: bool
) -> None:
    monkeypatch.setenv("TRTLLM_M31_DISABLE_ONEDIV_CAP", disable)
    try:
        importlib.reload(m31_switches)
        lc = importlib.reload(m3_attn_lc)
        assert lc.one_div_applies(1) and lc.one_div_applies(16)
        assert lc.one_div_applies(32) is applies_at_32
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_ONEDIV_CAP", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(m3_attn_lc)


@pytest.mark.parametrize("cap", ["", "1"])
def test_onediv_kill_switch(monkeypatch: pytest.MonkeyPatch, cap: str) -> None:
    """TRTLLM_M31_DISABLE_LC_ONEDIV=1 turns the one-division merge off at every row count, whatever the cap switch."""
    monkeypatch.setenv("TRTLLM_M31_DISABLE_LC_ONEDIV", "1")
    monkeypatch.setenv("TRTLLM_M31_DISABLE_ONEDIV_CAP", cap)
    try:
        importlib.reload(m31_switches)
        lc = importlib.reload(m3_attn_lc)
        assert lc.ONE_DIV_MAX_ROWS == 0
        assert not any(lc.one_div_applies(rows) for rows in (1, 8, 16, 32))
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_LC_ONEDIV", raising=False)
        monkeypatch.delenv("TRTLLM_M31_DISABLE_ONEDIV_CAP", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(m3_attn_lc)
