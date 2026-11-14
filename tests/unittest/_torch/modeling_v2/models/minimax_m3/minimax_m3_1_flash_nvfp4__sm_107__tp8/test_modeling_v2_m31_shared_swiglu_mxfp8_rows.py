# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 TP8 shared SwiGLU MXFP8 rows: the shared expert's SwiGLU writes the down projection's MXFP8 activation
past the MoE engine's
8-row fold up to the 32-row decode step (DECODE_LAYER_WIDE_ROWS); TRTLLM_M31_DISABLE_SHARED_SWIGLU_MXFP8=1 empties the
table."""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        m31_switches,
        modeling,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def test_swq_rows_cover_the_unfolded_shared_down_rows() -> None:
    if not modeling.R1VR8:
        pytest.skip("DECODE_LAYER_WIDE_ROWS is off: the decode step bound is the TP2 rows")
    assert modeling.SWQ_ROWS == {modeling.SWQ_SHARED_DOWN: frozenset(range(9, 33))}


@pytest.mark.parametrize("disable, empty", [("", False), ("0", False), ("1", True)])
def test_swq_kill_switch(monkeypatch: pytest.MonkeyPatch, disable: str, empty: bool) -> None:
    monkeypatch.setenv("TRTLLM_M31_DISABLE_SHARED_SWIGLU_MXFP8", disable)
    importlib.reload(m31_switches)  # the switches are read once, at import
    try:
        m = importlib.reload(modeling)
        assert (m.SWQ_ROWS == {}) is empty
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_SHARED_SWIGLU_MXFP8", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(modeling)
