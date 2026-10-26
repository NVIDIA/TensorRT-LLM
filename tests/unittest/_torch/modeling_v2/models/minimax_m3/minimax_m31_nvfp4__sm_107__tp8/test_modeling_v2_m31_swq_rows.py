# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VR TP8 SWQ rows: the shared expert's SwiGLU writes the down projection's MXFP8 activation past the MoE engine's
8-row fold up to R1VR8's 32-row step; TRTLLM_M31_DISABLE_SWQ=1 empties the table."""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling,
)


def test_swq_rows_cover_the_unfolded_shared_down_rows() -> None:
    if not modeling.R1VR8:
        pytest.skip("R1-VR-TP8 is off: the decode step bound is the TP2 rows")
    assert modeling.SWQ_ROWS == {modeling.SWQ_SHARED_DOWN: frozenset(range(9, 33))}


@pytest.mark.parametrize("disable, empty", [("", False), ("0", False), ("1", True)])
def test_swq_kill_switch(monkeypatch: pytest.MonkeyPatch, disable: str, empty: bool) -> None:
    monkeypatch.setenv("TRTLLM_M31_DISABLE_SWQ", disable)
    try:
        m = importlib.reload(modeling)
        assert (m.SWQ_ROWS == {}) is empty
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_SWQ", raising=False)
        importlib.reload(modeling)
