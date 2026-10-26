# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LC_TP2 (the tp2 baseline): the long-context rows run the Triton sparse decode; the C++
``sparse_decode_lc`` route is off by default, ``TRTLLM_M31_ENABLE_LC_TP2=1`` opts in and
``TRTLLM_M31_DISABLE_LC_TP2=1`` always wins. The switch sets are patched and the kernel module re-imported (read at
import); no GPU needed."""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    m31_switches,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
    triton_sparse_decode,
)

pytestmark = pytest.mark.cpu_only


def test_lc_tp2_is_off_by_default() -> None:
    assert triton_sparse_decode.LC_TP2 is False
    assert triton_sparse_decode.LC_TRITON_PATH == "m3_sparse_decode_lc_tp2_triton"


@pytest.mark.parametrize(
    "enabled, disabled, expected",
    [
        (frozenset(), frozenset(), False),
        (frozenset({"LC_TP2"}), frozenset(), True),
        (frozenset({"LC_TP2"}), frozenset({"LC_TP2"}), False),
        (frozenset(), frozenset({"LC_TP2"}), False),
    ],
    ids=["default", "enabled", "enabled_and_killed", "killed"],
)
def test_lc_tp2_switch(monkeypatch, enabled, disabled, expected) -> None:
    monkeypatch.setattr(m31_switches, "_ENABLED", enabled)
    monkeypatch.setattr(m31_switches, "_DISABLED", disabled)
    try:
        assert importlib.reload(triton_sparse_decode).LC_TP2 is expected
    finally:
        monkeypatch.undo()
        importlib.reload(triton_sparse_decode)
