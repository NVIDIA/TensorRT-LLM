# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MOE16_TP2: the 9..13-row MoE engine is off in the tp2 baseline.

The engine's kernel (kernels/m3_moe/m3_moe16_kernel.py) is the TP4 build, its routed and shared expert widths fixed at
the TP4 768 per rank; the tp2 target's experts are 1536 wide. So by default MOE16_ENGINE covers no rows, nothing builds
it and moe_path sends 9..13-row steps to the generic MoE; the opt-in (TRTLLM_M31_ENABLE_MOE16_TP2) restores the 9..13
rows, and TRTLLM_M31_DISABLE_MOE16_TP2 wins over it. CPU only.
"""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    m31_switches,
    modeling,
)

_ENV = ("TRTLLM_M31_ENABLE_MOE16_TP2", "TRTLLM_M31_DISABLE_MOE16_TP2")


def test_moe16_is_off_by_default() -> None:
    assert modeling.MOE16_TP2 is False
    assert modeling.MOE16_MAX_TOKENS == 0
    assert modeling.MOE16_ENGINE.keys == frozenset()
    for rows in range(9, 14):
        assert not modeling.MOE16_ENGINE.covers(rows)
        assert modeling.moe_path(rows) != modeling.MOE16_ENGINE.path
    # The MC finalize takes them (TP2_MC_FINALIZE_SMALL: from 9 rows with MoE16 off); with its kill (33) or
    # TP2W_MC_FINALIZE's (no rows) the generic MoE runs.
    want = (
        modeling.MC_FINALIZE.path
        if modeling.TP2W_MC_FINALIZE and modeling.MC_FINALIZE_MIN_TOKENS <= 9
        else None
    )
    assert all(modeling.moe_path(rows) == want for rows in range(9, 14))


def test_the_moe16_kernel_is_the_tp4_width_build() -> None:
    """Why it is off: the kernel's expert widths are TP4's, not the target's INTER."""
    kernel = pytest.importorskip(
        "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels"
        ".m3_moe.m3_moe16_kernel"
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.m3_moe import (  # noqa: E501
        op as moe_op,
    )

    assert kernel.I_TP == 768 and kernel.SH_I == 768
    assert moe_op.INTER == 1536


def _reload(monkeypatch, enable: bool, disable: bool):
    for name, on in zip(_ENV, (enable, disable)):
        if on:
            monkeypatch.setenv(name, "1")
        else:
            monkeypatch.delenv(name, raising=False)
    importlib.reload(m31_switches)
    return importlib.reload(modeling)


@pytest.mark.parametrize(
    "enable, disable, rows",
    [(False, False, ()), (True, False, tuple(range(9, 14))), (True, True, ())],
    ids=["default", "opt_in", "kill_wins"],
)
def test_switches(monkeypatch, enable, disable, rows) -> None:
    try:
        m = _reload(monkeypatch, enable, disable)
        assert m.MOE16_TP2 is bool(rows)
        assert m.MOE16_ENGINE.keys == frozenset(rows)
    finally:
        for name in _ENV:
            monkeypatch.delenv(name, raising=False)
        importlib.reload(m31_switches)
        importlib.reload(modeling)
