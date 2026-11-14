# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 9..13-row MoE engine is off at tp2 by default.

The engine's kernel (kernels/m3_moe/m3_moe16_kernel.py) is the TP4 build, its routed and shared expert widths fixed at
the TP4 768 per rank; the tp2 target's experts are 1536 wide. So by default MOE16_ENGINE covers no rows, nothing builds
it and moe_path sends 9..13-row steps to the MC finalize or the generic MoE; the opt-in
(TRTLLM_M31_ENABLE_MOE_ENGINE_9_TO_13_ROWS) gives the engine its 9..13 rows. CPU only.
"""

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

ENV = "TRTLLM_M31_ENABLE_MOE_ENGINE_9_TO_13_ROWS"


def test_moe16_is_off() -> None:
    assert modeling.MOE16_TP2 is m31_switches.m31_enabled("MOE_ENGINE_9_TO_13_ROWS")
    if modeling.MOE16_TP2:
        pytest.skip(f"{ENV} is set in this environment")
    assert modeling.MOE16_MAX_TOKENS == 0
    assert modeling.MOE16_ENGINE.keys == frozenset()
    for rows in range(9, 14):
        assert not modeling.MOE16_ENGINE.covers(rows)
        assert modeling.moe_path(rows) != modeling.MOE16_ENGINE.path
    # The MC finalize takes them (from 9 rows); with its kill switches the generic MoE runs.
    want = (
        modeling.MC_FINALIZE.path
        if modeling.TP2W_MC_FINALIZE and modeling.MC_FINALIZE_MIN_TOKENS <= 9
        else None
    )
    assert all(modeling.moe_path(rows) == want for rows in range(9, 14))


def test_the_moe16_kernel_is_the_tp4_width_build() -> None:
    """Why it is off: the kernel's expert widths are TP4's, not the target's INTER."""
    kernel = pytest.importorskip(
        "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels"
        ".m3_moe.m3_moe16_kernel"
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels.m3_moe import (  # noqa: E501
        op as moe_op,
    )

    assert kernel.I_TP == 768 and kernel.SH_I == 768
    assert modeling.MOE16_KERNEL_WIDTH == kernel.I_TP
    assert moe_op.INTER == 1536


@pytest.mark.parametrize("value, rows", [("", ()), ("0", ()), ("1", tuple(range(9, 14)))])
def test_the_opt_in(monkeypatch: pytest.MonkeyPatch, value: str, rows: tuple) -> None:
    original = os.environ.get(ENV)
    monkeypatch.setenv(ENV, value)
    try:
        importlib.reload(m31_switches)
        m = importlib.reload(modeling)
        assert m.MOE16_TP2 is bool(rows)
        assert m.MOE16_ENGINE.keys == frozenset(rows)
        # The MC finalize starts past the engine's rows.
        if m.TP2W_MC_FINALIZE and m.TP2_MC_FINALIZE_SMALL:
            assert m.MC_FINALIZE_MIN_TOKENS == (14 if rows else 9)
    finally:
        if original is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, original)
        importlib.reload(m31_switches)
        importlib.reload(modeling)
