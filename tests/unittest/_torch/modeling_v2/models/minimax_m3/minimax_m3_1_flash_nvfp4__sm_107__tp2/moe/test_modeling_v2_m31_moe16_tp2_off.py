# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 9..13-row MoE engine is off at tp2.

The engine's kernel (kernels/m3_moe/m3_moe16_kernel.py) is the TP4 build, its routed and shared expert widths fixed at
the TP4 768 per rank; the tp2 target's experts are 1536 wide. So MOE16_ENGINE covers no rows, nothing builds it and
moe_path sends 9..13-row steps to the MC finalize or the generic MoE. CPU only.
"""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        modeling,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def test_moe16_is_off() -> None:
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
