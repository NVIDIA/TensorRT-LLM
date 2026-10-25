# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP4 MoE engine's 4-token overrides (MOE_M4CFG): the B-image hand-off plus 8 ring stages, measured faster with
bit-identical outputs; other token counts keep the tables; TRTLLM_M31_DISABLE_MOE_M4CFG restores the tables."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4.ops.m3_moe import (
    op as moe_op,
)


def test_m4cfg_scope() -> None:
    assert moe_op.M4_CFG_BY_M == {4: {"b2img": 1, "stages": 8}}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("killed", [False, True])
def test_m4cfg_config_and_kill_switch(monkeypatch: pytest.MonkeyPatch, killed: bool) -> None:
    monkeypatch.setattr(moe_op, "m31_disabled", lambda item_id: killed and item_id == "MOE_M4CFG")
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    state = moe_op.M3MoeState(dev, num_tokens=4, num_ctas=sms - 24, shared=True)
    if killed:
        assert state.config["b2img"] == moe_op.B2IMG_BY_M[3] and "stages" not in state.config
    else:
        assert state.config["b2img"] == 1 and state.config["stages"] == 8
        assert state.mod.B2IMG and state.mod.STAGES == 8
