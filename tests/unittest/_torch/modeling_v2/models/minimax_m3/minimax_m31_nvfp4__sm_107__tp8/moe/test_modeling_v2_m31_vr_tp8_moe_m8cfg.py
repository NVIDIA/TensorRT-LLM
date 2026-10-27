# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 TP8 MoE engine's 8-token override (MOE_M8CFG): the tagged-record FC1 -> FC2 hand-off instead of the B
image, ported from the GB300 target (measured faster there with bit-identical outputs); other token counts keep the
tables; TRTLLM_M31_DISABLE_MOE_M8CFG restores the tables."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)


def test_m8cfg_scope() -> None:
    assert moe_op.M8_CFG_BY_M == {8: {"b2img": 0}}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("killed", [False, True])
def test_m8cfg_config_and_kill_switch(monkeypatch: pytest.MonkeyPatch, killed: bool) -> None:
    monkeypatch.setattr(moe_op, "m31_disabled", lambda item_id: killed and item_id == "MOE_M8CFG")
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    state = moe_op.M3MoeState(dev, num_tokens=8, num_ctas=sms - 24, shared=True)
    assert state.config["b2img"] == (moe_op.B2IMG_BY_M[7] if killed else 0)
    assert bool(state.mod.B2IMG) == bool(killed and moe_op.B2IMG_BY_M[7])
