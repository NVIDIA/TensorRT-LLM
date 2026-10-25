# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP4 MoE engine's early group-0 issue depth (MOE_EARLY0N): 6 stages at 4 tokens and 3 at 8, measured faster with
bit-identical outputs; every other token count keeps the kernel default; TRTLLM_M31_DISABLE_MOE_EARLY0N restores the
default everywhere."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4.ops.m3_moe import (
    op as moe_op,
)


def test_early0n_table_scope() -> None:
    assert moe_op.EARLY0_N_BY_M == (0, 0, 0, 6, 0, 0, 0, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize(("num_tokens", "expected"), [(1, None), (4, 6), (8, 3)])
@pytest.mark.parametrize("killed", [False, True])
def test_early0n_config_and_kill_switch(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, expected, killed: bool
) -> None:
    monkeypatch.setattr(moe_op, "m31_disabled", lambda item_id: killed and item_id == "MOE_EARLY0N")
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    state = moe_op.M3MoeState(dev, num_tokens=num_tokens, num_ctas=sms - 24, shared=True)
    assert state.config.get("early0_n") == (None if killed else expected)
