# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 TP8 MoE engine's VRPOLL override: 16 router logit replicas at 7 and 8 tokens instead of the kernel's 32;
up to 4 tokens keep LG_REPL; TRTLLM_M31_DISABLE_MOE_VRPOLL restores the kernel default."""

import importlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    m31_switches,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)


def test_vrpoll_scope() -> None:
    assert (
        moe_op.VRPOLL and moe_op.VRPOLL_LG_REPL == 16 and moe_op.VRPOLL_TOKENS == frozenset({7, 8})
    )
    assert max(moe_op.VRPOLL_TOKENS) > moe_op.LG_REPL_MAX_TOKENS >= 4


def test_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTLLM_M31_DISABLE_MOE_VRPOLL", "1")
    importlib.reload(m31_switches)
    try:
        assert not importlib.reload(moe_op).VRPOLL
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_MOE_VRPOLL", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(moe_op)
    assert moe_op.VRPOLL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("num_tokens", [4, 7, 8])
@pytest.mark.parametrize("killed", [False, True])
def test_vrpoll_config(monkeypatch: pytest.MonkeyPatch, num_tokens: int, killed: bool) -> None:
    monkeypatch.setattr(moe_op, "VRPOLL", not killed)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    state = moe_op.M3MoeState(
        dev,
        num_tokens=num_tokens,
        num_ctas=sms - moe_op.shared_ctas_for(num_tokens, True),
        shared=True,
    )
    if num_tokens <= moe_op.LG_REPL_MAX_TOKENS:
        want = moe_op.LG_REPL
    else:
        want = 32 if killed else 16
    assert state.mod.LG_REPL == want
    assert state.lgbuf.numel() == want * state.mod.LG_STRIDE
