# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 TP8 MoE engine's SH20 layout: at 8 tokens the shared expert's 24 work units run on 20 shared CTAs, so the
routed grid has 192 CTAs (one FC2 32-row block each); other token counts keep 24 shared CTAs, and an engine without the
shared expert has none. TRTLLM_M31_DISABLE_MOE_SH20 restores 24 everywhere."""

import importlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    m31_switches,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)


def test_sh20_scope() -> None:
    assert moe_op.SH20 and moe_op.SH20_CTAS == 20 and moe_op.SH20_TOKENS == frozenset({8})
    assert moe_op.SHARED_CTAS == 24
    # 6144 / 32 FC2 blocks: every routed CTA owns exactly one at 212 - 20 routed CTAs
    assert moe_op.HIDDEN // 32 == 212 - moe_op.SH20_CTAS


@pytest.mark.parametrize("num_tokens", range(1, 9))
def test_shared_ctas_for(num_tokens: int) -> None:
    want = 20 if num_tokens == 8 else 24
    assert moe_op.shared_ctas_for(num_tokens, True) == want
    assert moe_op.shared_ctas_for(num_tokens, False) == 0


def test_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTLLM_M31_DISABLE_MOE_SH20", "1")
    importlib.reload(m31_switches)
    try:
        m = importlib.reload(moe_op)
        assert not m.SH20
        assert [m.shared_ctas_for(n, True) for n in range(1, 9)] == [24] * 8
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_MOE_SH20", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(moe_op)
    assert moe_op.SH20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("killed", [False, True])
def test_sh20_config(monkeypatch: pytest.MonkeyPatch, killed: bool) -> None:
    monkeypatch.setattr(moe_op, "SH20", not killed)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    sh = moe_op.shared_ctas_for(8, True)
    state = moe_op.M3MoeState(dev, num_tokens=8, num_ctas=sms - sh, shared=True)
    assert state.config.get("sh_ctas", 24) == (24 if killed else 20)
    assert state.mod.NSHC == (24 if killed else 20) and state.mod.NSH == 24
    assert state.mod.SH_NU == (1 if killed else 2)
    assert state.epochs.numel() == state.num_ctas + state.mod.NSHC
