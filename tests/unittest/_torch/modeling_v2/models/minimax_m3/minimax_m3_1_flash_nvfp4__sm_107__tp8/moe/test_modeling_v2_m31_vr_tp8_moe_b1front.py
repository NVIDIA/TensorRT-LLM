# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MOE_XDEFER + MOE_B1SPLIT (default on at 8 tokens) on the sm_107 tp8 m3_moe engine: the non-GEMV routed CTAs issue
their x copy at the head of the B1 path, and warps 8..11 quantize B1 tokens 0..3 while they wait for the logits. Both
are scheduling only, so the outputs are bit for bit the kill-switched engine's (TRTLLM_M31_DISABLE_MOE_XDEFER /
_B1SPLIT); other token counts never take them."""

import importlib.util

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import hidden_states, layer_for  # noqa: E402

# The engine state compiles the CuTe DSL MoE kernel for the SM 10.x tcgen05 family only.
_NEEDS_SM107_CUTE = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or importlib.util.find_spec("cutlass") is None,
    reason="needs an SM107 GPU and CuTe DSL (the engine state compiles the kernel)",
)

BOTH = frozenset({"MOE_XDEFER", "MOE_B1SPLIT"})


def _state(monkeypatch, num_tokens: int, killed: frozenset):
    monkeypatch.setattr(moe_op, "m31_disabled", lambda item_id, k=killed: item_id in k)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return moe_op.M3MoeState(
        dev,
        num_tokens=num_tokens,
        num_ctas=sms - moe_op.shared_ctas_for(num_tokens, True),
        shared=True,
    )


@_NEEDS_SM107_CUTE
@pytest.mark.parametrize(
    "killed", [frozenset(), frozenset({"MOE_XDEFER"}), frozenset({"MOE_B1SPLIT"}), BOTH]
)
def test_config_and_kill_switches(monkeypatch: pytest.MonkeyPatch, killed: frozenset) -> None:
    state = _state(monkeypatch, 8, killed)
    xdefer, b1split = "MOE_XDEFER" not in killed, "MOE_B1SPLIT" not in killed
    assert state.config.get("xdefer", 0) == int(xdefer)
    assert state.config.get("b1split", 0) == int(b1split)
    assert state.config.get("b1s_tok", 0) == (4 if b1split else 0)
    assert bool(state.mod.XDEFER) == xdefer
    assert bool(state.mod.B1SPLIT) == b1split
    assert state.mod.XSTAGE, "both items ride the fused-router XSTAGE path"


@_NEEDS_SM107_CUTE
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 6, 7])
def test_other_token_counts_never_take_them(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int
) -> None:
    state = _state(monkeypatch, num_tokens, frozenset())
    assert "xdefer" not in state.config and "b1split" not in state.config
    assert not state.mod.XDEFER and not state.mod.B1SPLIT


@_NEEDS_SM107_CUTE
def test_engagement_line(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm.logger import logger

    seen = []
    monkeypatch.setattr(moe_op, "_B1FRONT_LOGGED", set())
    monkeypatch.setattr(logger, "info", lambda msg, *a, **k: seen.append(str(msg)))
    _state(monkeypatch, 8, frozenset())
    _state(monkeypatch, 8, frozenset())
    lines = [m for m in seen if m.startswith("[M31_TP8_B1FRONT]")]
    assert lines == ["[M31_TP8_B1FRONT] XDEFER=1 B1SPLIT=1 b1s_tok=4 tokens=8"], seen


@_NEEDS_SM107_CUTE
@pytest.mark.parametrize("case", ["random", "concentrated"])
@pytest.mark.parametrize(
    "killed", [frozenset({"MOE_XDEFER"}), frozenset({"MOE_B1SPLIT"}), frozenset()]
)
def test_bitwise_vs_both_killed(
    monkeypatch: pytest.MonkeyPatch, case: str, killed: frozenset
) -> None:
    hidden = hidden_states(8, seed=31)
    off = _state(monkeypatch, 8, BOTH)
    ref = off.run(layer_for(off, case), hidden).clone()
    on = _state(monkeypatch, 8, killed)
    for rep in range(3):
        got = on.run(layer_for(on, case), hidden)
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (
            case,
            sorted(killed),
            rep,
        )
