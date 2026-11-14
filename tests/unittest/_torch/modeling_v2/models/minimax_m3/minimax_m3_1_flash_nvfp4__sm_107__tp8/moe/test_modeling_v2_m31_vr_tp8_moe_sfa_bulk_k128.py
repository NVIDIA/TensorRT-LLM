# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SFA_BULK + K128 (default on) on the sm_107 tp8 m3_moe engine: the FC1 scale atoms by one bulk copy per stage from
the pre-spliced image, and the FC1 / FC2 MMAs as 2 x K128. Both are scheduling / instruction-shape only, so the
outputs are bit for bit the kill-switched engine's (TRTLLM_M31_DISABLE_MOE_SFA_BULK / _K128), at every token count."""

import importlib.util

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import m31_moe_sfa_image as sfa
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


def _state(monkeypatch, num_tokens: int, killed: frozenset):
    monkeypatch.setattr(moe_op, "m31_disabled", lambda item_id, k=killed: item_id in k)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return moe_op.M3MoeState(dev, num_tokens=num_tokens, num_ctas=sms - 24, shared=True)


def test_image_is_a_permutation() -> None:
    raw = torch.randint(
        0,
        255,
        (
            2,
            2 * 2048 * 4,
        ),
        dtype=torch.uint8,
    )
    img = sfa.fc1_scale_image(raw)
    w = raw.reshape(-1).view(torch.int32).view(-1, 4, 4, 32, 4)
    assert torch.equal(img.view(-1, 4, 4, 32, 4), w.permute(0, 4, 2, 3, 1))
    assert sfa.fc1_scale_image(raw) is img, "one image per scale tensor"


@_NEEDS_SM107_CUTE
@pytest.mark.parametrize(
    "killed", [frozenset(), frozenset({"MOE_SFA_BULK"}), frozenset({"MOE_K128"})]
)
def test_config_and_kill_switches(monkeypatch: pytest.MonkeyPatch, killed: frozenset) -> None:
    state = _state(monkeypatch, 8, killed)
    assert state.config.get("sfa_bulk", 0) == int("MOE_SFA_BULK" not in killed)
    assert state.config.get("k128", 0) == int("MOE_K128" not in killed)
    assert bool(state.mod.SFA_BULK) == ("MOE_SFA_BULK" not in killed)
    assert bool(state.mod.K128) == ("MOE_K128" not in killed)


@_NEEDS_SM107_CUTE
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 6, 7, 8])
def test_bitwise_vs_killed(monkeypatch: pytest.MonkeyPatch, num_tokens: int) -> None:
    hidden = hidden_states(num_tokens, seed=23 + num_tokens)
    off = _state(monkeypatch, num_tokens, frozenset({"MOE_SFA_BULK", "MOE_K128"}))
    ref = off.run(layer_for(off, "random"), hidden).clone()
    on = _state(monkeypatch, num_tokens, frozenset())
    layer = layer_for(on, "random")
    assert getattr(layer, "sfa1_img", None) is not None
    got = on.run(layer, hidden)
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
