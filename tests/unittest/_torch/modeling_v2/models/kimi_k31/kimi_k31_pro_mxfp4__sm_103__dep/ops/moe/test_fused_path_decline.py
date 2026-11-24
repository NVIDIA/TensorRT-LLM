# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the DEP16 target's fused-path choice (``fused_moe_decline``): the fused dispatch + experts + combine op
serves a layer only where the layer's exchange is the one-sided all-to-all, which only an NVLink fabric spanning the
EP group gives; elsewhere (e.g. two HGX nodes, where every local NVLink is up but the exchange fell back to the
all-gather) every call takes the separate path."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe import (  # noqa: E501
    w4a8_ep_moe_fused as fused,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _dep16(rank: int = 3) -> Mapping:
    return Mapping(
        world_size=16,
        rank=rank,
        tp_size=16,
        moe_tp_size=1,
        moe_ep_size=16,
        enable_attention_dp=True,
    )


def _fc1(local: int = 56, inter: int = 3072) -> torch.Tensor:
    return torch.empty(local, 2 * inter, 1792, dtype=torch.uint8)


@pytest.fixture
def served_group(monkeypatch):
    """A group the push kernels serve (the probe reads the GPU's compute capability)."""
    monkeypatch.setattr(fused._a2a, "push_all_to_all_serves", lambda *args: True)


def test_fused_path_needs_the_one_sided_exchange(served_group) -> None:
    assert fused.fused_moe_decline(_dep16(), _fc1(), one_sided_exchange=True) is None
    why = fused.fused_moe_decline(_dep16(), _fc1(), one_sided_exchange=False)
    assert why is not None and "one-sided" in why


def test_fused_path_declines_without_attention_dp(served_group) -> None:
    tp = Mapping(world_size=16, rank=3, tp_size=16, moe_tp_size=1, moe_ep_size=16)
    assert fused.fused_moe_decline(tp, _fc1(), one_sided_exchange=True) == "no all-to-all"


def test_fused_path_declines_split_experts(served_group) -> None:
    assert fused.fused_moe_decline(_dep16(), _fc1(inter=768), one_sided_exchange=True) is not None
