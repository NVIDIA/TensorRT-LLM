# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep's TRTLLM-runner experts build the NVLink one-sided all-to-all workspace only where MNNVL is supported, as the
engine's strategy selection does (`NVLinkOneSided.is_platform_supported`) and as dep2 does; elsewhere they keep no
workspace and run over the all-gather / reduce-scatter instead of failing at load."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the family ops (CUDA build)"
)
mod = importlib.import_module("tensorrt_llm._torch.modeling_v2.models.deepseek_v41.a2a_experts")


@pytest.mark.parametrize("supported", [False, True])
def test_the_workspace_needs_mnnvl(monkeypatch, supported):
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

    built = []
    monkeypatch.setattr(MnnvlMemory, "supports_mnnvl", staticmethod(lambda: supported))
    monkeypatch.setattr(mod, "moe_a2a_workspace", lambda *a, **k: built.append(a) or NS())
    mapping = NS(enable_attention_dp=True, moe_tp_size=1, moe_ep_size=8, world_size=8)
    a2a = mod.build_a2a_workspace(mapping, 8192, 5120, 6, 384)
    assert (a2a is not None) == supported and bool(built) == supported
