# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep2 builds its NVLink one-sided all-to-all workspace only where MNNVL is supported, as the engine's strategy
selection does (`NVLinkOneSided.is_platform_supported`); elsewhere it keeps no workspace and its experts run over the
all-gather / reduce-scatter. Before, it allocated the MNNVL workspace unconditionally, so such a host failed at load
(after construction, where `modeling_v2: auto` cannot fall back)."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target (CUDA build)"
)
mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling"
)


def _core():
    mapping = NS(enable_attention_dp=True, moe_tp_size=1, moe_ep_size=2, world_size=2)
    cfg = NS(hidden_size=5120, num_experts_per_tok=6, n_routed_experts=384)
    return NS(mapping=mapping, cfg=cfg, _a2a_rows=8192, layers=[])


@pytest.mark.parametrize("supported", [False, True])
def test_the_workspace_needs_mnnvl(monkeypatch, supported):
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

    built = []
    monkeypatch.setattr(MnnvlMemory, "supports_mnnvl", staticmethod(lambda: supported))
    monkeypatch.setattr(
        mod,
        "moe_a2a_workspace",
        lambda *a, **k: built.append(a)
        or NS(workspace=torch.empty(1, 1 << 20), sentinel_max_rows=0),
    )
    a2a = mod.ModelingV2Core._build_a2a(_core())
    assert (a2a is not None) == supported and bool(built) == supported
