# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for how Qwen3 MoE layers use the shared MoE finalize cutoff."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.models.modeling_qwen3_moe as m

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(("batch", "local"), [(47, False), (48, True), (128, True), (129, True)])
@pytest.mark.parametrize("nvfp4", [False, True])
def test_forward_uses_shared_cutoff_and_keeps_quant_guard(
    batch: int, local: bool, nvfp4: bool
) -> None:
    hidden = torch.zeros(batch, 8)
    residual = torch.ones_like(hidden)
    norm = SimpleNamespace(weight=torch.ones(8), variance_epsilon=1e-6)
    mlp = Mock(return_value=hidden if local or not nvfp4 else (hidden, hidden, hidden))
    mlp.experts = SimpleNamespace(has_nvfp4=nvfp4)
    moe_allreduce = Mock(return_value=(hidden, residual))
    moe_allreduce.finalize_max_tokens = 47
    moe_allreduce.max_token = 128
    layer = SimpleNamespace(
        fusion_config=SimpleNamespace(PRE_MOE_FUSION=True, POST_MOE_FUSION=True),
        self_attn=Mock(return_value=hidden),
        allreduce=Mock(return_value=(hidden, residual)),
        moe_allreduce=moe_allreduce,
        post_attention_layernorm=norm,
        next_layer_layernorm=norm,
        disable_attn_allreduce=True,
        model_config=SimpleNamespace(moe_backend="TRTLLM"),
        mlp=mlp,
        mapping=SimpleNamespace(tp_size=4),
        is_p2p_supported=True,
    )
    m.Qwen3MoEDecoderLayer.forward(layer, None, hidden, None, residual)
    assert mlp.call_args.kwargs["do_finalize"] is (local or not nvfp4)
    assert moe_allreduce.call_count == int(nvfp4 and not local)
    assert layer.allreduce.call_count == (2 if local or not nvfp4 else 1)
