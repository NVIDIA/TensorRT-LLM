# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from tensorrt_llm._torch.distributed import AllReduceFusionOp
from tensorrt_llm._torch.models.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextLinearDecoderLayer,
    _eager_fusion_enabled,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm


def _new_causal_lm() -> Qwen3NextForCausalLM:
    model = Qwen3NextForCausalLM.__new__(Qwen3NextForCausalLM)
    nn.Module.__init__(model)
    return model


@torch.no_grad()
def test_setup_aliases_does_not_read_meta_weights() -> None:
    model = _new_causal_lm()
    model.model_config = SimpleNamespace(pretrained_config=SimpleNamespace(num_hidden_layers=2))
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for layer in model.model.layers:
        layer.input_layernorm = RMSNorm(
            hidden_size=4,
            eps=1e-6,
            dtype=torch.bfloat16,
            device=torch.device("meta"),
            use_gemma=True,
        )
        layer.next_layer_layernorm = None
    model.model.norm = RMSNorm(
        hidden_size=4, eps=1e-6, dtype=torch.bfloat16, device=torch.device("meta"), use_gemma=True
    )

    model.setup_aliases()

    assert model.model.layers[0].next_layer_layernorm is model.model.layers[1].input_layernorm
    assert model.model.layers[1].next_layer_layernorm is model.model.norm
    assert not hasattr(model.model.norm, "_fused_norm_weight")


@torch.no_grad()
def test_cache_derived_state_refreshes_gemma_norm_weight() -> None:
    model = _new_causal_lm()
    model.gemma_norm = RMSNorm(hidden_size=4, eps=1e-6, dtype=torch.bfloat16, use_gemma=True)
    model.standard_norm = RMSNorm(hidden_size=4, eps=1e-6, dtype=torch.bfloat16)
    model.gemma_norm.weight.copy_(torch.tensor([-0.5, 0.0, 0.5, 1.0], dtype=torch.bfloat16))

    model.cache_derived_state()

    expected = (model.gemma_norm.weight.float() + 1.0).to(torch.bfloat16)
    # Exact: cache_derived_state bakes (1+weight) with the same fp32-add-then-
    # cast recomputed here, so the result must be bitwise-identical.
    torch.testing.assert_close(model.gemma_norm._fused_norm_weight, expected, atol=0.0, rtol=0.0)
    assert not hasattr(model.standard_norm, "_fused_norm_weight")

    model.gemma_norm.weight.add_(1.0)
    model.cache_derived_state()
    expected = (model.gemma_norm.weight.float() + 1.0).to(torch.bfloat16)
    # Exact: cache_derived_state bakes (1+weight) with the same fp32-add-then-
    # cast recomputed here, so the result must be bitwise-identical.
    torch.testing.assert_close(model.gemma_norm._fused_norm_weight, expected, atol=0.0, rtol=0.0)


@torch.no_grad()
def test_eager_fusion_is_enabled_for_gdn_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", raising=False)
    assert _eager_fusion_enabled(enable_attention_dp=False)
    assert not _eager_fusion_enabled(enable_attention_dp=True)

    monkeypatch.setenv("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", "1")
    assert not _eager_fusion_enabled(enable_attention_dp=False)


@torch.no_grad()
def test_gdn_fusion_has_single_allreduce_owner() -> None:
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden_states)
    # Use real RMSNorm modules (lightweight) rather than mocks: their
    # weight / use_gemma / variance_epsilon are exactly what the fused-norm
    # path reads. The linear_attn/allreduce/mlp below stay mocks because the
    # test asserts on *how they are called* (call_count / call_args).
    post_attention_norm = RMSNorm(hidden_size=4, eps=1e-6, dtype=torch.bfloat16, use_gemma=True)
    next_layer_norm = RMSNorm(hidden_size=4, eps=1e-6, dtype=torch.bfloat16, use_gemma=True)
    post_attention_norm.weight.copy_(torch.tensor([-0.5, 0.0, 0.5, 1.0], dtype=torch.bfloat16))
    next_layer_norm.weight.copy_(torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.bfloat16))
    linear_attn = MagicMock(side_effect=lambda hidden_states, *args, **kwargs: hidden_states)
    allreduce = MagicMock(
        side_effect=lambda hidden_states, *, all_reduce_params: (
            hidden_states,
            all_reduce_params.residual,
        )
    )
    mlp = MagicMock(side_effect=lambda hidden_states, *args, **kwargs: hidden_states)
    layer = SimpleNamespace(
        layer_idx=0,
        input_layernorm=MagicMock(),
        linear_attn=linear_attn,
        post_attention_layernorm=post_attention_norm,
        next_layer_layernorm=next_layer_norm,
        fusion_config=SimpleNamespace(PRE_MOE_FUSION=True, POST_MOE_FUSION=True),
        disable_attn_allreduce=True,
        allreduce=allreduce,
        mlp=mlp,
        mapping=SimpleNamespace(tp_size=2),
        moe_allreduce=MagicMock(),
    )

    Qwen3NextLinearDecoderLayer.forward(
        layer,
        position_ids=torch.arange(2),
        hidden_states=hidden_states,
        attn_metadata=SimpleNamespace(),
        residual=residual,
    )

    internal_ar_params = linear_attn.call_args.kwargs["all_reduce_params"]
    assert not internal_ar_params.enable_allreduce

    # Two module-level allreduces at the layer boundary: pre-MoE and post-MoE.
    # The GDN linear_attn's own allreduce is disabled (single-owner, asserted
    # just above), so it does not add a third.
    assert allreduce.call_count == 2
    pre_ar_params = allreduce.call_args_list[0].kwargs["all_reduce_params"]
    post_ar_params = allreduce.call_args_list[1].kwargs["all_reduce_params"]
    assert pre_ar_params.enable_allreduce
    assert post_ar_params.enable_allreduce
    assert pre_ar_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
    assert post_ar_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
    # Exact: norm_weight is the same baked (1+weight) recomputed here.
    torch.testing.assert_close(
        pre_ar_params.norm_weight,
        (post_attention_norm.weight.float() + 1.0).to(torch.bfloat16),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        post_ar_params.norm_weight,
        (next_layer_norm.weight.float() + 1.0).to(torch.bfloat16),
        atol=0.0,
        rtol=0.0,
    )

    mlp_ar_params = mlp.call_args.kwargs["all_reduce_params"]
    assert not mlp_ar_params.enable_allreduce
    assert mlp.call_args.kwargs["do_finalize"]
