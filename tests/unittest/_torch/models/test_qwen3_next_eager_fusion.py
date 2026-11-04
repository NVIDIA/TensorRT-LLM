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

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.distributed import AllReduceFusionOp, MoEAllReduceParams
from tensorrt_llm._torch.models.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextFullAttentionDecoderLayer,
    Qwen3NextMTP,
    _DeferredSharedExpertFinalize,
    _eager_fusion_enabled,
    _Qwen3NextDecoderLayerBase,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import EventType


def _new_causal_lm() -> Qwen3NextForCausalLM:
    model = Qwen3NextForCausalLM.__new__(Qwen3NextForCausalLM)
    nn.Module.__init__(model)
    model.spec_config = None
    model.spec_worker = None
    return model


def _new_mtp() -> Qwen3NextMTP:
    model = Qwen3NextMTP.__new__(Qwen3NextMTP)
    nn.Module.__init__(model)
    model.pre_fc_norm_embedding = nn.Identity()
    model.pre_fc_norm_hidden = nn.Identity()
    model.fc = nn.Identity()
    model.shared_head = nn.Module()
    model.shared_head.norm = MagicMock(
        side_effect=lambda hidden_states, residual: (hidden_states, None)
    )
    model.event_dict = {EventType.Main: None, EventType.MoeShared: None}
    model.aux_stream = None
    model.model_config = SimpleNamespace(
        mapping=SimpleNamespace(tp_size=1, tp_rank=0, enable_attention_dp=True)
    )
    model.decode_kernels = False
    model._fc_zero_residual = None
    return model


@torch.no_grad()
def test_mtp_forward_uses_and_restores_draft_rank_token_counts(monkeypatch) -> None:
    model = _new_mtp()
    target_rank_tokens = [1024, 1024]
    draft_rank_tokens = [1, 1]
    attn_metadata = SimpleNamespace(all_rank_num_tokens=target_rank_tokens)

    def decoder_forward(self, **kwargs):
        assert kwargs["attn_metadata"].all_rank_num_tokens is draft_rank_tokens
        return kwargs["hidden_states"], None

    monkeypatch.setattr(Qwen3NextFullAttentionDecoderLayer, "forward", decoder_forward)

    hidden_states = model(
        input_ids=torch.tensor([0]),
        position_ids=torch.tensor([0]),
        hidden_states=torch.ones(1, 2),
        embed_tokens=nn.Embedding(1, 2),
        attn_metadata=attn_metadata,
        all_rank_num_tokens=draft_rank_tokens,
    )

    assert hidden_states.shape == (1, 4)
    assert attn_metadata.all_rank_num_tokens is target_rank_tokens


@torch.no_grad()
def test_mtp_forward_restores_rank_token_counts_after_failure(monkeypatch) -> None:
    model = _new_mtp()
    target_rank_tokens = [1024, 1024]
    draft_rank_tokens = [1, 1]
    attn_metadata = SimpleNamespace(all_rank_num_tokens=target_rank_tokens)

    def decoder_forward(self, **kwargs):
        assert kwargs["attn_metadata"].all_rank_num_tokens is draft_rank_tokens
        raise RuntimeError("draft forward failed")

    monkeypatch.setattr(Qwen3NextFullAttentionDecoderLayer, "forward", decoder_forward)

    with pytest.raises(RuntimeError, match="draft forward failed"):
        model(
            input_ids=torch.tensor([0]),
            position_ids=torch.tensor([0]),
            hidden_states=torch.ones(1, 2),
            embed_tokens=nn.Embedding(1, 2),
            attn_metadata=attn_metadata,
            all_rank_num_tokens=draft_rank_tokens,
        )

    assert attn_metadata.all_rank_num_tokens is target_rank_tokens


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
def test_pre_moe_allreduce_runs_in_mixer_output_projection() -> None:
    """With PRE_MOE_FUSION the mixer's output projection owns the pre-MoE all-reduce + residual +
    post_attention_layernorm; the layer's own all-reduce runs only after the MoE."""
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden_states)
    post_attention_norm = RMSNorm(hidden_size=4, eps=1e-6, dtype=torch.bfloat16, use_gemma=True)
    next_layer_norm = RMSNorm(hidden_size=4, eps=1e-6, dtype=torch.bfloat16, use_gemma=True)
    post_attention_norm.weight.copy_(torch.tensor([-0.5, 0.0, 0.5, 1.0], dtype=torch.bfloat16))
    next_layer_norm.weight.copy_(torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.bfloat16))
    mixer_forward = MagicMock(
        side_effect=lambda pos, h, md, sm, lora, all_reduce_params, **kw: (
            h,
            all_reduce_params.residual,
        )
    )
    allreduce = MagicMock(
        side_effect=lambda hidden_states, *, all_reduce_params: (
            hidden_states,
            all_reduce_params.residual,
        )
    )
    mlp = MagicMock(side_effect=lambda hidden_states, *args, **kwargs: hidden_states)
    mlp.can_defer_finalize = False
    mlp.has_shared_expert = False
    layer = SimpleNamespace(
        layer_idx=0,
        hidden_size=4,
        input_layernorm=MagicMock(),
        mixer_forward=mixer_forward,
        post_attention_layernorm=post_attention_norm,
        next_layer_layernorm=next_layer_norm,
        next_layer_input_scale=None,
        next_layer_reads_bf16=False,
        allreduce_fp8_payload=False,
        fusion_config=SimpleNamespace(PRE_MOE_FUSION=True, POST_MOE_FUSION=True),
        disable_attn_allreduce=True,
        allreduce=allreduce,
        mlp=mlp,
        mapping=SimpleNamespace(tp_size=2),
    )

    _Qwen3NextDecoderLayerBase.forward(
        layer,
        position_ids=torch.arange(2),
        hidden_states=hidden_states,
        attn_metadata=SimpleNamespace(),
        residual=residual,
    )

    mixer_params = mixer_forward.call_args.args[5]
    assert mixer_params.enable_allreduce
    assert mixer_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
    assert mixer_params.residual is residual
    # Exact: norm_weight is the same baked (1+weight) recomputed here.
    torch.testing.assert_close(
        mixer_params.norm_weight,
        (post_attention_norm.weight.float() + 1.0).to(torch.bfloat16),
        atol=0.0,
        rtol=0.0,
    )

    assert allreduce.call_count == 1
    post_ar_params = allreduce.call_args.kwargs["all_reduce_params"]
    assert post_ar_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
    torch.testing.assert_close(
        post_ar_params.norm_weight,
        (next_layer_norm.weight.float() + 1.0).to(torch.bfloat16),
        atol=0.0,
        rtol=0.0,
    )

    mlp_ar_params = mlp.call_args.kwargs["all_reduce_params"]
    assert not mlp_ar_params.enable_allreduce
    assert mlp.call_args.kwargs["do_finalize"]


@pytest.mark.parametrize(
    "model_type,sm,enabled",
    [
        ("qwen3_5_moe", 103, True),
        ("qwen3_5_moe", 100, True),
        ("qwen3_next", 103, False),
        ("qwen3_5_moe", 107, False),
        ("qwen3_5_moe", 90, False),
    ],
)
def test_attention_takes_the_gqa_decode_kernel_on_qwen3_5_sm100_and_sm103(
    monkeypatch, model_type, sm, enabled
) -> None:
    from tensorrt_llm import _utils
    from tensorrt_llm._torch.models import modeling_qwen3_next
    from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3Attention

    built = {}

    def record_init(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        built.update(kwargs)

    monkeypatch.setattr(_utils, "get_sm_version", lambda: sm)
    monkeypatch.setattr(Qwen3Attention, "__init__", record_init)
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(model_type=model_type),
        mapping=SimpleNamespace(enable_attention_dp=False),
    )

    modeling_qwen3_next.Qwen3NextAttention(config, layer_idx=0, fuse_qk_norm_rope=True)

    assert built["use_gqa_decode_fmha"] is enabled
    assert built["cache_context_plan"] is enabled


@pytest.mark.parametrize("can_defer_finalize", [False, True])
def test_post_moe_allreduce_folds_the_shared_expert(can_defer_finalize) -> None:
    """Where the MoE-output all-reduce takes the fold, the MoE hands back its routed output with the shared
    expert and gate logits; a finalized routed output is folded without an expert index."""
    rows, hidden = 3, 4
    hidden_states = torch.randn(rows, hidden, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden_states)
    next_layer_norm = RMSNorm(hidden_size=hidden, eps=1e-6, dtype=torch.bfloat16, use_gemma=True)
    routed = torch.randn_like(hidden_states)
    shared = torch.randn_like(hidden_states)
    gate = torch.randn(rows, 1, dtype=torch.bfloat16)
    fc2, topw, perm = torch.randn(8, hidden), torch.rand(rows, 2), torch.zeros(rows, 2)

    def mlp_forward(hidden_states, *args, do_finalize, fold_shared_expert, **kwargs):
        assert fold_shared_expert
        assert do_finalize is not can_defer_finalize
        return _DeferredSharedExpertFinalize(
            (fc2, topw, perm) if not do_finalize else routed, shared, gate
        )

    mlp = MagicMock(side_effect=mlp_forward)
    mlp.can_defer_finalize = can_defer_finalize
    mlp.has_shared_expert = True
    allreduce = MagicMock(
        side_effect=lambda hidden_states, *, all_reduce_params: (
            hidden_states,
            all_reduce_params.residual,
        )
    )
    allreduce.supports_moe_fold.return_value = True
    layer = SimpleNamespace(
        layer_idx=0,
        hidden_size=hidden,
        input_layernorm=MagicMock(),
        mixer_forward=MagicMock(
            side_effect=lambda pos, h, md, sm, lora, all_reduce_params, **kw: (
                h,
                all_reduce_params.residual,
            )
        ),
        post_attention_layernorm=RMSNorm(
            hidden_size=hidden, eps=1e-6, dtype=torch.bfloat16, use_gemma=True
        ),
        next_layer_layernorm=next_layer_norm,
        next_layer_input_scale=None,
        next_layer_reads_bf16=False,
        allreduce_fp8_payload=False,
        fusion_config=SimpleNamespace(PRE_MOE_FUSION=True, POST_MOE_FUSION=True),
        disable_attn_allreduce=True,
        allreduce=allreduce,
        mlp=mlp,
        mapping=SimpleNamespace(tp_size=2),
    )

    _Qwen3NextDecoderLayerBase.forward(
        layer,
        position_ids=torch.arange(rows),
        hidden_states=hidden_states,
        attn_metadata=SimpleNamespace(),
        residual=residual,
    )

    allreduce.supports_moe_fold.assert_called_once_with(rows, hidden)
    ar_input = allreduce.call_args.args[0]
    params = allreduce.call_args.kwargs["all_reduce_params"]
    assert isinstance(params, MoEAllReduceParams)
    assert params.shared_expert_output is shared
    assert params.shared_expert_gate is gate
    if can_defer_finalize:
        assert ar_input is fc2
        assert params.expanded_idx_to_permuted_idx is perm
        assert params.expert_scale_factor is topw
    else:
        assert ar_input is routed
        assert params.expanded_idx_to_permuted_idx is None
        assert params.expert_scale_factor is None
