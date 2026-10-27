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

from tensorrt_llm._torch.distributed import AllReduceFusionOp
from tensorrt_llm._torch.models import modeling_qwen3_next
from tensorrt_llm._torch.models.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextFullAttentionDecoderLayer,
    Qwen3NextLinearDecoderLayer,
    Qwen3NextMTP,
    Qwen3NextSparseMoeBlock,
    _attach_norm_fp8_scales,
    _eager_fusion_enabled,
)
from tensorrt_llm._torch.models.modeling_utils import EagerFusionConfig
from tensorrt_llm._torch.modules.linear import Linear, W8A16FP8LinearMethod
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import EventType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _new_causal_lm() -> Qwen3NextForCausalLM:
    model = Qwen3NextForCausalLM.__new__(Qwen3NextForCausalLM)
    nn.Module.__init__(model)
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


_HIDDEN = 64


def _linear(quant_algo=QuantAlgo.FP8, **kwargs):
    quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo else None
    return Linear(
        _HIDDEN, 32, bias=False, dtype=torch.bfloat16, quant_config=quant_config, **kwargs
    )


def _norm():
    return RMSNorm(hidden_size=_HIDDEN, eps=1e-6, dtype=torch.bfloat16, use_gemma=True)


def _wiring_layer(gate_up_proj, has_pp=False, has_helix=False):
    layer = nn.Module()
    layer.mapping = SimpleNamespace(has_pp=lambda: has_pp, has_cp_helix=lambda: has_helix)
    layer.input_layernorm = _norm()
    layer.post_attention_layernorm = _norm()
    layer.mlp = nn.Module()
    if gate_up_proj is not None:
        layer.mlp.shared_expert = nn.Module()
        layer.mlp.shared_expert.gate_up_proj = gate_up_proj
    return layer


@pytest.mark.parametrize(
    "case",
    ("static_fp8", "unquantized", "dynamic_quant", "w8a16", "pp", "helix", "dense_mlp"),
)
def test_norms_share_the_static_fp8_scales_of_their_consumers(case):
    quant_algo = None if case == "unquantized" else QuantAlgo.FP8
    dynamic = {"force_dynamic_quantization": True} if case == "dynamic_quant" else {}
    qkv_proj, gate_up_proj = _linear(quant_algo, **dynamic), _linear(quant_algo, **dynamic)
    if case == "w8a16":
        qkv_proj.quant_method = gate_up_proj.quant_method = W8A16FP8LinearMethod()
    layer = _wiring_layer(
        None if case == "dense_mlp" else gate_up_proj,
        has_pp=case == "pp",
        has_helix=case == "helix",
    )

    _attach_norm_fp8_scales(layer, qkv_proj)

    for norm, consumer in (
        (layer.input_layernorm, qkv_proj if case in ("static_fp8", "dense_mlp") else None),
        (layer.post_attention_layernorm, gate_up_proj if case == "static_fp8" else None),
    ):
        if consumer is None:
            assert norm.fp8_scale is None
        else:
            assert norm.fp8_scale.data_ptr() == consumer.input_scale.data_ptr()
            assert "fp8_scale" not in norm.state_dict()


def _fp8_pair(num_tokens=3):
    bf16 = torch.randn((num_tokens, _HIDDEN), dtype=torch.bfloat16)
    return bf16.to(torch.float8_e4m3fn), bf16


def _moe_block():
    block = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    nn.Module.__init__(block)
    block.hidden_dim = _HIDDEN
    block.enable_attention_dp = True
    block.mapping = SimpleNamespace(tp_size=1)
    block.gate = MagicMock(side_effect=lambda h: torch.zeros(h.shape[0], 4))
    block.experts = MagicMock(side_effect=lambda h, logits, **kwargs: torch.zeros_like(h))
    block.shared_expert = MagicMock(
        side_effect=lambda h, lora_params=None: torch.ones(
            (h.shape[0], _HIDDEN), dtype=torch.bfloat16
        )
    )
    block.shared_expert_gate = MagicMock(side_effect=lambda h: torch.zeros(h.shape[0], 1))
    block.event_dict = {EventType.Main: None, EventType.MoeShared: None}
    block.aux_stream = None
    return block


@pytest.mark.parametrize("lora_params", (None, {"lora": True}))
@torch.no_grad()
def test_moe_block_feeds_fp8_copy_to_shared_expert_only(monkeypatch, lora_params):
    monkeypatch.setattr(
        modeling_qwen3_next,
        "fused_sigmoid_gate_mul_add",
        lambda routed, gate_logits, shared, output=None: routed + shared,
    )
    block = _moe_block()
    fp8, bf16 = _fp8_pair()

    out = block((fp8, bf16), SimpleNamespace(all_rank_num_tokens=None), lora_params=lora_params)

    expected_shared_input = bf16 if lora_params else fp8
    assert block.shared_expert.call_args.args[0].dtype == expected_shared_input.dtype
    assert block.shared_expert.call_args.args[0].data_ptr() == expected_shared_input.data_ptr()
    for module in (block.gate, block.experts, block.shared_expert_gate):
        assert module.call_args.args[0].data_ptr() == bf16.data_ptr()
    assert out.shape == bf16.shape


@torch.no_grad()
def test_gdn_in_proj_qkvz_reads_fp8_copy():
    gdn = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    nn.Module.__init__(gdn)
    gdn.conv_dim_per_tp, gdn.num_v_heads_per_tp, gdn.head_v_dim = 6, 2, 4
    gdn.in_proj_qkvz = MagicMock(side_effect=lambda h: torch.zeros(h.shape[0], 6 + 2 * 4))
    gdn.in_proj_ba = MagicMock(side_effect=lambda h: torch.zeros(h.shape[0], 2 * 2))
    gdn.event_dict = {EventType.Main: None, EventType.Attention: None}
    gdn.aux_stream = None
    fp8, bf16 = _fp8_pair()

    gdn._compute_tokenwise_inputs((fp8, bf16))

    assert gdn.in_proj_qkvz.call_args.args[0] is fp8
    assert gdn.in_proj_ba.call_args.args[0] is bf16


def _decoder_layer(cls):
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.layer_idx = 0
    layer.mapping = SimpleNamespace(tp_size=1)
    layer.fusion_config = EagerFusionConfig()
    layer.disable_attn_allreduce = True
    layer.post_attention_layernorm = MagicMock(side_effect=lambda h, r: (h, r))
    layer.mlp = MagicMock(side_effect=lambda h, *args, **kwargs: h)
    layer.next_layer_layernorm = None
    return layer


@pytest.mark.parametrize("lora_params", (None, {"lora": True}))
@torch.no_grad()
def test_full_attention_layer_feeds_fp8_copy_to_attention(lora_params):
    layer = _decoder_layer(Qwen3NextFullAttentionDecoderLayer)
    layer.self_attn = MagicMock(side_effect=lambda **kwargs: torch.zeros(3, _HIDDEN))
    fp8, bf16 = _fp8_pair()

    layer(
        position_ids=None,
        hidden_states=(fp8, bf16),
        attn_metadata=SimpleNamespace(),
        residual=torch.zeros_like(bf16),
        lora_params=lora_params,
    )

    assert layer.self_attn.call_args.kwargs["hidden_states"] is (bf16 if lora_params else fp8)


@pytest.mark.parametrize("num_tokens", (3, 0))
@torch.no_grad()
def test_linear_attention_layer_passes_pair_to_mixer(num_tokens):
    layer = _decoder_layer(Qwen3NextLinearDecoderLayer)
    layer.linear_attn = MagicMock(side_effect=lambda h, *args, **kwargs: torch.zeros(3, _HIDDEN))
    fp8, bf16 = _fp8_pair(num_tokens)

    layer(
        position_ids=None,
        hidden_states=(fp8, bf16),
        attn_metadata=SimpleNamespace(),
        residual=torch.zeros_like(bf16),
    )

    if num_tokens:
        pair = layer.linear_attn.call_args.args[0]
        assert pair[0] is fp8 and pair[1] is bf16
    else:
        layer.linear_attn.assert_not_called()
        assert layer.post_attention_layernorm.call_args.args[0] is bf16
