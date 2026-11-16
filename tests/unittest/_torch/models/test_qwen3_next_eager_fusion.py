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
    Qwen3NextSparseMoeBlock,
    _DeferredSharedExpertFinalize,
    _eager_fusion_enabled,
    _Qwen3NextDecoderLayerBase,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoEDecodeOptions
from tensorrt_llm._torch.utils import AuxStreamType, EventType


def _new_causal_lm() -> Qwen3NextForCausalLM:
    model = Qwen3NextForCausalLM.__new__(Qwen3NextForCausalLM)
    nn.Module.__init__(model)
    model.spec_config = None
    model.spec_worker = None
    return model


@pytest.mark.cpu_only
def test_sparse_moe_keeps_shared_expert_tokens_unpadded(monkeypatch) -> None:
    block = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    nn.Module.__init__(block)
    block.hidden_dim = 2
    block.enable_attention_dp = True
    block.mapping = SimpleNamespace(tp_size=2)
    block.event_dict = {EventType.Main: None, EventType.MoeShared: None}
    block.aux_stream = None
    block._shared_gate_aux = None
    block._decode_front = MagicMock(return_value=None)
    block._shared_expert_low_m_forward = MagicMock(return_value=None)
    block.gate = MagicMock(return_value=torch.zeros(4, 2))
    block.experts = MagicMock(side_effect=lambda x, *args, **kwargs: x[:2] + 3)
    block.shared_expert = MagicMock(side_effect=lambda x, **kwargs: x * 2)
    block.shared_expert_gate = MagicMock(side_effect=lambda x: x.new_zeros((x.shape[0], 1)))

    module = "tensorrt_llm._torch.models.modeling_qwen3_next"
    monkeypatch.setattr(f"{module}.is_sm_120f", lambda: True)
    monkeypatch.setattr(
        f"{module}.fused_sigmoid_gate_mul_add",
        lambda routed, gate, shared: routed + torch.sigmoid(gate) * shared,
    )
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output = block(hidden_states, SimpleNamespace(all_rank_num_tokens=[2, 4]))

    routed_input = block.experts.call_args.args[0]
    assert routed_input.shape == (4, 2)
    torch.testing.assert_close(routed_input[:2], hidden_states)
    assert torch.count_nonzero(routed_input[2:]) == 0
    assert block.gate.call_args.args[0].shape == (4, 2)
    assert block._decode_front.call_args.args[2].shape == (4, 2)
    assert block.experts.call_args.kwargs["use_dp_padding"]
    assert block.shared_expert.call_args.args[0].shape == (2, 2)
    assert block.shared_expert_gate.call_args.args[0].shape == (2, 2)
    assert output.shape == (2, 2)
    torch.testing.assert_close(output, hidden_states * 2 + 3)


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
    assert model.model.norm._fused_norm_weight is None


@torch.no_grad()
def test_cache_derived_state_refreshes_gemma_norm_weight() -> None:
    model = _new_causal_lm()
    # No decoder layers: only the norm weight precompute runs.
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    model.model_config = SimpleNamespace(pretrained_config=SimpleNamespace(num_hidden_layers=0))
    model.gemma_norm = RMSNorm(hidden_size=4, eps=1e-6, dtype=torch.bfloat16, use_gemma=True)
    model.standard_norm = RMSNorm(hidden_size=4, eps=1e-6, dtype=torch.bfloat16)
    model.gemma_norm.weight.copy_(torch.tensor([-0.5, 0.0, 0.5, 1.0], dtype=torch.bfloat16))

    model.cache_derived_state()

    expected = (model.gemma_norm.weight.float() + 1.0).to(torch.bfloat16)
    # Exact: cache_derived_state bakes (1+weight) with the same fp32-add-then-
    # cast recomputed here, so the result must be bitwise-identical.
    torch.testing.assert_close(model.gemma_norm._fused_norm_weight, expected, atol=0.0, rtol=0.0)
    assert model.standard_norm._fused_norm_weight is None

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
        next_layer_prefetch_weight=None,
        post_moe_norm_fold=False,
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


@pytest.mark.parametrize("enabled", [False, True])
def test_attention_takes_the_gqa_decode_kernel_where_the_model_declares_it(
    monkeypatch, enabled
) -> None:
    """The decode kernels the model class declares (test_decode_kernels_follow_the_model_declaration) reach the
    attention module as constructor arguments."""
    from tensorrt_llm._torch.models import modeling_qwen3_next
    from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3Attention

    built = {}

    def record_init(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        built.update(kwargs)

    monkeypatch.setattr(Qwen3Attention, "__init__", record_init)
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
        mapping=SimpleNamespace(enable_attention_dp=False),
        extra_attrs={"qwen3_next_decode_kernels": True} if enabled else {},
    )

    modeling_qwen3_next.Qwen3NextAttention(config, layer_idx=0, fuse_qk_norm_rope=True)

    assert built["use_gqa_decode_fmha"] is enabled
    assert built["cache_context_plan"] is enabled
    assert built["fp8_output_gate"] is False
    assert built["context_fmha_head_dim_per_stage_kv"] == 0


@pytest.mark.parametrize("declared", [False, True])
def test_attention_takes_the_prefill_kernels_where_the_model_declares_them(
    monkeypatch, declared
) -> None:
    """Qwen3_5MoeForCausalLM.prefill_kernels reaches the attention module as constructor arguments."""
    from tensorrt_llm._torch.models import modeling_qwen3_next
    from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3Attention

    built = {}

    def record_init(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        built.update(kwargs)

    monkeypatch.setattr(Qwen3Attention, "__init__", record_init)
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(model_type="qwen3_5_moe_text", head_dim=256),
        mapping=SimpleNamespace(enable_attention_dp=False),
        extra_attrs={"qwen3_next_prefill_kernels": True} if declared else {},
    )

    modeling_qwen3_next.Qwen3NextAttention(config, layer_idx=0, fuse_qk_norm_rope=True)

    assert built["fp8_output_gate"] is declared
    assert built["context_fmha_head_dim_per_stage_kv"] == (256 if declared else 0)
    assert built["context_last_rows_only"] is False


class _StopInit(Exception):
    pass


@pytest.mark.parametrize("declared", [False, True])
def test_mtp_attention_computes_only_last_context_rows_with_the_prefill_kernels(
    monkeypatch, declared
) -> None:
    """Only each context request's last MTP row feeds the draft token: with the prefill kernels the MTP layer
    builds its attention to compute only those rows."""
    built = {}

    def record_init(self, *args, **kwargs) -> None:
        built.update(kwargs)
        raise _StopInit

    monkeypatch.setattr(Qwen3NextFullAttentionDecoderLayer, "__init__", record_init)
    config = SimpleNamespace(
        extra_attrs={"qwen3_next_prefill_kernels": True} if declared else {},
    )
    with pytest.raises(_StopInit):
        Qwen3NextMTP(config, layer_idx=60, aux_stream_dict={AuxStreamType.Attention: None})

    assert built["context_last_rows_only"] is declared


@pytest.mark.parametrize("declared,blackwell", [(False, True), (True, False), (True, True)])
def test_decode_kernels_follow_the_model_declaration(monkeypatch, declared, blackwell) -> None:
    """The decode paths follow the model class's ``decode_kernels`` declaration (on SM 10.0 / 10.3), not the
    checkpoint's model type; both Qwen3.5 classes declare them."""
    from tensorrt_llm._torch.models import modeling_qwen3_next as m
    from tensorrt_llm._torch.models.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
    )

    assert Qwen3_5MoeForCausalLM.decode_kernels and Qwen3_5ForCausalLM.decode_kernels
    assert not Qwen3NextForCausalLM.decode_kernels

    def stop(model_config):
        raise _StopInit

    monkeypatch.setattr(m, "Qwen3NextModel", stop)
    monkeypatch.setattr(m, "is_blackwell_datacenter", lambda: blackwell)
    # A layout the decode kernels were tuned for (test_qwen3_next_decode_gates.py covers the layouts).
    monkeypatch.setattr(m, "_decode_kernels_layout", lambda model_config: True)
    model_cls = type("_Declared", (Qwen3NextForCausalLM,), {"decode_kernels": declared})
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(model_type="qwen3_next"),
        extra_attrs={},
        spec_config=None,
        cuda_graph_batch_sizes=[1, 2],
        mapping=SimpleNamespace(enable_attention_dp=False, dp_size=1),
    )
    with pytest.raises(_StopInit):
        model_cls(config)
    assert m._uses_decode_kernels(config) is (declared and blackwell)
    assert MoEDecodeOptions.of(config.extra_attrs).tuning is (declared and blackwell)


@pytest.mark.parametrize("sm", [90, 100, 103, 107, 120])
def test_decode_kernels_need_blackwell_datacenter(monkeypatch, sm) -> None:
    """The declared decode paths run on SM 10.0 / 10.3 only (not SM 10.7, 12.x or Hopper)."""
    import tensorrt_llm._utils as utils
    from tensorrt_llm._torch.models import modeling_qwen3_next as m

    def stop(model_config):
        raise _StopInit

    monkeypatch.setattr(utils, "get_sm_version", lambda: sm)
    monkeypatch.setattr(m, "Qwen3NextModel", stop)
    monkeypatch.setattr(m, "_decode_kernels_layout", lambda model_config: True)
    model_cls = type("_Declared", (Qwen3NextForCausalLM,), {"decode_kernels": True})
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(model_type="qwen3_next"),
        extra_attrs={},
        spec_config=None,
        cuda_graph_batch_sizes=[1, 2],
        mapping=SimpleNamespace(enable_attention_dp=False, dp_size=1),
    )
    with pytest.raises(_StopInit):
        model_cls(config)
    assert utils.is_blackwell_datacenter() is (sm in (100, 103))
    assert m._uses_decode_kernels(config) is (sm in (100, 103))


def test_next_layer_l2_prefetch_rows() -> None:
    """The MoE-output all-reduce prefetches the next projection's weight from l2_prefetch_min_rows rows, except in
    the rows the resident-weight GEMM takes it."""
    from tensorrt_llm._torch.models import modeling_qwen3_next as m

    weight = torch.zeros(8, 4)
    layer = SimpleNamespace(
        next_layer_prefetch_weight=weight,
        l2_prefetch_min_rows=14,
        next_layer_resident_rows=(14, 64),
    )
    assert m._next_layer_l2_prefetch(layer, 7) is None
    assert m._next_layer_l2_prefetch(layer, 14) is None
    assert m._next_layer_l2_prefetch(layer, 64) is None
    assert m._next_layer_l2_prefetch(layer, 65) is weight
    layer.next_layer_resident_rows = (1, 0)
    assert m._next_layer_l2_prefetch(layer, 14) is weight


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
        next_layer_prefetch_weight=None,
        post_moe_norm_fold=False,
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


@pytest.mark.parametrize("fold", [False, True])
def test_attention_dp_post_moe_chain_takes_the_norm_fold(monkeypatch, fold) -> None:
    """Under attention DP a layer with ``post_moe_norm_fold`` asks the MoE block to defer its shared-expert
    finalize and hands the deferred output to _post_moe_norm_fold instead of next_layer_layernorm."""
    from tensorrt_llm._torch.models import modeling_qwen3_next

    rows, hidden = 3, 4
    hidden_states = torch.randn(rows, hidden, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden_states)
    deferred = _DeferredSharedExpertFinalize(
        torch.randn_like(hidden_states), torch.randn_like(hidden_states), torch.randn(rows, 1)
    )
    mlp_out = torch.randn_like(hidden_states)

    def mlp_forward(hidden_states, *args, defer_shared_expert_finalize, fold_shared_expert, **kw):
        assert not fold_shared_expert
        assert defer_shared_expert_finalize is fold
        return deferred if defer_shared_expert_finalize else mlp_out

    folded = (torch.empty(rows, hidden, dtype=torch.float8_e4m3fn), residual)
    fold_calls = []

    def post_moe_norm_fold(layer, mlp_output, residual):
        fold_calls.append(mlp_output)
        return folded

    monkeypatch.setattr(modeling_qwen3_next, "_post_moe_norm_fold", post_moe_norm_fold)
    next_layer_norm = MagicMock(side_effect=lambda h, r: (h, r))
    layer = SimpleNamespace(
        layer_idx=0,
        hidden_size=hidden,
        post_moe_norm_fold=fold,
        mixer_forward=MagicMock(
            side_effect=lambda pos, h, md, sm, lora, all_reduce_params, **kw: h
        ),
        post_attention_layernorm=MagicMock(side_effect=lambda h, r: (h, r)),
        next_layer_layernorm=next_layer_norm,
        fusion_config=SimpleNamespace(PRE_MOE_FUSION=False, POST_MOE_FUSION=False),
        disable_attn_allreduce=True,
        mlp=MagicMock(side_effect=mlp_forward),
        mapping=SimpleNamespace(tp_size=4),
    )

    out = _Qwen3NextDecoderLayerBase.forward(
        layer,
        position_ids=torch.arange(rows),
        hidden_states=hidden_states,
        attn_metadata=SimpleNamespace(),
        residual=residual,
    )

    if fold:
        assert fold_calls == [deferred]
        assert out[0] is folded[0] and out[1] is folded[1]
        next_layer_norm.assert_not_called()
    else:
        assert not fold_calls
        assert out[0] is mlp_out
        next_layer_norm.assert_called_once()


def _kept_rows_layer(rows, hidden, norm_fold=False):
    """A decoder layer stub whose MLP is row-wise (its kept rows equal the full pass's rows) and records its input.
    ``norm_fold``: the MLP defers its shared-expert finalize to the post-MoE norm fold (attention DP)."""
    seen = {}

    def mlp_forward(hidden_states, attn_metadata, **kwargs):
        seen["rows"] = hidden_states.shape[0]
        seen["counts"] = attn_metadata.all_rank_num_tokens
        if kwargs["defer_shared_expert_finalize"]:
            return _DeferredSharedExpertFinalize(
                hidden_states * 2, hidden_states - 1, hidden_states * 0.5
            )
        return hidden_states * 2

    mlp = MagicMock(side_effect=mlp_forward)
    mlp.can_defer_finalize = False
    mlp.has_shared_expert = False
    layer = SimpleNamespace(
        layer_idx=0,
        hidden_size=hidden,
        input_layernorm=MagicMock(),
        mixer_forward=MagicMock(side_effect=lambda pos, h, md, sm, lora, arp, **kw: h + 1),
        post_attention_layernorm=MagicMock(side_effect=lambda h, r: (h, r + h)),
        next_layer_layernorm=MagicMock(side_effect=lambda h, r: (h + r, r)),
        fusion_config=SimpleNamespace(PRE_MOE_FUSION=False, POST_MOE_FUSION=False),
        disable_attn_allreduce=True,
        mlp=mlp,
        mapping=SimpleNamespace(tp_size=1),
        post_moe_norm_fold=norm_fold,
    )
    return layer, seen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the row gather is a Triton kernel")
@pytest.mark.parametrize("norm_fold", [False, True], ids=["norm", "norm_fold"])
@pytest.mark.parametrize("counts", [None, [3, 2]])
@torch.no_grad()
def test_kept_rows_run_the_mlp_tail_on_those_rows_only(monkeypatch, counts, norm_fold) -> None:
    """With ``kept_rows`` the layer runs its MLP and tail on the kept rows (under attention DP with every rank's
    kept-row count installed), also through the post-MoE norm fold: the result is the full pass's rows at those
    ids."""
    from tensorrt_llm._torch.models import modeling_qwen3_next as m
    from tensorrt_llm._torch.speculative.kept_rows import DraftKeptRows

    def fold(layer, mlp_output, residual):
        # row-wise stand-in for gate_add_rmsnorm_fp8: gate * shared + routed + residual
        h = mlp_output.routed_output + torch.sigmoid(mlp_output.shared_gate_logits) * (
            mlp_output.shared_output
        )
        return h + residual, residual + h

    monkeypatch.setattr(m, "_post_moe_norm_fold", fold)

    rows, hidden = 7, 4
    hidden_states = torch.randn(rows, hidden, device="cuda")
    residual = torch.randn(rows, hidden, device="cuda")
    ids = torch.tensor([6, 2, 3], device="cuda")

    def run(kept_rows):
        layer, seen = _kept_rows_layer(rows, hidden, norm_fold)
        md = SimpleNamespace(all_rank_num_tokens=[rows, rows])
        out = _Qwen3NextDecoderLayerBase.forward(
            layer,
            position_ids=torch.arange(rows, device="cuda"),
            hidden_states=hidden_states.clone(),
            attn_metadata=md,
            residual=residual.clone(),
            kept_rows=kept_rows,
        )
        return out, seen

    (full_h, full_r), full_seen = run(None)
    (kept_h, kept_r), kept_seen = run(DraftKeptRows(ids, counts, compact=True))
    assert full_seen["rows"] == rows and kept_seen["rows"] == ids.numel()
    assert kept_seen["counts"] == (counts if counts is not None else [rows, rows])
    assert torch.equal(kept_h, full_h[ids]) and torch.equal(kept_r, full_r[ids])


@pytest.mark.parametrize("compact", [True, False])
@torch.no_grad()
def test_mtp_forward_returns_the_kept_rows(monkeypatch, compact) -> None:
    """The MTP layer hands ``kept_rows`` to the decoder forward and returns its rows compact, or scattered back to
    their positions (other rows zero); the rank token counts are restored either way."""
    from tensorrt_llm._torch.speculative.kept_rows import DraftKeptRows

    model = _new_mtp()
    rows = 5
    ids = torch.tensor([4, 1])
    kept = DraftKeptRows(ids, [2, 2], compact=compact)
    target_rank_tokens = [rows, rows]
    attn_metadata = SimpleNamespace(all_rank_num_tokens=target_rank_tokens)

    def decoder_forward(self, **kwargs):
        assert kwargs["kept_rows"] is kept
        kwargs["attn_metadata"].all_rank_num_tokens = kept.all_rank_counts
        return kwargs["hidden_states"].index_select(0, ids), None

    monkeypatch.setattr(Qwen3NextFullAttentionDecoderLayer, "forward", decoder_forward)
    hidden_in = torch.randn(rows, 2)
    embed = nn.Embedding(1, 2)
    out = model(
        input_ids=torch.zeros(rows, dtype=torch.long),
        position_ids=torch.arange(rows),
        hidden_states=hidden_in,
        embed_tokens=embed,
        attn_metadata=attn_metadata,
        kept_rows=kept,
    )
    # identity norms and fc: the layer input is concat(embedding, hidden)
    expected = torch.cat([embed(torch.zeros(rows, dtype=torch.long)), hidden_in], dim=-1)[ids]
    if compact:
        assert torch.equal(out, expected)
    else:
        assert out.shape == (rows, 4) and torch.equal(out[ids], expected)
        assert not out[[0, 2, 3]].any()
    assert attn_metadata.all_rank_num_tokens is target_rank_tokens


@pytest.mark.parametrize(
    "batch,attention_dp,large_rows,kept_rows",
    [
        # (largest CUDA-graph decode batch, attention DP): reference GEN layouts at 7 rows per request
        (1, False, 448, False),  # TEP16 b1
        (2, False, 448, True),  # TEP8 b2
        (5, False, 448, True),  # TEP8 b5
        (8, False, 448, True),  # TEP8 b8
        (16, False, 448, True),  # TEP8 b16
        (64, False, 448, True),  # TEP8 b64: 448 rows from full steps
        (128, False, 448, True),
        (52, True, 448, True),  # DEP4 b52: 364 rows per rank, below 448
    ],
)
def test_decode_layout_decisions(monkeypatch, batch, attention_dp, large_rows, kept_rows) -> None:
    """Where the Qwen3.5 decode paths engage, derived from the layout (MTP rows per request, CUDA-graph batch
    sizes, attention DP): the large-step rows (tcgen05 GDN verify, the MoE engine's spare SMs), pass 0's kept rows,
    the state promotion beside the draft loop (TP decode kernels, every layout) and the expert prefetch's budgeted
    steps."""
    from tensorrt_llm._torch.models import modeling_qwen3_next as m

    monkeypatch.setattr(m, "_uses_decode_kernels", lambda model_config: True)
    model_config = SimpleNamespace(
        spec_config=SimpleNamespace(tokens_per_gen_step=7),
        cuda_graph_batch_sizes=list(range(1, batch + 1)),
        mapping=SimpleNamespace(enable_attention_dp=attention_dp),
    )
    assert m._large_step_rows(model_config) == large_rows
    # tcgen05 GDN verify: every step under attention DP, large steps with TP attention
    assert m._verify_tcgen05_min_rows(model_config) == (7 if attention_dp else large_rows)
    assert m.Qwen3NextMTP.takes_kept_rows(model_config) == kept_rows
    assert m.Qwen3NextMTP.promotes_target_state_beside(model_config) == (not attention_dp)
    # Expert prefetch rank + issue budget per step: 3..5 requests (TEP8 b5's steady step, not b2's or b8's)
    lo, hi = m._expert_prefetch_budget_rows(model_config)
    assert [r for r in range(1, 10) if lo < 7 * r <= hi] == [3, 4, 5]
