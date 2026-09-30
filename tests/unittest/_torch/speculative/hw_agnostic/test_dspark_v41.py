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
"""V4.1 DSpark stage, checkpoint, and captured-context attention contracts."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_deepseekv41 import _v41_load_coverage
from tensorrt_llm._torch.models.modeling_dspark_v41 import (
    DSv41DSparkDraftModel,
    remap_dspark_v41_draft_keys,
)
from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod
from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.utils import _build_spec_metadata


@pytest.mark.parametrize("layer_ids", [None, []])
def test_embedded_dspark_metadata_requires_resolved_capture_layers(layer_ids):
    spec = SimpleNamespace(
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        draft_is_embedded_in_target=True,
        target_layer_ids=layer_ids,
    )
    with pytest.raises(ValueError, match="requires nonempty target_layer_ids"):
        _build_spec_metadata(spec, SimpleNamespace(vocab_size=128), 2, 32)


def test_dspark_generation_never_silently_substitutes_missing_captures():
    worker = SimpleNamespace(max_draft_len=5)
    metadata = SimpleNamespace(get_hidden_states=Mock(return_value=None))
    accepted = torch.zeros(1, 6, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="requires captured target hidden states"):
        DSv4DSparkWorker._draft_gen_block_batched(
            worker,
            Mock(),
            metadata,
            Mock(),
            accepted,
            torch.ones(1, dtype=torch.int32),
            num_contexts=0,
            batch_size=1,
            total_target_tokens=6,
            position_ids=torch.arange(6),
        )


def test_v41_target_coverage_excludes_only_separately_loaded_draft_subtree():
    target = nn.Module()
    target.model = nn.Linear(2, 2, bias=False)
    target.draft_model = nn.Module()
    target.draft_model.stage = nn.Linear(2, 2, bias=False)
    # A similarly named target module must not accidentally escape the audit.
    target.draft_model_projection = nn.Linear(2, 2, bias=False)
    keys = {"model.weight", "draft_model_projection.weight"}
    assert _v41_load_coverage(target, keys, skip_modules=("draft_model",)) == ([], [])
    assert _v41_load_coverage(target, {"model.weight"}, skip_modules=("draft_model",)) == (
        [],
        ["draft_model_projection.weight"],
    )
    assert _v41_load_coverage(target.draft_model, ()) == ([], ["stage.weight"])
    assert _v41_load_coverage(target.draft_model, {"stage.weight"}) == ([], [])
    assert _v41_load_coverage(
        target, keys | {"draft_model.stage.weight"}, skip_modules=("draft_model",)
    ) == (["draft_model.stage.weight"], [])


def test_v41_remap_keeps_all_stages_and_maps_markov_factors():
    weights = {
        "mtp.0.main_proj.weight": torch.randn(8, 24),
        "mtp.0.main_norm.weight": torch.ones(8),
        "mtp.1.attn_norm.weight": torch.ones(8),
        "mtp.2.norm.weight": torch.ones(8),
        "mtp.2.markov_head.embed.weight": torch.randn(16, 4),
        "mtp.2.markov_head.head.weight": torch.randn(16, 4),
        "layers.0.attn_norm.weight": torch.ones(8),
    }
    remapped = remap_dspark_v41_draft_keys(weights, 3, 4)
    assert set(remapped) == {
        "mtp_layers.0.main_proj.weight",
        "mtp_layers.0.main_norm.weight",
        "mtp_layers.1.input_layernorm.weight",
        "mtp_layers.2.norm.weight",
        "mtp_layers.2.markov_head.markov_w1.weight",
        "mtp_layers.2.markov_head.markov_w2.weight",
    }
    assert (
        remapped["mtp_layers.2.markov_head.markov_w1.weight"]
        is weights["mtp.2.markov_head.embed.weight"]
    )


def test_v41_remap_rejects_missing_stage():
    with pytest.raises(ValueError, match=r"missing draft stages \[1\]"):
        remap_dspark_v41_draft_keys(
            {"mtp.0.norm.weight": torch.ones(4), "mtp.2.norm.weight": torch.ones(4)}, 3, 4
        )


@pytest.mark.parametrize(
    "tensor",
    [
        torch.ones(8, 8).to(torch.float8_e4m3fn),
        torch.ones(8, 8, dtype=torch.uint8),
        SimpleNamespace(get_dtype=lambda: "F8_E4M3"),
    ],
)
def test_v41_remap_rejects_quantized_draft_weight_without_scale(tensor):
    with pytest.raises(ValueError, match="incomplete quantized weight/scale pairs"):
        remap_dspark_v41_draft_keys({"mtp.0.main_proj.weight": tensor}, 1, 4)


def test_v41_remap_rejects_orphan_draft_scale():
    with pytest.raises(ValueError, match="incomplete quantized weight/scale pairs"):
        remap_dspark_v41_draft_keys({"mtp.0.main_proj.scale": torch.ones(1, 1)}, 1, 4)


@pytest.mark.parametrize("v41", [False, True])
def test_dspark_confidence_and_lm_head_use_variant_correct_hidden_states(v41):
    cls = DSv41DSparkDraftModel if v41 else modeling_dspark.DSv4DSparkDraftModel
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.block_size = 2
    collapsed = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])
    normalized = collapsed / 2
    confidence = Mock(return_value=torch.full((1, 2), 10.0))
    model.lm_head = nn.Identity()
    model.mtp_layers = [
        SimpleNamespace(
            hc_ffn=SimpleNamespace(collapse=Mock(return_value=collapsed)),
            hc_head=Mock(return_value=collapsed),
            norm=Mock(return_value=normalized),
            markov_head=None,
            confidence_head=confidence,
        )
    ]
    _, num_proposed, logits = model.forward_head(
        torch.empty(1, 2, 4, 2),
        torch.tensor([1]),
        pre_mix=torch.empty(1, 2, 4, 1),
        confidence_threshold=0.5,
        return_logits=True,
    )
    torch.testing.assert_close(logits, normalized)
    assert num_proposed.tolist() == [2]
    assert confidence.call_args.args[0] is (collapsed if v41 else normalized)


def _draft_routing(bias, *, hashed=False, table=None, top_k=3):
    return DeepSeekV4MoeRoutingMethod(
        top_k=top_k,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        callable_e_score_correction_bias=lambda: bias,
        callable_tid2eid=lambda: table,
        is_hashed=hashed,
    )


@pytest.mark.parametrize("num_tokens", [0, 1, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v41_draft_top3_routing_matches_reference(num_tokens, dtype, monkeypatch):
    torch.manual_seed(41)
    logits = torch.randn(num_tokens, 128, dtype=dtype)
    # The strong correction deliberately changes selection, not mixture weights.
    bias = torch.linspace(-10, 10, 128)
    fused = Mock(side_effect=AssertionError("top-3 must not call target-only gate_forward"))
    monkeypatch.setattr(torch.ops.trtllm, "gate_forward", fused)
    indices, weights = _draft_routing(bias).apply(logits)
    scores = torch.log1p(torch.exp(logits.float())).sqrt()
    expected_indices = torch.argsort(scores + bias, dim=-1, descending=True)[:, :3]
    expected_weights = torch.gather(scores, 1, expected_indices)
    expected_weights /= expected_weights.sum(-1, keepdim=True) + 1e-20
    torch.testing.assert_close(indices.long(), expected_indices)
    torch.testing.assert_close(weights, expected_weights * 2.5)
    assert indices.dtype == torch.int32
    assert weights.dtype == torch.float32
    fused.assert_not_called()


def test_v41_draft_hashed_routing_does_not_read_bias():
    logits = torch.arange(256, dtype=torch.float32).reshape(2, 128) / 20
    table = torch.tensor([[100, 50, 3], [4, 16, 127], [32, 41, 9]], dtype=torch.int32)
    token_ids = torch.tensor([2, 0])
    indices, weights = _draft_routing(None, hashed=True, table=table).apply(logits, token_ids)
    expected = F.softplus(logits).sqrt().gather(1, table[token_ids].long())
    torch.testing.assert_close(indices, table[token_ids])
    torch.testing.assert_close(weights, expected / expected.sum(-1, keepdim=True) * 2.5)


@pytest.mark.parametrize("experts", [256, 384])
def test_v4_target_top6_routing_keeps_fused_kernel(experts, monkeypatch):
    fused = Mock()
    monkeypatch.setattr(torch.ops.trtllm, "gate_forward", fused)
    logits = torch.zeros(1, experts, dtype=torch.bfloat16)
    _draft_routing(torch.zeros(experts), top_k=6).apply(logits)
    fused.assert_called_once()
    assert fused.call_args.args[0].dtype == torch.float32
    assert fused.call_args.args[6:9] == (6, 2.5, False)
    # The target gate also accepts the optional vision bias and image mask.
    assert fused.call_args.args[9:] == (None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires a GPU")
def test_v41_draft_top3_routing_cuda_graph_replay():
    logits = torch.randn(17, 128, dtype=torch.bfloat16, device="cuda")
    routing = _draft_routing(torch.linspace(-1, 1, 128, device="cuda"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices, weights = routing.apply(logits)
    logits.add_(torch.randn_like(logits))
    graph.replay()
    expected_indices, expected_weights = routing.apply(logits)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights)


def test_v41_block32_dequant_uses_each_tile_scale():
    weight = torch.ones(64, 64).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    actual = DSv41DSparkDraftModel._block_dequant(
        weight, scales, DSv41DSparkDraftModel.checkpoint_fp8_block_size
    )
    for row in range(2):
        for col in range(2):
            assert torch.all(
                actual[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32] == scales[row, col]
            )


def test_v41_draft_uses_smaller_expert_bank_without_changing_target(monkeypatch):
    captured = {}

    def fake_init(
        self, model_config, aux_stream_dict, num_stages, block_size, *, draft_moe_backend
    ):
        nn.Module.__init__(self)
        captured["config"] = model_config.pretrained_config
        self._attn_params = {}

    monkeypatch.setattr(modeling_dspark.DSv4DSparkDraftModel, "__init__", fake_init)
    target = SimpleNamespace(
        n_routed_experts=384,
        num_experts_per_tok=6,
        dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3,
        dspark_target_layer_ids=[0, 20, 39],
        engram_layer_ids=[1, 14],
        engram_config=object(),
        sliding_window=128,
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
    )
    model = DSv41DSparkDraftModel(SimpleNamespace(pretrained_config=target), {}, 3, 5)
    draft = captured["config"]
    assert (draft.n_routed_experts, draft.num_experts_per_tok) == (128, 3)
    assert (target.n_routed_experts, target.num_experts_per_tok) == (384, 6)
    assert target.engram_layer_ids == [1, 14]
    assert draft.engram_layer_ids == []
    assert draft.candidate_source_layer_id is None
    assert draft.compress_ratios == [0, 0, 0]
    assert draft.kv_source_layer_ids == draft.index_source_layer_ids == []
    assert draft.candidate_topk_blocks > 0
    assert not hasattr(target, "compress_ratios")
    assert model._attn_params["q_b_norm_enabled"] is False


@pytest.mark.parametrize("batched", [False, True])
def test_v41_three_stages_carry_lagged_pre_into_final_head(batched):
    model = DSv41DSparkDraftModel.__new__(DSv41DSparkDraftModel)
    nn.Module.__init__(model)
    model.hc_mult = 2
    model.block_size = 2
    model.lm_head = nn.Identity()
    model.mtp_layers = [SimpleNamespace(_dspark_attn={}) for _ in range(3)]
    model.mtp_layers[-1].hc_ffn = SimpleNamespace(collapse=mHC.collapse)
    model.mtp_layers[-1].norm = nn.Identity()
    model.mtp_layers[-1].markov_head = None
    model.mtp_layers[-1].confidence_head = None
    initial = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    mixes = [torch.full((1, 2, 2, 1), value) for value in (0.2, 0.3, 0.4)]
    seen = []

    def fake_stage(self, stage, hidden, *args, pre_mix, **kwargs):
        index = len(seen)
        seen.append(pre_mix.clone())
        return hidden + 1, mixes[index]

    model._forward_stage = MethodType(fake_stage, model)
    model.forward_embed = Mock(return_value=(initial, torch.zeros(1, 3), torch.zeros(1, 2)))
    model._dspark_freqs_table = Mock(return_value=torch.empty(0))
    call = model.forward_batched if batched else model.forward
    kwargs = {}
    if batched:
        kwargs = {"kv_windows": torch.zeros(1, 3, 4, 2), "slots": torch.zeros(1, dtype=torch.long)}
    _, _, logits = call(
        torch.zeros(1, 9),
        torch.ones(1, dtype=torch.long),
        torch.ones(1, dtype=torch.long) if batched else 1,
        return_logits=True,
        **kwargs,
    )
    torch.testing.assert_close(seen[0][..., 0, :], torch.ones(1, 2, 1))
    torch.testing.assert_close(seen[0][..., 1, :], torch.zeros(1, 2, 1))
    torch.testing.assert_close(seen[1], mixes[0])
    torch.testing.assert_close(seen[2], mixes[1])
    torch.testing.assert_close(logits, ((initial + 3) * mixes[-1]).sum(-2))


def test_v41_stage_collapses_with_previous_not_current_mix(monkeypatch):
    hidden = torch.randn(1, 2, 2, 3)
    previous_pre, attention_pre, ffn_pre = [torch.randn(1, 2, 2, 1) for _ in range(3)]
    attention_output = torch.randn(1, 2, 3)
    mid_residual = torch.randn_like(hidden)
    final_residual = torch.randn_like(hidden)
    attention_mixer = SimpleNamespace(
        pre_mapping_lagged=Mock(return_value=(attention_pre, None, None, hidden[:, :, 0])),
        post_mapping=Mock(return_value=mid_residual),
    )
    ffn_mixer = SimpleNamespace(
        pre_mapping_lagged=Mock(return_value=(ffn_pre, None, None, mid_residual[:, :, 0])),
        post_mapping=Mock(return_value=final_residual),
    )
    stage = SimpleNamespace(
        hc_attn=attention_mixer,
        hc_ffn=ffn_mixer,
        input_layernorm=nn.Identity(),
        post_attention_layernorm=nn.Identity(),
        mlp=Mock(return_value=torch.zeros(2, 3)),
        _dspark_attn={},
        enable_fused_hc=True,
    )
    model = SimpleNamespace(
        _attn_params={"window_size": 4, "head_dim": 2},
        model_config=SimpleNamespace(mapping=SimpleNamespace(enable_attention_dp=False, tp_size=1)),
    )
    monkeypatch.setattr(
        modeling_dspark, "dspark_attention_forward", Mock(return_value=attention_output)
    )
    result, next_pre = modeling_dspark.DSv4DSparkDraftModel._forward_stage(
        model,
        stage,
        hidden,
        torch.zeros(1, 3),
        1,
        torch.empty(0),
        torch.zeros(2),
        pre_mix=previous_pre,
    )
    assert attention_mixer.pre_mapping_lagged.call_args.args[1] is previous_pre
    assert ffn_mixer.pre_mapping_lagged.call_args.args[1] is attention_pre
    assert result is final_residual
    assert next_pre is ffn_pre


@pytest.mark.parametrize("batched", [False, True])
def test_v41_attention_query_has_no_per_head_rmsnorm(monkeypatch, batched):
    torch.manual_seed(24)
    monkeypatch.setattr(modeling_dspark, "IS_CUTLASS_DSL_AVAILABLE", False)
    hidden, heads, head_dim, q_rank, block = 8, 2, 4, 4, 3
    x = torch.randn(1, block, hidden)
    q_a = torch.randn(q_rank, hidden)
    q_b = torch.randn(heads * head_dim, q_rank) * 2
    norm_weight = torch.randn(q_rank)
    freqs = modeling_dspark.precompute_dspark_freqs_cis(2, 32)
    captured = {}

    def capture_attention(q, *args):
        captured["q"] = q
        return torch.zeros_like(q)

    monkeypatch.setattr(modeling_dspark, "dspark_sparse_attn", capture_attention)
    kwargs = dict(
        wq_a=q_a,
        wq_b=q_b,
        q_norm_w=norm_weight,
        wkv=torch.randn(head_dim, hidden),
        kv_norm_w=torch.ones(head_dim),
        wo_a=torch.randn(4, heads * head_dim),
        wo_b=torch.randn(hidden, 4),
        attn_sink=torch.ones(heads),
        n_heads=heads,
        head_dim=head_dim,
        rope_head_dim=2,
        n_groups=1,
        o_lora_rank=4,
        window_size=4,
        eps=1e-20,
        softmax_scale=0.5,
        freqs_cis=freqs,
        q_b_norm_enabled=False,
    )
    main_x, window = torch.randn(1, 1, hidden), torch.randn(1, 4, head_dim)
    if batched:
        modeling_dspark.dspark_attention_forward_batched(
            x, main_x, torch.tensor([1]), window, torch.tensor([0]), **kwargs
        )
    else:
        modeling_dspark.dspark_attention_forward(x, main_x, 1, window, **kwargs)
    q = F.linear(x, q_a)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-20) * norm_weight
    q = F.linear(q, q_b).unflatten(-1, (heads, head_dim))
    # The non-RoPE lanes are an independent check: their scale must survive wq_b.
    torch.testing.assert_close(captured["q"][..., :2], q[..., :2])


def test_v41_draft_csa2_layout_has_no_target_cache_sources(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode
    from tensorrt_llm.llmapi.llm_args import CSA2SparseAttentionConfig

    captured = {}

    def fake_init(
        self, model_config, aux_stream_dict, num_stages, block_size, *, draft_moe_backend
    ):
        nn.Module.__init__(self)
        captured["params"] = model_config.sparse_attention_config.to_sparse_params(
            pretrained_config=model_config.pretrained_config
        )
        self._attn_params = {}

    monkeypatch.setattr(modeling_dspark.DSv4DSparkDraftModel, "__init__", fake_init)
    target = SimpleNamespace(
        n_routed_experts=384,
        num_experts_per_tok=6,
        dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3,
        dspark_target_layer_ids=[0, 20, 39],
        engram_layer_ids=[1, 14],
        engram_config=None,
        sliding_window=128,
        compress_ratios=[1, 1],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        index_topk=512,
    )
    config = SimpleNamespace(
        pretrained_config=target, sparse_attention_config=CSA2SparseAttentionConfig()
    )
    DSv41DSparkDraftModel(config, {}, 3, 5)
    layout = captured["params"].layout
    assert [layout.layer(i).mode for i in range(3)] == [CSA2Mode.SWA] * 3
    assert target.compress_ratios == [1, 1] and target.kv_source_layer_ids == [0]
    assert DSv41DSparkDraftModel._draft_sparse_config(config, 40, 3) is None


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("cached_descriptors", [False, True])
def test_v41_draft_real_config_keeps_outer_and_text_geometry_in_sync(
    monkeypatch, composite, cached_descriptors
):
    import copy

    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode
    from tensorrt_llm._torch.configs.deepseek_v41 import DeepseekV41Config, DeepseekV41TextConfig
    from tensorrt_llm.llmapi.llm_args import CSA2SparseAttentionConfig

    text = DeepseekV41TextConfig(
        num_hidden_layers=4,
        compress_ratios=[0, 0, 1, 1],
        kv_source_layer_ids=[2],
        index_source_layer_ids=[2],
        candidate_source_layer_id=2,
        engram_layer_ids=[],
        engram_num_embeddings=[],
        dspark_target_layer_ids=[0, 2, 3],
        dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3,
    )
    target = DeepseekV41Config(text_config=text) if composite else text
    if cached_descriptors:
        assert text.layer_descriptors[2].has_long_range
    before = copy.deepcopy(target.to_dict())
    captured = {}

    def fake_init(
        self, model_config, aux_stream_dict, num_stages, block_size, *, draft_moe_backend
    ):
        nn.Module.__init__(self)
        captured["config"] = model_config.pretrained_config
        captured["layout"] = model_config.sparse_attention_config.to_sparse_params(
            pretrained_config=model_config.pretrained_config
        ).layout
        self._attn_params = {}

    monkeypatch.setattr(modeling_dspark.DSv4DSparkDraftModel, "__init__", fake_init)
    DSv41DSparkDraftModel(
        SimpleNamespace(
            pretrained_config=target, sparse_attention_config=CSA2SparseAttentionConfig()
        ),
        {},
        3,
        5,
    )
    draft = captured["config"]
    inner = draft.text_config if composite else draft
    for value in (draft, inner):
        assert value.compress_ratios == [0, 0, 0]
        assert value.kv_source_layer_ids == value.index_source_layer_ids == []
        assert value.candidate_source_layer_id is None
        assert value.engram_layer_ids == [] and value.engram_config is None
        assert (value.n_routed_experts, value.num_experts_per_tok) == (128, 3)
    assert [captured["layout"].layer(i).mode for i in range(3)] == [CSA2Mode.SWA] * 3
    assert len(inner.layer_descriptors) == 3
    assert all(not row.has_long_range for row in inner.layer_descriptors)
    assert target.to_dict() == before
    assert text.compress_ratios == [0, 0, 1, 1] and text.kv_source_layer_ids == [2]
    if cached_descriptors:
        assert text.layer_descriptors[2].has_long_range
