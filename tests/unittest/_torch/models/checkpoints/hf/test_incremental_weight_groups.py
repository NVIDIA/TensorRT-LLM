# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import WeightGroup
from tensorrt_llm._torch.models.checkpoints.hf.llama4_weight_mapper import Llama4HfWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextGate
from tensorrt_llm._torch.models.modeling_utils import _load_weights_impl_v2


class _QualifiedGenericModel:
    _supports_generic_hf_incremental_loading = True

    def __init__(self, config=None):
        self.config = config or SimpleNamespace()


class _Step3StyleUnqualifiedVlm:
    """VLM-style wrapper whose partial signature alone is not qualification."""

    def __init__(self):
        self.config = SimpleNamespace()

    def load_weights(self, weights, allow_partial_loading=False):
        del weights, allow_partial_loading


class _DerivedGenericModel(_QualifiedGenericModel):
    pass


def _qualified_generic_mapper(config=None) -> HfWeightMapper:
    mapper = HfWeightMapper()
    mapper._model = _QualifiedGenericModel(config)
    mapper._config = _borrow_safe_model_config()
    return mapper


def _borrow_safe_model_config(**updates):
    values = {
        "quant_config": SimpleNamespace(quant_algo=None),
        "quant_config_dict": {},
        "force_dynamic_quantization": False,
        "moe_load_balancer": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _qwen3_5_mapper() -> Qwen3_5MoeHfWeightMapper:
    mapper = Qwen3_5MoeHfWeightMapper()
    mapper._config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_algo=None),
        quant_config_dict={},
        force_dynamic_quantization=False,
        moe_load_balancer=None,
        mapping=SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            enable_attention_dp=False,
        ),
        pretrained_config=SimpleNamespace(
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            num_hidden_layers=2,
            num_experts=1,
            torch_dtype=torch.float32,
        ),
    )
    return mapper


def _qwen3_5_synthetic_weights() -> dict[str, torch.Tensor]:
    weights = {
        "model.visual.patch_embed.proj.weight": torch.arange(4).reshape(2, 2),
        "model.visual.merger.linear_fc1.weight": torch.arange(6).reshape(2, 3),
        "model.language_model.embed_tokens.weight": torch.arange(12).reshape(4, 3),
        "model.language_model.layers.0.input_layernorm.weight": torch.arange(3),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": torch.arange(6).reshape(
            2, 3
        ),
        "model.layers.1.input_layernorm.weight": torch.arange(3) + 10,
        "model.norm.weight": torch.arange(3) + 20,
        "lm_head.weight": torch.arange(12).reshape(4, 3) + 30,
        "lm_head.weight_scale": torch.ones(4),
        "mtp.layers.0.input_layernorm.weight": torch.arange(3) + 40,
        "mtp.fc.weight": torch.arange(9).reshape(3, 3),
    }

    projection_rows = {
        "q": 2,
        "k": 2,
        "v": 4,
        "z": 4,
        "b": 2,
        "a": 2,
    }
    for projection, rows in projection_rows.items():
        weights[f"model.language_model.layers.0.linear_attn.in_proj_{projection}.weight"] = (
            torch.arange(rows * 3).reshape(rows, 3) + rows
        )
    return weights


def test_qwen3_5_groups_partition_source_keys_and_preserve_dependencies():
    mapper = _qwen3_5_mapper()
    weights = _qwen3_5_synthetic_weights()

    groups = mapper.get_weight_groups(weights)

    assert groups is not None
    flattened_keys = [key for group in groups for key in group.keys]
    assert set(flattened_keys) == set(weights)
    assert len(flattened_keys) == len(weights)
    assert len(flattened_keys) == len(set(flattened_keys))

    qkvz = next(
        group
        for group in groups
        if group.group_id == "qwen3_5.model.layers.0.linear_attn.in_proj_qkvz"
    )
    assert any(".linear_attn.in_proj_q." in key for key in qkvz.keys)
    assert any(".linear_attn.in_proj_z." in key for key in qkvz.keys)
    assert not any(".linear_attn.in_proj_b." in key for key in qkvz.keys)

    experts = next(group for group in groups if group.group_id == "qwen3_5.model.layers.0.mlp")
    assert experts.keys == ("model.language_model.layers.0.mlp.experts.0.gate_proj.weight",)

    vision = next(group for group in groups if group.group_id == "qwen3_5.vision_and_projector")
    assert set(vision.keys) == {
        "model.visual.patch_embed.proj.weight",
        "model.visual.merger.linear_fc1.weight",
    }

    mtp_layer_norm = next(
        group for group in groups if group.group_id == "qwen3_5.mtp.layers.0.input_layernorm"
    )
    assert mtp_layer_norm.keys == ("mtp.layers.0.input_layernorm.weight",)

    mtp_fc = next(group for group in groups if group.group_id == "qwen3_5.mtp.fc")
    assert mtp_fc.keys == ("mtp.fc.weight",)

    lm_head = next(group for group in groups if group.group_id == "qwen3_5.lm_head")
    assert set(lm_head.keys) == {
        "lm_head.weight",
        "lm_head.weight_scale",
    }


def test_qwen3_5_moe_group_keeps_fused_gate_up_and_down_together():
    mapper = _qwen3_5_mapper()
    mapper._config.pretrained_config = SimpleNamespace(text_config=mapper._config.pretrained_config)
    keys = [
        "model.language_model.layers.0.mlp.gate_up_proj",
        "model.language_model.layers.0.mlp.gate_up_proj_scale_inv",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.mlp.down_proj_scale_inv",
        "model.language_model.layers.0.mlp.router.weight",
    ]

    groups = mapper.get_weight_groups(keys)

    assert groups is not None
    assert len(groups) == 1
    assert groups[0].group_id == "qwen3_5.model.layers.0.mlp"
    assert groups[0].keys == tuple(keys)
    processed_keys = [key.replace("model.language_model.", "model.") for key in keys]
    assert mapper.get_incremental_load_roots(processed_keys) == ("model.layers.0.mlp",)


def test_qwen3_5_moe_gate_group_supports_partial_dispatch(monkeypatch):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = nn.Module()
    model.model.layers[0].mlp.gate = Qwen3NextGate(
        hidden_size=3,
        num_experts=2,
        top_k=1,
        dtype=torch.float32,
    )
    mapper = _qwen3_5_mapper()
    mapper._model = model
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    weights = {"model.layers.0.mlp.gate.weight": expected}
    groups = mapper.get_weight_groups(weights)

    assert groups == [WeightGroup("qwen3_5.model.layers.0.mlp", tuple(weights))]
    assert model.model.layers[0].mlp.gate.supports_partial_weight_loading

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    _load_weights_impl_v2(model, weights, mapper, allow_partial_loading=True)

    torch.testing.assert_close(model.model.layers[0].mlp.gate.weight, expected)


def test_qwen3_5_full_and_grouped_preprocessing_match():
    mapper = _qwen3_5_mapper()
    weights = _qwen3_5_synthetic_weights()

    full_result = mapper.preprocess_weights(weights)
    grouped_result = {}
    groups = mapper.get_weight_groups(weights)
    assert groups is not None
    for group in groups:
        partial_result = mapper.preprocess_weights({key: weights[key] for key in group.keys})
        duplicate_keys = grouped_result.keys() & partial_result.keys()
        assert not duplicate_keys
        grouped_result.update(partial_result)

    assert grouped_result.keys() == full_result.keys()
    for key in full_result:
        torch.testing.assert_close(grouped_result[key], full_result[key])


def test_incremental_lifecycle_rejects_missing_duplicate_and_unknown_groups():
    mapper = HfWeightMapper()
    groups = [
        WeightGroup("layer.0", ("model.layers.0.weight",)),
        WeightGroup("layer.1", ("model.layers.1.weight",)),
    ]

    mapper.begin_incremental_load(groups)
    mapper.record_incremental_group_loaded("layer.0")
    with pytest.raises(RuntimeError, match="layer.1"):
        mapper.finalize_incremental_load()
    with pytest.raises(ValueError, match="loaded twice"):
        mapper.record_incremental_group_loaded("layer.0")
    with pytest.raises(ValueError, match="Unknown"):
        mapper.record_incremental_group_loaded("layer.2")

    mapper.abort_incremental_load()
    mapper.begin_incremental_load(groups)
    for group in groups:
        mapper.record_incremental_group_loaded(group.group_id)
    mapper.finalize_incremental_load()


def test_generic_hf_groups_atomic_destination_dependencies():
    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.fc.weight",
    ]

    groups = _qualified_generic_mapper().get_weight_groups(keys)

    assert groups is not None
    by_id = {group.group_id: set(group.keys) for group in groups}
    assert by_id["hf.model.embed_tokens"] == {keys[0]}
    assert by_id["hf.model.layers.0.self_attn.qkv_proj"] == set(keys[1:4])
    assert by_id["hf.model.layers.0.mlp.gate_up_proj"] == set(keys[4:6])
    assert by_id["hf.model.layers.1.self_attn.qkv_proj"] == {keys[6]}
    assert by_id["hf.model.norm"] == {keys[7]}
    assert by_id["hf.lm_head"] == {keys[8]}
    assert by_id["hf.mtp"] == {keys[9], keys[10]}
    flattened_keys = [key for group in groups for key in group.keys]
    assert set(flattened_keys) == set(keys)
    assert len(flattened_keys) == len(keys)


def test_generic_hf_routed_moe_keeps_complete_mlp_atomic():
    mapper = _qualified_generic_mapper(SimpleNamespace(num_experts=8, tie_word_embeddings=False))
    keys = [
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight_scale_inv",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight_scale_inv",
        "model.layers.0.mlp.router.weight",
    ]

    groups = mapper.get_weight_groups(keys)

    assert groups is not None
    assert len(groups) == 1
    assert groups[0].group_id == "hf.model.layers.0.mlp"
    assert groups[0].keys == tuple(keys)


class _UnqualifiedHfWeightMapper(HfWeightMapper):
    pass


def test_unqualified_mapper_subclass_does_not_inherit_generic_grouping():
    keys = ["model.layers.0.self_attn.q_proj.weight"]
    default_mapper = HfWeightMapper()
    qualified_mapper = _qualified_generic_mapper()

    assert _UnqualifiedHfWeightMapper().get_weight_groups(keys) is None
    assert not HfWeightMapper.single_tensor_groups_safe
    assert not _UnqualifiedHfWeightMapper.single_tensor_groups_safe
    assert default_mapper.get_weight_groups(keys) is None
    assert default_mapper.get_incremental_load_roots(keys) is None
    assert not default_mapper.borrowed_source_tensors_safe
    assert not _UnqualifiedHfWeightMapper().borrowed_source_tensors_safe
    assert qualified_mapper.get_weight_groups(keys) is not None
    assert qualified_mapper.borrowed_source_tensors_safe
    llama_mapper = Llama4HfWeightMapper()
    llama_mapper._config = _borrow_safe_model_config()
    assert llama_mapper.borrowed_source_tensors_safe
    assert _qwen3_5_mapper().borrowed_source_tensors_safe


@pytest.mark.parametrize(
    "config",
    [
        _borrow_safe_model_config(quant_config=SimpleNamespace(quant_algo="FP8")),
        _borrow_safe_model_config(
            quant_config_dict={"model.layers.0.mlp": SimpleNamespace(quant_algo="NVFP4")}
        ),
        _borrow_safe_model_config(force_dynamic_quantization=True),
        _borrow_safe_model_config(moe_load_balancer=SimpleNamespace(layer_updates_per_iter=1)),
    ],
)
def test_quantized_or_dynamic_profiles_stage_transport_source_tensors(config):
    generic_mapper = _qualified_generic_mapper()
    generic_mapper._config = config
    llama_mapper = Llama4HfWeightMapper()
    llama_mapper._config = config
    qwen_mapper = _qwen3_5_mapper()
    qwen_mapper._config = config

    assert not generic_mapper.borrowed_source_tensors_safe
    assert not llama_mapper.borrowed_source_tensors_safe
    assert not qwen_mapper.borrowed_source_tensors_safe


def test_unqualified_vlm_partial_signature_does_not_enable_generic_streaming():
    mapper = HfWeightMapper()
    mapper._model = _Step3StyleUnqualifiedVlm()
    keys = [
        "vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_model.encoder.layers.1.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
    ]

    assert mapper.get_weight_groups(keys) is None
    assert mapper.get_incremental_load_roots(keys) is None
    assert not mapper.borrowed_source_tensors_safe


def test_derived_architecture_does_not_inherit_generic_qualification():
    mapper = HfWeightMapper()
    mapper._model = _DerivedGenericModel()
    keys = ["model.layers.0.self_attn.q_proj.weight"]

    assert mapper.get_weight_groups(keys) is None
    assert mapper.get_incremental_load_roots(keys) is None
    assert not mapper.borrowed_source_tensors_safe


def test_llama4_groups_vision_atomically_and_text_by_layer():
    keys = [
        "vision_model.patch_embedding.weight",
        "multi_modal_projector.linear_1.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.1.feed_forward.gate_proj.weight",
        "language_model.lm_head.weight",
    ]

    groups = Llama4HfWeightMapper().get_weight_groups(keys)

    assert groups is not None
    by_id = {group.group_id: set(group.keys) for group in groups}
    assert by_id["llama4.vision_and_projector"] == set(keys[:2])
    assert by_id["llama4.model.layers.0.self_attn.qkv_proj"] == set(keys[3:5])
    assert by_id["llama4.model.layers.1.feed_forward.gate_up_proj"] == {keys[5]}
    flattened_keys = [key for group in groups for key in group.keys]
    assert set(flattened_keys) == set(keys)
    assert len(flattened_keys) == len(keys)


def test_llama4_routed_moe_keeps_complete_mlp_atomic():
    mapper = Llama4HfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(text_config=SimpleNamespace(num_local_experts=16))
    )
    keys = [
        "language_model.model.layers.0.feed_forward.experts.gate_up_proj",
        "language_model.model.layers.0.feed_forward.experts.down_proj",
        "language_model.model.layers.0.feed_forward.router.weight",
        "language_model.model.layers.0.feed_forward.shared_expert.gate_proj.weight",
        "language_model.model.layers.0.feed_forward.shared_expert.up_proj.weight",
        "language_model.model.layers.0.feed_forward.shared_expert.down_proj.weight",
    ]

    groups = mapper.get_weight_groups(keys)

    assert groups is not None
    assert len(groups) == 1
    assert groups[0].group_id == "llama4.model.layers.0.feed_forward"
    assert groups[0].keys == tuple(keys)


class _RecordingFusedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))
        self.calls = 0

    def load_weights(self, weights, allow_partial_loading=False):
        del weights, allow_partial_loading
        self.calls += 1


class _NoQuantMode:
    @staticmethod
    def has_nvfp4():
        return False


class _SyntheticFusedLinear(nn.Module):
    def __init__(self, output_rows: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_rows, 2))
        self.quant_config = SimpleNamespace(quant_mode=_NoQuantMode())

    def load_weights(self, weights, allow_partial_loading=False):
        if not allow_partial_loading:
            assert all("weight" in source_weights for source_weights in weights)
        present = [
            source_weights["weight"] for source_weights in weights if "weight" in source_weights
        ]
        if present:
            self.weight.data.copy_(torch.cat(present))


class _SyntheticDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Module()
        self.input_layernorm.weight = nn.Parameter(torch.empty(2))
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = _SyntheticFusedLinear(6)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = _SyntheticFusedLinear(4)


class _SyntheticDecoder(nn.Module):
    _supports_generic_hf_incremental_loading = True

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Module()
        self.model.embed_tokens.weight = nn.Parameter(torch.empty(3, 2))
        self.model.layers = nn.ModuleList([_SyntheticDecoderLayer(), _SyntheticDecoderLayer()])
        self.model.norm = nn.Module()
        self.model.norm.weight = nn.Parameter(torch.empty(2))
        self.lm_head = nn.Module()
        self.lm_head.weight = nn.Parameter(torch.empty(3, 2))
        self.config = SimpleNamespace(
            num_key_value_heads=1,
            num_attention_heads=1,
            tie_word_embeddings=False,
        )


def _generic_synthetic_weights() -> dict[str, torch.Tensor]:
    weights = {
        "model.embed_tokens.weight": torch.arange(6).reshape(3, 2),
        "model.norm.weight": torch.arange(2) + 100,
        "lm_head.weight": torch.arange(6).reshape(3, 2) + 110,
    }
    for layer_index in range(2):
        prefix = f"model.layers.{layer_index}"
        offset = layer_index * 20
        weights[f"{prefix}.input_layernorm.weight"] = torch.arange(2) + offset
        for projection_index, projection in enumerate(("q_proj", "k_proj", "v_proj")):
            weights[f"{prefix}.self_attn.{projection}.weight"] = (
                torch.arange(4).reshape(2, 2) + offset + projection_index * 4
            )
        for projection_index, projection in enumerate(("gate_proj", "up_proj")):
            weights[f"{prefix}.mlp.{projection}.weight"] = (
                torch.arange(4).reshape(2, 2) + offset + projection_index * 4 + 12
            )
    return weights


def _generic_mapper_for(model: _SyntheticDecoder) -> HfWeightMapper:
    mapper = HfWeightMapper()
    mapper._model = model
    mapper._tp_size = 1
    mapper.map_weights()
    return mapper


def test_generic_full_and_grouped_model_loading_match(monkeypatch):
    monkeypatch.setenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "1")
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    weights = _generic_synthetic_weights()

    full_model = _SyntheticDecoder()
    full_mapper = _generic_mapper_for(full_model)
    _load_weights_impl_v2(full_model, weights, full_mapper)

    grouped_model = _SyntheticDecoder()
    grouped_mapper = _generic_mapper_for(grouped_model)
    groups = grouped_mapper.get_weight_groups(weights)
    assert groups is not None
    for group in groups:
        _load_weights_impl_v2(
            grouped_model,
            {key: weights[key] for key in group.keys},
            grouped_mapper,
            allow_partial_loading=True,
        )

    full_params = dict(full_model.named_parameters())
    grouped_params = dict(grouped_model.named_parameters())
    assert grouped_params.keys() == full_params.keys()
    for name in full_params:
        torch.testing.assert_close(grouped_params[name], full_params[name])


def test_partial_dispatch_avoids_full_model_walk_and_thread_pool(monkeypatch):
    model = _SyntheticDecoder()
    mapper = _generic_mapper_for(model)
    weight = torch.arange(2) + 40

    def fail_full_walk(*args, **kwargs):
        del args, kwargs
        raise AssertionError("partial loading walked the full model")

    def fail_thread_pool(*args, **kwargs):
        del args, kwargs
        raise AssertionError("partial loading created a concurrent executor")

    monkeypatch.delenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", raising=False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(model, "named_modules", fail_full_walk)
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_utils.run_concurrently",
        fail_thread_pool,
    )

    _load_weights_impl_v2(
        model,
        {"model.layers.1.input_layernorm.weight": weight},
        mapper,
        allow_partial_loading=True,
    )

    torch.testing.assert_close(model.model.layers[1].input_layernorm.weight, weight)


class _FusedMapper(HfWeightMapper):
    def apply_callbacks(self, module, module_name, module_names_breakdown, weights):
        del module, module_name
        return [
            self.filter_weights(".".join(module_names_breakdown + [source_name]), weights)
            for source_name in ("q_proj", "k_proj", "v_proj")
        ]


def test_partial_loading_skips_absent_fused_modules_without_weakening_full_load(monkeypatch):
    model = nn.Module()
    model.qkv_proj = _RecordingFusedModule()
    model.config = SimpleNamespace(tie_word_embeddings=False)
    mapper = _FusedMapper()
    mapper._model = model
    mapper._mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    monkeypatch.setenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "1")
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)

    unrelated_weights = {"other.weight": torch.ones(1)}
    _load_weights_impl_v2(model, unrelated_weights, mapper, allow_partial_loading=True)
    assert model.qkv_proj.calls == 0

    _load_weights_impl_v2(model, unrelated_weights, mapper, allow_partial_loading=False)
    assert model.qkv_proj.calls == 1
