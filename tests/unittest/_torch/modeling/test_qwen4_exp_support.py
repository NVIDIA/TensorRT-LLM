# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _text_config_dict() -> dict:
    return {
        "model_type": "qwen4_exp_text",
        "architectures": ["Qwen4ExpForCausalLM"],
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "full_attention_interval": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "mamba_ssm_dtype": "float32",
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "hidden_act": "silu",
        "hc_count": 4,
        "hc_lowrank": 32,
        "ple_layer_ids": [2],
        "ple_embed_dim": 128,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 2048,
        "make_ngram_vocab_size_divisible_by": 128,
        "split_ngram_parts": 128,
        "output_gate_type": "sigmoid",
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 32,
        "indexer_budget": 64,
        "indexer_compress_ratio": 4,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [3, 3, 2],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default",
        },
    }


def test_config_types_are_registered_with_transformers() -> None:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    import tensorrt_llm._torch.configs  # noqa: F401
    from tensorrt_llm._torch.configs import Qwen4ExpConfig, Qwen4ExpTextConfig, Qwen4ExpVisionConfig

    assert CONFIG_MAPPING["qwen4_exp"] is Qwen4ExpConfig
    assert CONFIG_MAPPING["qwen4_exp_text"] is Qwen4ExpTextConfig
    assert CONFIG_MAPPING["qwen4_exp_vision"] is Qwen4ExpVisionConfig


def test_config_normalizes_hf_qsa_layer_alias() -> None:
    from tensorrt_llm._torch.configs import Qwen4ExpTextConfig

    fields = _text_config_dict()
    fields["layer_types"][-1] = "deepseek_sparse_attention"
    config = Qwen4ExpTextConfig.from_dict(fields)

    assert config.layer_types[-1] == "full_attention"


def test_language_only_config_flattens_to_text_without_remote_code(tmp_path) -> None:
    from tensorrt_llm._torch.configs import Qwen4ExpTextConfig
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    config_dict = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "language_model_only": True,
        "text_config": _text_config_dict(),
        "vision_config": {
            "model_type": "qwen4_exp_vision",
            "depth": 2,
            "hidden_size": 64,
            "num_heads": 4,
            "out_hidden_size": 128,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config_dict))

    config = load_pretrained_config(str(tmp_path))

    assert isinstance(config, Qwen4ExpTextConfig)
    assert config.architectures == ["Qwen4ExpForCausalLM"]


def test_composite_config_preserves_vision_without_remote_code(tmp_path) -> None:
    from tensorrt_llm._torch.configs import Qwen4ExpConfig, Qwen4ExpTextConfig, Qwen4ExpVisionConfig
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    config_dict = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "text_config": _text_config_dict(),
        "vision_config": {
            # Match the early checkpoint spelling normalized by the adapter.
            "model_type": "qwen4_exp",
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 128,
            "deepstack_visual_indexes": [],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config_dict))

    config = load_pretrained_config(str(tmp_path))

    assert isinstance(config, Qwen4ExpConfig)
    assert isinstance(config.text_config, Qwen4ExpTextConfig)
    assert isinstance(config.vision_config, Qwen4ExpVisionConfig)
    assert config.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert config.text_config.architectures == ["Qwen4ExpForCausalLM"]
    assert config.vision_config.model_type == "qwen4_exp_vision"
    assert config.vision_config.out_hidden_size == config.text_config.hidden_size


def test_vision_attention_does_not_inherit_text_sparse_config(monkeypatch) -> None:
    from tensorrt_llm._torch.models import modeling_qwen3vl
    from tensorrt_llm._torch.models.modeling_utils import ModelConfig

    captured = {}

    def mock_parent_init(self, model_config, *, layer_idx, reduce_output):
        captured["model_config"] = model_config
        captured["layer_idx"] = layer_idx
        captured["reduce_output"] = reduce_output

    monkeypatch.setattr(
        modeling_qwen3vl.Qwen2_5_VLVisionAttention,
        "__init__",
        mock_parent_init,
    )
    sparse_config = object()
    pretrained_config = SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=SimpleNamespace(
            max_position_embeddings=4096,
            dtype=torch.bfloat16,
        ),
        vision_config=SimpleNamespace(),
    )
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        sparse_attention_config=sparse_config,
    )

    modeling_qwen3vl.Qwen3VLVisionAttention(model_config, layer_idx=3)

    assert captured["model_config"] is not model_config
    assert captured["model_config"].sparse_attention_config is None
    assert model_config.sparse_attention_config is sparse_config
    assert captured["layer_idx"] == 3
    assert captured["reduce_output"] is False


def test_text_model_registration_and_defaults() -> None:
    from tensorrt_llm._torch.models.modeling_qwen4_exp import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class

    assert get_registered_model_class("Qwen4ExpForCausalLM") is Qwen4ExpForCausalLM
    assert (
        get_registered_model_class("Qwen4ExpForConditionalGeneration")
        is Qwen4ExpForConditionalGeneration
    )
    defaults = Qwen4ExpForCausalLM.get_model_defaults(None)
    assert defaults["sparse_attention_config"] == {"algorithm": "qsa"}
    assert defaults["kv_cache_config"]["enable_block_reuse"] is False
    assert "moe_config" not in defaults
    assert "allreduce_strategy" not in defaults
    assert Qwen4ExpForCausalLM.get_preferred_kv_cache_manager_version() == "V2"


def test_local_multimodal_embedding_is_not_treated_as_encoder_handoff() -> None:
    from tensorrt_llm._torch.models.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
    from tensorrt_llm.inputs.multimodal import MultimodalParams

    model = object.__new__(Qwen4ExpForConditionalGeneration)
    nn.Module.__init__(model)
    model.mm_encoder = nn.Identity()
    param = MultimodalParams(multimodal_data={"multimodal_embedding": torch.empty(1, 128)})

    assert model.select_multimodal_params([param], 1) == [param]


def test_multimodal_embedding_without_local_encoder_requires_handoff_support() -> None:
    from tensorrt_llm._torch.models.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
    from tensorrt_llm.inputs.multimodal import MultimodalParams

    model = object.__new__(Qwen4ExpForConditionalGeneration)
    nn.Module.__init__(model)
    model.mm_encoder = None
    param = MultimodalParams(multimodal_data={"multimodal_embedding": torch.empty(1, 128)})

    with pytest.raises(NotImplementedError, match="does not support disaggregated inference"):
        model.select_multimodal_params([param], 1)


def test_text_model_is_eligible_for_online_eplb() -> None:
    from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import moe_model_arch_list

    assert "Qwen4ExpForCausalLM" in moe_model_arch_list
    assert "Qwen4ExpForConditionalGeneration" in moe_model_arch_list


def test_hybrid_and_ple_layout_is_derived_from_config() -> None:
    from tensorrt_llm._torch.configs import Qwen4ExpTextConfig
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        extract_mamba_kv_cache_params,
        extract_qwen4_exp_ple_cache_params,
        get_qwen3_hybrid_layer_types,
    )

    config = Qwen4ExpTextConfig.from_dict(_text_config_dict())
    assert get_qwen3_hybrid_layer_types(config) == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    mamba = extract_mamba_kv_cache_params(config)
    assert mamba.num_mamba_layers == 3
    ple = extract_qwen4_exp_ple_cache_params(config)
    assert ple.ple_layer_mask == [False, True, False, False]
    assert ple.short_conv_channels == 4 * 128
    assert ple.short_conv_state_len == 9
    assert ple.ngram_context_len == 2


def test_ple_cache_layout_excludes_separate_mtp_draft() -> None:
    from tensorrt_llm._torch.configs import Qwen4ExpTextConfig
    from tensorrt_llm._torch.pyexecutor._util import _get_qwen4_exp_ple_cache_params

    config = Qwen4ExpTextConfig.from_dict(_text_config_dict())

    target = _get_qwen4_exp_ple_cache_params(config, total_layers=4, is_draft=False)
    unified = _get_qwen4_exp_ple_cache_params(config, total_layers=5, is_draft=False)
    draft = _get_qwen4_exp_ple_cache_params(config, total_layers=5, is_draft=True)

    assert target.ple_layer_mask == [False, True, False, False]
    assert unified.ple_layer_mask == [False, True, False, False, False]
    assert unified.num_ple_layers == 1
    assert draft is None


def test_v2_cache_estimator_counts_ple_lifecycle_state() -> None:
    from tensorrt_llm._torch.configs import Qwen4ExpTextConfig
    from tensorrt_llm._torch.pyexecutor.config_utils import extract_qwen4_exp_ple_cache_params
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    config = Qwen4ExpTextConfig.from_dict(_text_config_dict())
    no_ple_config = deepcopy(config)
    no_ple_config.ple_layer_ids = []
    common = {
        "mapping": Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        "max_batch_size": 2,
        "kv_cache_config": KvCacheConfig(enable_block_reuse=False),
    }
    with_ple = MambaHybridCacheManagerV2.get_cache_size_per_token(
        SimpleNamespace(pretrained_config=config, quant_config=None), **common
    )
    without_ple = MambaHybridCacheManagerV2.get_cache_size_per_token(
        SimpleNamespace(pretrained_config=no_ple_config, quant_config=None), **common
    )

    ple = extract_qwen4_exp_ple_cache_params(config)
    bytes_per_slot = (
        ple.short_conv_channels * ple.short_conv_state_len * ple.conv_state_dtype.itemsize
        + ple.ngram_context_len * torch.int64.itemsize
    )
    assert with_ple[0] == without_ple[0]
    # Two live request slots plus one non-speculative CUDA-graph dummy slot.
    assert with_ple[1] - without_ple[1] == 3 * bytes_per_slot


def test_ple_states_use_v2_lifecycle_buffers(monkeypatch) -> None:
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
        MambaRole,
    )

    ngram_context = torch.full((12, 2), 11, dtype=torch.int64)
    conv_state = torch.zeros((12, 16, 6), dtype=torch.bfloat16)
    requested = []

    def fake_get_state_buffer(self, local_layer_idx, role, dtype, state_shape):
        del self
        requested.append((local_layer_idx, role, dtype, state_shape))
        if role == MambaRole.PLE_NGRAM_CONTEXT:
            return ngram_context
        if role == MambaRole.PLE_CONV_STATE:
            return conv_state
        raise AssertionError(f"unexpected role {role}")

    monkeypatch.setattr(MambaHybridCacheManagerV2, "_get_state_buffer", fake_get_state_buffer)
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager._ple_layer_ids = [1]
    manager._ple_ngram_context_shape = [2]
    manager._ple_conv_state_shape = [16, 6]
    manager._ple_conv_state_dtype = torch.bfloat16
    manager._ple_ngram_contexts = {}
    manager._ple_conv_states = {}
    manager.layer_offsets = {1: 0}

    manager._setup_ple_states(num_state_slots=12)

    actual_conv, actual_context = manager.ple_layer_cache(1)
    assert actual_conv is conv_state
    assert actual_context is ngram_context
    assert requested == [
        (0, MambaRole.PLE_CONV_STATE, torch.bfloat16, [16, 6]),
        (0, MambaRole.PLE_NGRAM_CONTEXT, torch.int64, [2]),
    ]


def test_attention_dp_does_not_enable_tp_output_reduction() -> None:
    from tensorrt_llm._torch.models.modeling_qwen4_exp import _qwen4_exp_tp_output_reduction_enabled

    assert not _qwen4_exp_tp_output_reduction_enabled(
        SimpleNamespace(tp_size=1, enable_attention_dp=False)
    )
    assert _qwen4_exp_tp_output_reduction_enabled(
        SimpleNamespace(tp_size=4, enable_attention_dp=False)
    )
    assert not _qwen4_exp_tp_output_reduction_enabled(
        SimpleNamespace(tp_size=4, enable_attention_dp=True)
    )


def test_mapper_normalizes_bf16_and_per_expert_fp8_weights() -> None:
    from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import (
        _normalize_moe_module_weights,
        _rank_block,
    )
    from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode

    q = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    z = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
    blocked = _rank_block([q, z], tp_size=2)
    expected = torch.cat((q[:2], z[:2], q[2:], z[2:]))
    torch.testing.assert_close(blocked, expected)

    config = SimpleNamespace(hidden_size=4, moe_intermediate_size=3)
    fused, mode = _normalize_moe_module_weights(
        {
            "gate_up_proj": torch.randn(2, 6, 4),
            "down_proj": torch.randn(2, 4, 3),
        },
        config,
    )
    assert mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ
    assert fused["gate_up_proj"].shape == (2, 4, 6)
    assert fused["down_proj"].shape == (2, 3, 4)

    per_expert, mode = _normalize_moe_module_weights(
        {
            "0.gate_proj.weight": torch.empty(1),
            "0.gate_proj.weight_scale_inv": torch.empty(1),
            "0.up_proj.weight": torch.empty(1),
            "0.down_proj.weight": torch.empty(1),
        },
        config,
    )
    assert mode == MoEWeightLoadingMode.VANILLA
    assert set(per_expert) == {
        "0.w1.weight",
        "0.w1.weight_scale_inv",
        "0.w3.weight",
        "0.w2.weight",
    }


def test_mapper_streams_only_local_ple_row_overlap(monkeypatch) -> None:
    from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import (
        Qwen4ExpHfWeightMapper,
    )

    module = nn.Module()
    module.padded_vocab_size = 10
    module.vocab_start_index = 3
    module.vocab_end_index = 8
    module.ngram_embedding = nn.Embedding(5, 2)
    with torch.no_grad():
        module.ngram_embedding.weight.fill_(-1)

    mapper = Qwen4ExpHfWeightMapper()
    monkeypatch.setattr(mapper, "_ngram_module_for_prefix", lambda _prefix: module)
    full_table = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    leaves = {
        "ngram_embedding.shard_0.weight": full_table[:4],
        "ngram_embedding.shard_1.weight": full_table[4:],
    }

    table_ptr = module.ngram_embedding.weight.data_ptr()
    mapper._load_ngram_tables({"model.layers.1.ple": leaves})

    assert module.ngram_embedding.weight.data_ptr() == table_ptr
    torch.testing.assert_close(module.ngram_embedding.weight, full_table[3:8])


def test_mapper_keeps_fp8_ple_table_quantized(monkeypatch) -> None:
    from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import (
        Qwen4ExpHfWeightMapper,
    )
    from tensorrt_llm._torch.modules.qwen4_exp_ple import Qwen4ExpNGramEmbedding

    config = SimpleNamespace(
        ngram_size=2,
        heads_per_ngram=1,
        vocab_size=16,
        eos_token_id=2,
        seed=1234,
        ngram_vocab_size_base=3,
        make_ngram_vocab_size_divisible_by=4,
        quantization_config={
            "quant_method": "fp8",
            "modules_to_not_convert": ["model.language_model.layers.1.ple.key_proj"],
        },
    )
    module = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=2,
        dtype=torch.bfloat16,
    )
    assert module.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    excluded_config = SimpleNamespace(**vars(config))
    excluded_config.quantization_config = {
        "quant_method": "fp8",
        "modules_to_not_convert": [
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"
        ],
    }
    excluded_module = Qwen4ExpNGramEmbedding(
        excluded_config,
        embedding_dim=2,
        dtype=torch.bfloat16,
    )
    assert excluded_module.ngram_embedding.weight.dtype == torch.bfloat16

    mapper = Qwen4ExpHfWeightMapper()
    monkeypatch.setattr(mapper, "_ngram_module_for_prefix", lambda _prefix: module)
    fp8_table = torch.tensor(
        [[-48.0, 72.0], [-80.0, 64.0], [-36.0, 36.0], [-26.0, 30.0]],
        dtype=torch.float8_e4m3fn,
    )
    scale = torch.tensor([0.0002], dtype=torch.bfloat16)
    leaves = {
        "ngram_embedding.shard_0.weight": fp8_table[:2],
        "ngram_embedding.shard_1.weight": fp8_table[2:],
        "ngram_embedding.weight_scale": scale,
    }

    mapper._load_ngram_tables({"model.layers.1.ple": leaves})

    assert module.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    assert module.ngram_embedding.weight.element_size() == 1
    torch.testing.assert_close(module.ngram_embedding.weight, fp8_table)
    expected = (fp8_table.float() * scale.item()).to(torch.bfloat16)
    torch.testing.assert_close(module.embed(torch.arange(4)), expected)


def test_pipeline_mapper_drops_nonlocal_layer_weights() -> None:
    from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import (
        Qwen4ExpHfWeightMapper,
    )

    local_layer = nn.Linear(1, 1, bias=False)
    remote_layer = nn.Linear(1, 1, bias=False)
    remote_layer._weights_removed = True
    fake_model = nn.Module()
    fake_model.model = nn.Module()
    fake_model.model.layers = nn.ModuleList((local_layer, remote_layer))

    mapper = Qwen4ExpHfWeightMapper()
    mapper._model = fake_model
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=2,
            linear_key_head_dim=4,
            linear_num_key_heads=1,
            linear_value_head_dim=4,
            linear_num_value_heads=1,
        ),
        mapping=SimpleNamespace(
            enable_attention_dp=False,
            tp_size=1,
            tp_rank=0,
            has_pp=lambda: True,
        ),
    )
    weights = {
        "model.language_model.layers.0.marker.weight": torch.ones(1),
        "model.language_model.layers.1.marker.weight": torch.full((1,), 2.0),
        "lm_head.weight": torch.full((1,), 3.0),
    }

    mapped = mapper.preprocess_weights(weights)

    assert set(mapped) == {"model.layers.0.marker.weight", "lm_head.weight"}


def test_mapper_packs_hc_down_and_injection_with_alignment() -> None:
    from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import (
        Qwen4ExpHfWeightMapper,
    )

    mapper = Qwen4ExpHfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=1,
            linear_key_head_dim=4,
            linear_num_key_heads=1,
            linear_value_head_dim=4,
            linear_num_value_heads=1,
        ),
        mapping=SimpleNamespace(
            enable_attention_dp=False,
            tp_size=1,
            tp_rank=0,
            has_pp=lambda: False,
        ),
        spec_config=None,
    )
    down = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    inject = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
    final_down = torch.full((6, 4), 7.0)
    weights = {
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight": down,
        "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight": inject,
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": final_down,
    }

    mapped = mapper.preprocess_weights(weights)

    packed_name = "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    assert mapped[packed_name].shape == (16, 4)
    torch.testing.assert_close(mapped[packed_name][:6], down)
    torch.testing.assert_close(mapped[packed_name][6:8], inject)
    torch.testing.assert_close(mapped[packed_name][8:], torch.zeros(8, 4))
    torch.testing.assert_close(
        mapped["model.hyper_connection_mixer.input_mix_weight_down.weight"],
        final_down,
    )


def test_mapper_packs_fused_hc_lowrank_padding_before_injection(monkeypatch) -> None:
    from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import (
        Qwen4ExpHfWeightMapper,
    )

    monkeypatch.setenv("TRTLLM_QWEN4_EXP_HC_FUSED_MIX", "1")
    mapper = Qwen4ExpHfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=1,
            linear_key_head_dim=4,
            linear_num_key_heads=1,
            linear_value_head_dim=4,
            linear_num_value_heads=1,
        ),
        mapping=SimpleNamespace(
            enable_attention_dp=False,
            tp_size=1,
            tp_rank=0,
            has_pp=lambda: False,
        ),
        spec_config=None,
    )
    down = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    inject = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
    weights = {
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight": down,
        "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight": inject,
    }

    mapped = mapper.preprocess_weights(weights)

    packed_name = "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    packed = mapped[packed_name]
    assert packed.shape == (144, 4)
    torch.testing.assert_close(packed[:6], down)
    torch.testing.assert_close(packed[6:128], torch.zeros(122, 4))
    torch.testing.assert_close(packed[128:130], inject)
    torch.testing.assert_close(packed[130:], torch.zeros(14, 4))


def test_mtp_checkpoint_names_map_to_recurrent_runtime_layer() -> None:
    from tensorrt_llm._torch.models.checkpoints.hf.qwen4_exp_weight_mapper import (
        Qwen4ExpHfWeightMapper,
    )

    class _MTPMode:
        @staticmethod
        def is_mtp_one_model() -> bool:
            return True

    mapper = Qwen4ExpHfWeightMapper()
    mapper._config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=48,
            linear_key_head_dim=4,
            linear_num_key_heads=2,
            linear_value_head_dim=4,
            linear_num_value_heads=2,
        ),
        mapping=SimpleNamespace(
            enable_attention_dp=False,
            tp_size=1,
            tp_rank=0,
            has_pp=lambda: False,
        ),
        spec_config=SimpleNamespace(spec_dec_mode=_MTPMode()),
    )
    weights = {
        "mtp.fc_embedding.weight": torch.ones(4, 4),
        "mtp.pre_fc_norm_hidden.weight": torch.ones(16),
        "mtp.hyper_connection_mixer.hc_norm.weight": torch.ones(16),
        "mtp.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
    }

    mapped = mapper.preprocess_weights(weights)

    assert set(mapped) == {
        "model.layers.48.fc_embedding.weight",
        "model.layers.48.pre_fc_norm_hidden.weight",
        "model.layers.48.shared_head.hyper_connection_mixer.hc_norm.weight",
        "model.layers.48.self_attn.o_proj.weight",
    }


@pytest.mark.parametrize("wrapped", [False, True])
def test_mtp_resource_hidden_size_includes_all_hc_streams(wrapped) -> None:
    from tensorrt_llm._torch.speculative.utils import get_mtp_hidden_size

    text_config = SimpleNamespace(
        model_type="qwen4_exp_text",
        hidden_size=2560,
        hc_count=4,
    )
    pretrained_config = (
        SimpleNamespace(model_type="qwen4_exp", text_config=text_config) if wrapped else text_config
    )
    model_config = SimpleNamespace(pretrained_config=pretrained_config)

    assert get_mtp_hidden_size(model_config) == 10240


def test_mtp_local_full_vocab_head_collapses_hc_streams(monkeypatch) -> None:
    from tensorrt_llm._torch.configs import Qwen4ExpTextConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_qwen4_exp
    from tensorrt_llm._torch.models.modeling_qwen4_exp import Qwen4ExpMTPHead
    from tensorrt_llm.mapping import Mapping

    config = Qwen4ExpTextConfig.from_dict(_text_config_dict())
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(
            world_size=4,
            rank=0,
            tp_size=4,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=True,
        ),
    )
    head = Qwen4ExpMTPHead(model_config)

    def unexpected_allgather(*args, **kwargs):
        del args, kwargs
        raise AssertionError("local full-vocabulary MTP logits must not all-gather")

    monkeypatch.setattr(modeling_qwen4_exp, "allgather", unexpected_allgather)

    class CaptureLMHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_shape = None

        def forward(self, hidden_states):
            self.input_shape = hidden_states.shape
            return hidden_states

    lm_head = CaptureLMHead()
    hidden_states = torch.zeros(
        2,
        config.hc_count * config.hidden_size,
        dtype=config.torch_dtype,
    )

    logits = head.forward_local_full_vocab(
        hidden_states, lm_head, attn_metadata=None, return_context_logits=True
    )

    assert lm_head.input_shape == (2, config.hidden_size)
    assert logits.shape == (2, config.hidden_size)


@pytest.mark.parametrize("draft_len", [3, 5, 7])
def test_mtp_uses_one_recurrent_checkpoint_layer(monkeypatch, draft_len) -> None:
    from tensorrt_llm._torch.models import modeling_qwen4_exp
    from tensorrt_llm._torch.models.modeling_speculative import MTPForCausalLM
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    created = []

    class _FakeQwen4ExpMTP(nn.Module):
        def __init__(self, model_config, layer_idx, aux_stream_dict):
            super().__init__()
            del model_config, aux_stream_dict
            self.layer_idx = layer_idx
            created.append(layer_idx)

    monkeypatch.setattr(modeling_qwen4_exp, "Qwen4ExpMTP", _FakeQwen4ExpMTP)
    spec_config = MTPDecodingConfig(max_draft_len=draft_len)
    config = SimpleNamespace(
        model_type="qwen4_exp_text",
        num_hidden_layers=48,
        num_nextn_predict_layers=1,
    )
    model_config = SimpleNamespace(
        pretrained_config=config,
        spec_config=spec_config,
    )
    model = SimpleNamespace(aux_stream_dict={}, embed_tokens=nn.Identity())

    draft_model = MTPForCausalLM(
        model_config,
        start_layer_idx=config.num_hidden_layers,
        lm_head=nn.Identity(),
        model=model,
    )

    assert spec_config.spec_dec_mode.is_mtp_eagle_one_model()
    assert len(draft_model.mtp_layers) == 1
    assert created == [config.num_hidden_layers]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PP construct smoke requires CUDA")
@pytest.mark.parametrize(
    "rank,owned_layers,owns_embedding,owns_epilogue",
    [
        (0, {0, 1}, True, False),
        (1, {2, 3}, False, True),
    ],
)
def test_pp2_stage_ownership_and_handoff_width(
    rank,
    owned_layers,
    owns_embedding,
    owns_epilogue,
) -> None:
    from tensorrt_llm._torch.configs import Qwen4ExpTextConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_qwen4_exp import Qwen4ExpForCausalLM
    from tensorrt_llm.mapping import Mapping

    torch.cuda.set_device(0)
    config = Qwen4ExpTextConfig.from_dict(_text_config_dict())
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(world_size=2, rank=rank, tp_size=1, pp_size=2),
        attn_backend="TRTLLM",
        moe_backend="CUTLASS",
    )
    with torch.device("cuda:0"):
        model = Qwen4ExpForCausalLM(model_config)

    assert bool(model.model.embed_tokens._parameters) is owns_embedding
    assert any(True for _ in model.model.hyper_connection_mixer.parameters()) is owns_epilogue
    assert bool(model.lm_head._parameters) is owns_epilogue
    assert model.model.has_ple is (rank == 0)
    for layer_index, layer in enumerate(model.model.layers[: config.num_hidden_layers]):
        has_parameters = any(True for _ in layer.parameters())
        assert has_parameters is (layer_index in owned_layers)

    if rank == 1:
        skipped_embedding = model.model.embed_tokens.skip_forward(torch.arange(3, device="cuda:0"))
        handoff = skipped_embedding.new_empty(skipped_embedding.shape[0], model.model.hc_dim)
        assert handoff.shape == (3, config.hc_count * config.hidden_size)
