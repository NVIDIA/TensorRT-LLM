# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the transformers AutoConfig / AutoTokenizer dispatch
for TRT-LLM-only model_types (cosmos3, deepseek_v32, kimi_k2, deepseek_v41).

Before this registration, transformers >= 5.5 falls back to a bare
PreTrainedConfig that lacks `max_position_embeddings`, and
AutoTokenizer.from_pretrained then raises AttributeError on it. The
broken test that motivated this — perf/test_perf_sanity.py disagg gen_only
on GB200 — only runs in L0_PostMerge because it needs 12 GB200 GPUs across
3 nodes, so a cheap pre-merge unit test is the right place to catch
regressions.
"""

import json

import pytest
from transformers import AutoConfig

import tensorrt_llm  # noqa: F401  triggers AutoConfig registration
from tensorrt_llm._torch.configs import (
    Cosmos3Config,
    Cosmos3EdgeConfig,
    Cosmos3EdgeProjectorConfig,
    Cosmos3EdgeTextConfig,
    Cosmos3EdgeVisionConfig,
    DeepseekV3Config,
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)

pytestmark = pytest.mark.cpu_only


_COSMOS3_EDGE_MIN_CONFIG = {
    "architectures": ["Cosmos3EdgeForConditionalGeneration"],
    "model_type": "cosmos3_edge",
    "image_token_id": 19,
    "video_token_id": 18,
    "vision_start_token_id": 20,
    "vision_end_token_id": 21,
    "tie_word_embeddings": False,
    "projector_config": {
        "model_type": "cosmos3_edge_projector",
        "input_hidden_size": 1152,
        "merger_intermediate_size": 11520,
        "out_hidden_size": 2048,
        "spatial_merge_size": 2,
        "use_postshuffle_norm": False,
    },
    "text_config": {
        "model_type": "cosmos3_edge_text",
        "hidden_size": 2048,
        "intermediate_size": 9216,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_act": "relu2",
        "max_position_embeddings": 131072,
        "vocab_size": 131072,
        "rms_norm_eps": 1e-5,
        "dtype": "bfloat16",
        "rope_parameters": {
            "mrope_section": [24, 20, 20],
            "rope_theta": 100000000,
            "rope_type": "default",
        },
    },
    "vision_config": {
        "model_type": "cosmos3_edge_vision",
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_patches": 256,
        "patch_size": 16,
        "hidden_act": "gelu_pytorch_tanh",
        "layer_norm_eps": 1e-6,
    },
}


_COSMOS3_MIN_CONFIG = {
    "architectures": ["Cosmos3ForConditionalGeneration"],
    "model_type": "cosmos3",
    "model": {"_target": "omni_mot_model"},
    "text_config": {
        "model_type": "qwen3_vl_text",
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "max_position_embeddings": 8192,
        "vocab_size": 151936,
    },
    "vision_config": {
        "model_type": "qwen3_vl",
        "depth": 2,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_heads": 4,
        "out_hidden_size": 1024,
    },
}


def _deepseek_min_config(model_type: str) -> dict:
    return {
        "model_type": model_type,
        "max_position_embeddings": 16384,
    }


# DeepSeek-V4.1 nests its text tower, so `max_position_embeddings` lives one level down
# and the shared assertion below reads it off the composite. That is deliberate: it makes
# this test also cover the composite's attribute forwarding, which is the mechanism
# `modeling_deepseekv4.py`'s `config.<field>` read sites depend on.
_DEEPSEEK_V41_MIN_CONFIG = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "model_type": "deepseek_v41",
    "text_config": {
        "model_type": "deepseek_v41_text",
        "max_position_embeddings": 16384,
    },
    "vision_config": {"model_type": "deepseek_v41_vision"},
}


def _verify_autoconfig_from_pretrained(cfg, model_type: str, config_cls) -> None:
    assert isinstance(cfg, config_cls)
    assert cfg.model_type == model_type

    if model_type == "cosmos3":
        assert not isinstance(cfg.text_config, dict)
        assert not isinstance(cfg.vision_config, dict)
        assert cfg.text_config.hidden_size == 1024
        assert cfg.text_config.max_position_embeddings == 8192
        assert cfg.vision_config.hidden_size == 256
        assert cfg.vision_config.out_hidden_size == 1024
    elif model_type == "cosmos3_edge":
        assert cfg.text_config.hidden_act == "relu2"
        assert cfg.text_config.rope_scaling["mrope_interleaved"] is True
        assert cfg.text_config.rope_scaling["type"] == "mrope"
        assert cfg.text_config.rope_scaling["mrope_section"] == [24, 20, 20]
        assert cfg.projector_config.merger_intermediate_size == 11520
        assert cfg.vision_config.out_hidden_size == 2048
        assert cfg.vision_config.temporal_patch_size == 1
        assert cfg.vision_config.num_heads == 16
    elif model_type == "deepseek_v41":
        assert not isinstance(cfg.text_config, dict)
        assert not isinstance(cfg.vision_config, dict)
        assert cfg.text_config.model_type == "deepseek_v41_text"
        assert cfg.text_config.max_position_embeddings == 16384
        # Read off the *composite*: the field only exists on the text tower, so this
        # passes only if attribute forwarding is wired.
        assert cfg.max_position_embeddings == 16384
    else:
        assert cfg.max_position_embeddings == 16384


@pytest.mark.parametrize(
    ("model_type", "config_cls"),
    [
        ("cosmos3", Cosmos3Config),
        ("cosmos3_omni", Cosmos3Config),  # backward-compat alias
        ("cosmos3_edge", Cosmos3EdgeConfig),
        ("cosmos3_edge_text", Cosmos3EdgeTextConfig),
        ("cosmos3_edge_vision", Cosmos3EdgeVisionConfig),
        ("cosmos3_edge_projector", Cosmos3EdgeProjectorConfig),
        ("deepseek_v32", DeepseekV3Config),
        ("kimi_k2", DeepseekV3Config),
        # The composite is covered by from_pretrained below; check subtypes directly.
        ("deepseek_v41_text", DeepseekV41TextConfig),
        ("deepseek_v41_vision", DeepseekV41VisionConfig),
    ],
)
def test_custom_model_type_registered_with_autoconfig(model_type, config_cls):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    assert model_type in CONFIG_MAPPING
    assert CONFIG_MAPPING[model_type] is config_cls


@pytest.mark.parametrize(
    ("model_type", "config_cls", "config_dict"),
    [
        ("cosmos3", Cosmos3Config, _COSMOS3_MIN_CONFIG),
        ("cosmos3_edge", Cosmos3EdgeConfig, _COSMOS3_EDGE_MIN_CONFIG),
        ("deepseek_v32", DeepseekV3Config, _deepseek_min_config("deepseek_v32")),
        ("kimi_k2", DeepseekV3Config, _deepseek_min_config("kimi_k2")),
        ("deepseek_v41", DeepseekV41Config, _DEEPSEEK_V41_MIN_CONFIG),
    ],
)
def test_autoconfig_from_pretrained_resolves_to_local_config(
    tmp_path, model_type, config_cls, config_dict
):
    # Mirrors what the benchmark_serving subprocess does under the hood:
    # AutoTokenizer.from_pretrained -> AutoConfig.from_pretrained. Without
    # the registration this fails through to a bare PreTrainedConfig that
    # lacks expected fields (e.g. `max_position_embeddings`, nested
    # `text_config` for Cosmos3).
    model_dir = tmp_path / model_type
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config_dict))

    cfg = AutoConfig.from_pretrained(str(model_dir))
    _verify_autoconfig_from_pretrained(cfg, model_type, config_cls)
    if model_type in ("cosmos3", "cosmos3_edge"):
        assert cfg._name_or_path == str(model_dir)


def test_legacy_cosmos3_omni_model_type_still_resolves(tmp_path):
    # "cosmos3_omni" is the pre-rename model_type. Checkpoints created before
    # the rename to "cosmos3" must keep loading via the backward-compat alias.
    legacy_config = {**_COSMOS3_MIN_CONFIG, "model_type": "cosmos3_omni"}
    model_dir = tmp_path / "cosmos3_omni_legacy"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(legacy_config))

    cfg = AutoConfig.from_pretrained(str(model_dir))
    assert isinstance(cfg, Cosmos3Config)
    assert not isinstance(cfg.text_config, dict)
    assert cfg.text_config.hidden_size == 1024


@pytest.mark.parametrize("model_type", ["deepseek_v32", "kimi_k2"])
def test_load_hf_model_config_uses_autoconfig_dispatch(tmp_path, model_type):
    # ModelLoader.load_hf_model_config is the llmapi/llm_utils entry point used
    # by trtllm-serve to pre-load HF model configs. On transformers 5.5.x it
    # must dispatch via AutoConfig (which CONFIG_MAPPING.register affects), not
    # directly via PretrainedConfig.from_pretrained — the latter bypasses the
    # mapping and returns a bare PretrainedConfig without
    # `max_position_embeddings`, causing downstream AttributeError on the V3.2
    # disagg gen_only GB200 post-merge perf-sanity test.
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    model_dir = tmp_path / model_type
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "max_position_embeddings": 16384,
            }
        )
    )

    cfg = ModelLoader.load_hf_model_config(str(model_dir))
    assert cfg is not None
    assert isinstance(cfg, DeepseekV3Config)
    assert cfg.max_position_embeddings == 16384
