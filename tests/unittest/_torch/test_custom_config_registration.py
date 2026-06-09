# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the transformers AutoConfig / AutoTokenizer dispatch
for TRT-LLM-only model_types (cosmos3, deepseek_v32, kimi_k2).

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

import tensorrt_llm  # noqa: F401  triggers AutoConfig registration
from tensorrt_llm._torch.configs import Cosmos3Config, DeepseekV3Config

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
    else:
        assert cfg.max_position_embeddings == 16384


@pytest.mark.parametrize(
    ("model_type", "config_cls"),
    [
        ("cosmos3", Cosmos3Config),
        ("cosmos3_omni", Cosmos3Config),  # backward-compat alias
        ("deepseek_v32", DeepseekV3Config),
        ("kimi_k2", DeepseekV3Config),
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
        ("deepseek_v32", DeepseekV3Config, _deepseek_min_config("deepseek_v32")),
        ("kimi_k2", DeepseekV3Config, _deepseek_min_config("kimi_k2")),
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
    from transformers import AutoConfig

    model_dir = tmp_path / model_type
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config_dict))

    cfg = AutoConfig.from_pretrained(str(model_dir))
    _verify_autoconfig_from_pretrained(cfg, model_type, config_cls)
    if model_type == "cosmos3":
        assert cfg._name_or_path == str(model_dir)


def test_legacy_cosmos3_omni_model_type_still_resolves(tmp_path):
    # "cosmos3_omni" is the pre-rename model_type. Checkpoints created before
    # the rename to "cosmos3" must keep loading via the backward-compat alias.
    from transformers import AutoConfig

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
