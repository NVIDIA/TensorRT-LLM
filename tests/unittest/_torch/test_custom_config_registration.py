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
import subprocess
import sys
from pathlib import Path

import pytest
import torch
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


def test_edge_text_config_keeps_legacy_torch_dtype():
    cfg = Cosmos3EdgeTextConfig(torch_dtype="float16")
    assert cfg.dtype == torch.float16


def test_edge_rope_parameters_keep_checkpoint_keys():
    cfg = Cosmos3EdgeTextConfig(
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1e8,
            "factor": 2.0,
            "original_max_position_embeddings": 32768,
        }
    )
    assert cfg.rope_parameters["rope_type"] == "yarn"
    assert cfg.rope_parameters["factor"] == 2.0
    assert cfg.rope_parameters["original_max_position_embeddings"] == 32768
    assert cfg.rope_parameters["type"] == "mrope"
    assert cfg.rope_parameters["mrope_interleaved"] is True


def test_edge_rope_parameters_reject_unknown_type():
    with pytest.raises(ValueError, match="not supported"):
        Cosmos3EdgeTextConfig(rope_parameters={"rope_type": "custom_rope"})


def test_edge_vision_merge_must_match_projector():
    with pytest.raises(ValueError, match="spatial_merge_size"):
        Cosmos3EdgeConfig(
            vision_config={"spatial_merge_size": 4},
            projector_config={"spatial_merge_size": 2},
        )


def test_edge_from_dict_returns_unused_kwargs():
    config, unused = Cosmos3EdgeConfig.from_dict(
        {"model_type": "cosmos3_edge", "text_config": {"torch_dtype": "float16"}},
        return_unused_kwargs=True,
        _name_or_path="/tmp/cosmos3-edge",
        not_a_config_field="kept",
    )
    assert config.text_config.dtype == torch.float16
    assert config._name_or_path == "/tmp/cosmos3-edge"
    assert unused["not_a_config_field"] == "kept"


def test_edge_attn_implementation_reaches_sub_configs():
    cfg = Cosmos3EdgeConfig(attn_implementation="sdpa")
    assert cfg.text_config._attn_implementation == "sdpa"
    assert cfg.vision_config._attn_implementation == "sdpa"
    assert cfg.projector_config._attn_implementation == "sdpa"


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


# The published nvidia/NVIDIA-Nemotron-Nano-9B-v2 pattern: "-" marks MLP layers.
_NEMOTRON_H_DENSE_PATTERN = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"


def _nemotron_h_min_config(pattern: str) -> dict:
    return {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "hybrid_override_pattern": pattern,
        "num_hidden_layers": len(pattern),
        "hidden_size": 4480,
        "intermediate_size": 15680,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 131072,
        "vocab_size": 131072,
        "ssm_state_size": 128,
        "conv_kernel": 4,
        "chunk_size": 128,
        "n_groups": 8,
        "mamba_num_heads": 128,
        "mamba_head_dim": 80,
        "rms_norm_eps": 1e-5,
    }


def _write_config(tmp_path, name: str, config_dict: dict) -> str:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config_dict))
    return str(model_dir)


def test_nemotron_h_dense_pattern_survives_autoconfig(tmp_path):
    # transformers 5.5.x NemotronHConfig converts hybrid_override_pattern to
    # layers_block_type with a mapping that only knows "M"/"E"/"*", so every
    # dense Nemotron-H checkpoint raises `KeyError: '-'` inside
    # AutoConfig.from_pretrained (and therefore AutoTokenizer.from_pretrained).
    # TRT-LLM builds the model by iterating hybrid_override_pattern, so the
    # literal string has to survive config loading intact.
    from tensorrt_llm._torch.configs import NemotronHConfig

    model_dir = _write_config(
        tmp_path, "nemotron_h_dense", _nemotron_h_min_config(_NEMOTRON_H_DENSE_PATTERN)
    )

    cfg = AutoConfig.from_pretrained(model_dir)

    assert isinstance(cfg, NemotronHConfig)
    assert cfg.hybrid_override_pattern == _NEMOTRON_H_DENSE_PATTERN
    assert cfg.num_hidden_layers == len(_NEMOTRON_H_DENSE_PATTERN)
    # Layer routing in modeling_nemotron_h / model_config counts these chars.
    assert cfg.hybrid_override_pattern.count("*") == 4
    assert cfg.hybrid_override_pattern.count("M") == 27
    assert cfg.hybrid_override_pattern.count("-") == 25
    assert cfg.layers_block_type.count("mlp") == 25
    # Attributes modeling_nemotron_h reads off the pretrained config.
    assert cfg.ssm_state_size == 128
    assert cfg.mamba_head_dim == 80
    assert cfg.rms_norm_eps == 1e-5
    assert cfg.num_nextn_predict_layers == 0


def test_nemotron_h_representable_pattern_uses_native_config(tmp_path):
    # Patterns the installed transformers can express must keep loading through
    # the native class, so this shim cannot regress checkpoints that work today.
    from transformers.models.nemotron_h.configuration_nemotron_h import (
        NemotronHConfig as HFNemotronHConfig,
    )

    model_dir = _write_config(tmp_path, "nemotron_h_moe", _nemotron_h_min_config("M*E" * 4))

    cfg = AutoConfig.from_pretrained(model_dir)

    assert type(cfg) is HFNemotronHConfig
    assert cfg.hybrid_override_pattern == "M*E" * 4


def test_nemotron_h_invalid_pattern_rejected():
    from tensorrt_llm._torch.configs import NemotronHConfig

    with pytest.raises(ValueError, match="hybrid_override_pattern"):
        NemotronHConfig(
            hybrid_override_pattern="M*XZ-",
            layers_block_type=["mamba", "attention", "moe", "mlp", "mlp"],
        )


@pytest.mark.parametrize(
    "layers_block_type",
    [
        ["moe", "moe", "moe"],
        ["mamba"],
    ],
)
def test_nemotron_h_pattern_overrides_explicit_block_types(layers_block_type):
    from tensorrt_llm._torch.configs import NemotronHConfig

    config = NemotronHConfig(hybrid_override_pattern="M-*", layers_block_type=layers_block_type)

    assert config.layers_block_type == ["mamba", "mlp", "attention"]
    assert len(config.layers_block_type) == config.num_hidden_layers


def test_nemotron_h_mtp_pattern_follows_block_types():
    # speculative/utils.py overlays a draft checkpoint's mtp_layers_block_type
    # onto the target config; the pattern modeling_nemotron_h builds MTP layers
    # from must follow it, as it does on the native config.
    from tensorrt_llm._torch.configs import NemotronHConfig

    config = NemotronHConfig(hybrid_override_pattern="M-*", mtp_hybrid_override_pattern="*E*")
    assert config.mtp_layers_block_type == ["attention", "moe", "attention"]

    # An explicit pattern wins over a block-type list left at its default.
    config = NemotronHConfig(
        hybrid_override_pattern="M-*",
        mtp_hybrid_override_pattern="*E*",
        mtp_layers_block_type=["attention", "moe"],
    )
    assert config.mtp_hybrid_override_pattern == "*E*"

    config.mtp_layers_block_type = ["moe"]
    assert config.mtp_hybrid_override_pattern == "E"
    with pytest.raises(AttributeError):
        config.mtp_hybrid_override_pattern = "*E"


def test_nemotron_h_keeps_use_cache():
    from tensorrt_llm._torch.configs import NemotronHConfig

    assert NemotronHConfig(hybrid_override_pattern="-").use_cache is True
    assert NemotronHConfig(hybrid_override_pattern="-", use_cache=False).use_cache is False


def test_nemotron_h_block_types_only_dense_checkpoint(tmp_path):
    # transformers >= 5.13 re-saves only layers_block_type; "mlp" entries there
    # are as unrepresentable natively as "-" in the pattern.
    from tensorrt_llm._torch.configs import NemotronHConfig

    config_dict = _nemotron_h_min_config("M-M*")
    del config_dict["hybrid_override_pattern"]
    config_dict["layers_block_type"] = ["mamba", "mlp", "mamba", "attention"]

    cfg = AutoConfig.from_pretrained(_write_config(tmp_path, "nemotron_h_list", config_dict))

    assert isinstance(cfg, NemotronHConfig)
    assert cfg.hybrid_override_pattern == "M-M*"


def test_nemotron_h_missing_pattern_rejected():
    from tensorrt_llm._torch.configs import NemotronHConfig

    with pytest.raises(ValueError, match="hybrid_override_pattern"):
        NemotronHConfig(hybrid_override_pattern=None)


def test_nemotron_h_invalid_block_type_rejected():
    from tensorrt_llm._torch.configs import NemotronHConfig

    with pytest.raises(ValueError, match="layers_block_type"):
        NemotronHConfig(hybrid_override_pattern=None, layers_block_type=["unknown"])


def test_transformers_tokenizer_registers_dense_nemotron_h_in_clean_process(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    config_dict = _nemotron_h_min_config(_NEMOTRON_H_DENSE_PATTERN)
    config_dict["tokenizer_class"] = "PreTrainedTokenizerFast"
    model_dir = Path(_write_config(tmp_path, "nemotron_h_tokenizer", config_dict))

    backend = Tokenizer(WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(model_dir / "tokenizer.json"))

    script = """
import sys

from tensorrt_llm.tokenizer.tokenizer import TransformersTokenizer

assert "tensorrt_llm._torch.configs" not in sys.modules
tokenizer = TransformersTokenizer.from_pretrained(sys.argv[1])
assert tokenizer.encode("hello") == [1]
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(model_dir)],
        # Do not put the source checkout first on sys.path: CPU L0 runs this
        # test against the built wheel, while the checkout has no compiled
        # tensorrt_llm.bindings extension for a fresh interpreter to import.
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


def test_load_hf_model_config_handles_dense_nemotron_h(tmp_path):
    # ModelLoader.load_hf_model_config swallows exceptions and returns None,
    # which is how the KeyError surfaced as a confusing downstream failure.
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    model_dir = _write_config(
        tmp_path, "nemotron_h_loader", _nemotron_h_min_config(_NEMOTRON_H_DENSE_PATTERN)
    )

    cfg = ModelLoader.load_hf_model_config(model_dir)

    assert cfg is not None
    assert cfg.hybrid_override_pattern == _NEMOTRON_H_DENSE_PATTERN
