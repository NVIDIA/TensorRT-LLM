# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Spark2_5 (XHToken Spark-X2.5/X2.6) config class and its
registration with transformers AutoConfig and TRT-LLM's config registry."""

import pytest
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

import tensorrt_llm  # noqa: F401  triggers AutoConfig registration
from tensorrt_llm._torch.configs import Spark2_5Config
from tensorrt_llm._torch.pyexecutor.config_utils import (
    _CONFIG_REGISTRY,
    get_layer_attention_window,
    uses_vswa_kv_cache_layout,
)

pytestmark = pytest.mark.cpu_only

# Verbatim from XHToken/Spark-X2.5-1.7B(-Base) config.json (Spark-X2.6 shares
# the architecture; only sizes differ).
SPARK2_5_1_7B_CONFIG = {
    "architectures": ["Spark2_5ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_spark.Spark2_5Config",
        "AutoModel": "modeling_spark.Spark2_5Model",
        "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
    },
    "bos_token_id": 0,
    "dtype": "bfloat16",
    "eos_token_id": 1,
    "gate_attn_act_mode": "sigmoid",
    "head_dim": 256,
    "headwise_attn_output_gate": True,
    "hidden_act": "gelu",
    "hidden_size": 2048,
    "initializer_range": 0.0221,
    "intermediate_size": 6656,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
    * 7,
    "max_position_embeddings": 1048576,
    "mlp_bias": False,
    "model_type": "spark2_5",
    "num_attention_heads": 8,
    "num_hidden_layers": 28,
    "num_key_value_heads": 2,
    "pad_token_id": 2,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 5000000,
        },
        "sliding_attention": {
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000,
        },
    },
    "sliding_window": 512,
    "tie_word_embeddings": True,
    "transformers_version": "4.57.1",
    "use_cache": True,
    "vocab_size": 131072,
}


def _make_config() -> Spark2_5Config:
    return Spark2_5Config(**SPARK2_5_1_7B_CONFIG)


def test_spark2_5_config_preserves_hf_fields():
    config = _make_config()
    assert config.model_type == "spark2_5"
    assert config.vocab_size == 131072
    assert config.hidden_size == 2048
    assert config.intermediate_size == 6656
    assert config.num_hidden_layers == 28
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 2
    assert config.head_dim == 256
    assert config.hidden_act == "gelu"
    assert config.mlp_bias is False
    assert config.attention_bias is False
    assert config.rms_norm_eps == 1e-06
    assert config.max_position_embeddings == 1048576
    assert config.sliding_window == 512
    assert config.tie_word_embeddings is True
    assert config.headwise_attn_output_gate is True
    assert config.gate_attn_act_mode == "sigmoid"
    assert config.layer_types == SPARK2_5_1_7B_CONFIG["layer_types"]
    assert config.rope_parameters == SPARK2_5_1_7B_CONFIG["rope_parameters"]
    # RoPE helpers used by the modeling code.
    assert config.get_rope_theta("full_attention") == 5000000
    assert config.get_rope_theta("sliding_attention") == 10000
    assert config.get_partial_rotary_factor("full_attention") == 0.25
    assert config.get_partial_rotary_factor("sliding_attention") == 1.0


def test_spark2_5_config_defaults():
    config = Spark2_5Config()
    # head_dim falls back to hidden_size // num_attention_heads.
    assert config.head_dim == config.hidden_size // config.num_attention_heads
    # num_key_value_heads falls back to num_attention_heads (MHA).
    assert config.num_key_value_heads == config.num_attention_heads
    # layer_types defaults to all full attention.
    assert config.layer_types == ["full_attention"] * config.num_hidden_layers
    assert config.headwise_attn_output_gate is False


def test_spark2_5_config_rejects_bad_inputs():
    with pytest.raises(ValueError, match="divisible"):
        Spark2_5Config(num_attention_heads=8, num_key_value_heads=3)
    with pytest.raises(ValueError, match="layer_types"):
        Spark2_5Config(num_hidden_layers=4, layer_types=["full_attention"] * 3)


def test_spark2_5_config_rejects_sliding_without_window():
    # Sliding layers with no (or a non-positive) window would silently run as
    # full attention; the config must reject them instead.
    with pytest.raises(ValueError, match="sliding_window"):
        Spark2_5Config(
            num_hidden_layers=4,
            layer_types=["sliding_attention"] * 4,
            sliding_window=None,
        )
    with pytest.raises(ValueError, match="sliding_window"):
        Spark2_5Config(
            num_hidden_layers=4,
            layer_types=["sliding_attention"] * 4,
            sliding_window=0,
        )
    # bool is not a valid window size either.
    with pytest.raises(ValueError, match="sliding_window"):
        Spark2_5Config(
            num_hidden_layers=4,
            layer_types=["sliding_attention"] * 4,
            sliding_window=True,
        )
    # Full-attention-only configs legitimately omit sliding_window.
    Spark2_5Config(num_hidden_layers=4, sliding_window=None)


def test_spark2_5_layer_windows():
    config = _make_config()
    windows = [get_layer_attention_window(config, i) for i in range(config.num_hidden_layers)]
    expected = [512 if lt == "sliding_attention" else None for lt in config.layer_types]
    assert windows == expected
    # Full-attention layers land on indices 3, 7, ..., 27 for the 3:1 pattern.
    assert [i for i, w in enumerate(windows) if w is None] == list(range(3, 28, 4))
    assert uses_vswa_kv_cache_layout(windows) is True


def test_spark2_5_registered_with_autoconfig():
    assert CONFIG_MAPPING["spark2_5"] is Spark2_5Config
    config = AutoConfig.for_model("spark2_5", **SPARK2_5_1_7B_CONFIG)
    assert isinstance(config, Spark2_5Config)
    assert config.sliding_window == 512


def test_spark2_5_config_registry_lookup():
    assert _CONFIG_REGISTRY["spark2_5"] is Spark2_5Config
