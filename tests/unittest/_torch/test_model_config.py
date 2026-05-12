import json
import types

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_kv_cache_quant
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
from tensorrt_llm.quantization.mode import QuantMode


def make_pretrained_config(
    *,
    num_attention_heads: int = 16,
    num_key_value_heads=8,
    head_dim: int | None = None,
    num_hidden_layers: int = 1,
    vocab_size: int = 3000,
):
    # A minimal config object that provides the attributes used by
    # ModelConfig.get_bindings_model_config().
    hidden_size = head_dim * num_attention_heads
    intermediate_size = hidden_size * 4

    return types.SimpleNamespace(
        architectures=["DummyArchitecture"],
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        torch_dtype=torch.float16,
    )


@pytest.mark.parametrize(
    "num_key_value_heads",
    [
        pytest.param(8, id="kv_heads_scalar"),
        pytest.param([8, 20], id="kv_heads_per_layer_varied"),
    ],
)
@pytest.mark.parametrize("enable_attention_dp", [False, True])
@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        # Same tp/cp sizes, but different ways of setting attention TP:
        # - No explicit attn_tp_size: Mapping infers it.
        # - Explicit attn_tp_size: Mapping uses the provided value.
        dict(world_size=8, tp_size=4, cp_size=2),
        dict(world_size=4, tp_size=2, cp_size=2, attn_tp_size=4),
    ],
)
def test_get_bindings_model_config_attention_dp_attn_tp_override(
    enable_attention_dp, mapping_kwargs, num_key_value_heads
):
    mapping = Mapping(enable_attention_dp=enable_attention_dp, **mapping_kwargs)
    cfg = make_pretrained_config(
        # Keep values consistent:
        # hidden_size = num_attention_heads * head_dim.
        num_attention_heads=16,
        head_dim=4,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=2,
    )
    model_config = ModelConfig(pretrained_config=cfg, mapping=mapping)

    tokens_per_block = 32
    bindings_cfg = model_config.get_bindings_model_config(tokens_per_block=tokens_per_block)

    # bindings hidden_size is sharded by attn_tp_size and attn_cp_size.
    attn_tp_size = mapping.attn_tp_size if not mapping.enable_attention_dp else 1
    attn_cp_size = mapping.attn_cp_size

    def ceil_div(a, b):
        return (a + b - 1) // b

    assert bindings_cfg.num_heads == ceil_div(cfg.num_attention_heads, attn_tp_size * attn_cp_size)
    # bindings hidden_size is sharded by attn_tp_size.
    assert bindings_cfg.hidden_size == ceil_div(cfg.hidden_size, attn_tp_size)
    if isinstance(cfg.num_key_value_heads, (list, tuple)):
        expected_num_kv_heads_per_layer = [
            ceil_div(kv, attn_tp_size * attn_cp_size) for kv in cfg.num_key_value_heads
        ]
        assert list(bindings_cfg.num_kv_heads_per_layer) == expected_num_kv_heads_per_layer
        assert bindings_cfg.num_kv_heads(0) == expected_num_kv_heads_per_layer[0]
    else:
        assert bindings_cfg.num_kv_heads(0) == ceil_div(
            cfg.num_key_value_heads, attn_tp_size * attn_cp_size
        )

    # tp_size-dependent value (uses mapping.tp_size, not attn_tp_size).
    assert bindings_cfg.mlp_hidden_size == ceil_div(cfg.intermediate_size, mapping.tp_size)
    assert bindings_cfg.tokens_per_block == tokens_per_block


def _make_model_config_with_kv_quant(kv_cache_quant_algo):
    return ModelConfig(quant_config=QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo))


def test_validate_and_set_kv_cache_quant_auto_uses_checkpoint():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "auto")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_validate_and_set_kv_cache_quant_explicit_dtype_overrides():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "nvfp4")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4


def test_validate_and_set_kv_cache_quant_rejects_invalid_dtype():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    with pytest.raises(ValueError, match="Accepted types are"):
        validate_and_set_kv_cache_quant(model_config, "invalid_dtype")


def test_quant_mode_distinguishes_w4a16_nvfp4_from_w4a4_nvfp4():
    w4a4_mode = QuantMode.from_quant_algo(QuantAlgo.NVFP4)
    assert w4a4_mode.has_nvfp4()
    assert not w4a4_mode.has_w4a16_nvfp4()

    w4a16_mode = QuantMode.from_quant_algo(QuantAlgo.W4A16_NVFP4)
    assert w4a16_mode.has_w4a16_nvfp4()
    assert not w4a16_mode.has_nvfp4()
    assert w4a16_mode.has_any_quant(exclude_kv_cache=True)


def test_load_hf_quant_config_detects_nvfp4_w4a16_compressed_tensors():
    hf_quant_config = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "ignore": ["mtp.layers"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear", "lm_head"],
                "weights": {
                    "type": "float",
                    "num_bits": 4,
                    "strategy": "group",
                    "group_size": 16,
                },
                "input_activations": None,
            },
        },
    }

    quant_config, layer_quant_config = ModelConfig.load_hf_quant_config(
        hf_quant_config, moe_backend="CUTLASS"
    )

    assert quant_config.quant_algo == QuantAlgo.W4A16_NVFP4
    assert quant_config.group_size == 16
    assert quant_config.exclude_modules == ["mtp.layers"]
    assert layer_quant_config is None


def test_load_modelopt_quant_config_respects_hf_w4a16_metadata(tmp_path):
    modelopt_quant_config = {
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": 16,
            "exclude_modules": ["mtp*"],
        }
    }
    hf_quant_config = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "ignore": ["mtp.layers.0.mixer.q_proj"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear", "lm_head"],
                "weights": {
                    "type": "float",
                    "num_bits": 4,
                    "strategy": "tensor_group",
                    "group_size": 16,
                },
                "input_activations": None,
            },
        },
    }
    quant_config_file = tmp_path / "hf_quant_config.json"
    quant_config_file.write_text(json.dumps(modelopt_quant_config), encoding="utf-8")

    quant_config, layer_quant_config = ModelConfig.load_modelopt_quant_config(
        str(quant_config_file),
        str(tmp_path),
        moe_backend="CUTLASS",
        hf_quant_config=hf_quant_config,
    )

    assert quant_config.quant_algo == QuantAlgo.W4A16_NVFP4
    assert quant_config.group_size == 16
    assert quant_config.exclude_modules == ["mtp.layers.0.mixer.q_proj"]
    assert layer_quant_config is None
