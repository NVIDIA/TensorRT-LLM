import json
import tempfile
from pathlib import Path

import pytest

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.llm_utils import CalibConfig, QuantAlgo, QuantConfig

# isort: off
from .test_llm import cnn_dailymail_path, llama_model_path, get_model_path
from utils.util import skip_blackwell, skip_pre_blackwell, skip_pre_hopper
# isort: on


@skip_blackwell
def test_llm_int4_awq_quantization():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
    assert quant_config.quant_mode.has_any_quant()
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)

    llm = LLM(llama_model_path,
              quant_config=quant_config,
              calib_config=calib_config)

    sampling_params = SamplingParams(max_tokens=6)
    for output in llm.generate(["A B C"], sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G H I"


@skip_pre_hopper
def test_llm_fp8_quantization():
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8)
    assert quant_config.quant_mode.has_any_quant()
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)

    llm = LLM(llama_model_path,
              quant_config=quant_config,
              calib_config=calib_config)
    sampling_params = SamplingParams(max_tokens=6)
    for output in llm.generate(["A B C"], sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G H I"


@skip_pre_blackwell
def test_llm_nvfp4_quantization():
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4,
                               kv_cache_quant_algo=QuantAlgo.FP8)
    assert quant_config.quant_mode.has_any_quant()
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)

    llm = LLM(llama_model_path,
              quant_config=quant_config,
              calib_config=calib_config)
    sampling_params = SamplingParams(max_tokens=6)
    for output in llm.generate(["A B C"], sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G H I"


@skip_pre_hopper
@pytest.mark.skip("https://nvbugs/5027953")
def test_llm_fp8_quantization_modelOpt_ckpt():
    llama_fp8_model_path = get_model_path(
        "llama-3.1-model/Llama-3.1-8B-Instruct-FP8")
    llm = LLM(llama_fp8_model_path,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))
    sampling_params = SamplingParams(max_tokens=6)
    for output in llm.generate(["A B C"], sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == " D E F G H I"


def test_quant_cfg_from_quant_cfg_json():
    """
    Test loading MIXED_PRECISION config from quant_cfg.json with per-layer quantization.
    This supports the workflow from examples/quantization/quantize_mixed_precision_moe.py.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)

        # Create dummy quant_cfg.json
        quant_cfg_content = {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": "FP8",
            "quantized_layers": {
                "model.layers.0.self_attn.q_proj": {
                    "quant_algo": "FP8"
                },
                "model.layers.0.self_attn.k_proj": {
                    "quant_algo": "FP8"
                },
                "model.layers.1.mlp.gate_proj": {
                    "quant_algo": "W4A8_AWQ",
                    "group_size": 128,
                    "has_zero_point": False,
                    "pre_quant_scale": True,
                }
            }
        }

        quant_cfg_file = model_dir / "quant_cfg.json"
        with open(quant_cfg_file, 'w') as f:
            json.dump(quant_cfg_content, f)

        # Create dummy hf_quant_config.json
        hf_quant_config_content = {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
            }
        }

        hf_quant_config_file = model_dir / "hf_quant_config.json"
        with open(hf_quant_config_file, 'w') as f:
            json.dump(hf_quant_config_content, f)

        quant_config, layer_quant_config = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)

        # Verify quant_cfg.json was loaded
        assert quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
        assert quant_config.kv_cache_quant_algo == "FP8"

        # Verify layer configs were created correctly
        assert layer_quant_config[
            "model.layers.0.self_attn.q_proj"].quant_algo == "FP8"
        assert layer_quant_config[
            "model.layers.0.self_attn.q_proj"].kv_cache_quant_algo == "FP8"
        awq_layer = layer_quant_config["model.layers.1.mlp.gate_proj"]
        assert awq_layer.quant_algo == "W4A8_AWQ"
        assert awq_layer.group_size == 128
        assert awq_layer.has_zero_point is False
        assert awq_layer.pre_quant_scale is True


def test_quant_cfg_top_level_overlay():
    """quant_cfg.json's top-level group_size/exclude_modules override hf_quant_config.json."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)

        # quant_cfg.json overrides top-level group_size and exclude_modules.
        quant_cfg_content = {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": "FP8",
            "group_size": 64,
            "exclude_modules": ["lm_head", "model.embed_tokens"],
            "quantized_layers": {
                "model.layers.0.self_attn.q_proj": {
                    "quant_algo": "FP8"
                }
            },
        }
        quant_cfg_file = model_dir / "quant_cfg.json"
        with open(quant_cfg_file, 'w') as f:
            json.dump(quant_cfg_content, f)

        hf_quant_config_content = {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": "FP8",
                "group_size": 128,
                "exclude_modules": ["foo"],
            }
        }
        hf_quant_config_file = model_dir / "hf_quant_config.json"
        with open(hf_quant_config_file, 'w') as f:
            json.dump(hf_quant_config_content, f)

        quant_config, _ = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)

        assert quant_config.group_size == 64
        assert quant_config.exclude_modules == ["lm_head", "model.embed_tokens"]


def test_quant_cfg_from_hf_quant_config():
    """Test fallback to hf_quant_config.json when quant_cfg.json is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)

        # Create dummy hf_quant_config.json
        hf_quant_config_content = {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": "FP8",
                "quantized_layers": {
                    "model.layers.0.self_attn.q_proj": {
                        "quant_algo": "FP8"
                    },
                    "model.layers.0.mlp.up_proj": {
                        "quant_algo": "W4A16_AWQ",
                        "group_size": 64
                    }
                }
            }
        }
        hf_quant_config_file = model_dir / "hf_quant_config.json"
        with open(hf_quant_config_file, 'w') as f:
            json.dump(hf_quant_config_content, f)
        quant_config, layer_quant_config = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)

        # Verify layer configs
        assert quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
        assert quant_config.kv_cache_quant_algo == "FP8"
        assert layer_quant_config[
            "model.layers.0.self_attn.q_proj"].quant_algo == "FP8"
        assert layer_quant_config[
            "model.layers.0.mlp.up_proj"].quant_algo == "W4A16_AWQ"
        assert layer_quant_config["model.layers.0.mlp.up_proj"].group_size == 64


def _write_hf_quant_config(model_dir: Path, content: dict) -> Path:
    """Write a ``hf_quant_config.json`` under ``model_dir`` and return its path."""
    path = model_dir / "hf_quant_config.json"
    with open(path, 'w') as f:
        json.dump(content, f)
    return path


def test_quant_cfg_fp8_legacy_shape():
    """Plain FP8 modelopt 0.x checkpoint: legacy 'quantization' wrapper."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "producer": {
                    "name": "modelopt",
                    "version": "0.43.0"
                },
                "quantization": {
                    "quant_algo": "FP8",
                    "kv_cache_quant_algo": "FP8",
                    "exclude_modules": ["lm_head"],
                },
            })
        quant_config, layer_quant_config = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)
        assert quant_config.quant_algo == QuantAlgo.FP8
        assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8
        assert quant_config.exclude_modules == ["lm_head"]
        assert layer_quant_config is None


def test_quant_cfg_flat_shape_with_ignore_rename():
    """Modelopt 1.x flat shape: ``ignore`` is renamed to ``exclude_modules``."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "producer": {
                    "name": "modelopt",
                    "version": "1.0.0"
                },
                "quant_method": "modelopt",
                "quant_algo": "FP8",
                "ignore": ["lm_head", "model.embed_tokens"],
            })
        quant_config, _ = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)
        assert quant_config.quant_algo == QuantAlgo.FP8
        assert quant_config.exclude_modules == ["lm_head", "model.embed_tokens"]


def test_quant_cfg_flat_shape_kv_cache_scheme_dict():
    """Flat shape with compressed-tensors-style kv_cache_scheme dict (FP8)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "producer": {
                    "name": "modelopt",
                    "version": "1.0.0"
                },
                "quant_method": "modelopt",
                "quant_algo": "FP8",
                "kv_cache_scheme": {
                    "dynamic": False,
                    "num_bits": 8,
                    "type": "float",
                },
                "ignore": [],
            })
        quant_config, _ = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)
        assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_quant_cfg_flat_shape_kv_cache_scheme_string_nvfp4():
    """Flat shape with bare-string kv_cache_scheme fallback (NVFP4)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "producer": {
                    "name": "modelopt",
                    "version": "1.0.0"
                },
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "kv_cache_scheme": "NVFP4",
                "ignore": [],
            })
        quant_config, _ = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)
        assert quant_config.quant_algo == QuantAlgo.NVFP4
        assert quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4


def test_quant_cfg_fp8_pb_wo_alias_canonicalized():
    """Legacy ``fp8_pb_wo`` alias is canonicalized to FP8_BLOCK_SCALES."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "producer": {
                    "name": "modelopt"
                },
                "quantization": {
                    "quant_algo": "fp8_pb_wo"
                },
            })
        quant_config, _ = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)
        assert quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        # FP8_BLOCK_SCALES default group_size.
        assert quant_config.group_size == 128


def test_quant_cfg_fp8_block_scales_trtllm_default_excludes():
    """TRTLLM moe_backend + FP8_BLOCK_SCALES + no excludes → defaults applied."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "producer": {
                    "name": "modelopt"
                },
                "quantization": {
                    "quant_algo": "FP8_BLOCK_SCALES"
                },
            })
        quant_config, _ = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, "TRTLLM")
        assert quant_config.exclude_modules == [
            "*kv_b_proj*", "*k_b_proj*", "*eh_proj"
        ]


def test_quant_cfg_explicit_empty_excludes_preserved():
    """Explicit ``exclude_modules: []`` is preserved (no defaults applied)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "producer": {
                    "name": "modelopt"
                },
                "quantization": {
                    "quant_algo": "FP8_BLOCK_SCALES",
                    "exclude_modules": [],
                },
            })
        quant_config, _ = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, "TRTLLM")
        # Explicit [] must NOT trigger the TRTLLM default-excludes branch.
        assert quant_config.exclude_modules == []


def test_quant_cfg_mixed_precision_kv_cache_conflict_raises():
    """quant_cfg.json kv_cache_quant_algo conflicting with hf_quant_config.json raises."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        with open(model_dir / "quant_cfg.json", 'w') as f:
            json.dump(
                {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_quant_algo": "NVFP4",
                    "quantized_layers": {
                        "l0": {
                            "quant_algo": "FP8"
                        }
                    },
                }, f)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "producer": {
                    "name": "modelopt"
                },
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_quant_algo": "FP8",
                },
            })
        with pytest.raises(RuntimeError, match="kvcache config"):
            ModelConfig.load_modelopt_quant_config(hf_quant_config_file,
                                                   model_dir, None)


def test_quant_cfg_awq_extra_fields_preserved_via_load_hf_quant_config():
    """AWQ extras (``has_zero_point``, ``pre_quant_scale``) flow through ``load_hf_quant_config``."""
    inline_modelopt_awq = {
        "producer": {
            "name": "modelopt"
        },
        "quantization": {
            "quant_algo": "W4A16_AWQ",
            "group_size": 128,
            "has_zero_point": False,
            "pre_quant_scale": True,
        },
    }
    quant_config, _ = ModelConfig.load_hf_quant_config(inline_modelopt_awq,
                                                       moe_backend=None,
                                                       checkpoint_dir=None)
    assert quant_config.quant_algo == QuantAlgo.W4A16_AWQ
    assert quant_config.group_size == 128
    assert quant_config.has_zero_point is False
    assert quant_config.pre_quant_scale is True


@pytest.mark.parametrize("config,expected", [
    ({
        "producer": {
            "name": "modelopt"
        }
    }, True),
    ({
        "quant_method": "modelopt"
    }, True),
    ({
        "quant_method": "modelopt-flat"
    }, True),
    ({
        "producer": {
            "name": "other"
        }
    }, False),
    ({
        "quant_method": "fp8"
    }, False),
    ({}, False),
    ("not a dict", False),
    (None, False),
])
def test_is_modelopt_quant_config(config, expected):
    """Producer name or quant_method prefix must signal modelopt."""
    from tensorrt_llm.quantization.modelopt_config import \
        is_modelopt_quant_config
    assert is_modelopt_quant_config(config) is expected


@pytest.mark.parametrize(
    "scheme,expected",
    [
        (None, None),
        # Bare-string fallback.
        ("FP8", "FP8"),
        ("NVFP4", "NVFP4"),
        ("INT8", "INT8"),
        ("int8", "INT8"),
        # Compressed-tensors dict form.
        ({
            "type": "float",
            "num_bits": 8
        }, "FP8"),
        ({
            "type": "float",
            "num_bits": 4
        }, "NVFP4"),
        ({
            "type": "int",
            "num_bits": 8
        }, "INT8"),
        # Unrecognized -> None.
        ("UNKNOWN_ALGO", None),
        ({
            "type": "float",
            "num_bits": 16
        }, None),
        (123, None),
    ])
def test_kv_cache_scheme_to_algo(scheme, expected):
    """``_kv_cache_scheme_to_algo`` covers string + dict + None inputs."""
    from tensorrt_llm.quantization.modelopt_config import \
        _kv_cache_scheme_to_algo
    assert _kv_cache_scheme_to_algo(scheme) == expected


@pytest.mark.parametrize("raw,match", [
    ("not a dict", "Expected dict"),
    ({
        "producer": {
            "name": "other"
        }
    }, "Not a modelopt quant config"),
    ({
        "producer": {
            "name": "modelopt"
        },
        "quantization": "not a dict",
    }, "'quantization' must be a dict"),
])
def test_read_modelopt_quant_config_invalid_raises(raw, match):
    """Non-dict / non-modelopt / malformed configs raise ValueError."""
    from tensorrt_llm.quantization.modelopt_config import \
        read_modelopt_quant_config
    with pytest.raises(ValueError, match=match):
        read_modelopt_quant_config(raw)


def test_quant_cfg_quant_algo_fields_are_enum_typed():
    """Top-level and per-layer ``quant_algo``/``kv_cache_quant_algo`` are QuantAlgo enums."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "producer": {
                    "name": "modelopt"
                },
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_quant_algo": "FP8",
                    "quantized_layers": {
                        "layer.0": {
                            "quant_algo": "NVFP4"
                        },
                    },
                },
            })
        quant_config, layer_quant_config = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)
        # Top-level.
        assert isinstance(quant_config.quant_algo, QuantAlgo)
        assert isinstance(quant_config.kv_cache_quant_algo, QuantAlgo)
        assert quant_config.kv_cache_quant_algo is QuantAlgo.FP8
        # Per-layer: quant_algo from the layer dict; kv_cache_quant_algo inherited.
        layer = layer_quant_config["layer.0"]
        assert isinstance(layer.quant_algo, QuantAlgo)
        assert isinstance(layer.kv_cache_quant_algo, QuantAlgo)
        assert layer.quant_algo is QuantAlgo.NVFP4
        assert layer.kv_cache_quant_algo is QuantAlgo.FP8


@pytest.mark.parametrize("scheme", ["INT8", {"type": "int", "num_bits": 8}])
def test_quant_cfg_flat_shape_kv_cache_scheme_int8(scheme):
    """Flat shape: INT8 ``kv_cache_scheme`` honored via both string and dict forms."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "quant_method": "modelopt",
                "quant_algo": "FP8",
                "kv_cache_scheme": scheme,
                "ignore": [],
            })
        quant_config, _ = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)
        assert quant_config.kv_cache_quant_algo is QuantAlgo.INT8


def test_quant_cfg_awq_extras_default_when_absent():
    """When AWQ extras are absent from the JSON, ``QuantConfig`` defaults are preserved."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir)
        hf_quant_config_file = _write_hf_quant_config(
            model_dir, {
                "producer": {
                    "name": "modelopt"
                },
                "quantization": {
                    "quant_algo": "FP8"
                },
            })
        quant_config, _ = ModelConfig.load_modelopt_quant_config(
            hf_quant_config_file, model_dir, None)
        # QuantConfig defaults: has_zero_point=False, pre_quant_scale=False.
        assert quant_config.has_zero_point is False
        assert quant_config.pre_quant_scale is False


def test_load_hf_quant_config_fp8_block_scales_deepseek_v3():
    """DeepSeek V3 ``quant_method=fp8`` with weight_block_size=(128,128)."""
    quant_config, _ = ModelConfig.load_hf_quant_config(
        {
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        },
        moe_backend=None)
    assert quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert quant_config.group_size == 128
    # Default excludes for FP8_BLOCK_SCALES include kv/eh-proj patterns.
    assert "*kv_b_proj*" in quant_config.exclude_modules


@pytest.mark.parametrize("weights_strategy,inputs_strategy,expected_algo", [
    ("channel", "token", QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN),
    ("block", "group", QuantAlgo.FP8_BLOCK_SCALES),
])
def test_load_hf_quant_config_compressed_tensors(weights_strategy,
                                                 inputs_strategy,
                                                 expected_algo):
    """LLM-compressor ``compressed-tensors``: strategy combinations map to TRT-LLM algos."""
    inputs_cfg = {"num_bits": 8, "strategy": inputs_strategy}
    if inputs_strategy == "group":
        inputs_cfg["group_size"] = 128
    quant_config, _ = ModelConfig.load_hf_quant_config(
        {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "strategy": weights_strategy
                    },
                    "input_activations": inputs_cfg,
                },
            },
            "ignore": ["lm_head"],
        },
        moe_backend=None)
    assert quant_config.quant_algo == expected_algo
    assert quant_config.exclude_modules == ["lm_head"]


def test_load_hf_quant_config_nvfp4_native_with_modules_to_not_convert():
    """HF nvfp4 schema: ``modules_to_not_convert`` is merged into ``exclude_modules``."""
    quant_config, _ = ModelConfig.load_hf_quant_config(
        {
            "quant_method": "nvfp4",
            "group_size": 16,
            "modules_to_not_convert": ["custom_layer"],
        },
        moe_backend=None)
    assert quant_config.quant_algo == QuantAlgo.NVFP4
    assert quant_config.group_size == 16
    assert "custom_layer" in quant_config.exclude_modules
    assert "*.mlp.gate" in quant_config.exclude_modules  # default
    assert "lm_head" in quant_config.exclude_modules  # default


def test_load_hf_quant_config_no_match_returns_empty_quant_config():
    """An unrecognized ``quant_method`` returns an empty QuantConfig (no algo set)."""
    quant_config, _ = ModelConfig.load_hf_quant_config(
        {"quant_method": "unknown_format"}, moe_backend=None)
    assert quant_config.quant_algo is None


if __name__ == "__main__":
    test_llm_int4_awq_quantization()
    test_llm_fp8_quantization_modelOpt_ckpt()
    test_quant_cfg_from_quant_cfg_json()
    test_quant_cfg_from_hf_quant_config()
