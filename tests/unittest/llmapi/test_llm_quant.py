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
                    "group_size": 128
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
        assert layer_quant_config[
            "model.layers.1.mlp.gate_proj"].quant_algo == "W4A8_AWQ"
        assert layer_quant_config[
            "model.layers.1.mlp.gate_proj"].group_size == 128


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


if __name__ == "__main__":
    test_llm_int4_awq_quantization()
    test_llm_fp8_quantization_modelOpt_ckpt()
    test_quant_cfg_from_quant_cfg_json()
    test_quant_cfg_from_hf_quant_config()
