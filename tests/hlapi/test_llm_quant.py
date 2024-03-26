import os
import sys
import tempfile

from tensorrt_llm.hlapi.llm import LLM, ModelConfig
from tensorrt_llm.quantization import QuantAlgo

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import skip_pre_ampere, skip_pre_hopper

llama_model_path = str(llm_models_root() / "llama-models/llama-7b-hf")


@skip_pre_ampere
def test_llm_int4_awq_quantization():
    config = ModelConfig(llama_model_path)
    config.quant_config.quant_algo = QuantAlgo.W4A16_AWQ
    assert config.quant_config.quant_mode.has_any_quant()

    llm = LLM(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm.save(tmpdir)


@skip_pre_hopper
def test_llm_fp8_quantization():
    config = ModelConfig(llama_model_path)
    config.quant_config.quant_algo = QuantAlgo.FP8
    config.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
    config.quant_config.exclude_modules = ["lm_head"]

    assert config.quant_config.quant_mode.has_any_quant()

    llm = LLM(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm.save(tmpdir)


if __name__ == "__main__":
    test_llm_int4_awq_quantization()
    test_llm_fp8_quantization()
