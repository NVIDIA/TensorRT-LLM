import os
import sys

from tensorrt_llm.hlapi.llm import LLM, ModelConfig
from tensorrt_llm.quantization import QuantAlgo

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import skip_pre_ampere, skip_pre_hopper

try:
    from .test_llm import llama_model_path
except ImportError:
    from test_llm import llama_model_path


@skip_pre_ampere
def test_llm_int4_awq_quantization():
    config = ModelConfig(llama_model_path)
    config.quant_config.quant_algo = QuantAlgo.W4A16_AWQ
    assert config.quant_config.quant_mode.has_any_quant()

    llm = LLM(config)

    sampling_config = llm.get_default_sampling_config()
    sampling_config.max_new_tokens = 6
    for output in llm.generate(["A B C"], sampling_config=sampling_config):
        print(output)
        assert output.text == "D E F G H I"


@skip_pre_hopper
def test_llm_fp8_quantization():
    config = ModelConfig(llama_model_path)
    config.quant_config.quant_algo = QuantAlgo.FP8
    config.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
    config.quant_config.exclude_modules = ["lm_head"]

    assert config.quant_config.quant_mode.has_any_quant()

    llm = LLM(config)
    sampling_config = llm.get_default_sampling_config()
    sampling_config.max_new_tokens = 6
    for output in llm.generate(["A B C"], sampling_config=sampling_config):
        print(output)
        assert output.text == "D E F G H I"


if __name__ == "__main__":
    test_llm_int4_awq_quantization()
    test_llm_fp8_quantization()
