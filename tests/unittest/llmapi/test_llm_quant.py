import pytest

from tensorrt_llm._tensorrt_engine import LLM
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


if __name__ == "__main__":
    test_llm_int4_awq_quantization()
    test_llm_fp8_quantization_modelOpt_ckpt()
