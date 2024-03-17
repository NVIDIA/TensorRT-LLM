import os
import sys
import tempfile

from tensorrt_llm.hlapi.llm import LLM, ModelConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import skip_pre_ampere, skip_pre_hopper

llama_model_path = str(llm_models_root() / "llama-models/llama-7b-hf")


@skip_pre_ampere
def test_llm_int4_awq_quantization():
    config = ModelConfig(llama_model_path)
    config.quant_config.init_from_description(quantize_weights=True,
                                              use_int4_weights=True,
                                              per_group=True)
    config.quant_config.quantize_lm_head = True
    assert config.quant_config.has_any_quant()

    llm = LLM(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm.save(tmpdir)


@skip_pre_hopper
def test_llm_fp8_quantization():
    config = ModelConfig(llama_model_path)
    config.quant_config.set_fp8_qdq()
    config.quant_config.set_fp8_kv_cache()

    assert config.quant_config.has_any_quant()

    llm = LLM(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm.save(tmpdir)


if __name__ == "__main__":
    test_llm_int4_awq_quantization()
    test_llm_fp8_quantization()
