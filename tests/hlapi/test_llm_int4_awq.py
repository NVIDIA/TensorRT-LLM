import os
import sys
import tempfile

from tensorrt_llm.hlapi.llm import LLM, ModelConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root

llama_model_path = str(llm_models_root() / "llama-models/llama-7b-hf")


def _test_llm_int4_awq_quantization():
    config = ModelConfig(llama_model_path)
    config.quant_config.init_from_description(quantize_weights=True,
                                              use_int4_weights=True,
                                              per_group=True)
    assert config.quant_config.has_any_quant()

    llm = LLM(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm.save(tmpdir)
