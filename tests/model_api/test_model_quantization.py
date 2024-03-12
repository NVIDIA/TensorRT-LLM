import os
import sys
import tempfile

import tensorrt_llm
from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.quantization.mode import QuantMode

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import skip_no_ammo, skip_pre_ada, skip_pre_ampere

tensorrt_llm.logger.set_level('info')


@skip_pre_ampere
@skip_no_ammo
def test_int4_awq_quantization():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    quant_mode_int4_awq = QuantMode.from_description(quantize_weights=True,
                                                     quantize_activations=False,
                                                     per_token=False,
                                                     per_channel=False,
                                                     per_group=True,
                                                     use_int4_weights=True)

    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir,
                                               'float16',
                                               quant_mode=quant_mode_int4_awq,
                                               quantize_lm_head=True)
    engine = build(
        llama,
        BuildConfig(max_batch_size=max_batch_size,
                    max_input_len=max_isl,
                    max_output_len=max_osl))

    engine_dir = "llama-awq-quantized"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    engine.save(engine_dir)
    executor = GenerationExecutor(engine_dir, tokenizer_dir)
    for idx, output in enumerate(executor.generate(input_text, 10)):
        print(f"Input: {input_text[idx]}")
        print(f'Output: {output.text}')
        # TODO: TRTLLM-185, check the score when the test infra is ready, hard coded value is not stable, cause flaky tests in L0


@skip_pre_ada
@skip_no_ammo
def test_fp8_quantization():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    quant_mode = QuantMode(0)
    quant_mode = quant_mode.set_fp8_qdq()
    quant_mode = quant_mode.set_fp8_kv_cache()

    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir,
                                               'float16',
                                               quant_mode=quant_mode)
    engine = build(
        llama,
        BuildConfig(max_batch_size=max_batch_size,
                    max_input_len=max_isl,
                    max_output_len=max_osl))
    engine_dir = "llama-fp8-quantized"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    engine.save(engine_dir)
    executor = GenerationExecutor(engine_dir, tokenizer_dir)
    for idx, output in enumerate(executor.generate(input_text, 10)):
        print(f"Input: {input_text[idx]}")
        print(f'Output: {output.text}')
        # TODO: TRTLLM-185, check the score when the test infra is ready, hard coded value is not stable, cause flaky tests in L0


if __name__ == "__main__":
    test_int4_awq_quantization()
    test_fp8_quantization()
