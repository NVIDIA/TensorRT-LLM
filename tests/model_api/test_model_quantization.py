import os
import sys
import tempfile
from pathlib import Path

import tensorrt_llm
from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.executor import GenerationExecutor, SamplingConfig
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import force_ampere, skip_no_modelopt, skip_pre_ada

tensorrt_llm.logger.set_level('info')


@force_ampere
@skip_no_modelopt
def test_int4_awq_quantization():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir
    checkpoint_dir = tempfile.TemporaryDirectory("llama-checkpoint").name
    quant_config = QuantConfig(QuantAlgo.W4A16_AWQ)
    LLaMAForCausalLM.quantize(hf_model_dir,
                              checkpoint_dir,
                              quant_config=quant_config,
                              calib_batches=32,
                              calib_batch_size=32)
    llama = LLaMAForCausalLM.from_checkpoint(checkpoint_dir)
    engine = build(
        llama,
        BuildConfig(max_batch_size=max_batch_size,
                    max_input_len=max_isl,
                    max_output_len=max_osl))

    engine_dir = "llama-awq-quantized"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    engine.save(engine_dir)
    with GenerationExecutor.create(Path(engine_dir), tokenizer_dir) as executor:
        for idx, output in enumerate(
                executor.generate(
                    input_text,
                    sampling_config=SamplingConfig(max_new_tokens=10))):
            print(f"Input: {input_text[idx]}")
            print(f'Output: {output.text}')
            # TODO: TRTLLM-185, check the score when the test infra is ready, hard coded value is not stable, cause flaky tests in L0


@skip_pre_ada
@skip_no_modelopt
def test_fp8_quantization():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    checkpoint_dir = tempfile.TemporaryDirectory("llama-checkpoint").name
    quant_config = QuantConfig(QuantAlgo.FP8, exclude_modules=["lm_head"])
    LLaMAForCausalLM.quantize(hf_model_dir,
                              checkpoint_dir,
                              quant_config=quant_config,
                              calib_batches=32)
    llama = LLaMAForCausalLM.from_checkpoint(checkpoint_dir)

    engine = build(
        llama,
        BuildConfig(max_batch_size=max_batch_size,
                    max_input_len=max_isl,
                    max_output_len=max_osl,
                    strongly_typed=True))
    engine_dir = "llama-fp8-quantized"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    engine.save(engine_dir)
    with GenerationExecutor.create(Path(engine_dir), tokenizer_dir) as executor:
        for idx, output in enumerate(
                executor.generate(
                    input_text,
                    sampling_config=SamplingConfig(max_new_tokens=10))):
            print(f"Input: {input_text[idx]}")
            print(f'Output: {output.text}')
            # TODO: TRTLLM-185, check the score when the test infra is ready, hard coded value is not stable, cause flaky tests in L0


if __name__ == "__main__":
    test_int4_awq_quantization()
    test_fp8_quantization()
