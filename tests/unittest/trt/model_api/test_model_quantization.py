import tempfile

import pytest
from transformers import AutoTokenizer
from utils.llm_data import llm_models_root
from utils.util import force_ampere, skip_no_modelopt, skip_pre_ada

import tensorrt_llm
from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams

tensorrt_llm.logger.set_level('info')

batch_input_text = [
    "Born in north-east France, Soyer trained as a",
    "What is large language model?"
]


@pytest.mark.skip(reason="https://nvbugs/5488280")
@force_ampere
@skip_no_modelopt
def test_int4_awq_quantization():

    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")
    cnn_dailymail_path = str(llm_models_root() / "datasets/cnn_dailymail")

    checkpoint_dir = tempfile.TemporaryDirectory("llama-checkpoint").name
    quant_config = QuantConfig(QuantAlgo.W4A16_AWQ)
    LLaMAForCausalLM.quantize(hf_model_dir,
                              checkpoint_dir,
                              quant_config=quant_config,
                              calib_dataset=cnn_dailymail_path,
                              calib_batches=32,
                              calib_batch_size=32)
    llama = LLaMAForCausalLM.from_checkpoint(checkpoint_dir)
    engine = build(
        llama,
        BuildConfig(
            max_batch_size=max_batch_size,
            max_input_len=max_isl,
            max_seq_len=max_osl + max_isl,
            max_num_tokens=max_batch_size * max_isl,
        ))

    engine_dir = "llama-awq-quantized"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    engine.save(engine_dir)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
    with GenerationExecutor.create(engine_dir) as executor:
        batch_input_ids = [tokenizer.encode(inp) for inp in batch_input_text]
        outputs = executor.generate(
            batch_input_ids, sampling_params=SamplingParams(max_tokens=10))
        for idx, output in enumerate(outputs):
            print(f"Input: {batch_input_text[idx]}")
            output_text = tokenizer.decode(output.outputs[0].token_ids)
            print(f'Output: {output_text}')
            # TODO: TRTLLM-185, check the score when the test infra is ready, hard coded value is not stable, cause flaky tests in L0


@pytest.mark.skip(reason="https://nvbugs/5488280")
@skip_pre_ada
@skip_no_modelopt
def test_fp8_quantization():
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")
    cnn_dailymail_path = str(llm_models_root() / "datasets/cnn_dailymail")

    checkpoint_dir = tempfile.TemporaryDirectory("llama-checkpoint").name
    quant_config = QuantConfig(QuantAlgo.FP8)
    LLaMAForCausalLM.quantize(hf_model_dir,
                              checkpoint_dir,
                              quant_config=quant_config,
                              calib_dataset=cnn_dailymail_path,
                              calib_batches=32)
    llama = LLaMAForCausalLM.from_checkpoint(checkpoint_dir)

    engine = build(
        llama,
        BuildConfig(max_batch_size=max_batch_size,
                    max_input_len=max_isl,
                    max_seq_len=max_osl + max_isl,
                    max_num_tokens=max_batch_size * max_isl,
                    strongly_typed=True))
    engine_dir = "llama-fp8-quantized"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    engine.save(engine_dir)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
    with GenerationExecutor.create(engine_dir) as executor:
        batch_input_ids = [tokenizer.encode(inp) for inp in batch_input_text]
        outputs = executor.generate(
            batch_input_ids, sampling_params=SamplingParams(max_tokens=10))

        for idx, output in enumerate(outputs):
            print(f"Input: {batch_input_text[idx]}")
            output_text = tokenizer.decode(output.outputs[0].token_ids)
            print(f'Output: {output_text}')
            # TODO: TRTLLM-185, check the score when the test infra is ready, hard coded value is not stable, cause flaky tests in L0


if __name__ == "__main__":
    test_int4_awq_quantization()
    test_fp8_quantization()
