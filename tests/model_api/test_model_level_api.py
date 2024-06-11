import asyncio
import os
import sys
import tempfile
from contextlib import contextmanager

from profile_utils import profile

import tensorrt_llm
from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.executor import GenerationExecutor, SamplingParams
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.models.llama.config import LLaMAConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import force_ampere

tensorrt_llm.logger.set_level('verbose')

input_text = [
    'Born in north-east France, Soyer trained as a',
    "What is large language model?"
]
expected_output = [
    "chef in Paris and London before moving to New York",
    "\nLarge language model is a model that is"
]


@contextmanager
def workspace(suffix, prefix="./trtllm_workspace"):
    keep_workspace = os.environ.get("TRTLLM_KEEP", False)
    if not keep_workspace:
        temp = tempfile.TemporaryDirectory(suffix)
        yield temp.name
    else:
        temp = f"{prefix}/{suffix}"
        os.makedirs(temp, exist_ok=True)
        yield temp


# 233s on ipp1-1197: loading weights 37s, network/engine 27s, save engine: 35s, load engine (14GB) about 100s
@profile("save-and-load")
@force_ampere
def test_save_load():
    '''When the engine_dir parameter of to_trt and generate is not None
        to_trt() saves the engine to disk.
        generate() loads engine from the disk.
        This is optional, but users can store the engine into any folder they want, and use later
    '''
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    with workspace("llama-save-load") as engine_dir:
        # build and run by one llama object
        llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, 'float16')
        build_config = BuildConfig(max_batch_size=max_batch_size,
                                   max_input_len=max_isl,
                                   max_output_len=max_osl,
                                   plugin_config=llama.default_plugin_config())
        build_config.plugin_config.gemm_plugin = 'float16'  # faster build
        engine = build(llama, build_config)
        engine.save(engine_dir)

        # use context manager to make sure the __exit__ can release the resources immediately
        with GenerationExecutor.create(engine_dir, tokenizer_dir) as executor:
            for idx, output in enumerate(
                    executor.generate(
                        input_text,
                        sampling_params=SamplingParams(max_new_tokens=10))):
                tensorrt_llm.logger.info(f"Input: {input_text[idx]}")
                tensorrt_llm.logger.info(f'Output: {output.text}')
                # note the output.text contains everything from the input, so only compare the suffix here.
                assert output.text.endswith(
                    expected_output[idx]
                ), f"Expecting and got:'{expected_output[idx]}' Got: '{output.text}'"


@profile(tag="fake-weights")
@force_ampere
def test_high_level_fake_weights():
    '''sanity to make sure the flow works.
    '''
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"

    # Fake weights, skipping save and load engine. Make it faster to sanity test
    config = LLaMAConfig.from_hugging_face(hf_model_dir, dtype='float16')
    llama = LLaMAForCausalLM(config)
    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_input_len=max_isl,
                               max_output_len=max_osl,
                               plugin_config=llama.default_plugin_config())
    build_config.plugin_config.gemm_plugin = 'float16'  # faster build
    build(llama, build_config)


@force_ampere
def test_inflight_batching():
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, 'float16')
    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_input_len=max_isl,
                               max_output_len=max_osl)
    build_config.plugin_config.gemm_plugin = 'float16'  # faster build
    engine = build(llama, build_config)

    engine_dir = "llama-ifb"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    engine.save(engine_dir)

    async def main():
        with GenerationExecutor.create(engine_dir,
                                       tokenizer_dir) as async_engine:

            async def generate_and_print(idx, inp):
                result = async_engine.generate_async(
                    inp,
                    streaming=False,
                    sampling_params=SamplingParams(max_new_tokens=10))
                await result.aresult()
                tensorrt_llm.logger.info(result.text)
                assert result.text.endswith(expected_output[idx])

                output = ""
                async for stream in async_engine.generate_async(
                        inp,
                        streaming=True,
                        sampling_params=SamplingParams(max_new_tokens=10)):
                    output += stream.text + ' '
                    tensorrt_llm.logger.info(
                        f"prompt: '{inp}', generation: '{output}'")

            loop = asyncio.get_running_loop()
            tasks = []
            # submit many request concurrently
            for idx, inp in enumerate(input_text):
                task = loop.create_task(generate_and_print(idx, inp))
                tasks.append(task)

            # wait all task done
            await asyncio.gather(*tasks)

    asyncio.run(main())


if __name__ == "__main__":
    test_save_load()
    test_inflight_batching()
    test_high_level_fake_weights()
