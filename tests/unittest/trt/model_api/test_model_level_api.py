import asyncio
import os
import tempfile
from contextlib import contextmanager

import pytest
from profile_utils import profile
from transformers import AutoTokenizer
from utils.llm_data import llm_models_root
from utils.util import force_ampere

import tensorrt_llm
from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.models.llama.config import LLaMAConfig
from tensorrt_llm.sampling_params import SamplingParams

tensorrt_llm.logger.set_level('verbose')

batch_input_text = [
    "Born in north-east France, Soyer trained as a",
    "What is large language model?"
]
batch_output_text_expected = [
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
@pytest.mark.skip(reason="https://nvbugs/5488280")
def test_save_load():
    '''When the engine_dir parameter of to_trt and generate is not None
        to_trt() saves the engine to disk.
        generate() loads engine from the disk.
        This is optional, but users can store the engine into any folder they want, and use later
    '''
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")

    with workspace("llama-save-load") as engine_dir:
        # build and run by one llama object
        llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir)
        build_config = BuildConfig(max_batch_size=max_batch_size,
                                   max_input_len=max_isl,
                                   max_seq_len=max_osl + max_isl,
                                   plugin_config=llama.default_plugin_config())
        build_config.plugin_config.gemm_plugin = 'auto'  # faster build
        engine = build(llama, build_config)
        engine.save(engine_dir)

        tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)

        # use context manager to make sure the __exit__ can release the resources immediately
        with GenerationExecutor.create(engine_dir) as executor:
            batch_input_ids = [
                tokenizer.encode(inp) for inp in batch_input_text
            ]
            outputs = executor.generate(
                batch_input_ids, sampling_params=SamplingParams(max_tokens=10))

            for idx, output in enumerate(outputs):
                tensorrt_llm.logger.info(f"Input: {batch_input_text[idx]}")
                output_text = tokenizer.decode(output.outputs[0].token_ids)
                tensorrt_llm.logger.info(f'Output: {output_text}')
                # note the output.text contains everything from the input, so only compare the suffix here.
                assert output_text.endswith(
                    batch_output_text_expected[idx]
                ), f"Expecting and got: {batch_output_text_expected[idx]!r} Got: {output_text!r}"


@profile(tag="fake-weights")
@force_ampere
def test_high_level_fake_weights():
    '''sanity to make sure the flow works.
    '''
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")

    # Fake weights, skipping save and load engine. Make it faster to sanity test
    config = LLaMAConfig.from_hugging_face(hf_model_dir)
    llama = LLaMAForCausalLM(config)
    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_input_len=max_isl,
                               max_seq_len=max_osl + max_isl,
                               plugin_config=llama.default_plugin_config())
    build_config.plugin_config.gemm_plugin = 'auto'  # faster build
    build(llama, build_config)


@force_ampere
@pytest.mark.skip(reason="https://nvbugs/5488280")
def test_async_io():
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")

    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir)
    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_input_len=max_isl,
                               max_seq_len=max_osl + max_isl)
    build_config.plugin_config.gemm_plugin = 'auto'  # faster build
    engine = build(llama, build_config)

    engine_dir = "llama-ifb"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    engine.save(engine_dir)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)

    async def main():
        with GenerationExecutor.create(engine_dir) as async_engine:

            async def generate_and_print(idx, inp):
                result = async_engine.generate_async(
                    tokenizer.encode(inp),
                    sampling_params=SamplingParams(max_tokens=10),
                    streaming=False)
                await result.aresult()
                output_text = tokenizer.decode(result.outputs[0].token_ids)
                tensorrt_llm.logger.info(output_text)
                assert output_text.endswith(batch_output_text_expected[idx])

                async for stream in async_engine.generate_async(
                        tokenizer.encode(inp),
                        sampling_params=SamplingParams(max_tokens=10),
                        streaming=True):
                    output_text = tokenizer.decode(stream.outputs[0].token_ids)
                    tensorrt_llm.logger.info(
                        f"prompt: {inp!r}, generation: {output_text!r}")

            loop = asyncio.get_running_loop()
            tasks = []
            # submit many request concurrently
            for idx, inp in enumerate(batch_input_text):
                task = loop.create_task(generate_and_print(idx, inp))
                tasks.append(task)

            # wait all task done
            await asyncio.gather(*tasks)

    asyncio.run(main())


if __name__ == "__main__":
    test_save_load()
    test_async_io()
    test_high_level_fake_weights()
