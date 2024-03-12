import asyncio
import os
import sys
import tempfile
from contextlib import contextmanager

from profile_utils import profile

import tensorrt_llm
from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.models import LLaMAForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import skip_pre_ampere

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
@skip_pre_ampere
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
        engine = build(
            llama,
            BuildConfig(max_batch_size=max_batch_size,
                        max_input_len=max_isl,
                        max_output_len=max_osl))
        engine.save(engine_dir)

        executor = GenerationExecutor(engine_dir, tokenizer_dir)
        for idx, output in enumerate(
                executor.generate(input_text, [10] * len(input_text))):
            tensorrt_llm.logger.info(f"Input: {input_text[idx]}")
            tensorrt_llm.logger.info(f'Output: {output.text}')
            # note the output.text contains everything from the input, so only compare the suffix here.
            assert output.text.endswith(
                expected_output[idx]
            ), f"Expecting and got:'{expected_output[idx]}' Got: '{output.text}'"


# 76s on ipp1-1197, loading weights 18s (varies based on network speed), network/engine creation 27s
@profile("all-in-one-step")
@skip_pre_ampere
def test_all_in_one_step():
    '''Do not save the engine, all in one LLaMAForCausalLM object
    '''
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"

    # build and run by one llama object
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, 'float16')
    build(
        llama,
        BuildConfig(max_batch_size=max_batch_size,
                    max_input_len=max_isl,
                    max_output_len=max_osl))

    # TODO (tali): init the generation executor from the in-memory engine
    # This is depending on WIP MR https://gitlab-master.nvidia.com/ftp/tekit/-/merge_requests/2785


@profile(tag="fake-weights")
@skip_pre_ampere
def test_high_level_fake_weights():
    '''sanity to make sure the flow works. The key is "skip_loading_weights" param
    '''
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"

    # Fake weights, skipping save and load engine. Make it faster to sanity test
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir,
                                               'float16',
                                               skip_loading_weights=True)
    build(
        llama,
        BuildConfig(max_batch_size=max_batch_size,
                    max_input_len=max_isl,
                    max_output_len=max_osl))


@skip_pre_ampere
def _test_inflight_batching():
    # TODO[chunweiy]: Enable it later
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, 'float16')
    engine = build(
        llama,
        BuildConfig(max_batch_size=max_batch_size,
                    max_input_len=max_isl,
                    max_output_len=max_osl))
    engine_dir = "llama-ifb"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    engine.save(engine_dir)

    async def main():
        async_engine = GenerationExecutor(engine_dir, tokenizer_dir)

        async def generate_and_print(idx, inp):
            result = async_engine.generate_async(inp,
                                                 streaming=False,
                                                 max_new_tokens=10)
            await result.aresult()
            tensorrt_llm.logger.info(result.text)
            assert result.text.endswith(expected_output[idx])

            output = ""
            async for stream in async_engine.generate_async(inp,
                                                            streaming=True,
                                                            max_new_tokens=10):
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
    test_all_in_one_step()
    test_high_level_fake_weights()
    test_save_load()
    test_inflight_batching()
