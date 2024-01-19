import asyncio
import tempfile

import torch
from llm_data import llm_models_root
from profile_utils import profile

import tensorrt_llm
from tensorrt_llm.models import LLaMAForCausalLM


def ammo_installed():
    try:
        # isort: off
        import ammo.torch.quantization as atq
        from ammo.torch.export import export_model_config
        print(type(atq))
        print(type(export_model_config))
        # isort: on
        return True
    except Exception:
        return False
    return False


tensorrt_llm.logger.set_level('verbose')

input_text = [
    'Born in north-east France, Soyer trained as a',
    "What is large language model?"
]
expected_output = [
    "chef in Paris and London before moving to New York",
    "\nLarge language model is a model that is"
]


# 233s on ipp1-1197: loading weights 37s, network/engine 27s, save engine: 35s, load engine (14GB) about 100s
@profile("save-and-load")
def test_save_load():
    '''When the engine_dir parameter of to_trt and generate is not None
        to_trt() saves the engine to disk.
        generate() loads engine from the disk.
        This is optional, but users can store the engine into any folder they want, and use later
    '''
    major, minor = torch.cuda.get_device_capability()
    if not (major >= 8):
        print("Test supported on post Ampere")
        return
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir
    engine_dir = tempfile.TemporaryDirectory("llama-save-load")

    # build and run by one llama object
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, 'float16')
    llama.to_trt(max_batch_size, max_isl, max_osl)
    llama.save(engine_dir.name)

    for idx, (inp, output) in enumerate(
            llama._generate(input_text,
                            10,
                            tokenizer_dir=tokenizer_dir,
                            engine_dir=engine_dir.name)):
        print(f"Input: {inp}")
        print(f'Output: {output}')
        assert output == expected_output[
            idx], f"Expecting {expected_output[idx]}, got {output}"


# 76s on ipp1-1197, loading weights 18s (varies based on network speed), network/engine creation 27s
@profile("all-in-one-step")
def test_all_in_one_step():
    '''Do not save the engine, all in one LLaMAForCausalLM object
    '''
    major, minor = torch.cuda.get_device_capability()
    if not (major >= 8):
        print("Test supported on post Ampere")
        return
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    # build and run by one llama object
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, 'float16')
    llama.to_trt(max_batch_size, max_isl, max_osl)

    for idx, (inp, output) in enumerate(
            llama._generate(input_text, 10, tokenizer_dir=tokenizer_dir)):
        print(f"Input: {inp}")
        print(f'Output: {output}')
        assert output == expected_output[
            idx], f"Expecting {expected_output[idx]}, got {output}"


@profile(tag="fake-weights")
def test_high_level_fake_weights():
    '''sanity to make sure the flow works. The key is "skip_loading_weights" param
    '''
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    major, minor = torch.cuda.get_device_capability()
    if not (major >= 8):
        print("Test supported on post Ampere")
        return
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    # Fake weights, skipping save and load engine. Make it faster to sanity test
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir,
                                               'float16',
                                               skip_loading_weights=True)
    llama.to_trt(max_batch_size, max_isl, max_osl)
    llama._generate(input_text, 10, tokenizer_dir=tokenizer_dir)


def _test_inflight_batching():
    # TODO[chunweiy]: Enable it later
    major, minor = torch.cuda.get_device_capability()
    if not (major >= 8):
        print("Test supported on post Ampere")
        return
    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir

    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, 'float16')
    llama.to_trt(max_batch_size, max_isl, max_osl)
    engine_dir = "llama-ifb"
    engine_temp = tempfile.TemporaryDirectory(engine_dir)
    engine_dir = engine_temp.name
    llama.save(engine_dir)
    from tensorrt_llm.engine import AsyncLLMEngine

    async def main():
        async_engine = AsyncLLMEngine(engine_dir, tokenizer_dir)

        async def generate_and_print(idx, inp):
            # streaming
            output = ""
            async for diff_str in async_engine.generate(inp, 10):
                output += diff_str.text + ' '
                print(f"primpt: '{inp}', generation: '{output}'")
            # output may have additional whitespace in prefix
            assert output.strip() == expected_output[idx] or output == expected_output[idx] or ''.join(output.strip().split()) == ''.join(expected_output[idx].split()), \
                f"Expecting '{expected_output[idx]}', got '{output}'"

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
