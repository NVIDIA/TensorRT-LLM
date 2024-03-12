import asyncio
import os
import tempfile
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from tensorrt_llm.hlapi.llm import (LLM, DecodingMode, KvCacheConfig,
                                    ModelConfig, SamplingConfig, TokenizerBase)
from tensorrt_llm.hlapi.utils import get_total_gpu_memory


def get_model_path(model_name):
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.llm_data import llm_models_root
    return str(llm_models_root() / model_name)


default_model_name = "llama-models/llama-7b-hf"
mixtral_model_name = "Mixtral-8x7B-v0.1"

llama_model_path = get_model_path(default_model_name)
llm_engine_dir = os.environ.get('LLM_ENGINE_DIR', './tmp.engine')
prompts = ["A B C"]

cur_dir = os.path.dirname(os.path.abspath(__file__))
models_root = os.path.join(cur_dir, '../../models')

skip_single_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="The test needs at least 2 GPUs, skipping")


def test_llm_loading_from_hf():
    config = ModelConfig(llama_model_path)
    # The performance-related flags are turned on eagerly to check the functionality

    devices = config.parallel_config.get_devices()
    if torch.cuda.get_device_properties(devices[0]).major >= 8:
        # only available for A100 or newer GPUs
        config.multi_block_mode = True
    # TODO[chunweiy]: Change to a larger value once SamplingConfig is connected to cpp runtime
    config.max_beam_width = 1

    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        enable_chunked_context=False,
        enable_trt_overlap=True,
        decoding_mode=DecodingMode.top_k,
    )

    sampling_config = llm.get_default_sampling_config()
    assert sampling_config is not None
    sampling_config.num_beams = 1

    for output in llm.generate(prompts):
        print(output)


class MyTokenizer(TokenizerBase):
    ''' A wrapper for the Transformers' tokenizer.
    This is the default tokenizer for LLM. '''

    @classmethod
    def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                                                  **kwargs)
        return MyTokenizer(tokenizer)

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def encode(self, text: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_encode_plus(self, texts: List[str], **kwargs) -> dict:
        return self.tokenizer.batch_encode_plus(texts, **kwargs)


def test_llm_with_customized_tokenizer():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        # a customized tokenizer is passed to override the default one
        tokenizer=MyTokenizer.from_pretrained(config.model_dir),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    for output in llm.generate(prompts):
        print(output)


def test_llm_without_tokenizer():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    sampling_config = SamplingConfig(end_id=2,
                                     pad_id=2,
                                     output_sequence_lengths=True,
                                     return_dict=True)

    prompts = [[23, 14, 3]]

    for output in llm.generate(prompts, sampling_config=sampling_config):
        assert not output.text, "The output should be empty since the tokenizer is missing"
        print(output)


@skip_single_gpu
def test_llm_build_engine_for_tp2(model_name=default_model_name):
    config = ModelConfig(get_model_path(model_name))
    config.parallel_config.tp_size = 2
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        llm.save(tmpdir)


@skip_single_gpu
@pytest.mark.parametrize("use_auto_parallel", [True, False],
                         ids=["enable_auto_parallel", "disable_auto_parallel"])
def test_llm_generate_for_tp2(use_auto_parallel):
    config = ModelConfig(llama_model_path)
    if use_auto_parallel:
        config.parallel_config.world_size = 2
        config.parallel_config.auto_parallel = True
    else:
        config.parallel_config.tp_size = 2
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    for output in llm.generate(prompts):
        print(output)


# TODO[chunweiy]: Move mixtral test to the e2e test
def is_memory_enough_for_mixtral():
    if torch.cuda.device_count() < 2:
        return False
    try:
        total_memory = get_total_gpu_memory(0) + get_total_gpu_memory(1)
        if total_memory >= 160 * 1024**3:
            return True
    except:
        return False


@skip_single_gpu
@pytest.mark.skipif(not is_memory_enough_for_mixtral(),
                    reason="The test needs at least 160GB memory, skipping")
def test_llm_generate_mixtral_for_tp2():
    config = ModelConfig(get_model_path(mixtral_model_name))
    config.parallel_config.tp_size = 2
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    for output in llm.generate(prompts):
        print(output)


def test_llm_generate_async(model_name=default_model_name,
                            tp_size: int = 1,
                            use_auto_parallel: bool = False):
    if "Mixtral" in model_name and use_auto_parallel:
        pytest.skip("Auto parallel is not supported for Mixtral models")
    config = ModelConfig(llama_model_path)
    if use_auto_parallel:
        config.parallel_config.world_size = tp_size
        config.parallel_config.auto_parallel = True
    else:
        config.parallel_config.tp_size = tp_size

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
    devices = config.parallel_config.get_devices()
    if torch.cuda.get_device_properties(devices[0]).major >= 8:
        kv_cache_config.enable_block_reuse = True

    llm = LLM(
        config,
        kv_cache_config=kv_cache_config,
    )

    def test_async(streaming: bool):

        async def task(prompt: str):
            outputs = []
            async for output in llm.generate_async(prompt, streaming=streaming):
                print('output', output)
                outputs.append(output.text)
            print(' '.join(outputs))

        async def main():
            tasks = [task(prompt) for prompt in prompts]
            await asyncio.gather(*tasks)

        asyncio.run(main())

    def test_wait(streaming: bool):
        for prompt in prompts:
            future = llm.generate_async(prompt, streaming=streaming)
            for output in future:
                print('wait', output)

    def test_non_streaming_usage_wait():
        for prompt in prompts:
            output = llm.generate_async(prompt, streaming=False)
            print(output.text)

    def test_future(streaming: bool):
        for prompt in prompts:
            future = llm.generate_async(prompt, streaming=streaming)
            if streaming is True:
                for output in future:
                    # Do something else and then wait for the result if needed
                    output = output.result(timeout=10)
                    print('future', output.text)
            else:
                # Do something else and then wait for the result if needed
                output = future.result(timeout=10)
                print('future', output.text)

    def test_future_async():

        async def task(prompt: str):
            future = llm.generate_async(prompt, streaming=False)
            output = await future.aresult()
            print('future', output.text)

        async def main():
            tasks = [task(prompt) for prompt in prompts]
            await asyncio.gather(*tasks)

        asyncio.run(main())

    test_async(streaming=True)
    test_async(streaming=False)
    test_wait(streaming=True)
    test_wait(streaming=False)
    test_future(streaming=True)
    test_future(streaming=False)
    test_future_async()
    test_non_streaming_usage_wait()


@skip_single_gpu
@pytest.mark.parametrize("use_auto_parallel", [True, False],
                         ids=["enable_auto_parallel", "disable_auto_parallel"])
def test_llm_generate_async_tp2(use_auto_parallel):
    test_llm_generate_async(default_model_name,
                            tp_size=2,
                            use_auto_parallel=use_auto_parallel)


# TODO[chunweiy]: Add test for loading inmemory model

if __name__ == '__main__':
    # test_llm_generate_async_tp2()
    test_llm_loading_from_hf()
