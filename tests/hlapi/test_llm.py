import asyncio
import os
import pickle
import sys
import tempfile
from typing import List

import pytest
import torch
from parameterized import parameterized
from transformers import AutoTokenizer

from tensorrt_llm.hlapi.llm import (LLM, KvCacheConfig, ModelConfig,
                                    ParallelConfig, SamplingConfig,
                                    StreamingLLMParam, TokenizerBase)
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer
from tensorrt_llm.hlapi.utils import GpuArch, get_total_gpu_memory

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import force_ampere, unittest_name_func

from tensorrt_llm.models.llama.model import LLaMAForCausalLM

# The unittests are based on the tiny-llama, which is fast to build and run.
# There are other tests based on llama-7B model, such as the end-to-end tests in test_e2e.py, and parallel tests in test_llm_multi_gpu.py.


def get_model_path(model_name):
    engine_dir = os.environ.get('LLM_ENGINE_DIR', None)
    if engine_dir:
        return engine_dir
    return str(llm_models_root() / model_name)


default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
mixtral_model_name = "Mixtral-8x7B-v0.1"

llama_model_path = get_model_path(default_model_name)
llm_engine_dir = os.environ.get('LLM_ENGINE_DIR', './tmp.engine')
prompts = ["A B C"]


@pytest.mark.parametrize("enable_executor", [True, False])
def test_llm_loading_from_hf(enable_executor: bool):
    config = ModelConfig(llama_model_path)
    # The performance-related flags are turned on eagerly to check the functionality

    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        enable_chunked_context=False,
        enable_trt_overlap=True if not enable_executor else False,
        enable_executor=enable_executor,
    )

    sampling_config = llm.get_default_sampling_config()
    assert sampling_config is not None
    sampling_config.max_new_tokens = 8

    for output in llm.generate(prompts, sampling_config=sampling_config):
        print(output)
        if enable_executor:
            assert output.text == "D E F G H I J K"
        else:
            assert output.text == "<s> A B C D E F G H I J K"


@force_ampere
def test_llm_loading_from_ckpt():
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    assert tokenizer is not None
    with tempfile.TemporaryDirectory() as ckpt_dir:
        llama = LLaMAForCausalLM.from_hugging_face(llama_model_path)
        llama.save_checkpoint(ckpt_dir)
        del llama

        config = ModelConfig(ckpt_dir)
        llm = LLM(
            config,
            tokenizer=tokenizer,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        )

        sampling_config = llm.get_default_sampling_config()
        assert sampling_config is not None
        sampling_config.max_new_tokens = 8

        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)
            assert output.text == "D E F G H I J K"


def llm_end2end_cases():
    yield {},  # Default options
    yield {'trt_strongly_typed': False},
    if GpuArch.is_post_ampere():
        yield {'multi_block_mode': True},
    yield {'use_fused_mlp': True},


@parameterized.expand(llm_end2end_cases(), name_func=unittest_name_func)
def test_llm_end2end(llm_additional_options):
    model_path = get_model_path(default_model_name)
    config = ModelConfig(model_path)
    llm = LLM(config, **llm_additional_options)

    if 'trt_strongly_typed' in llm_additional_options:
        assert llm._build_config.strongly_typed == llm_additional_options.get(
            'trt_strongly_typed')
    else:
        assert llm._build_config.strongly_typed is True
    if 'use_fused_mlp' in llm_additional_options:
        assert llm._build_config.use_fused_mlp == llm_additional_options.pop(
            'use_fused_mlp')
    else:
        assert llm._build_config.use_fused_mlp is False

    sampling_config = llm.get_default_sampling_config()
    sampling_config.max_new_tokens = 8
    assert sampling_config is not None
    for output in llm.generate(prompts, sampling_config=sampling_config):
        print(output)
        assert output.text == "D E F G H I J K"


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

    sampling_config = SamplingConfig(end_id=2, pad_id=2, max_new_tokens=8)

    prompts = [[23, 14, 3]]

    for output in llm.generate(prompts, sampling_config=sampling_config):
        assert not output.text, "The output should be empty since the tokenizer is missing"
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


@pytest.mark.parametrize("enable_executor", [True, False])
def test_llm_generate_async(enable_executor):
    _test_llm_generate_async()


def _test_llm_generate_async(model_name=default_model_name,
                             tp_size: int = 1,
                             use_auto_parallel: bool = False,
                             tokenizer=None,
                             enable_executor=True):
    if "Mixtral" in model_name and use_auto_parallel:
        pytest.skip("Auto parallel is not supported for Mixtral models")

    config = ModelConfig(llama_model_path)
    if use_auto_parallel:
        config.parallel_config.auto_parallel = True
        config.parallel_config.world_size = tp_size
    else:
        config.parallel_config.tp_size = tp_size

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
    devices = config.parallel_config.devices
    if torch.cuda.get_device_properties(devices[0]).major >= 8:
        kv_cache_config.enable_block_reuse = True

    llm = LLM(
        config,
        tokenizer=tokenizer,
        kv_cache_config=kv_cache_config,
        enable_executor=enable_executor,
    )

    sampling_config = llm.get_default_sampling_config()
    sampling_config.max_new_tokens = 6

    def test_async(streaming: bool):

        async def task(prompt: str):
            outputs = []
            async for output in llm.generate_async(
                    prompt, streaming=streaming,
                    sampling_config=sampling_config):
                print('output', output)
                outputs.append(output.text)
            print(' '.join(outputs))

        async def main():
            tasks = [task(prompt) for prompt in prompts]
            await asyncio.gather(*tasks)

        asyncio.run(main())

    def test_wait(streaming: bool):
        for prompt in prompts:
            future = llm.generate_async(prompt,
                                        streaming=streaming,
                                        sampling_config=sampling_config)
            for output in future:
                print('wait', output)

    def test_non_streaming_usage_wait():
        for prompt in prompts:
            output = llm.generate_async(prompt,
                                        streaming=False,
                                        sampling_config=sampling_config)
            print(output.text)

    def test_future(streaming: bool):
        for prompt in prompts:
            future = llm.generate_async(prompt,
                                        streaming=streaming,
                                        sampling_config=sampling_config)
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
            future = llm.generate_async(prompt,
                                        streaming=False,
                                        sampling_config=sampling_config)
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


@force_ampere
def test_generate_with_sampling_config():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    def test_sampling_config_per_prompt():
        sampling_configs = [llm.get_default_sampling_config() for _ in range(2)]
        sampling_configs[0].max_new_tokens = 4
        sampling_configs[1].max_new_tokens = 8
        for sc in sampling_configs:
            sc.end_id = -1
            sc.pad_id = -1

        for off, output in enumerate(
                llm.generate(prompts, sampling_config=sampling_configs)):
            output_len = len(output.token_ids)
            print(f"output_len: {output_len}")
            assert output_len <= sampling_configs[off].max_new_tokens

    def test_temperature():
        sampling_config = llm.get_default_sampling_config()
        sampling_config.max_new_tokens = 6
        sampling_config.temperature = [0.5]
        sampling_config.beam_search_diversity_rate = [0.5]
        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)

    def test_top_k():
        sampling_config = llm.get_default_sampling_config()
        sampling_config.max_new_tokens = 6
        sampling_config.top_k = [10]
        sampling_config.top_p = [0.92]
        print('top_k')
        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)

    def test_top_p():
        sampling_config = llm.get_default_sampling_config()
        sampling_config.max_new_tokens = 6
        sampling_config.top_p = [0.92]
        print('top_p')
        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)

    def test_penalty():
        sampling_config = llm.get_default_sampling_config()
        sampling_config.max_new_tokens = 8
        sampling_config.length_penalty = [1.0]
        sampling_config.presence_penalty = [0.0]
        sampling_config.repetition_penalty = [1.0]
        sampling_config.min_length = [5]
        print('penalty')

        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)

    def test_early_stopping():
        sampling_config = llm.get_default_sampling_config()
        sampling_config.max_new_tokens = 6
        sampling_config.early_stopping = [5]
        print('early stop')
        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)

    test_top_k()
    test_top_p()
    test_early_stopping()

    test_sampling_config_per_prompt()
    test_temperature()
    test_penalty()


@force_ampere
def test_generate_with_beam_search():
    config = ModelConfig(llama_model_path, max_beam_width=2)
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    prompt = ["Tell me a story"]

    sampling_config = llm.get_default_sampling_config()
    sampling_config.max_new_tokens = 6
    sampling_config.beam_width = 2

    for output in llm.generate(prompts, sampling_config=sampling_config):
        print(output)
        assert len(output.text) == 2
        assert len(output.token_ids) == 2
        assert len(output.token_ids[0]) <= len(
            prompt[0].split()) + sampling_config.max_new_tokens


@force_ampere
def test_generate_with_streaming_llm():
    config = ModelConfig(llama_model_path)
    # TODO[chunweiy]: Test with larger size when the underlying support is ready
    llm = LLM(config, streaming_llm=StreamingLLMParam(64, 4))

    for output in llm.generate(prompts):
        print(output)


def test_sampling_config():
    sc = SamplingConfig()
    sc.max_new_tokens = 1024

    sc0 = pickle.loads(pickle.dumps(sc))
    assert sc0.max_new_tokens == 1024


def test_parallel_config():
    config = ParallelConfig()
    config.tp_size = 2
    config.pp_size = 2
    assert config.world_size == 4
    config.world_size = 4  # should not raise exception


# TODO[chunweiy]: Add test for loading inmemory model

if __name__ == '__main__':
    test_llm_loading_from_hf(True)
    test_llm_generate_async(True)
    test_llm_without_tokenizer()
    test_generate_with_streaming_llm()
    test_generate_with_sampling_config()
