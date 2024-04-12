import asyncio
import os
import pickle
import sys
import tempfile
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from tensorrt_llm.hlapi.llm import (LLM, KvCacheConfig, ModelConfig,
                                    SamplingConfig, StreamingLLMParam,
                                    TokenizerBase)
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer
from tensorrt_llm.hlapi.utils import get_total_gpu_memory

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import force_ampere

from tensorrt_llm.models.llama.model import LLaMAForCausalLM


def get_model_path(model_name):
    engine_dir = os.environ.get('LLM_ENGINE_DIR', None)
    if engine_dir:
        return engine_dir
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

    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        enable_chunked_context=False,
        enable_trt_overlap=True,
    )

    sampling_config = llm.get_default_sampling_config()
    assert sampling_config is not None

    for output in llm.generate(prompts):
        print(output)
        assert output.text == "<s> A B C D E F G H I J K L M N O P Q R S T U V W X Y Z\nA B C D E F G H"


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

        for output in llm.generate(prompts):
            print(output)
            assert output.text == "<s> A B C D E F G H I J K L M N O P Q R S T U V W X Y Z\nA B C D E F G H"


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


def test_llm_generate_async(model_name=default_model_name,
                            tp_size: int = 1,
                            use_auto_parallel: bool = False,
                            tokenizer=None):
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
        tokenizer=tokenizer,
        kv_cache_config=kv_cache_config,
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

    prompt = ["Tell me a story"]

    def test_sampling_config_per_prompt():
        sampling_configs = [llm.get_default_sampling_config() for _ in range(2)]
        sampling_configs[0].max_new_tokens = 6
        sampling_configs[1].max_new_tokens = 10
        for sc in sampling_configs:
            sc.end_id = -1
            sc.pad_id = -1

        prompts = ["Tell me a story"] * 2

        input_len = len(prompt[0].split())
        for off, output in enumerate(
                llm.generate(prompts, sampling_config=sampling_configs)):
            output_len = len(output.token_ids) - input_len - 1
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
        sampling_config.top_k = [1]
        sampling_config.top_p = [0.92]
        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)

    def test_top_p():
        sampling_config = llm.get_default_sampling_config()
        sampling_config.max_new_tokens = 6
        sampling_config.top_p = [0.92]
        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)

    def test_penalty():
        sampling_config = llm.get_default_sampling_config()
        sampling_config.max_new_tokens = 6
        sampling_config.length_penalty = [0.8]
        sampling_config.presence_penalty = [0.8]
        sampling_config.repetition_penalty = [0.8]
        sampling_config.min_length = [5]

        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)

    def test_early_stopping():
        sampling_config = llm.get_default_sampling_config()
        sampling_config.max_new_tokens = 6
        sampling_config.early_stopping = [True]
        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)

    test_sampling_config_per_prompt()
    test_temperature()
    test_penalty()
    test_early_stopping()
    # TODO[chunweiy]: Enable the top_k and top_p test on the new Executor, currently on gptManager, something wrong.
    #test_top_k()
    #test_top_p()


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


# TODO[chunweiy]: Add test for loading inmemory model

if __name__ == '__main__':
    test_llm_without_tokenizer()
    test_generate_with_streaming_llm()
    test_generate_with_sampling_config()
    test_llm_loading_from_hf()
    test_llm_generate_async_tp2(use_auto_parallel=True)
    test_llm_generate_async_tp2(use_auto_parallel=False)
    test_sampling_config()
