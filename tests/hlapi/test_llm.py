import asyncio
import os
import tempfile
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from tensorrt_llm.hlapi.llm import (LLM, ModelConfig, SamplingConfig,
                                    TokenizerBase, TransformersTokenizer)

llm_models_root = os.environ.get('LLM_MODELS_ROOT', None)
llama_model_path = os.path.join(llm_models_root, "llama-models/llama-7b-hf")
llm_engine_dir = os.environ.get('LLM_ENGINE_DIR', './tmp.engine')

prompts = ["Tell a story", "Who are you"]

cur_dir = os.path.dirname(os.path.abspath(__file__))
models_root = os.path.join(cur_dir, '../../models')


def test_tokenizer():
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)

    res = tokenizer("hello world")
    assert res


def test_llm_loadding_from_hf():
    config = ModelConfig(llama_model_path)
    llm = LLM(config)

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

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def batch_encode_plus(self, texts: List[str]) -> dict:
        return self.tokenizer.batch_encode_plus(texts)


def test_llm_with_customized_tokenizer():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        # a customized tokenizer is passed to override the default one
        tokenizer=MyTokenizer.from_pretrained(config.model_dir))

    for output in llm.generate(prompts):
        print(output)


def test_llm_without_tokenizer():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        # this will turn off tokenizer for pre-processing and post-processing
        enable_tokenizer=False,
    )

    sampling_config = SamplingConfig(end_id=2,
                                     pad_id=2,
                                     output_sequence_lengths=True,
                                     return_dict=True)

    prompts = [[23, 14, 3]]

    for output in llm.generate(prompts, sampling_config=sampling_config):
        print(output)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="The test needs at least 2 GPUs, skipping")
def test_llm_build_engine_for_tp2():
    config = ModelConfig(llama_model_path)
    config.parallel_config.tp_size = 2
    llm = LLM(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        llm.save(tmpdir)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="The test needs at least 2 GPUs, skipping")
def test_llm_generate_for_tp2():
    config = ModelConfig(llama_model_path)
    config.parallel_config.tp_size = 2
    llm = LLM(config)
    for output in llm.generate(prompts):
        print(output)


def test_llm_generate_async():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        async_mode=True,
        # set to 40%, since by default, the executor will occupy all the free memory, making some other tests OOM in CI
        kvcahe_free_gpu_memory_fraction=0.4)

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
                print(output)

    test_async(streaming=True)
    test_async(streaming=False)
    test_wait(streaming=True)
    test_wait(streaming=False)


# TODO[chunweiy]: Add test for loading inmemory model
# TODO[chunweiy]: Add a multi-gpu test on loading engine

if __name__ == '__main__':
    test_llm_generate_async()
