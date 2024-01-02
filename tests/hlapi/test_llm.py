import os
import tempfile
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from tensorrt_llm.hlapi.llm import (LLM, ModelConfig, SamplingConfig,
                                    TokenIdsTy, TokenizerBase)

llm_models_root = os.environ.get('LLM_MODELS_ROOT',
                                 '/scratch.trt_llm_data/llm-models/')
llm_engine_root = os.environ.get('LLM_ENGINE_ROOT', None)

llama_model_path = os.path.join(llm_models_root, "llama-models/llama-7b-hf")

prompts = ["Tell a story", "Who are you"]


def test_llm_loadding_from_hf():
    config = ModelConfig(model_dir=llama_model_path)
    llm = LLM(config)

    for output in llm(prompts):
        print(output)


def _test_llm_loading_from_engine():
    # TODO[chunweiy]: Enable this test later, OOM
    # build the engine
    config = ModelConfig(model_dir=llama_model_path)
    llm = LLM(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        llm.save(tmpdir)
        del llm

        config = ModelConfig(model_dir=tmpdir)
        new_llm = LLM(config)

        for output in new_llm(prompts):
            print(output)


class MyTokenizer(TokenizerBase):
    ''' A wrapper for the Transformers' tokenizer.
    This is the default tokenizer for LLM. '''

    @classmethod
    def from_pretrained(self, pretrained_model_dir: str, **kwargs):
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

    def encode(self, text: str) -> TokenIdsTy:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: TokenIdsTy) -> str:
        return self.tokenizer.decode(token_ids)

    def batch_encode_plus(self, texts: List[str]) -> dict:
        return self.tokenizer.batch_encode_plus(texts)


def test_llm_with_customized_tokenizer():
    config = ModelConfig(model_dir=llama_model_path)
    llm = LLM(
        config,
        # a customized tokenizer is passed to override the default one
        tokenizer=MyTokenizer.from_pretrained(config.model_dir))

    for output in llm(prompts):
        print(output)


def test_llm_without_tokenizer():
    config = ModelConfig(model_dir=llama_model_path)
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

    for output in llm(prompts, sampling_config=sampling_config):
        print(output)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="The test needs at least 2 GPUs, skipping")
def test_llm_build_engine_for_tp2():
    config = ModelConfig(model_dir=llama_model_path)
    config.parallel_config.tp_size = 2
    llm = LLM(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        engine_path = llm_engine_root or tmpdir
        llm.save(engine_path)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="The test needs at least 2 GPUs, skipping")
def test_llm_generate_for_tp2():
    config = ModelConfig(model_dir=llama_model_path)
    config.parallel_config.tp_size = 2
    llm = LLM(config)
    for output in llm(prompts):
        print(output)


# TODO[chunweiy]: Add a multi-gpu test on loading engine

if __name__ == '__main__':
    test_llm_loadding_from_hf()
