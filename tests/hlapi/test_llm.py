import os
import subprocess
import tempfile
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from tensorrt_llm.hlapi.llm import (LLM, ModelConfig, SamplingConfig,
                                    TokenizerBase, TransformersTokenizer)

llm_models_root = os.environ.get('LLM_MODELS_ROOT',
                                 '/scratch.trt_llm_data/llm-models/')
llm_engine_dir = os.environ.get('LLM_ENGINE_DIR', None)

llama_model_path = os.path.join(llm_models_root, "llama-models/llama-7b-hf")

prompts = ["Tell a story", "Who are you"]

cur_dir = os.path.dirname(os.path.abspath(__file__))
models_root = os.path.join(cur_dir, '../../models')


def test_tokenizer():
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)

    res = tokenizer("hello world")
    assert res


def test_llm_loadding_from_hf():
    config = ModelConfig(model_dir=llama_model_path)
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
    config = ModelConfig(model_dir=llama_model_path)
    llm = LLM(
        config,
        # a customized tokenizer is passed to override the default one
        tokenizer=MyTokenizer.from_pretrained(config.model_dir))

    for output in llm.generate(prompts):
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

    for output in llm.generate(prompts, sampling_config=sampling_config):
        print(output)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="The test needs at least 2 GPUs, skipping")
def test_llm_build_engine_for_tp2():
    config = ModelConfig(model_dir=llama_model_path)
    config.parallel_config.tp_size = 2
    llm = LLM(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        llm.save(tmpdir)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="The test needs at least 2 GPUs, skipping")
def test_llm_generate_for_tp2():
    config = ModelConfig(model_dir=llama_model_path)
    config.parallel_config.tp_size = 2
    llm = LLM(config)
    for output in llm.generate(prompts):
        print(output)


# TODO[chunweiy]: Add test for loading inmemory model

# TODO[chunweiy]: Add a multi-gpu test on loading engine


def _test_llm_int4_awq_quantization():
    # TODO[chunweiy]: Enable it on L0 tests
    config = ModelConfig(model_dir=llama_model_path)
    config.quant_config.init_from_description(quantize_weights=True,
                                              use_int4_weights=True,
                                              per_group=True)
    assert config.quant_config.has_any_quant()

    llm = LLM(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm.save(tmpdir)


if __name__ == '__main__':

    def get_faked_engine():
        temp_dir = tempfile.TemporaryDirectory()
        subprocess.run([
            'bash',
            os.path.join(cur_dir, './fake.sh'), llama_model_path, temp_dir.name
        ],
                       check=True)

        return temp_dir

    engine = get_faked_engine()
    test_llm_generate_async(engine)
