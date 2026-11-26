# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from typing import List

import pytest
import transformers

from tensorrt_llm import LLM
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.llmapi import KvCacheRetentionConfig
from tensorrt_llm.llmapi.llm_utils import _ParallelConfig
from tensorrt_llm.llmapi.tokenizer import TokenizerBase
from tensorrt_llm.sampling_params import SamplingParams

from .llm_test_utils import global_kvcache_config, llm_test_harness, prompts
from .model_paths import llama_model_path

pytestmark = pytest.mark.threadleak(enabled=False)


def test_llm_loading_from_hf():
    sampling_params = SamplingParams(max_tokens=8)
    llm_test_harness(llama_model_path,
                     prompts, ["D E F G H I J K"],
                     sampling_params=sampling_params,
                     kv_cache_config=global_kvcache_config)


class MyTokenizer(TokenizerBase):
    """A wrapper for the Transformers' tokenizer.

    This is the default tokenizer for LLM.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_dir, **kwargs)
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
        return self.tokenizer(texts, **kwargs)


def test_llm_with_customized_tokenizer():
    llm = LLM(
        model=llama_model_path,
        # a customized tokenizer is passed to override the default one
        tokenizer=MyTokenizer.from_pretrained(llama_model_path),
        kv_cache_config=global_kvcache_config,
    )

    for output in llm.generate(prompts):
        print(output)


def test_llm_without_tokenizer():
    llm = LLM(
        model=llama_model_path,
        skip_tokenizer_init=True,
        kv_cache_config=global_kvcache_config,
    )

    sampling_params = SamplingParams(end_id=2, pad_id=2, max_tokens=8)

    prompts = [[23, 14, 3]]

    for output in llm.generate(prompts, sampling_params=sampling_params):
        assert not output.outputs[0].text, \
            "The output should be empty since the tokenizer is missing"
        print(output)


def test_llm_with_kv_cache_retention_config():
    kv_cache_retention_config = KvCacheRetentionConfig([
        KvCacheRetentionConfig.TokenRangeRetentionConfig(
            0, 2, 30, datetime.timedelta(seconds=30))
    ], 80, None, tllm.KvCacheTransferMode.DRAM, "test_dir")

    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)

    for output in llm.generate(
            prompts, kv_cache_retention_config=kv_cache_retention_config):
        print(output)


def test_parallel_config():
    config = _ParallelConfig()
    config.tp_size = 2
    config.pp_size = 2
    assert config.world_size == 4
    config.world_size = 4  # should not raise exception

    with pytest.raises(ValueError):
        config.world_size = 5
