# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scheduled pooling: a non-generation model served through LLM.generate()."""

import random

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.executor.utils import RequestError, is_context_length_exceeded_message

from .test_llm import get_model_path

BERT_MODEL_PATH = "bert/bert-base-uncased-yelp-polarity"
CLS_ID, SEP_ID = 101, 102


def _prompt(num_tokens: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [CLS_ID, *(rng.randrange(1000, 30522) for _ in range(num_tokens - 2)), SEP_ID]


def _llm(**kwargs) -> LLM:
    return LLM(model=get_model_path(BERT_MODEL_PATH), dtype="float32", **kwargs)


def _pooled(llm: LLM, prompt: list[int]) -> torch.Tensor:
    params = SamplingParams(max_tokens=1, return_context_logits=True)
    return llm.generate(prompt, sampling_params=params).context_logits.float().cpu()


def test_over_long_request_fails_alone_with_chunked_prefill():
    prompts = [_prompt(16, seed=1), _prompt(80), _prompt(24, seed=2)]
    params = SamplingParams(max_tokens=1, return_context_logits=True)

    with _llm(
        max_num_tokens=64, enable_chunked_prefill=True, disable_overlap_scheduler=True
    ) as llm:
        futures = [llm.generate_async(prompt, sampling_params=params) for prompt in prompts]
        with pytest.raises(RequestError) as error:
            futures[1].result()
        assert is_context_length_exceeded_message(str(error.value))
        batched = [futures[i].result().context_logits.float().cpu() for i in (0, 2)]
        alone = [_pooled(llm, prompts[i]) for i in (0, 2)]

    for got, expected in zip(batched, alone):
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
