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
import threading

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.scheduled_encoder_executor import ScheduledEncoderExecutor
from tensorrt_llm.executor.utils import RequestError, is_context_length_exceeded_message

from .test_llm import get_model_path

BERT_MODEL_PATH = "bert/bert-base-uncased-yelp-polarity"
BERT_MAX_SEQ_LEN = 512
CLS_ID, SEP_ID = 101, 102


def _prompt(num_tokens: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [CLS_ID, *(rng.randrange(1000, 30522) for _ in range(num_tokens - 2)), SEP_ID]


def _llm(**kwargs) -> LLM:
    return LLM(model=get_model_path(BERT_MODEL_PATH), dtype="float32", **kwargs)


def _pooled(llm: LLM, prompt: list[int], max_tokens: int = 1) -> torch.Tensor:
    params = SamplingParams(max_tokens=max_tokens, return_context_logits=True)
    return llm.generate(prompt, sampling_params=params).context_logits.float().cpu()


def test_over_long_request_fails_alone_with_chunked_prefill():
    # The prompts that fit max_num_tokens add up to more than it, so they span batches unsplit.
    prompts = [
        _prompt(16, seed=1),
        _prompt(80),
        _prompt(24, seed=2),
        _prompt(40, seed=3),
        _prompt(40, seed=4),
    ]
    fitting = (0, 2, 3, 4)
    params = SamplingParams(max_tokens=1, return_context_logits=True)

    with _llm(
        max_num_tokens=64, enable_chunked_prefill=True, disable_overlap_scheduler=True
    ) as llm:
        futures = [llm.generate_async(prompt, sampling_params=params) for prompt in prompts]
        with pytest.raises(RequestError) as error:
            futures[1].result()
        assert is_context_length_exceeded_message(str(error.value))
        batched = [futures[i].result().context_logits.float().cpu() for i in fitting]
        alone = [_pooled(llm, prompts[i]) for i in fitting]

    for got, expected in zip(batched, alone):
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_prompt_that_fills_max_seq_len_matches_encode():
    prompt = _prompt(BERT_MAX_SEQ_LEN)

    with _llm(disable_overlap_scheduler=True) as llm:
        scheduled = _pooled(llm, prompt)
        with pytest.raises(RequestError) as error:
            _pooled(llm, _prompt(BERT_MAX_SEQ_LEN + 1))
        assert is_context_length_exceeded_message(str(error.value))
    with _llm(encode_only=True) as llm:
        encoded = llm.encode(prompt).logits.float().cpu()

    torch.testing.assert_close(scheduled.reshape(-1), encoded.reshape(-1), rtol=1e-4, atol=1e-4)


def test_requests_that_ask_for_more_tokens_get_the_same_pooled_row():
    prompts = [_prompt(16, seed=seed) for seed in range(4)]

    with _llm() as llm:
        longer = [_pooled(llm, prompt, max_tokens=4) for prompt in prompts]
        reference = [_pooled(llm, prompt) for prompt in prompts]

    for got, expected in zip(longer, reference):
        assert torch.equal(got, expected)


def test_cancelling_a_queued_request_finishes_it(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    params = SamplingParams(max_tokens=1, return_context_logits=True)

    with _llm(max_batch_size=1) as llm:
        engine = llm._executor.engine
        assert isinstance(engine, ScheduledEncoderExecutor)
        # The first forward waits for the abort, so the last request is still queued when it lands.
        aborted = threading.Event()
        forward = engine._runner.forward

        def forward_after_abort(*args, **kwargs):
            assert aborted.wait(timeout=60)
            return forward(*args, **kwargs)

        monkeypatch.setattr(engine._runner, "forward", forward_after_abort)
        futures = [
            llm.generate_async(_prompt(32, seed=seed), sampling_params=params) for seed in range(50)
        ]
        futures[-1].abort()
        aborted.set()
        finished = [future.result(timeout=60).outputs[0].finish_reason for future in futures]

    assert finished == ["length"] * 49 + ["cancelled"]


def test_startup_metrics_report_model_loading_and_executor_phases():
    with _llm() as llm:
        metrics = llm.startup_metrics

    assert {
        "scheduled_encoder_creation_seconds",
        "worker_start_seconds",
        "total_py_executor_creation_seconds",
    } <= metrics["py_executor"].keys()
    assert metrics["model_loader"]
