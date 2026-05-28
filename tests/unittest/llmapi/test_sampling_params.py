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
import pytest
import torch

from tensorrt_llm.llmapi.thinking_budget import (
    ThinkingBudgetLogitsProcessor,
    add_thinking_budget_logits_processor,
)
from tensorrt_llm.sampling_params import MAX_TOP_LOGPROBS, SamplingParams, check_logprobs_limit
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest


@pytest.mark.parametrize("field", ["logprobs", "prompt_logprobs", "top_logprobs"])
def test_check_logprobs_limit(field):
    check_logprobs_limit(field, None)
    check_logprobs_limit(field, 0)
    check_logprobs_limit(field, MAX_TOP_LOGPROBS)

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        check_logprobs_limit(field, -1)

    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        check_logprobs_limit(field, MAX_TOP_LOGPROBS + 1)


@pytest.mark.parametrize("field", ["logprobs", "prompt_logprobs"])
def test_logprobs_request_limit(field):
    SamplingParams(**{field: MAX_TOP_LOGPROBS})

    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        SamplingParams(**{field: MAX_TOP_LOGPROBS + 1})


def test_chat_top_logprobs_request_limit():
    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            logprobs=True,
            top_logprobs=MAX_TOP_LOGPROBS + 1,
        )


def test_completion_logprobs_assignment_revalidates():
    request = CompletionRequest(model="test", prompt="hi", logprobs=MAX_TOP_LOGPROBS)
    request.logprobs = MAX_TOP_LOGPROBS + 1

    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        request.to_sampling_params(backend="pytorch")


@pytest.mark.parametrize("value", [None, -1])
def test_thinking_token_budget_unlimited_values(value):
    assert SamplingParams(thinking_token_budget=value).thinking_token_budget is None


@pytest.mark.parametrize("value", [-2, True, 1.5, "1"])
def test_thinking_token_budget_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="thinking_token_budget"):
        SamplingParams(thinking_token_budget=value)


def test_chat_thinking_token_budget_request_validation():
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        thinking_token_budget=8,
    )

    assert request.to_sampling_params().thinking_token_budget == 8

    with pytest.raises(ValueError, match="thinking_token_budget"):
        ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking_token_budget=True,
        )


def test_thinking_budget_logits_processor_forces_end_token():
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=2,
        reasoning_start_token_ids=[1],
        reasoning_end_token_ids=[2, 3],
    )
    logits = torch.zeros(1, 1, 8)

    processor(0, logits, [[1, 5, 6]], None, None)

    assert logits[0, 0, 2] == 0
    mask = torch.ones(8, dtype=torch.bool)
    mask[2] = False
    assert torch.isneginf(logits[0, 0, mask]).all()


def test_thinking_budget_logits_processor_completes_partial_end_sequence():
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=2,
        reasoning_start_token_ids=[1],
        reasoning_end_token_ids=[2, 3],
    )
    logits = torch.zeros(1, 1, 8)

    processor(0, logits, [[1, 5, 6, 2]], None, None)

    assert logits[0, 0, 3] == 0
    mask = torch.ones(8, dtype=torch.bool)
    mask[3] = False
    assert torch.isneginf(logits[0, 0, mask]).all()


def test_thinking_budget_logits_processor_ignores_partial_end_before_budget():
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=4,
        reasoning_start_token_ids=[1],
        reasoning_end_token_ids=[2, 3],
    )
    logits = torch.zeros(1, 1, 8)

    processor(0, logits, [[1, 5, 6, 2]], None, None)

    assert torch.equal(logits, torch.zeros(1, 1, 8))


def test_thinking_budget_logits_processor_ignores_closed_reasoning_block():
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=1,
        reasoning_start_token_ids=[1],
        reasoning_end_token_ids=[2],
    )
    logits = torch.zeros(1, 1, 8)

    processor(0, logits, [[1, 5, 2, 6]], None, None)

    assert torch.equal(logits, torch.zeros(1, 1, 8))


def test_add_thinking_budget_logits_processor_uses_reasoning_parser_tokens():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return {
                "<think>": [1],
                "</think>": [2],
            }[text]

    sampling_params = SamplingParams(thinking_token_budget=4)

    add_thinking_budget_logits_processor(
        sampling_params,
        reasoning_parser="qwen3",
        tokenizer=FakeTokenizer(),
    )

    assert isinstance(sampling_params.logits_processor, ThinkingBudgetLogitsProcessor)


@pytest.mark.parametrize("thinking_token_budget", [0, 2])
def test_thinking_budget_text_generation_caps_reasoning_tokens(thinking_token_budget):
    class FakeTokenizer:
        token_to_id = {
            "Question:": 0,
            "<think>": 1,
            "</think>": 2,
            "alpha": 3,
            "beta": 4,
            "gamma": 5,
            "delta": 6,
            "answer": 7,
        }
        id_to_token = {token_id: token for token, token_id in token_to_id.items()}

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [self.token_to_id[token] for token in text.split()]

        def decode(self, token_ids):
            return " ".join(self.id_to_token[token_id] for token_id in token_ids)

    tokenizer = FakeTokenizer()
    sampling_params = SamplingParams(thinking_token_budget=thinking_token_budget)
    add_thinking_budget_logits_processor(
        sampling_params,
        reasoning_parser="qwen3",
        tokenizer=tokenizer,
    )

    prompt_ids = tokenizer.encode("Question:")
    generated_ids = []
    reasoning_plan = tokenizer.encode("alpha beta gamma delta")
    start_id = tokenizer.encode("<think>")[0]
    end_id = tokenizer.encode("</think>")[0]
    answer_id = tokenizer.encode("answer")[0]

    def next_model_token():
        if not generated_ids:
            return start_id
        if end_id in generated_ids:
            return answer_id
        reasoning_tokens = generated_ids[generated_ids.index(start_id) + 1 :]
        if len(reasoning_tokens) < len(reasoning_plan):
            return reasoning_plan[len(reasoning_tokens)]
        return end_id

    for _ in range(8):
        logits = torch.full((1, 1, len(tokenizer.token_to_id)), -100.0)
        logits[0, 0, next_model_token()] = 100.0
        sampling_params.logits_processor(0, logits, [prompt_ids + generated_ids], None, None)
        generated_ids.append(int(torch.argmax(logits[0, 0]).item()))
        if generated_ids[-1] == answer_id:
            break

    start_idx = generated_ids.index(start_id)
    end_idx = generated_ids.index(end_id)
    reasoning_ids = generated_ids[start_idx + 1 : end_idx]
    expected_reasoning_ids = reasoning_plan[:thinking_token_budget]

    assert tokenizer.decode(prompt_ids) == "Question:"
    assert reasoning_ids == expected_reasoning_ids
    assert len(reasoning_ids) <= thinking_token_budget
    assert generated_ids[end_idx + 1 :] == [answer_id]
    assert tokenizer.decode(generated_ids) == tokenizer.decode(
        [start_id, *expected_reasoning_ids, end_id, answer_id]
    )
