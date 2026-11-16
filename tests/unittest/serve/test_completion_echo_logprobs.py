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
"""`echo` with `logprobs` on /v1/completions reports the prompt's logprobs.

As in the OpenAI completions API, the logprobs of an echoed prompt start with
one entry per prompt token (None for the first, which has no context) and go
on with the generated tokens. lm-eval's loglikelihood requests read the
continuation's logprobs from these prompt entries.
"""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.serve.openai_protocol import CompletionRequest
from tensorrt_llm.serve.postprocess_handlers import (
    CompletionPostprocArgs,
    completion_response_post_processor,
    completion_stream_post_processor,
    create_completion_logprobs,
)

pytestmark = pytest.mark.cpu_only

_TEXT = {1: "<s>", 10: "The", 11: " sky", 12: " is", 13: " cloud", 20: " blue", 21: " green"}

# "<s>The sky is", as the model sees it.
PROMPT_TOKEN_IDS = [1, 10, 11, 12]
# What the engine returns for prompt_logprobs=1: entry t scores prompt token t + 1, and the
# last entry scores the first generated token.
PROMPT_LOGPROBS = [
    {10: Logprob(logprob=-2.0, rank=1)},
    {13: Logprob(logprob=-1.0, rank=1), 11: Logprob(logprob=-3.0, rank=2)},
    {12: Logprob(logprob=-0.5, rank=1)},
    {20: Logprob(logprob=-0.25, rank=1)},
]
GENERATED_LOGPROBS = [{20: Logprob(logprob=-0.25, rank=1)}]


class _Tokenizer:
    def decode(self, token_ids, **kwargs):
        ids = token_ids if isinstance(token_ids, list) else [token_ids]
        return "".join(_TEXT[t] for t in ids)


def _args(stream=False):
    request = CompletionRequest(
        model="m", prompt="The sky is", echo=True, logprobs=1, max_tokens=2, stream=stream
    )
    args = CompletionPostprocArgs.from_request(request)
    args.prompt = request.prompt
    args.prompt_token_ids = PROMPT_TOKEN_IDS
    args.num_prompt_tokens = len(PROMPT_TOKEN_IDS)
    args.tokenizer = _Tokenizer()
    return args


def _rsp(output):
    return SimpleNamespace(
        outputs=[output], context_logits=None, cached_tokens=0, id=1, _done=False
    )


def _stream_logprobs(args, **output_fields):
    output = SimpleNamespace(
        index=0,
        finish_reason=None,
        stop_reason=None,
        prompt_logprobs=PROMPT_LOGPROBS,
        **output_fields,
    )
    (event,) = completion_stream_post_processor(_rsp(output), args)
    return json.loads(event[len("data: ") :])["choices"][0]["logprobs"]


@pytest.mark.parametrize(
    "echo, logprobs, prompt_logprobs", [(True, 1, 1), (True, None, None), (False, 1, None)]
)
def test_echo_with_logprobs_requests_prompt_logprobs(echo, logprobs, prompt_logprobs):
    request = CompletionRequest(model="m", prompt="The sky is", echo=echo, logprobs=logprobs)
    assert request.to_sampling_params(backend="pytorch").prompt_logprobs == prompt_logprobs


def test_response_reports_prompt_then_generated_logprobs():
    output = SimpleNamespace(
        index=0,
        text=" blue",
        token_ids=[20],
        logprobs=GENERATED_LOGPROBS,
        prompt_logprobs=PROMPT_LOGPROBS,
        length=1,
        disaggregated_params=None,
        stop_reason=None,
        finish_reason="length",
    )
    choice = completion_response_post_processor(_rsp(output), _args()).choices[0]

    assert choice.text == "The sky is blue"
    assert choice.logprobs.tokens == ["<s>", "The", " sky", " is", " blue"]
    assert choice.logprobs.token_logprobs == [None, -2.0, -3.0, -0.5, -0.25]
    assert choice.logprobs.top_logprobs[:3] == [None, {"The": -2.0}, {" cloud": -1.0, " sky": -3.0}]
    assert choice.logprobs.text_offset == [0, 3, 6, 10, 13]
    # lm-eval sends context + continuation with max_tokens=1 and sums the continuation's entries.
    context_length = 3
    assert sum(choice.logprobs.token_logprobs[context_length:-1]) == -0.5


def test_stream_reports_prompt_logprobs_in_the_first_chunk_only():
    args = _args(stream=True)
    first = _stream_logprobs(
        args,
        text_diff=" blue",
        token_ids_diff=[20],
        logprobs_diff=GENERATED_LOGPROBS,
        _last_text_len=0,
        length=1,
    )
    second = _stream_logprobs(
        args,
        text_diff=" green",
        token_ids_diff=[21],
        logprobs_diff=[{21: Logprob(logprob=-1.5, rank=1)}],
        _last_text_len=5,
        length=2,
    )

    assert first["tokens"] == ["<s>", "The", " sky", " is", " blue"]
    assert first["token_logprobs"] == [None, -2.0, -3.0, -0.5, -0.25]
    assert first["text_offset"] == [0, 3, 6, 10, 13]
    assert second["tokens"] == [" green"]
    assert second["text_offset"] == [18]


def test_text_offset_starts_each_token_after_the_previous_one():
    logprobs = create_completion_logprobs([20, 21], _Tokenizer(), [-0.25, -1.5], initial_offset=4)
    assert logprobs.text_offset == [4, 9]
