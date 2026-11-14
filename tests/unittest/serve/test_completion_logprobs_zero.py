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
"""`logprobs: 0` on /v1/completions asks for the sampled token's logprob only.

OpenAI documents `logprobs` as the number of most likely tokens to return in
addition to the chosen token, so 0 is a request for the chosen token alone and
only an absent `logprobs` (null) means "no logprobs". `SamplingParams.logprobs`
uses the same convention.
"""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.openai_protocol import CompletionRequest
from tensorrt_llm.serve.postprocess_handlers import (
    CompletionPostprocArgs,
    completion_response_post_processor,
    completion_stream_post_processor,
)

pytestmark = pytest.mark.cpu_only


class _Tokenizer:
    def decode(self, token_id, *args, **kwargs):
        return f"t{token_id}"


def _logprobs(token_ids):
    return [{token_id: SimpleNamespace(logprob=-0.5, rank=1)} for token_id in token_ids]


def _args(logprobs):
    request = CompletionRequest(model="m", prompt="hi", logprobs=logprobs, stream=True)
    args = CompletionPostprocArgs.from_request(request)
    args.prompt = "hi"
    args.num_prompt_tokens = 3
    args.tokenizer = _Tokenizer()
    return args


@pytest.mark.parametrize("logprobs", [None, 0, 1, 5])
def test_request_passes_logprobs_through_to_sampling_params(logprobs):
    request = CompletionRequest(model="m", prompt="hi", logprobs=logprobs)

    assert request.to_sampling_params(backend="pytorch").logprobs == logprobs


def test_zero_logprobs_kept_without_generation_logits_on_non_pytorch_backend():
    request = CompletionRequest(model="m", prompt="hi", logprobs=0)

    sampling_params = request.to_sampling_params(backend="tensorrt", gather_generation_logits=False)

    assert sampling_params.logprobs == 0
    assert sampling_params._get_output_config().return_log_probs


def test_logprobs_above_one_still_rejected_without_generation_logits():
    request = CompletionRequest(model="m", prompt="hi", logprobs=2)

    with pytest.raises(ValueError, match="logprobs"):
        request.to_sampling_params(backend="tensorrt", gather_generation_logits=False)


@pytest.mark.parametrize("logprobs", [None, 0, 2])
def test_non_streaming_response_carries_logprobs_unless_absent(logprobs):
    token_ids = [3, 4]
    output = SimpleNamespace(
        index=0,
        text="ab",
        token_ids=token_ids,
        logprobs=_logprobs(token_ids),
        length=2,
        disaggregated_params=None,
        stop_reason=None,
        finish_reason="length",
    )
    rsp = SimpleNamespace(outputs=[output], context_logits=None, cached_tokens=0, id=1)

    response = completion_response_post_processor(rsp, _args(logprobs))

    choice_logprobs = response.choices[0].logprobs
    if logprobs is None:
        assert choice_logprobs is None
        return
    assert choice_logprobs.tokens == ["t3", "t4"]
    assert choice_logprobs.token_logprobs == [-0.5, -0.5]
    assert [len(top) for top in choice_logprobs.top_logprobs] == [1, 1]


@pytest.mark.parametrize("logprobs", [None, 0, 2])
def test_streaming_chunk_carries_logprobs_unless_absent(logprobs):
    token_ids = [3, 4]
    output = SimpleNamespace(
        index=0,
        text_diff="ab",
        token_ids_diff=token_ids,
        token_ids=token_ids,
        logprobs_diff=_logprobs(token_ids),
        length=2,
        _last_text_len=0,
        stop_reason=None,
        finish_reason="length",
    )
    rsp = SimpleNamespace(outputs=[output], cached_tokens=0, id=1, _done=True)

    frames = completion_stream_post_processor(rsp, _args(logprobs))

    choice = json.loads(frames[0].removeprefix("data: "))["choices"][0]
    if logprobs is None:
        assert choice["logprobs"] is None
        return
    assert choice["logprobs"]["tokens"] == ["t3", "t4"]
    assert choice["logprobs"]["token_logprobs"] == [-0.5, -0.5]
