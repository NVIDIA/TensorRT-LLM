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
"""`echo: true` on /v1/completions with a prompt sent as token ids.

The completions API takes the prompt as text or as token ids. With `echo` the
reply starts with the prompt, which for token ids is the decoded text.
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
    """Stand-in whose token 1 is a special token."""

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return "".join(f"<{t}>" for t in token_ids if not (skip_special_tokens and t == 1))


def _args(prompt, **request_fields):
    request = CompletionRequest(model="m", prompt=prompt, echo=True, **request_fields)
    args = CompletionPostprocArgs.from_request(request)
    args.prompt = prompt
    args.num_prompt_tokens = 3
    args.tokenizer = _Tokenizer()
    return args


def _response(args):
    output = SimpleNamespace(
        index=0,
        text="ab",
        token_ids=[7, 8],
        length=2,
        disaggregated_params=None,
        stop_reason=None,
        finish_reason="length",
    )
    rsp = SimpleNamespace(outputs=[output], context_logits=None, cached_tokens=0, id=1)
    return completion_response_post_processor(rsp, args)


def _first_stream_text(args):
    output = SimpleNamespace(
        index=0,
        text_diff="ab",
        token_ids_diff=[7, 8],
        length=2,
        stop_reason=None,
        finish_reason=None,
    )
    rsp = SimpleNamespace(outputs=[output], cached_tokens=0, id=1, _done=False)
    frames = completion_stream_post_processor(rsp, args)
    return json.loads(frames[0].removeprefix("data: "))["choices"][0]["text"]


def test_text_prompt_is_echoed_as_sent():
    assert _response(_args("hi")).choices[0].text == "hiab"
    assert _first_stream_text(_args("hi", stream=True)) == "hiab"


def test_token_id_prompt_is_echoed_as_decoded_text():
    assert _response(_args([5, 6])).choices[0].text == "<5><6>ab"
    assert _first_stream_text(_args([5, 6], stream=True)) == "<5><6>ab"


@pytest.mark.parametrize(
    "skip_special_tokens, echoed", [(True, "<2><3>ab"), (False, "<1><2><3>ab")]
)
def test_echoed_token_id_prompt_follows_skip_special_tokens(skip_special_tokens, echoed):
    args = _args([1, 2, 3], skip_special_tokens=skip_special_tokens)

    assert _response(args).choices[0].text == echoed
