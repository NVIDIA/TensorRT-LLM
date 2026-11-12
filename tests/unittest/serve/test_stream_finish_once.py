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
"""A streamed choice reports its finish_reason once, also when n > 1.

Without beam search every response of an `n > 1` request carries one sequence,
yet `rsp.outputs` lists all `n` outputs each time. An output that finished in an
earlier response is therefore listed, still finished, in every later response.
The stream must not repeat its finish_reason chunk (OpenAI and vLLM send one per
choice).
"""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    CompletionPostprocArgs,
    chat_stream_post_processor,
    completion_stream_post_processor,
)

pytestmark = pytest.mark.cpu_only


def _output(index, text, token_ids, finish_reason):
    return SimpleNamespace(
        index=index,
        text_diff=text,
        token_ids_diff=token_ids,
        finish_reason=finish_reason,
        stop_reason=None,
        length=3,
        _last_text_len=0,
    )


def _rsp(outputs, done=False):
    return SimpleNamespace(outputs=outputs, cached_tokens=0, _done=done, id=7)


def _responses():
    # Sequence 0 finishes in the first response. Later responses carry sequence 1
    # and list sequence 0 again with nothing new, still finished.
    return [
        _rsp([_output(0, "A", [1], "stop"), _output(1, "", [], None)]),
        _rsp([_output(0, "", [], "stop"), _output(1, "B", [2], None)]),
        _rsp([_output(0, "", [], "stop"), _output(1, "", [], "length")], done=True),
    ]


def _choices(frames):
    for frame in frames:
        yield from json.loads(frame.removeprefix("data: "))["choices"]


def _chat_choices():
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], n=2, stream=True
    )
    args = ChatPostprocArgs.from_request(request)
    args.num_prompt_tokens = 4
    return [c for rsp in _responses() for c in _choices(chat_stream_post_processor(rsp, args))]


def _completion_choices():
    request = CompletionRequest(model="m", prompt="hi", n=2, stream=True)
    args = CompletionPostprocArgs.from_request(request)
    args.prompt = "hi"
    args.num_prompt_tokens = 3
    return [
        c for rsp in _responses() for c in _choices(completion_stream_post_processor(rsp, args))
    ]


@pytest.mark.parametrize("collect", [_chat_choices, _completion_choices])
def test_each_choice_reports_finish_reason_once(collect):
    finished = [(c["index"], c["finish_reason"]) for c in collect() if c.get("finish_reason")]

    assert sorted(finished) == [(0, "stop"), (1, "length")]


@pytest.mark.parametrize("collect", [_chat_choices, _completion_choices])
def test_a_finished_choice_streams_nothing_after_its_finish(collect):
    choices = collect()
    first_finish = min(
        position for position, c in enumerate(choices) if c["index"] == 0 and c.get("finish_reason")
    )

    assert not [c for c in choices[first_finish + 1 :] if c["index"] == 0]
