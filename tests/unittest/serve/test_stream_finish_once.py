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

from tensorrt_llm.executor.result import DetokenizedGenerationResultBase, GenerationResultBase
from tensorrt_llm.sampling_params import SamplingParams
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
    # and list sequence 0 again with nothing new, still finished. The last
    # response carries a text delta of its own.
    return [
        _rsp([_output(0, "A", [1], "stop"), _output(1, "", [], None)]),
        _rsp([_output(0, "", [], "stop"), _output(1, "B", [2], None)]),
        _rsp([_output(0, "", [], "stop"), _output(1, "C", [3], "length")], done=True),
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


def _text(choice):
    # chat chunks carry the delta under "delta", completion chunks under "text"
    if "text" in choice:
        return choice["text"]
    return choice["delta"].get("content") or ""


@pytest.mark.parametrize("collect", [_chat_choices, _completion_choices])
def test_terminal_text_of_each_choice_is_kept(collect):
    choices = collect()

    for index, expected in [(0, "A"), (1, "BC")]:
        assert "".join(_text(c) for c in choices if c["index"] == index) == expected


def test_chat_terminal_text_precedes_the_finish_chunk():
    choices = _chat_choices()

    for index in (0, 1):
        own = [c for c in choices if c["index"] == index and (_text(c) or c.get("finish_reason"))]
        assert own[-1].get("finish_reason")
        assert not any(c.get("finish_reason") for c in own[:-1])
        assert any(_text(c) for c in own[:-1])


class _HoldingTokenizer:
    """Incremental decoder that keeps the newest piece back until it is flushed."""

    def decode_incrementally(self, token_ids, prev_text=None, states=None, *, flush=False, **_):
        held = (states or {}).get("held", "")
        pieces = held + "".join(f"t{token_id}" for token_id in token_ids)
        if flush or not token_ids:
            return (prev_text or "") + pieces, {"held": ""}
        return (prev_text or "") + pieces[:-2], {"held": pieces[-2:]}

    def decode(self, token_ids, **_):
        return "".join(f"t{token_id}" for token_id in token_ids)


def test_finished_choice_streams_its_held_decoder_text_once(monkeypatch):
    # n=2: choice 0 finishes while the decoder still holds its last piece and
    # choice 1 keeps generating. The held piece must leave with choice 0's
    # finishing chunk: later responses skip choice 0.
    script = [
        ({0: ([1, 2], "stop"), 1: ([5], None)}, False),
        ({1: ([6], None)}, False),
        ({1: ([7], "length")}, True),
    ]
    steps = iter(script)

    def scripted_base_handler(self, response):
        updates, done = next(steps)
        self._done = done
        for output in self._outputs:
            output._last_token_ids_len = len(output.token_ids)
        for index, (ids, finish_reason) in updates.items():
            output = self._outputs[index]
            output.token_ids = output.token_ids + ids
            if finish_reason is not None:
                output.finish_reason = finish_reason

    monkeypatch.setattr(GenerationResultBase, "_handle_response", scripted_base_handler)
    result = DetokenizedGenerationResultBase(
        1,
        SamplingParams(n=2, detokenize=True, temperature=0.8, top_k=5),
        tokenizer=_HoldingTokenizer(),
        streaming=True,
    )
    request = CompletionRequest(model="m", prompt="hi", n=2, stream=True)
    args = CompletionPostprocArgs.from_request(request)
    args.prompt = "hi"
    args.num_prompt_tokens = 3

    choices = []
    for _ in script:
        result._handle_response(None)
        choices += list(_choices(completion_stream_post_processor(result, args)))

    assert "".join(c["text"] for c in choices if c["index"] == 0) == "t1t2"
    assert sorted((c["index"], c["finish_reason"]) for c in choices if c["finish_reason"]) == [
        (0, "stop"),
        (1, "length"),
    ]
