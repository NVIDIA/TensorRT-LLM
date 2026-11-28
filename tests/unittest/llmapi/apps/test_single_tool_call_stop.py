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
"""A request for at most one tool call per turn stops generation after the first call.

``parallel_tool_calls=false`` (Chat Completions, Responses) and Anthropic's
``disable_parallel_tool_use`` are honoured with a stop on the text that ends one
call (``BaseToolParser.single_call_stops``), kept in the output so the tool
parser still reads that call. These tests cover where the stop is configured,
what a client sees of it, and that it matches the tokens the model writes.
"""

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.responses.tool import FunctionTool
from test_common.serve.app_factory import AggConfig, isolated_serve_env
from utils.llm_data import llm_models_root

from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve import openai_server as openai_server_module
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionToolsParam,
    ChatMessage,
    FunctionDefinition,
    ResponsesRequest,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import _single_tool_call_stops, _stop_after_first_tool_call
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    chat_response_post_processor,
    chat_stream_post_processor,
)
from tensorrt_llm.serve.render import chat as render_chat_module

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
V41_STOPS = ("</｜DSML｜ invoke>\n", "</｜DSML｜ invoke>")
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
    },
}
WEATHER_INVOKE = (
    '<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="location" string="true">NYC</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>"
)


# ---------------------------------------------------------------------------
# The stop and the request's own stops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_parser, stops",
    [
        ("deepseek_v41", V41_STOPS),
        ("DeepSeek_V41", V41_STOPS),
        ("deepseek_v4", ("</｜DSML｜invoke>\n", "</｜DSML｜invoke>")),
        ("deepseek_v32", ("</｜DSML｜invoke>\n", "</｜DSML｜invoke>")),
        ("qwen3", ()),
        ("not_a_parser", ()),
        (None, ()),
    ],
)
def test_the_stops_come_from_the_tool_parser(tool_parser, stops):
    assert _single_tool_call_stops(tool_parser) == stops


def test_the_stops_are_added_and_kept_in_the_output():
    sampling_params = SamplingParams()
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    assert sampling_params.stop == list(V41_STOPS)
    assert sampling_params.include_stop_str_in_output is True


@pytest.mark.parametrize(
    "fields",
    [{"stop": "END"}, {"stop": ["END"]}, {"stop_token_ids": [7]}],
    ids=["stop_string", "stop_list", "stop_id"],
)
def test_nothing_changes_when_the_request_drops_its_own_stops_from_the_output(fields):
    """Keeping stop text applies to every stop, so the client's would show."""
    sampling_params = SamplingParams(**fields)
    assert not _stop_after_first_tool_call(sampling_params, V41_STOPS)
    assert sampling_params.stop == fields.get("stop")
    assert sampling_params.include_stop_str_in_output is False


def test_the_stops_follow_the_request_stops_it_keeps_in_the_output():
    sampling_params = SamplingParams(stop=["END", V41_STOPS[0]], include_stop_str_in_output=True)
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    assert sampling_params.stop == ["END", *V41_STOPS]


@pytest.mark.parametrize(
    "model_name, tool_parser",
    [
        ("DeepSeek-V3.2-hf", "deepseek_v32"),
        ("DeepSeek-V4-Flash", "deepseek_v4"),
        ("DeepSeek-V4-Pro", "deepseek_v4"),
        ("DeepSeek-V4.1-Flash", "deepseek_v41"),
    ],
)
def test_the_stop_matches_the_tokens_the_model_writes(model_name, tool_parser):
    """Stops match token ids, and '>' plus the newline after it are one token.

    So in each encoder's layout of DeepSeek tool calls the stop with the
    newline matches once per call, and the bare closing tag never does.
    """
    root = llm_models_root()
    tokenizer_file = None if root is None else root / model_name / "tokenizer.json"
    if tokenizer_file is None or not tokenizer_file.is_file():
        pytest.skip(f"{model_name} is not under LLM_MODELS_ROOT")
    from transformers import PreTrainedTokenizerFast

    from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer
    from tensorrt_llm.tokenizer.deepseek_v32 import DeepseekV32Tokenizer
    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    tokenizer_class = {
        "deepseek_v32": DeepseekV32Tokenizer,
        "deepseek_v4": DeepseekV4Tokenizer,
        "deepseek_v41": DeepseekV41Tokenizer,
    }[tool_parser]

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    # The ids the engine matches: SamplingParams tokenizes its stop strings.
    stops = _single_tool_call_stops(tool_parser)
    sampling_params = SamplingParams(stop=list(stops))
    sampling_params._setup(tokenizer, None, None)
    with_newline, bare = sampling_params._get_stop_words()

    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        for index, (name, arguments) in enumerate([("exec", {"cmd": "ls"}), ("wait", {"s": 2})])
    ]
    prompt = tokenizer_class(tokenizer).apply_chat_template(
        [
            {"role": "user", "content": "List the files, then wait."},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ],
        thinking=False,
        tokenize=False,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    def matches(word):
        return sum(ids[i : i + len(word)] == word for i in range(len(ids)))

    assert prompt.count(stops[0]) == 2
    assert matches(with_newline) == 2
    assert matches(bare) == 0


# ---------------------------------------------------------------------------
# Chat Completions and Anthropic Messages
# ---------------------------------------------------------------------------


@contextmanager
def _chat_server(tmp_path, monkeypatch, tool_parser):
    """A server whose real ``openai_chat`` records what reaches the engine."""

    async def render(**kwargs):
        return "rendered prompt"

    monkeypatch.setattr(render_chat_module, "async_apply_chat_template", render)
    seen = {}

    def generate_async(*, inputs, **kwargs):
        seen["sampling_params"] = kwargs["sampling_params"]
        return SimpleNamespace(request_id=1, prompt_token_ids=[1, 2, 3], finished=True)

    async def create_chat_response(promise, postproc_params, raw_request, disaggregated_params):
        seen["postproc_args"] = postproc_params.postproc_args
        return ChatCompletionResponse(
            model=MODEL,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )

    with isolated_serve_env(tmp_path) as env:
        server = env.build_openai_server(AggConfig(model_name=MODEL, tool_parser=tool_parser))
        server.generator.generate_async = generate_async
        server.await_disconnected = AsyncMock()
        server._create_chat_response = create_chat_response
        yield server, seen


def _raw_request():
    return SimpleNamespace(state=SimpleNamespace(), headers={})


@pytest.fixture
def server_logger(monkeypatch):
    log = Mock()
    monkeypatch.setattr(openai_server_module, "logger", log)
    return log


def _warned(log, key):
    return key in [call.kwargs.get("key") for call in log.warning_once.call_args_list]


@pytest.mark.parametrize(
    "fields, tool_parser, stop, enforced",
    [
        ({"parallel_tool_calls": False}, "deepseek_v41", list(V41_STOPS), True),
        ({}, "deepseek_v41", [], False),
        ({"parallel_tool_calls": True}, "deepseek_v41", [], False),
        ({"parallel_tool_calls": False, "tool_choice": "none"}, "deepseek_v41", [], False),
        ({"parallel_tool_calls": False, "tools": None}, "deepseek_v41", [], False),
        ({"parallel_tool_calls": False}, "qwen3", [], False),
        ({"parallel_tool_calls": False, "stop": ["END"]}, "deepseek_v41", ["END"], False),
        (
            {"parallel_tool_calls": False, "stop": ["END"], "include_stop_str_in_output": True},
            "deepseek_v41",
            ["END", *V41_STOPS],
            True,
        ),
    ],
    ids=[
        "asked",
        "omitted",
        "parallel_allowed",
        "tool_choice_none",
        "no_tools",
        "format_without_stop",
        "own_stops_dropped",
        "own_stops_kept",
    ],
)
def test_chat_configures_the_stop_only_when_asked_and_possible(
    tmp_path, monkeypatch, server_logger, fields, tool_parser, stop, enforced
):
    request = ChatCompletionRequest(
        **{
            "model": MODEL,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEATHER_TOOL],
            **fields,
        }
    )
    with _chat_server(tmp_path, monkeypatch, tool_parser) as (server, seen):
        response = asyncio.run(server.openai_chat(request, _raw_request()))

    assert response.status_code == 200, response.body
    sampling_params = seen["sampling_params"]
    assert sampling_params.stop == stop
    assert sampling_params.include_stop_str_in_output is (
        "include_stop_str_in_output" in fields or enforced
    )
    assert seen["postproc_args"].single_call_stops == (V41_STOPS if enforced else ())
    asked = fields.get("parallel_tool_calls") is False and fields.get("tools", [WEATHER_TOOL])
    asked = asked and fields.get("tool_choice") != "none"
    assert _warned(server_logger, "chat_parallel_tool_calls_unenforced") is bool(
        asked and not enforced
    )


def test_a_stop_the_request_lists_itself_keeps_its_stop_reason(
    tmp_path, monkeypatch, server_logger
):
    """Only the stops the server adds are hidden from the reply's stop reason."""
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "weather?"}],
        tools=[WEATHER_TOOL],
        parallel_tool_calls=False,
        stop=[V41_STOPS[0]],
        include_stop_str_in_output=True,
    )
    with _chat_server(tmp_path, monkeypatch, "deepseek_v41") as (server, seen):
        response = asyncio.run(server.openai_chat(request, _raw_request()))

    assert response.status_code == 200, response.body
    assert seen["sampling_params"].stop == list(V41_STOPS)
    assert seen["postproc_args"].single_call_stops == V41_STOPS[1:]


@pytest.mark.parametrize(
    "tool_parser, stop_sequences, enforced",
    [
        ("deepseek_v41", [], True),
        ("deepseek_v41", ["END"], False),
        ("qwen3", [], False),
        (None, [], False),
    ],
    ids=["supported", "own_stops_dropped", "format_without_stop", "no_tool_parser"],
)
def test_anthropic_disable_parallel_tool_use_configures_the_stop_when_possible(
    tmp_path, monkeypatch, server_logger, tool_parser, stop_sequences, enforced
):
    request = AnthropicMessagesRequest(
        model=MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": "weather?"}],
        tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        stop_sequences=stop_sequences,
    )
    with _chat_server(tmp_path, monkeypatch, tool_parser) as (server, seen):
        response = asyncio.run(server.anthropic_messages(request, _raw_request()))

    assert response.status_code == 200, response.body
    assert seen["sampling_params"].stop == (list(V41_STOPS) if enforced else stop_sequences)
    assert seen["sampling_params"].include_stop_str_in_output is enforced
    assert seen["postproc_args"].single_call_stops == (V41_STOPS if enforced else ())
    assert _warned(server_logger, "chat_parallel_tool_calls_unenforced") is not enforced


# ---------------------------------------------------------------------------
# What a chat client receives
# ---------------------------------------------------------------------------


def _chat_args(single_call_stops=V41_STOPS):
    args = ChatPostprocArgs(
        role="assistant",
        model=MODEL,
        tools=[ChatCompletionToolsParam(function=FunctionDefinition(**WEATHER_TOOL["function"]))],
        tool_choice="auto",
        single_call_stops=single_call_stops,
    )
    args.tool_parser = "deepseek_v41"
    args.num_prompt_tokens = 3
    return args


def _output(text, finish_reason, stop_reason):
    return SimpleNamespace(
        index=0,
        text=text,
        text_diff=text,
        token_ids=[1, 2],
        token_ids_diff=[1, 2],
        length=2,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        logprobs=None,
        logprobs_diff=None,
        disaggregated_params=None,
    )


# The output a generation stopped after its first call leaves: the calls block
# is open and the kept stop ends it.
STOPPED_REPLY = "Checking.\n\n<｜DSML｜ calls>\n" + WEATHER_INVOKE + "\n"


@pytest.mark.parametrize(
    "stop_reason, reported", [(V41_STOPS[0], None), (V41_STOPS[1], None), ("END", "END")]
)
def test_a_chat_reply_reports_its_call_and_no_stop_the_server_added(stop_reason, reported):
    rsp = SimpleNamespace(
        outputs=[_output(STOPPED_REPLY, "stop", stop_reason)], prompt_token_ids=[1], cached_tokens=0
    )
    choice = chat_response_post_processor(rsp, _chat_args()).choices[0]

    assert choice.finish_reason == "tool_calls"
    assert choice.stop_reason == reported
    assert choice.message.content == "Checking."
    calls = [(c.function.name, json.loads(c.function.arguments)) for c in choice.message.tool_calls]
    assert calls == [("get_weather", {"location": "NYC"})]


def test_a_streamed_chat_reply_reports_its_call_and_no_stop_the_server_added():
    args = _chat_args()
    head, tail = STOPPED_REPLY.split("<｜DSML｜ invoke", 1)
    frames = []
    for diff, done in [(head, False), ("<｜DSML｜ invoke" + tail, True)]:
        output = _output(diff, "stop" if done else None, V41_STOPS[0] if done else None)
        rsp = SimpleNamespace(id=1, outputs=[output], cached_tokens=0, _done=done)
        frames += chat_stream_post_processor(rsp, args)

    choices = [json.loads(frame[len("data: ") :])["choices"][0] for frame in frames]
    deltas = [call for choice in choices for call in choice["delta"].get("tool_calls") or []]
    assert [d["function"]["name"] for d in deltas if d["function"].get("name")] == ["get_weather"]
    assert json.loads("".join(d["function"].get("arguments") or "" for d in deltas)) == {
        "location": "NYC"
    }
    assert choices[-1]["finish_reason"] == "tool_calls"
    assert all(choice.get("stop_reason") is None for choice in choices)
    assert "｜DSML｜" not in "".join(choice["delta"].get("content") or "" for choice in choices)


def test_streaming_n2_single_call_stop_finishes_only_the_matching_choice():
    """With n=2 streaming, stopping choice 0 on a DSML stop leaves choice 1 running."""
    from tensorrt_llm.bindings import executor as tllm
    from tensorrt_llm.executor.result import DetokenizedGenerationResultBase

    vocab = {
        1: "Checking.\n\n<｜DSML｜ calls>\n",
        2: WEATHER_INVOKE + "\n",
        3: "Second.\n\n<｜DSML｜ calls>\n",
        4: WEATHER_INVOKE.replace("NYC", "SFO") + "\n",
        5: " trailing-after-stop",
    }

    class _FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, **_kwargs):
            return [99]

        def decode(self, token_ids, **_kwargs):
            return "".join(vocab[t] for t in token_ids)

    sampling_params = SamplingParams(n=2, temperature=0.7)
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    sampling_params._setup(_FakeTokenizer(), hf_model_config=None, generation_config=None)

    result = DetokenizedGenerationResultBase(
        id=7,
        sampling_params=sampling_params,
        tokenizer=_FakeTokenizer(),
        streaming=True,
    )
    args = _chat_args()
    args.num_choices = 2

    def _response(seq_idx, token_ids, finish_reason, is_final=False):
        return SimpleNamespace(
            has_error=lambda: False,
            result=SimpleNamespace(
                is_final=is_final,
                context_phase_params=None,
                decoding_iter=1,
                cached_tokens=0,
                avg_decoded_tokens_per_iter=1.0,
                finish_reasons=[finish_reason],
                output_token_ids=[token_ids],
                request_perf_metrics=None,
                sequence_index=seq_idx,
                cum_log_probs=None,
                log_probs=None,
                per_pos_drafted=None,
                per_pos_accepted=None,
                spec_dec_totals=None,
                generation_logits=None,
                additional_context_outputs=None,
                additional_generation_outputs=None,
                time_breakdown_metrics=None,
                context_logits=None,
            ),
        )

    frames = []
    # Step 1: choice 0 hits the injected DSML stop; choice 1 has only started.
    result._handle_response(_response(0, [1, 2], tllm.FinishReason.NOT_FINISHED, is_final=False))
    assert result.outputs[0].finish_reason == "stop"
    assert result.outputs[0].stop_reason == V41_STOPS[0]
    assert result.outputs[1].finish_reason is None
    assert not result._done
    frames += chat_stream_post_processor(result, args)

    result._handle_response(_response(1, [3], tllm.FinishReason.NOT_FINISHED, is_final=False))
    assert result.outputs[1].finish_reason is None
    assert not result._done
    frames += chat_stream_post_processor(result, args)

    # Extra tokens arriving for already-stopped choice 0 do not unfreeze it or inflate token_ids.
    result._handle_response(_response(0, [5], tllm.FinishReason.NOT_FINISHED, is_final=False))
    assert result.outputs[0].text == STOPPED_REPLY
    assert result.outputs[0].text_diff == ""
    assert result.outputs[0].token_ids == [1, 2]
    assert not result._done
    frames += chat_stream_post_processor(result, args)

    # Step 2: choice 1 hits the injected DSML stop on a later step -> now whole result is done.
    result._handle_response(_response(1, [4], tllm.FinishReason.NOT_FINISHED, is_final=False))
    assert result.outputs[1].finish_reason == "stop"
    assert result.outputs[1].stop_reason == V41_STOPS[0]
    assert result._done
    assert result._aborted
    frames += chat_stream_post_processor(result, args)

    by_index = {0: [], 1: []}
    for frame in frames:
        for choice in json.loads(frame[len("data: ") :])["choices"]:
            by_index[choice["index"]].append(choice)

    for idx, expected_loc in [(0, "NYC"), (1, "SFO")]:
        deltas = [c for ch in by_index[idx] for c in ch["delta"].get("tool_calls") or []]
        assert [d["function"]["name"] for d in deltas if d["function"].get("name")] == [
            "get_weather"
        ]
        assert json.loads("".join(d["function"].get("arguments") or "" for d in deltas)) == {
            "location": expected_loc
        }
        finish_reasons = [ch["finish_reason"] for ch in by_index[idx] if ch.get("finish_reason")]
        assert finish_reasons == ["tool_calls"]
        assert all(ch.get("stop_reason") is None for ch in by_index[idx])


def _make_llm_response(
    seq_idx,
    token_ids,
    finish_reason,
    is_final=False,
    cum_log_probs=None,
    per_pos_drafted=None,
    per_pos_accepted=None,
    spec_dec_totals=None,
):
    return SimpleNamespace(
        has_error=lambda: False,
        result=SimpleNamespace(
            is_final=is_final,
            context_phase_params=None,
            decoding_iter=1,
            cached_tokens=0,
            avg_decoded_tokens_per_iter=1.0,
            finish_reasons=finish_reason if isinstance(finish_reason, list) else [finish_reason],
            output_token_ids=token_ids
            if (token_ids and isinstance(token_ids[0], list))
            else [token_ids],
            request_perf_metrics=None,
            sequence_index=seq_idx,
            cum_log_probs=cum_log_probs,
            log_probs=None,
            per_pos_drafted=per_pos_drafted,
            per_pos_accepted=per_pos_accepted,
            spec_dec_totals=spec_dec_totals,
            generation_logits=None,
            additional_context_outputs=None,
            additional_generation_outputs=None,
            time_breakdown_metrics=None,
            context_logits=None,
        ),
    )


def test_streaming_n2_incremental_decoder_does_not_refeed_other_choice():
    """Incremental detokenization only feeds token_ids_diff for the responding sequence_index."""
    from tensorrt_llm.bindings import executor as tllm
    from tensorrt_llm.executor.result import DetokenizedGenerationResultBase

    vocab = {
        1: "Alpha ",
        2: "Beta ",
        3: "Gamma\n" + V41_STOPS[0] + "\n",
        4: "Delta\n" + V41_STOPS[0] + "\n",
    }

    class _IncrementalTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __init__(self):
            self.calls = []

        def encode(self, text, **_kwargs):
            return [99]

        def decode_incrementally(
            self, token_ids, prev_text="", states=None, flush=False, **_kwargs
        ):
            self.calls.append((list(token_ids), prev_text, flush))
            return prev_text + "".join(vocab[t] for t in token_ids), states or {"seen": True}

    tok = _IncrementalTokenizer()
    sampling_params = SamplingParams(n=2, temperature=0.7)
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    sampling_params._setup(tok, hf_model_config=None, generation_config=None)

    result = DetokenizedGenerationResultBase(
        id=8, sampling_params=sampling_params, tokenizer=tok, streaming=True
    )

    result._handle_response(_make_llm_response(0, [1], tllm.FinishReason.NOT_FINISHED))
    assert result.outputs[0].text == "Alpha "
    assert result.outputs[1].text == ""

    # Responding on seq 1 must NOT re-feed seq 0's previous token_ids_diff ([1])
    # and must clear seq 0's stale token_ids_diff / logprobs_diff.
    result._handle_response(_make_llm_response(1, [2], tllm.FinishReason.NOT_FINISHED))
    assert result.outputs[0].text == "Alpha "
    assert result.outputs[0].text_diff == ""
    assert result.outputs[0].token_ids_diff == []
    assert result.outputs[0].logprobs_diff == []
    assert result.outputs[1].text == "Beta "
    assert len(tok.calls) == 2

    # Seq 0 stops on string stop; seq 1 continues and finishes via engine EOS.
    result._handle_response(_make_llm_response(0, [3], tllm.FinishReason.NOT_FINISHED))
    assert result.outputs[0].finish_reason == "stop"
    assert not result._done

    result._handle_response(_make_llm_response(1, [4], tllm.FinishReason.END_ID, is_final=False))
    assert result.outputs[1].finish_reason == "stop"
    assert result._done
    assert result._aborted


def test_request_output_from_generation_result_aborts_when_all_choices_string_stop():
    """RequestOutput constructed via _from_generation_result (__new__) handles per-choice string stops."""
    from tensorrt_llm.bindings import executor as tllm
    from tensorrt_llm.executor.result import GenerationResult
    from tensorrt_llm.llmapi.llm import RequestOutput

    vocab = {1: "ok" + V41_STOPS[0], 2: "more"}

    class _FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, **_kwargs):
            return [99]

        def decode(self, token_ids, **_kwargs):
            return "".join(vocab[t] for t in token_ids)

    aborted = []

    class _FakeExecutor:
        def abort_request(self, req_id):
            aborted.append(req_id)

    sampling_params = SamplingParams(n=2, temperature=0.7)
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    sampling_params._setup(_FakeTokenizer(), hf_model_config=None, generation_config=None)

    gen_req = SimpleNamespace(
        id=42,
        sampling_params=sampling_params,
        postproc_params=None,
        streaming=True,
        trace_headers=None,
        multimodal_params=None,
        prompt_token_ids=[10, 11],
    )
    executor = _FakeExecutor()
    gen_result = GenerationResult(gen_req, executor=executor)
    req_out = RequestOutput._from_generation_result(gen_result, tokenizer=_FakeTokenizer())

    req_out._handle_response(_make_llm_response(0, [1], tllm.FinishReason.NOT_FINISHED))
    assert req_out.outputs[0].finish_reason == "stop"
    assert not req_out._done
    assert aborted == []

    req_out._handle_response(_make_llm_response(1, [1], tllm.FinishReason.NOT_FINISHED))
    assert req_out.outputs[1].finish_reason == "stop"
    assert req_out._done
    assert aborted == [42]


def test_best_of_greater_than_n_waits_for_all_underlying_candidates():
    """With best_of > n, provisional candidates are buffered until all best_of finish."""
    from tensorrt_llm.bindings import executor as tllm
    from tensorrt_llm.executor.result import DetokenizedGenerationResultBase

    vocab = {
        1: "Checking.\n\n<｜DSML｜ calls>\n" + WEATHER_INVOKE + "\n",
        2: "Plain winning answer.",
    }

    class _FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, **_kwargs):
            return [99]

        def decode(self, token_ids, **_kwargs):
            return "".join(vocab[t] for t in token_ids)

    sampling_params = SamplingParams(n=1, best_of=2, temperature=0.7)
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    sampling_params._setup(_FakeTokenizer(), hf_model_config=None, generation_config=None)

    result = DetokenizedGenerationResultBase(
        id=9, sampling_params=sampling_params, tokenizer=_FakeTokenizer(), streaming=True
    )
    args = _chat_args()
    args.num_choices = 1

    # Candidate 0 leads temporarily and hits a DSML tool-call stop, while
    # candidate 1 is still unfinished. Streaming must not emit candidate 0's
    # provisional tool call or finish_reason yet.
    result._handle_response(
        _make_llm_response(0, [1], tllm.FinishReason.NOT_FINISHED, cum_log_probs=[-0.5])
    )
    assert result._outputs[0].finish_reason == "stop"
    assert result._outputs[1].finish_reason is None
    assert not result._done
    step1_frames = chat_stream_post_processor(result, args)
    step1_choices = [
        ch for frame in step1_frames for ch in json.loads(frame[len("data: ") :])["choices"]
    ]
    assert all(ch.get("finish_reason") is None for ch in step1_choices)
    assert all(not ch["delta"].get("tool_calls") for ch in step1_choices)
    assert all(not ch["delta"].get("content") for ch in step1_choices)

    # Candidate 1 finishes with plain text and a higher cumulative logprob,
    # overtaking candidate 0. Only candidate 1's plain text and finish_reason="stop"
    # (not candidate 0's tool_calls) are emitted.
    result._handle_response(
        _make_llm_response(1, [2], tllm.FinishReason.END_ID, is_final=False, cum_log_probs=[-0.1])
    )
    assert result._outputs[1].finish_reason == "stop"
    assert result._done
    assert result._aborted
    step2_frames = chat_stream_post_processor(result, args)
    step2_choices = [
        ch for frame in step2_frames for ch in json.loads(frame[len("data: ") :])["choices"]
    ]
    assert (
        "".join(ch["delta"].get("content") or "" for ch in step2_choices) == "Plain winning answer."
    )
    assert all(not ch["delta"].get("tool_calls") for ch in step2_choices)
    assert [ch["finish_reason"] for ch in step2_choices if ch.get("finish_reason")] == ["stop"]


def test_streaming_best_of_preserves_spec_decode_stats_on_late_stopped_response():
    """A late engine response for a string-stopped winner keeps its stop-time spec-decode stats."""
    from tensorrt_llm.bindings import executor as tllm
    from tensorrt_llm.executor.result import DetokenizedGenerationResultBase

    vocab = {
        1: "Checking.\n\n<｜DSML｜ calls>\n" + WEATHER_INVOKE + "\n",
        2: " late-ignored-token",
        3: "Lower-scored candidate.",
    }

    class _FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, **_kwargs):
            return [99]

        def decode(self, token_ids, **_kwargs):
            return "".join(vocab[t] for t in token_ids)

    sampling_params = SamplingParams(n=1, best_of=2, temperature=0.7)
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    sampling_params._setup(_FakeTokenizer(), hf_model_config=None, generation_config=None)

    result = DetokenizedGenerationResultBase(
        id=12, sampling_params=sampling_params, tokenizer=_FakeTokenizer(), streaming=True
    )
    args = _chat_args()
    args.num_choices = 1
    args.return_spec_decode_stats = True
    args.spec_decode_num_spec_tokens = 2

    # Candidate 0 string-stops with higher cum_log_prob and spec-decode counters.
    result._handle_response(
        _make_llm_response(
            0,
            [1],
            tllm.FinishReason.NOT_FINISHED,
            cum_log_probs=[-0.5],
            per_pos_drafted=[4, 4],
            per_pos_accepted=[3, 1],
            spec_dec_totals=(4, 8),
        )
    )
    step0_frames = chat_stream_post_processor(result, args)
    step0_choices = [json.loads(frame[len("data: ") :])["choices"][0] for frame in step0_frames]
    assert all(ch.get("finish_reason") is None for ch in step0_choices)
    assert all("speculative_decoding" not in ch for ch in step0_choices)

    # A late engine response arrives for candidate 0 while candidate 1 is still running.
    result._handle_response(
        _make_llm_response(
            0,
            [2],
            tllm.FinishReason.NOT_FINISHED,
            cum_log_probs=[-0.5],
        )
    )
    assert chat_stream_post_processor(result, args) == []

    # Candidate 1 finishes with lower score so candidate 0 wins; winner keeps its spec stats.
    result._handle_response(
        _make_llm_response(
            1,
            [3],
            tllm.FinishReason.END_ID,
            is_final=True,
            cum_log_probs=[-3.0],
            per_pos_drafted=[3, 3],
            per_pos_accepted=[1],
            spec_dec_totals=(1, 6),
        )
    )
    final_frames = chat_stream_post_processor(result, args)
    choices = [json.loads(frame[len("data: ") :])["choices"][0] for frame in final_frames]
    spec_payloads = [ch["speculative_decoding"] for ch in choices if "speculative_decoding" in ch]
    assert len(spec_payloads) == 1
    assert spec_payloads[0] == {
        "acceptance_rate": 0.5,
        "total_accepted_draft_tokens": 4,
        "total_draft_tokens": 8,
        "num_spec_steps": 4,
        "acceptance_histogram": [1, 2, 1],
        "num_spec_tokens": 2,
    }


def test_beam_search_hidden_beam_does_not_terminate_visible_answer():
    """With multi-beam search (n=1, best_of=2), a hidden beam matching a stop string must not end the request."""
    from tensorrt_llm.bindings import executor as tllm
    from tensorrt_llm.executor.result import DetokenizedGenerationResultBase

    vocab = {1: "beam0", 2: "beam1" + V41_STOPS[0]}

    class _FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, **_kwargs):
            return [99]

        def decode(self, token_ids, **_kwargs):
            return "".join(vocab[t] for t in token_ids)

    sampling_params = SamplingParams(n=1, best_of=2, use_beam_search=True)
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    sampling_params._setup(_FakeTokenizer(), hf_model_config=None, generation_config=None)

    result = DetokenizedGenerationResultBase(
        id=10, sampling_params=sampling_params, tokenizer=_FakeTokenizer(), streaming=False
    )
    result._handle_response(
        _make_llm_response(
            0,
            [[1], [2]],
            [tllm.FinishReason.NOT_FINISHED, tllm.FinishReason.NOT_FINISHED],
            is_final=False,
        )
    )
    assert result._str_stopped_outputs == {}
    assert not result._done
    assert result.outputs[0].finish_reason is None


def test_completion_stream_n2_emits_finish_reason_once_per_choice():
    """On /v1/completions with n=2, a string-stopped choice does not repeat its finish chunk."""
    from tensorrt_llm.bindings import executor as tllm
    from tensorrt_llm.executor.result import DetokenizedGenerationResultBase
    from tensorrt_llm.serve.postprocess_handlers import (
        CompletionPostprocArgs,
        completion_stream_post_processor,
    )

    vocab = {1: "first" + V41_STOPS[0], 2: "second", 3: " done" + V41_STOPS[0]}

    class _FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, **_kwargs):
            return [99]

        def decode(self, token_ids, **_kwargs):
            return "".join(vocab[t] for t in token_ids)

    sampling_params = SamplingParams(n=2, temperature=0.7)
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    sampling_params._setup(_FakeTokenizer(), hf_model_config=None, generation_config=None)

    result = DetokenizedGenerationResultBase(
        id=11, sampling_params=sampling_params, tokenizer=_FakeTokenizer(), streaming=True
    )
    args = CompletionPostprocArgs(model="m", num_choices=2, num_prompt_tokens=3)

    frames = []
    result._handle_response(_make_llm_response(0, [1], tllm.FinishReason.NOT_FINISHED))
    frames += completion_stream_post_processor(result, args)
    result._handle_response(_make_llm_response(1, [2], tllm.FinishReason.NOT_FINISHED))
    frames += completion_stream_post_processor(result, args)
    result._handle_response(_make_llm_response(1, [3], tllm.FinishReason.NOT_FINISHED))
    frames += completion_stream_post_processor(result, args)

    finishes = {0: [], 1: []}
    for frame in frames:
        for ch in json.loads(frame[len("data: ") :])["choices"]:
            if ch.get("finish_reason") is not None:
                finishes[ch["index"]].append(ch["finish_reason"])
    assert finishes == {0: ["stop"], 1: ["stop"]}


def test_postproc_worker_discards_late_inflight_responses_after_local_stop():
    """After a local string stop pops a PostprocWorker record, late engine responses are discarded."""
    from tensorrt_llm.bindings import executor as tllm
    from tensorrt_llm.executor.postproc_worker import PostprocWorker

    vocab = {1: "ok" + V41_STOPS[0], 2: "late"}

    class _FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, **_kwargs):
            return [99]

        def decode(self, token_ids, **_kwargs):
            return "".join(vocab[t] for t in token_ids)

    sampling_params = SamplingParams(n=1, temperature=0.7)
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    sampling_params._setup(_FakeTokenizer(), hf_model_config=None, generation_config=None)

    def _rsp(token_ids, finish_reason, is_final=False):
        r = _make_llm_response(0, token_ids, finish_reason, is_final=is_final)
        r.client_id = 77
        return r

    # Drive PostprocWorker._mainloop with 3 queued inputs:
    # 1) triggers local string stop (is_final=False from engine -> record popped, client marked locally finished)
    # 2) late in-flight response without sampling_params (must be silently dropped, no ErrorResponse)
    # 3) engine's final CANCELLED response (clears the discard marker)
    inputs = [
        [
            PostprocWorker.Input(
                rsp=_rsp([1], tllm.FinishReason.NOT_FINISHED, is_final=False),
                sampling_params=sampling_params,
                streaming=True,
            ),
            PostprocWorker.Input(
                rsp=_rsp([2], tllm.FinishReason.NOT_FINISHED, is_final=False),
                sampling_params=None,
                streaming=True,
            ),
            PostprocWorker.Input(
                rsp=_rsp([], tllm.FinishReason.CANCELLED, is_final=True),
                sampling_params=None,
                streaming=True,
            ),
        ],
        None,
    ]

    class _FakePullPipe:
        async def get_async(self):
            return inputs.pop(0)

    from collections import OrderedDict

    worker = object.__new__(PostprocWorker)
    worker._records = {}
    worker._locally_finished_client_ids = OrderedDict()
    worker._record_creator = PostprocWorker.default_record_creator
    worker._pull_pipe = _FakePullPipe()
    worker._tokenizer = _FakeTokenizer()
    worker._post_processor_hook = None
    worker._to_stop = asyncio.Event()

    async def _collect():
        batches = []
        async for batch in worker._mainloop():
            if batch is None:
                break
            batches.append(batch)
        return batches

    batches = asyncio.run(_collect())
    assert len(batches) == 1
    assert len(batches[0]) == 1
    out = batches[0][0]
    assert isinstance(out, PostprocWorker.Output)
    assert out.is_final is True
    assert out.should_abort is True
    assert worker._records == {}
    assert len(worker._locally_finished_client_ids) == 0


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class _EngineReached(Exception):
    pass


@contextmanager
def _responses_server(tmp_path, monkeypatch, tool_parser):
    """A server whose real ``openai_responses`` records what reaches the engine."""
    seen = {}

    async def preprocess(*, request, **kwargs):
        return [1, 2, 3], request.to_sampling_params(default_sampling_params={"stop_token_ids": []})

    def generate_async(**kwargs):
        seen["sampling_params"] = kwargs["sampling_params"]
        raise _EngineReached()

    monkeypatch.setattr(openai_server_module, "responses_api_request_preprocess", preprocess)
    with isolated_serve_env(tmp_path) as env:
        server = env.build_openai_server(
            AggConfig(
                model_name=MODEL,
                tool_parser=tool_parser,
                env=(("TRTLLM_RESPONSES_API_DISABLE_STORE", "1"),),
            )
        )
        server.generator.generate_async = generate_async
        yield server, seen


@pytest.mark.parametrize(
    "fields, tool_parser, enforced, warned",
    [
        ({"parallel_tool_calls": False}, "deepseek_v41", True, False),
        ({}, "deepseek_v41", False, False),
        ({"parallel_tool_calls": True}, "deepseek_v41", False, False),
        ({"parallel_tool_calls": False, "tool_choice": "none"}, "deepseek_v41", False, False),
        ({"parallel_tool_calls": False, "tools": []}, "deepseek_v41", False, False),
        ({"parallel_tool_calls": False}, "qwen3", False, True),
        ({"parallel_tool_calls": False}, None, False, True),
    ],
    ids=[
        "asked",
        "omitted",
        "parallel_allowed",
        "tool_choice_none",
        "no_tools",
        "format_without_stop",
        "no_tool_parser",
    ],
)
def test_responses_configures_the_stop_only_when_asked_and_possible(
    tmp_path, monkeypatch, server_logger, fields, tool_parser, enforced, warned
):
    tool = FunctionTool(
        name="get_weather",
        type="function",
        strict=False,
        parameters=WEATHER_TOOL["function"]["parameters"],
    )
    request = ResponsesRequest(**{"model": MODEL, "input": "weather?", "tools": [tool], **fields})
    raw_request = SimpleNamespace(
        state=SimpleNamespace(),
        headers={},
        url=SimpleNamespace(path="/v1/responses"),
        json=AsyncMock(return_value={}),
    )
    with _responses_server(tmp_path, monkeypatch, tool_parser) as (server, seen):
        asyncio.run(server.openai_responses(request, raw_request))

    sampling_params = seen["sampling_params"]
    assert sampling_params.stop == (list(V41_STOPS) if enforced else None)
    assert sampling_params.include_stop_str_in_output is enforced
    assert _warned(server_logger, "responses_parallel_tool_calls_unenforced") is warned
