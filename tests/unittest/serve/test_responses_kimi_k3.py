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
"""Kimi-K3 on the Responses API, driven through its serving extension.

The handler tests build a ``kimi_k3`` server with the factory and give it a
probe chat template: a byte-level tokenizer (one token per byte, so the prompt
the engine receives decodes back exactly) whose template echoes the
conversation it is handed and every chat-template kwarg it reads. What the
extension derives from a Responses request is therefore visible in the prompt
the engine gets, and what it leaves alone is visibly absent. The hooks are also
checked directly on small inline conversations.
"""

import copy
import json
import logging
from unittest import mock

import pytest
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import ScriptedOutput, ScriptedStep
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.logger import logger
from tensorrt_llm.serve import chat_utils, serving_extensions
from tensorrt_llm.serve.extensions.kimi_k3 import KimiK3ServingExtension, enforce_kimi_param_policy
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ResponsesRequest
from tensorrt_llm.serve.responses_utils import request_preprocess
from tensorrt_llm.serve.router import KvCacheAwareRouter
from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension

# threadleak is off: the factory's TestClient and worker pools are closed by the
# serve_env fixture's teardown, after the leak check (see test_system_endpoints.py).
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

TOOLS = [{"type": "function", "name": "exec", "parameters": {"type": "object"}}]
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}

# Echoes each message as <|role|>content with its calls and result binding,
# then every chat-template kwarg the kimi_k3 extension can derive, then the
# generation prompt.
PROBE_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}"
    "{% if m['tool_calls'] %}{% for c in m['tool_calls'] %}<call:{{ c['id'] }}>{% endfor %}{% endif %}"
    "{% if m['tool_call_id'] %}<result:{{ m['tool_call_id'] }}>{% endif %}{% endfor %}"
    "{% if thinking is defined %}<thinking:{{ thinking }}>{% endif %}"
    "{% if thinking_effort is defined %}<effort:{{ thinking_effort }}>{% endif %}"
    "{% if tool_choice is defined %}<tool_choice:{{ tool_choice }}>{% endif %}"
    "{% if response_format is defined %}<response_format:{{ response_format }}>{% endif %}"
    "{% if response_schema is defined %}<response_schema:{{ response_schema | tojson }}>{% endif %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _probe_tokenizer() -> TransformersTokenizer:
    vocab = {
        token: index for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    vocab["</s>"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    return TransformersTokenizer(
        PreTrainedTokenizerFast(
            tokenizer_object=backend, eos_token="</s>", chat_template=PROBE_TEMPLATE
        )
    )


def _user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _call(call_id):
    return {"type": "function_call", "call_id": call_id, "name": "exec", "arguments": "{}"}


def _result(call_id, output):
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _output_items(response, stream):
    """The output items as the client receives them, in either mode."""
    assert response.status_code == 200, response.text
    if not stream:
        return response.json()["output"]
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    return [event["item"] for event in events if event["type"] == "response.output_item.done"]


class _Served:
    """One factory-built server behind a client, with the probe template installed."""

    def __init__(self, serve_env, config: AggConfig) -> None:
        self.server = serve_env.build_openai_server(config)
        self.tokenizer = _probe_tokenizer()
        self.server.tokenizer = self.tokenizer
        self.client = serve_env.client(self.server.app)

    def respond(self, body: dict, *, reply: str = "ok"):
        """POST one Responses request whose engine reply is ``reply``."""
        if reply != "ok":
            self.server.generator.script(
                ScriptedOutput(
                    steps=(ScriptedStep(text=reply, token_ids=(7,), finish_reason="stop"),)
                )
            )
        return self.client.post(
            "/v1/responses", json={"model": "fake-model", "store": False, **body}
        )

    @property
    def engine_calls(self):
        return self.server.generator.generate_calls

    def prompt(self) -> str:
        """The prompt of the last engine call, decoded."""
        inputs = self.engine_calls[-1].inputs
        ids = inputs if isinstance(inputs, list) else inputs["prompt_token_ids"]
        return self.tokenizer.decode(ids)


@pytest.fixture
def text_only(monkeypatch):
    """The multimodal placeholder registry answers for ``kimi_k3`` without importing its model.

    Production returns ``None`` too: the Kimi-K3 modeling module registers its
    placeholders without a content format. Looking the type up for real imports
    that module (the model zoo) into a CPU serving test, so the answer is given
    here, for ``kimi_k3`` only; a K3 content-format registration would turn this
    stub red instead of silently diverging.
    """

    def content_format_of(model_type):
        assert model_type == "kimi_k3", model_type
        return None

    monkeypatch.setattr(
        chat_utils.MULTIMODAL_PLACEHOLDER_REGISTRY, "get_content_format", content_format_of
    )


def _kimi_config(**overrides) -> AggConfig:
    return AggConfig(
        model_type="kimi_k3", reasoning_parser="kimi_k3", tool_parser="kimi_k3", **overrides
    )


@pytest.fixture
def kimi(serve_env, text_only):
    return _Served(serve_env, _kimi_config())


@pytest.fixture
def kimi_with_policy(serve_env, text_only):
    return _Served(serve_env, _kimi_config(env=(("TRTLLM_KIMI_PARAM_POLICY", "1"),)))


@pytest.fixture
def generic(serve_env):
    """A model without a serving extension of its own."""
    return _Served(serve_env, AggConfig(model_type="llama"))


@pytest.fixture
def grammar_enforced(serve_env):
    """A model whose extension enforces ``required`` by grammar, which the Responses path does not do.

    The factory builds a generic server; its model config is swapped for one
    resolving to a scratch model type registered with a grammar-mode extension.
    """
    model_type = "responses-test-grammar-required"

    @register_serving_extension(model_types=(model_type,))
    class GrammarRequired(ServingExtension):
        def required_tool_choice_mode(self):
            return "grammar"

    served = _Served(serve_env, AggConfig(model_type="llama"))
    served.server.model_config = type("GrammarConfig", (), {"model_type": model_type})()
    try:
        yield served
    finally:
        serving_extensions._BY_MODEL_TYPE.pop(model_type, None)


# ---------------------------------------------------------------------------
# Request options reach the template as the kwargs the kimi_k3 template reads
# ---------------------------------------------------------------------------


class TestRequestOptionsReachTheTemplate:
    @pytest.mark.parametrize(
        ("effort", "thinking_effort"),
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "max"),
            ("max", "max"),
        ],
    )
    def test_reasoning_effort_maps_onto_thinking_effort(self, kimi, effort, thinking_effort):
        response = kimi.respond({"input": "hi", "reasoning": {"effort": effort}})

        assert response.status_code == 200, response.text
        assert f"<effort:{thinking_effort}>" in kimi.prompt()
        assert "<thinking:False>" not in kimi.prompt()

    def test_no_effort_leaves_the_template_default(self, kimi):
        response = kimi.respond({"input": "hi"})

        assert response.status_code == 200, response.text
        assert "<effort:" not in kimi.prompt()
        assert "<thinking:" not in kimi.prompt()

    @pytest.mark.parametrize("stream", [False, True])
    def test_effort_none_turns_thinking_off_for_the_prompt_and_the_parser(self, kimi, stream):
        """The same kwargs steer the reasoning parser: the reply is a message, not reasoning."""
        response = kimi.respond({"input": "hi", "reasoning": {"effort": "none"}, "stream": stream})

        prompt = kimi.prompt()
        assert "<thinking:False>" in prompt
        assert "<effort:" not in prompt
        items = _output_items(response, stream)
        assert [(item["type"], item["content"][0]["text"]) for item in items] == [("message", "ok")]

    def test_the_reply_is_reasoning_while_thinking_is_on(self, kimi):
        """The counterpart of the test above: with thinking on, K3 starts in its think channel."""
        response = kimi.respond({"input": "hi", "reasoning": {"effort": "high"}})

        assert [item["type"] for item in _output_items(response, False)] == ["reasoning"]

    @pytest.mark.parametrize("tool_choice", ["required", "none"])
    def test_tool_choice_reaches_the_template_when_tools_are_declared(self, kimi, tool_choice):
        response = kimi.respond({"input": "hi", "tools": TOOLS, "tool_choice": tool_choice})

        assert response.status_code == 200, response.text
        assert f"<tool_choice:{tool_choice}>" in kimi.prompt()

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"tools": TOOLS, "tool_choice": "auto"}, id="auto"),
            pytest.param({"tool_choice": "required"}, id="required_without_tools"),
        ],
    )
    def test_tool_choice_adds_nothing_otherwise(self, kimi, body):
        response = kimi.respond({"input": "hi", **body})

        assert response.status_code == 200, response.text
        assert "<tool_choice:" not in kimi.prompt()

    def test_text_format_json_schema_reaches_the_template(self, kimi):
        text = {"format": {"type": "json_schema", "name": "reply", "schema": SCHEMA}}
        response = kimi.respond({"input": "hi", "text": text})

        assert response.status_code == 200, response.text
        prompt = kimi.prompt()
        assert "<response_format:json_schema>" in prompt
        assert '"properties": {"answer": {"type": "string"}}' in prompt

    def test_text_format_json_object_reaches_the_template(self, kimi):
        response = kimi.respond({"input": "hi", "text": {"format": {"type": "json_object"}}})

        assert response.status_code == 200, response.text
        assert "<response_format:json_object>" in kimi.prompt()
        assert "<response_schema:" not in kimi.prompt()

    def test_plain_text_format_adds_nothing(self, kimi):
        response = kimi.respond({"input": "hi", "text": {"format": {"type": "text"}}})

        assert response.status_code == 200, response.text
        assert "<response_format:" not in kimi.prompt()

    def test_explicit_chat_template_kwargs_win_over_derived_values(self, kimi):
        response = kimi.respond(
            {
                "input": "hi",
                "reasoning": {"effort": "high"},
                "chat_template_kwargs": {"thinking_effort": "low"},
            }
        )

        assert response.status_code == 200, response.text
        assert "<effort:low>" in kimi.prompt()


# ---------------------------------------------------------------------------
# The sampling-parameter policy, on the fields this endpoint has
# ---------------------------------------------------------------------------


class TestSamplingPolicy:
    @pytest.mark.parametrize("sent", [None, 1.0], ids=["unset", "openai_default"])
    def test_top_p_is_pinned_when_the_policy_is_on(self, kimi_with_policy, sent):
        body = {"input": "hi"} if sent is None else {"input": "hi", "top_p": sent}
        response = kimi_with_policy.respond(body)

        assert response.status_code == 200, response.text
        assert kimi_with_policy.engine_calls[-1].sampling_params.top_p == pytest.approx(0.95)

    def test_an_out_of_range_temperature_is_rejected_before_generation(self, kimi_with_policy):
        response = kimi_with_policy.respond({"input": "hi", "temperature": 1.5})

        assert response.status_code == 400, response.text
        assert "temperature" in response.json()["message"]
        assert kimi_with_policy.engine_calls == []

    def test_the_policy_is_off_by_default(self, kimi):
        response = kimi.respond({"input": "hi", "top_p": 0.8, "temperature": 1.5})

        assert response.status_code == 200, response.text
        params = kimi.engine_calls[-1].sampling_params
        assert (params.top_p, params.temperature) == (0.8, 1.5)


# ---------------------------------------------------------------------------
# Tool results, as the K3 renderer binds them
# ---------------------------------------------------------------------------

SPLIT_AND_ORPHAN_INPUT = [
    _user("go"),
    _call("a"),
    _call("b"),
    _result("a", "A1"),
    _result("zzz", "LATE"),
    _result("a", "A2"),
    _result("b", "B"),
]


class TestToolResults:
    def test_split_results_are_joined_and_orphans_become_notes(self, kimi):
        response = kimi.respond({"input": SPLIT_AND_ORPHAN_INPUT, "tools": TOOLS})

        assert response.status_code == 200, response.text
        prompt = kimi.prompt()
        # One result per call, in the order of the first segment, then the note.
        joined = prompt.index("<|tool|>A1\nA2<result:a>")
        second = prompt.index("<|tool|>B<result:b>")
        note = prompt.index(
            "<|user|>[Output of tool call 'zzz', which matches no tool call before it:]"
        )
        assert joined < second < note
        assert "LATE" in prompt[note:]
        assert "<result:zzz>" not in prompt

    def test_a_generic_model_keeps_every_segment_and_orphan(self, generic):
        response = generic.respond({"input": SPLIT_AND_ORPHAN_INPUT, "tools": TOOLS})

        assert response.status_code == 200, response.text
        prompt = generic.prompt()
        for expected in (
            "<|tool|>A1<result:a>",
            "<|tool|>LATE<result:zzz>",
            "<|tool|>A2<result:a>",
        ):
            assert expected in prompt
        assert "matches no tool call" not in prompt


# ---------------------------------------------------------------------------
# tool_choice="required" is enforced by the template, so it is not reported
# ---------------------------------------------------------------------------


@pytest.fixture
def warnings_logged(monkeypatch):
    """Warning messages the TRT-LLM logger emits during the test.

    The logger's default level is error, and a once-per-process warning that
    an earlier test triggered would not repeat; both are reset for the test.
    """
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture(level=logging.WARNING)
    monkeypatch.setattr(logger, "_appeared_keys", set())
    monkeypatch.setattr(logger, "_set_from_env", False)
    previous = logger.level
    logger.set_level("warning")
    logger._logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger._logger.removeHandler(handler)
        logger.set_level(previous)


def _unenforced_tool_choice_warnings(messages):
    return [
        message for message in messages if "'tool_choice'" in message and "not enforced" in message
    ]


class TestToolChoiceWarning:
    def test_required_is_enforced_on_kimi_k3_and_not_reported(self, kimi, warnings_logged):
        response = kimi.respond({"input": "hi", "tools": TOOLS, "tool_choice": "required"})

        assert response.status_code == 200, response.text
        assert "<tool_choice:required>" in kimi.prompt()
        assert _unenforced_tool_choice_warnings(warnings_logged) == []

    def test_required_without_tools_is_still_reported_on_kimi_k3(self, kimi, warnings_logged):
        response = kimi.respond({"input": "hi", "tool_choice": "required"})

        assert response.status_code == 200, response.text
        assert len(_unenforced_tool_choice_warnings(warnings_logged)) == 1

    def test_required_is_reported_on_a_model_whose_template_does_not_enforce_it(
        self, generic, warnings_logged
    ):
        response = generic.respond({"input": "hi", "tools": TOOLS, "tool_choice": "required"})

        assert response.status_code == 200, response.text
        assert len(_unenforced_tool_choice_warnings(warnings_logged)) == 1

    def test_required_is_reported_on_a_model_that_enforces_it_by_grammar(
        self, grammar_enforced, warnings_logged
    ):
        """Only template enforcement reaches the Responses prompt; a grammar is not attached here."""
        response = grammar_enforced.respond(
            {"input": "hi", "tools": TOOLS, "tool_choice": "required"}
        )

        assert response.status_code == 200, response.text
        assert len(_unenforced_tool_choice_warnings(warnings_logged)) == 1


# ---------------------------------------------------------------------------
# Stored turns replay with their text, without their reasoning
# ---------------------------------------------------------------------------


class TestStoredTurns:
    def test_a_stored_turn_replays_its_text_but_not_its_reasoning(self, kimi):
        first = kimi.respond(
            {"input": "hi", "store": True, "reasoning": {"effort": "none"}}, reply="the answer"
        )
        assert first.status_code == 200, first.text
        second = kimi.respond(
            {"input": "more", "store": True, "previous_response_id": first.json()["id"]},
            reply="a thought",
        )
        assert second.status_code == 200, second.text
        assert [item["type"] for item in _output_items(second, False)] == ["reasoning"]

        third = kimi.respond(
            {"input": "next", "store": True, "previous_response_id": second.json()["id"]}
        )

        assert third.status_code == 200, third.text
        prompt = kimi.prompt()
        assert "<|user|>hi<|assistant|>the answer<|user|>more<|user|>next" in prompt
        assert "a thought" not in prompt


# ---------------------------------------------------------------------------
# The KV-aware router renders the prompt the worker renders
# ---------------------------------------------------------------------------

ROUTER_CASES = {
    "effort_high": {"input": "hi", "reasoning": {"effort": "high"}},
    "effort_medium": {"input": "hi", "reasoning": {"effort": "medium"}},
    "effort_minimal": {"input": "hi", "reasoning": {"effort": "minimal"}},
    "effort_xhigh": {"input": "hi", "reasoning": {"effort": "xhigh"}},
    "effort_none": {"input": "hi", "reasoning": {"effort": "none"}},
    "no_effort": {"input": "hi"},
    "tool_choice_required": {"input": "hi", "tools": TOOLS, "tool_choice": "required"},
    "json_schema": {
        "input": "hi",
        "text": {"format": {"type": "json_schema", "name": "reply", "schema": SCHEMA}},
    },
    "split_results": {
        "input": [
            _user("go"),
            _call("a"),
            _call("b"),
            _result("a", "A1"),
            _result("a", "A2"),
            _result("b", "B"),
        ],
        "tools": TOOLS,
    },
    "orphan_result": {"input": SPLIT_AND_ORPHAN_INPUT, "tools": TOOLS},
}


class TestRouterFollowsTheWorker:
    """Routing ids of a Kimi-K3 Responses request equal the worker's prompt ids.

    The router runs the same Responses hooks on a copy of the request and
    renders with the same template, so prefix affinity survives the kwargs and
    the history rules the extension adds.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", sorted(ROUTER_CASES))
    async def test_router_ids_equal_worker_ids(self, case, text_only):
        tokenizer = _probe_tokenizer()
        request = ResponsesRequest(
            model="m", store=False, instructions="Be brief.", **ROUTER_CASES[case]
        )
        as_sent = request.model_dump()
        router = KvCacheAwareRouter(
            server_role=None, servers=["s1", "s2"], tokens_per_block=4, use_harmony=False
        )
        with (
            mock.patch.object(router, "_get_tokenizer", return_value=tokenizer.tokenizer),
            mock.patch.object(router, "_get_model_type", return_value="kimi_k3"),
        ):
            router_ids = router._tokenize(request)
        assert request.model_dump() == as_sent

        worker_ids, _ = await request_preprocess(
            request.model_copy(deep=True),
            None,
            None,
            enable_store=False,
            use_harmony=False,
            tokenizer=tokenizer,
            model_config=type("KimiConfig", (), {"model_type": "kimi_k3"})(),
            reasoning_parser="kimi_k3",
        )
        assert router_ids == [worker_ids]
        assert len(worker_ids) > 0


# ---------------------------------------------------------------------------
# The hooks themselves, on inline inputs
# ---------------------------------------------------------------------------


def _request(**fields) -> ResponsesRequest:
    return ResponsesRequest(model="m", input="hi", **fields)


def _tool(call_id, content):
    return {"role": "tool", "content": content, "tool_call_id": call_id}


def _assistant_calls(*call_ids):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "exec", "arguments": "{}"}}
            for call_id in call_ids
        ],
    }


USER = {"role": "user", "content": "go"}


class TestExtensionHooksDirectly:
    extension = KimiK3ServingExtension()

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            pytest.param({}, None, id="nothing_to_derive"),
            pytest.param(
                {"reasoning": {"effort": "medium"}}, {"thinking_effort": "high"}, id="effort"
            ),
            pytest.param({"reasoning": {"effort": "none"}}, {"thinking": False}, id="effort_none"),
            pytest.param(
                {"reasoning": {"context": "all_turns"}}, None, id="reasoning_without_effort"
            ),
            pytest.param(
                {"tools": TOOLS, "tool_choice": "required"},
                {"tool_choice": "required"},
                id="required",
            ),
            pytest.param(
                {"tools": TOOLS, "tool_choice": "none"}, {"tool_choice": "none"}, id="none"
            ),
            pytest.param({"tools": TOOLS, "tool_choice": "auto"}, None, id="auto"),
            pytest.param({"tool_choice": "required"}, None, id="required_without_tools"),
            pytest.param(
                {"text": {"format": {"type": "json_object"}}},
                {"response_format": "json_object"},
                id="json_object",
            ),
            pytest.param(
                {"text": {"format": {"type": "json_schema", "name": "r", "schema": SCHEMA}}},
                {"response_format": "json_schema", "response_schema": SCHEMA},
                id="json_schema",
            ),
            pytest.param({"text": {"format": {"type": "text"}}}, None, id="text_format"),
            pytest.param(
                {
                    "reasoning": {"effort": "high"},
                    "chat_template_kwargs": {"thinking_effort": "low"},
                },
                {"thinking_effort": "low"},
                id="explicit_kwargs_win",
            ),
            pytest.param(
                {"reasoning": {"effort": "high"}, "chat_template_kwargs": {"custom": 1}},
                {"thinking_effort": "high", "custom": 1},
                id="explicit_kwargs_are_kept",
            ),
        ],
    )
    def test_derived_chat_template_kwargs(self, fields, expected):
        request = _request(**fields)

        self.extension.apply_responses_extensions(request)

        assert getattr(request, "chat_template_kwargs", None) == expected

    def test_a_generic_extension_derives_nothing(self):
        from tensorrt_llm.serve.serving_extensions import ServingExtension

        request = _request(reasoning={"effort": "high"}, tools=TOOLS, tool_choice="required")
        ServingExtension().apply_responses_extensions(request)
        assert not hasattr(request, "chat_template_kwargs")

    @pytest.mark.parametrize(
        ("results", "expected"),
        [
            pytest.param(
                [_tool("a", "A1"), _tool("a", "A2"), _tool("b", "B")],
                [("a", "A1\nA2"), ("b", "B")],
                id="two_segments",
            ),
            pytest.param(
                [_tool("a", "A1"), _tool("b", "B"), _tool("a", "A2\n")],
                [("a", "A1\nA2\n"), ("b", "B")],
                id="interleaved_segments_keep_the_first_segment_order",
            ),
            pytest.param(
                [_tool("a", "A1\n"), _tool("a", "A2")],
                [("a", "A1\nA2")],
                id="a_trailing_newline_is_not_doubled",
            ),
            pytest.param(
                [_tool("a", ""), _tool("a", "A2")],
                [("a", "A2")],
                id="an_empty_first_segment_adds_no_separator",
            ),
            pytest.param(
                [_tool("b", "B"), _tool("a", "A")],
                [("b", "B"), ("a", "A")],
                id="one_result_per_call_is_unchanged",
            ),
            pytest.param(
                [_tool("a", [{"type": "text", "text": "A1"}]), _tool("a", "A2")],
                [("a", [{"type": "text", "text": "A1"}, {"type": "text", "text": "A2"}])],
                id="content_parts_are_concatenated",
            ),
        ],
    )
    def test_split_results_are_joined_within_a_run(self, results, expected):
        messages = [USER, _assistant_calls("a", "b"), *results]

        prepared = self.extension.prepare_responses_history(messages)

        assert [
            (m["tool_call_id"], m["content"]) for m in prepared if m["role"] == "tool"
        ] == expected
        assert prepared[:2] == messages[:2]

    def test_a_segment_in_a_later_run_stays_there(self):
        later = {"role": "user", "content": "and?"}
        messages = [USER, _assistant_calls("a"), _tool("a", "A1"), later, _tool("a", "A2")]

        prepared = self.extension.prepare_responses_history(messages)

        # Segments are joined within a run only: the late one stays after the
        # user turn, a run of its own, and still binds to its call by id (the
        # renderer binds a run to the most recent assistant message's calls).
        assert prepared == messages

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([USER, _tool("zzz", "LATE")], id="no_call_anywhere"),
            pytest.param(
                [USER, _assistant_calls("call_1"), _tool("call_1", "ok"), _tool("zzz", "LATE")],
                id="unknown_call_id",
            ),
        ],
    )
    def test_a_result_without_a_call_becomes_a_user_note(self, messages):
        prepared = self.extension.prepare_responses_history(messages)

        assert [m for m in prepared if m.get("tool_call_id") == "zzz"] == []
        note = prepared[-1]
        assert note["role"] == "user"
        assert note["content"] == [
            {
                "type": "text",
                "text": "[Output of tool call 'zzz', which matches no tool call before it:]",
            },
            {"type": "text", "text": "LATE"},
        ]
        assert prepared[:-1] == messages[:-1]

    def test_an_assistant_turn_without_calls_ends_the_binding(self):
        """The renderer binds a run to the most recent assistant message, calls or not."""
        then = {"role": "assistant", "content": "then"}
        messages = [USER, _assistant_calls("a"), _tool("a", "ok"), then, _tool("a", "LATE")]

        prepared = self.extension.prepare_responses_history(messages)

        assert prepared[:4] == messages[:4]
        assert prepared[4]["role"] == "user"
        assert prepared[4]["content"] == [
            {
                "type": "text",
                "text": "[Output of tool call 'a', which matches no tool call before it:]",
            },
            {"type": "text", "text": "LATE"},
        ]
        assert len(prepared) == 5

    def test_the_input_conversation_is_not_mutated(self):
        messages = [
            USER,
            _assistant_calls("a"),
            _tool("a", "A1"),
            _tool("zzz", "LATE"),
            _tool("a", "A2"),
        ]
        as_given = copy.deepcopy(messages)

        prepared = self.extension.prepare_responses_history(messages)

        assert messages == as_given
        assert messages[2]["content"] == "A1"
        assert prepared is not messages
        assert [m.get("content") for m in prepared if m.get("role") == "tool"] == ["A1\nA2"]

    def test_the_note_follows_its_run_so_the_run_still_binds_by_id(self):
        messages = [
            USER,
            _assistant_calls("A", "B"),
            _tool("B", "RESULT-B"),
            _tool("gone", "RESULT-ORPHAN"),
            _tool("A", "RESULT-A"),
        ]

        prepared = self.extension.prepare_responses_history(messages)

        assert prepared[:2] == messages[:2]
        assert [m.get("tool_call_id") for m in prepared[2:4]] == ["B", "A"]
        assert prepared[4]["role"] == "user"
        assert prepared[4]["content"][1] == {"type": "text", "text": "RESULT-ORPHAN"}
        assert len(prepared) == 5

    def test_the_note_stays_with_its_run_when_the_conversation_goes_on(self):
        later = {"role": "user", "content": "and then?"}
        messages = [USER, _assistant_calls("a"), _tool("a", "A"), _tool("gone", "LATE"), later]

        prepared = self.extension.prepare_responses_history(messages)

        assert prepared[:3] == messages[:3]
        assert prepared[3]["role"] == "user"
        assert prepared[3]["content"][1] == {"type": "text", "text": "LATE"}
        assert prepared[4] is later

    def test_a_conversation_without_tool_results_is_returned_as_is(self):
        messages = [{"role": "system", "content": "Be brief."}, USER, _assistant_calls("a")]

        assert self.extension.prepare_responses_history(messages) == messages

    def test_the_policy_covers_the_fields_each_request_type_declares(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_KIMI_PARAM_POLICY", "1")

        # A Responses request has no penalties or n; extra fields a client sends
        # under those names are ignored, as this endpoint ignores unknown fields.
        request = _request(n=2, presence_penalty=0.5, frequency_penalty=0.5)
        enforce_kimi_param_policy(request)
        assert request.top_p == pytest.approx(0.95)
        with pytest.raises(ValueError, match="top_p is fixed at 0.95"):
            enforce_kimi_param_policy(_request(top_p=0.8))
        with pytest.raises(ValueError, match="temperature must be within"):
            enforce_kimi_param_policy(_request(temperature=1.5))

        # A chat request declares them, and the rules on them are unchanged.
        messages = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValueError, match="n is fixed at 1 for this model; got 2."):
            enforce_kimi_param_policy(ChatCompletionRequest(model="m", messages=messages, n=2))
        with pytest.raises(ValueError, match="presence_penalty is fixed at 0"):
            enforce_kimi_param_policy(
                ChatCompletionRequest(model="m", messages=messages, presence_penalty=0.5)
            )
        with pytest.raises(ValueError, match="frequency_penalty is fixed at 0"):
            enforce_kimi_param_policy(
                ChatCompletionRequest(model="m", messages=messages, frequency_penalty=0.5)
            )

    def test_the_policy_is_off_unless_opted_in(self, monkeypatch):
        monkeypatch.delenv("TRTLLM_KIMI_PARAM_POLICY", raising=False)
        request = _request(top_p=0.8, temperature=1.5)

        enforce_kimi_param_policy(request)

        assert (request.top_p, request.temperature) == (0.8, 1.5)
