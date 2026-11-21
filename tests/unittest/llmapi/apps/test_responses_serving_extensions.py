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
"""The serving-extension hooks of the Responses API request path.

A model's ``ServingExtension`` preprocesses a Responses request
(``apply_responses_extensions``), decides how each stored message of a
``previous_response_id`` conversation is replayed (``replay_stored_turn``) and
adapts the converted chat messages to its renderer
(``prepare_responses_history``). The default hooks reproduce the generic path;
these tests pin that, and prove the path consults the hooks where it must: the
request hook before the sampling parameters are built and before the prompt is
rendered, the history hooks on the conversation the template renders.
"""

import asyncio
import copy
from types import SimpleNamespace
from typing import Optional

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.serve import serving_extensions
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    ConversationHistoryStore,
    _create_input_messages,
    _serving_extension_for,
    chat_messages_from_responses_input,
    request_preprocess,
)
from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension

# The CPU-* CI stages run pytest with -m 'cpu_only'; without the marker every
# test here is deselected and the stage reports exit code 5.
pytestmark = pytest.mark.cpu_only

USER_TURN = {"role": "user", "content": "earlier question"}
ASSISTANT_TEXT = {"role": "assistant", "content": "earlier answer"}
TOOL_CALLS = [
    {"id": "call_1", "type": "function", "function": {"name": "exec", "arguments": "{}"}},
]
REASONING_WITH_CALLS = {"role": "assistant", "reasoning": "thought", "tool_calls": TOOL_CALLS}
REASONING_WITH_TEXT = {"role": "assistant", "reasoning": "thought", "content": "earlier answer"}
REASONING_ONLY = {"role": "assistant", "reasoning": "thought"}
TOOL_RESULT = {"role": "tool", "content": "ok", "tool_call_id": "call_1"}


def _request(**overrides) -> ResponsesRequest:
    fields = dict(model="m", input="hi", instructions="Be brief.", store=False)
    fields.update(overrides)
    return ResponsesRequest(**fields)


# ---------------------------------------------------------------------------
# The defaults are the generic path
# ---------------------------------------------------------------------------


class TestDefaultHooks:
    def test_apply_responses_extensions_leaves_the_request_as_sent(self) -> None:
        request = _request(
            reasoning={"effort": "high"},
            tool_choice="required",
            tools=[{"type": "function", "name": "exec", "parameters": {"type": "object"}}],
            text={"format": {"type": "json_object"}},
            temperature=0.3,
            top_p=0.8,
        )
        as_sent = request.model_dump()

        assert ServingExtension().apply_responses_extensions(request) is None

        assert request.model_dump() == as_sent
        assert not hasattr(request, "chat_template_kwargs")

    def test_prepare_responses_history_returns_the_conversation_itself(self) -> None:
        messages = [USER_TURN, ASSISTANT_TEXT, {"role": "user", "content": "next"}]
        as_built = copy.deepcopy(messages)

        assert ServingExtension().prepare_responses_history(messages) is messages
        assert messages == as_built

    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            pytest.param(USER_TURN, USER_TURN, id="user_message_as_stored"),
            pytest.param(ASSISTANT_TEXT, ASSISTANT_TEXT, id="assistant_text_as_stored"),
            pytest.param(TOOL_RESULT, TOOL_RESULT, id="tool_result_as_stored"),
            pytest.param(
                REASONING_WITH_CALLS,
                {"role": "assistant", "content": None, "tool_calls": TOOL_CALLS},
                id="reasoning_keeps_only_its_calls",
            ),
            pytest.param(REASONING_ONLY, None, id="reasoning_alone_is_left_out"),
            pytest.param(REASONING_WITH_TEXT, None, id="reasoning_with_text_is_left_out"),
        ],
    )
    def test_replay_stored_turn(self, stored, expected) -> None:
        replayed = ServingExtension().replay_stored_turn(stored)

        assert replayed == expected
        if expected is stored:
            # Replayed as stored: the same object, not a copy.
            assert replayed is stored


class TestConversionWithTheDefaultExtension:
    """What the generic conversion produces, with and without an extension argument."""

    HISTORY = [USER_TURN, REASONING_WITH_CALLS, TOOL_RESULT, REASONING_WITH_TEXT, REASONING_ONLY]
    EXPECTED = [
        {"role": "system", "content": "Be brief."},
        USER_TURN,
        {"role": "assistant", "content": None, "tool_calls": TOOL_CALLS},
        TOOL_RESULT,
        {"role": "user", "content": "hi"},
    ]

    def test_stored_history_replays_as_before(self) -> None:
        assert chat_messages_from_responses_input(_request(), self.HISTORY) == self.EXPECTED

    def test_the_default_extension_is_the_no_argument_conversion(self) -> None:
        default = serving_extensions.get_serving_extension(None)
        assert chat_messages_from_responses_input(_request(), self.HISTORY, default) == (
            self.EXPECTED
        )

    def test_the_async_wrapper_takes_the_extension(self) -> None:
        messages = asyncio.run(_create_input_messages(_request(), self.HISTORY, None))
        assert messages == self.EXPECTED


# ---------------------------------------------------------------------------
# The conversion consults the extension it is given
# ---------------------------------------------------------------------------


class _RecordingExtension(ServingExtension):
    """Records every hook call; replays and prepares in a recognizable way."""

    def __init__(self) -> None:
        self.applied: list = []
        self.replayed: list = []
        self.prepared: list = []
        self.top_p: Optional[float] = None
        self.chat_template_kwargs: Optional[dict] = None
        self.history_note: Optional[str] = None

    def apply_responses_extensions(self, request) -> None:
        self.applied.append(request)
        if self.top_p is not None:
            request.top_p = self.top_p
        if self.chat_template_kwargs is not None:
            request.chat_template_kwargs = dict(self.chat_template_kwargs)

    def replay_stored_turn(self, message):
        self.replayed.append(message)
        if message.get("role") == "tool":
            return None
        return {**message, "content": f"replayed:{message.get('content')}"}

    def prepare_responses_history(self, messages):
        self.prepared.append(list(messages))
        if self.history_note is None:
            return messages
        return [*messages, {"role": "user", "content": self.history_note}]


class TestConversionDispatch:
    def test_each_stored_message_is_replayed_as_the_extension_says(self) -> None:
        extension = _RecordingExtension()
        history = [USER_TURN, TOOL_RESULT, ASSISTANT_TEXT]

        messages = chat_messages_from_responses_input(_request(), history, extension)

        # Every stored message, in order; a None leaves the message out.
        assert extension.replayed == history
        assert messages == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "replayed:earlier question"},
            {"role": "assistant", "content": "replayed:earlier answer"},
            {"role": "user", "content": "hi"},
        ]

    def test_the_finished_conversation_is_what_the_extension_returns(self) -> None:
        extension = _RecordingExtension()
        extension.history_note = "note from the extension"

        messages = chat_messages_from_responses_input(_request(), [USER_TURN], extension)

        # Prepared once, with instructions, replayed history and input in order.
        assert extension.prepared == [
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "replayed:earlier question"},
                {"role": "user", "content": "hi"},
            ]
        ]
        assert messages == [
            *extension.prepared[0],
            {"role": "user", "content": extension.history_note},
        ]

    def test_the_extension_prepares_structured_input_too(self) -> None:
        extension = _RecordingExtension()
        request = _request(
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                }
            ]
        )

        chat_messages_from_responses_input(request, [], extension)

        assert extension.prepared == [
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "go"}]
        ]


# ---------------------------------------------------------------------------
# request_preprocess runs the request hook first, and renders what the history
# hooks return
# ---------------------------------------------------------------------------

# Renders each message as <|role|>content, a `probe` chat-template kwarg when
# one is passed, and the generation prompt.
PROBE_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
    "{% if probe is defined %}<probe:{{ probe }}>{% endif %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _probe_tokenizer() -> TransformersTokenizer:
    """A byte-level tokenizer: one token per byte, so any rendered prompt decodes back exactly."""
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


@pytest.fixture
def scratch_model():
    """A registered recording extension and a model config that resolves to it."""
    model_type = "responses-hooks-test-model"
    register_serving_extension(model_types=(model_type,))(_RecordingExtension)
    try:
        yield SimpleNamespace(
            extension=serving_extensions.get_serving_extension(model_type),
            model_config=type("ScratchConfig", (), {"model_type": model_type})(),
        )
    finally:
        serving_extensions._BY_MODEL_TYPE.pop(model_type, None)


def _preprocess(request, model_config, tokenizer=None, **overrides):
    kwargs = dict(
        request=request,
        prev_response=None,
        conversation_store=ConversationHistoryStore(),
        enable_store=False,
        use_harmony=False,
        tokenizer=tokenizer,
        model_config=model_config,
    )
    kwargs.update(overrides)
    return asyncio.run(request_preprocess(**kwargs))


class TestRequestPreprocess:
    def test_the_extension_runs_before_the_sampling_parameters_are_built(
        self, scratch_model
    ) -> None:
        scratch_model.extension.top_p = 0.25
        request = _request(top_p=1.0)

        _, sampling_params = _preprocess(request, scratch_model.model_config, _probe_tokenizer())

        assert scratch_model.extension.applied == [request]
        assert sampling_params.top_p == pytest.approx(0.25)

    def test_the_extension_runs_for_a_relayed_prompt_too(self, scratch_model) -> None:
        """A generation worker renders nothing, but its sampling parameters are still the extension's."""
        scratch_model.extension.top_p = 0.25
        request = _request(
            disaggregated_params={"request_type": "generation_only"}, prompt_token_ids=[7, 8, 9]
        )

        input_tokens, sampling_params = _preprocess(request, scratch_model.model_config)

        assert input_tokens == [7, 8, 9]
        assert sampling_params.top_p == pytest.approx(0.25)

    def test_the_prompt_renders_what_the_extension_derives_and_prepares(
        self, scratch_model
    ) -> None:
        scratch_model.extension.chat_template_kwargs = {"probe": "on"}
        scratch_model.extension.history_note = "note from the extension"
        tokenizer = _probe_tokenizer()

        input_tokens, _ = _preprocess(_request(), scratch_model.model_config, tokenizer)

        assert tokenizer.decode(input_tokens) == (
            "<|system|>Be brief.<|user|>hi<|user|>note from the extension<probe:on><|assistant|>"
        )

    def test_the_store_receives_the_prepared_conversation(self, scratch_model) -> None:
        """A previous_response_id follow-up replays what was rendered, notes included."""
        scratch_model.extension.history_note = "note from the extension"
        store = ConversationHistoryStore()
        request = _request(store=True)

        _preprocess(
            request,
            scratch_model.model_config,
            _probe_tokenizer(),
            conversation_store=store,
            enable_store=True,
        )

        stored = asyncio.run(store.get_conversation_history(request.request_id))
        assert stored == [
            *scratch_model.extension.prepared[0],
            {"role": "user", "content": "note from the extension"},
        ]

    def test_a_followup_replays_every_stored_turn_before_preparing_history(
        self, scratch_model
    ) -> None:
        extension = scratch_model.extension
        extension.history_note = "note from the extension"
        store = ConversationHistoryStore()
        tokenizer = _probe_tokenizer()
        first = _request(store=True, instructions=None, input="first")
        _preprocess(
            first,
            scratch_model.model_config,
            tokenizer,
            conversation_store=store,
            enable_store=True,
        )
        stored = asyncio.run(store.get_conversation_history(first.request_id))
        followup = _request(store=True, previous_response_id=first.request_id, input="next")

        input_tokens, _ = _preprocess(
            followup,
            scratch_model.model_config,
            tokenizer,
            conversation_store=store,
            enable_store=True,
        )

        assert extension.replayed == stored
        assert tokenizer.decode(input_tokens) == (
            "<|system|>Be brief.<|user|>replayed:first"
            "<|user|>replayed:note from the extension<|user|>next"
            "<|user|>note from the extension<|assistant|>"
        )
        assert asyncio.run(store.get_conversation_history(followup.request_id)) == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "replayed:first"},
            {"role": "user", "content": "replayed:note from the extension"},
            {"role": "user", "content": "next"},
            {"role": "user", "content": "note from the extension"},
        ]

    @pytest.mark.parametrize("reads_effort", [False, True])
    def test_hooks_keep_native_tools_and_template_aware_musespark_effort(
        self, monkeypatch, scratch_model, reads_effort
    ) -> None:
        # Exercise the real serving extension without loading a GPU model implementation.
        model_type = scratch_model.model_config.model_type
        muse_extension_type = type(serving_extensions.get_serving_extension("muse_spark"))
        register_serving_extension(model_types=(model_type,))(muse_extension_type)
        extension = serving_extensions.get_serving_extension(model_type)
        calls = []

        def prepare(messages):
            calls.append(list(messages))
            return [*messages, {"role": "user", "content": "hook note"}]

        monkeypatch.setattr(extension, "prepare_responses_history", prepare)
        template = (
            "{% set deferred_render_mode = deferred_render_mode | default('native') %}"
            "{% for tool in tools %}[{{ tool.type }}:{{ tool.name }}]{% endfor %}"
            + ("[effort:{{ reasoning_effort }}]" if reads_effort else "")
            + PROBE_TEMPLATE
        )
        tokenizer = _probe_tokenizer()
        tokenizer.tokenizer.chat_template = template
        request = _request(
            reasoning={"effort": "none"},
            tools=[
                {
                    "type": "namespace",
                    "name": "agent",
                    "description": "Agent tools.",
                    "tools": [{"type": "function", "name": "exec"}],
                }
            ],
        )
        input_tokens, _ = _preprocess(request, scratch_model.model_config, tokenizer)

        assert calls == [
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ]
        ]
        effort = "[effort:low]" if reads_effort else ""
        assert tokenizer.decode(input_tokens) == (
            f"[namespace:agent]{effort}<|system|>Be brief.<|user|>hi<|user|>hook note<|assistant|>"
        )

    def test_a_generic_model_is_preprocessed_as_before(self) -> None:
        """Without a registered extension the request and its sampling parameters are untouched."""
        request = _request(top_p=0.8, temperature=0.3)
        as_sent = request.model_dump()
        tokenizer = _probe_tokenizer()

        input_tokens, sampling_params = _preprocess(
            request, type("GenericConfig", (), {"model_type": "llama"})(), tokenizer
        )

        assert request.model_dump() == as_sent
        assert (sampling_params.top_p, sampling_params.temperature) == (0.8, 0.3)
        assert tokenizer.decode(input_tokens) == "<|system|>Be brief.<|user|>hi<|assistant|>"

    def test_the_harmony_path_is_gpt_oss(self) -> None:
        """The Harmony path passes no model config; its extension is gpt-oss's."""
        assert _serving_extension_for(None, use_harmony=True) is (
            serving_extensions.get_serving_extension("gpt_oss")
        )
        assert type(_serving_extension_for(None, use_harmony=False)) is ServingExtension

    def test_a_rejected_request_raises_before_anything_is_built(self, scratch_model) -> None:
        class Rejecting(ServingExtension):
            def apply_responses_extensions(self, request) -> None:
                raise ValueError("top_p is fixed for this model")

        # Replaces the fixture's registration under the same key; the fixture removes it.
        register_serving_extension(model_types=(scratch_model.model_config.model_type,))(Rejecting)

        with pytest.raises(ValueError, match="top_p is fixed"):
            _preprocess(_request(), scratch_model.model_config, _probe_tokenizer())


def test_a_registered_extension_hooks_are_reachable_through_the_registry() -> None:
    """The decorator registers one instance whose Responses hooks the registry hands back."""
    model_type = "responses-hooks-registered-model"

    @register_serving_extension(model_types=(model_type,))
    class Registered(ServingExtension):
        def prepare_responses_history(self, messages):
            return [*messages, {"role": "user", "content": "registered"}]

    try:
        extension = serving_extensions.get_serving_extension(model_type)
        assert isinstance(extension, Registered)
        assert chat_messages_from_responses_input(_request(), [], extension)[-1] == {
            "role": "user",
            "content": "registered",
        }
    finally:
        serving_extensions._BY_MODEL_TYPE.pop(model_type, None)
