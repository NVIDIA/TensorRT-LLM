# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the unused chat_template_kwargs guard.

A chat-template kwarg the active template never reads is a silent no-op;
the guard rejects such requests at the render site (with an env-var escape
hatch that downgrades the rejection to a one-per-key warning).
"""

from copy import deepcopy
from unittest.mock import patch

import jinja2
import pytest

from tensorrt_llm.inputs import chat_template_guard
from tensorrt_llm.inputs.chat_template_guard import (
    ALLOW_UNUSED_CHAT_TEMPLATE_KWARGS_ENV_VAR,
    ALWAYS_ALLOWED_CHAT_TEMPLATE_KWARGS,
    UnusedChatTemplateKwargsError,
    validate_chat_template_kwargs,
)
from tensorrt_llm.inputs.utils import apply_chat_template

pytestmark = pytest.mark.cpu_only

# A minimal template that reads `enable_thinking` but nothing else unusual.
TEMPLATE_WITH_TOGGLE = (
    "{%- if enable_thinking %}<think>{%- endif %}"
    "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"
)

# GLM-style retention control: reads `clear_thinking` but not DeepSeek-V4's
# `drop_thinking`, which the Anthropic adapter always sends alongside it.
TEMPLATE_GLM_RETENTION = (
    "{%- if enable_thinking %}<think>{%- endif %}"
    "{%- if clear_thinking %}[pruned]{%- endif %}"
    "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"
)

# A template that self-defaults a variable: `disable_reasoning` is read
# before it is assigned, so it must count as referenced.
TEMPLATE_WITH_SELF_DEFAULT = (
    "{%- set disable_reasoning = disable_reasoning | default(false) -%}"
    "{%- if not disable_reasoning %}<think>{%- endif %}"
    "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"
)

# A template that reads no caller-controllable variables at all.
TEMPLATE_PLAIN = "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"

# A template in the style transformers' renderer accepts but core Jinja does
# not: the assistant body sits inside `{% generation %}` (the tag transformers
# registers for assistant-token masks). It reads `reasoning_effort` but not
# `disable_reasoning`.
TEMPLATE_WITH_GENERATION_TAG = (
    "{%- if reasoning_effort %}Reasoning: {{ reasoning_effort }}.{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}{% endgeneration %}"
    "{%- else %}{{ message['content'] }}{%- endif %}"
    "{%- endfor %}"
)

# An unsupported extension must remain the renderer's responsibility.
TEMPLATE_UNPARSEABLE = "{% bogus_tag %}{{ messages }}{% endbogus_tag %}"


@pytest.fixture(autouse=True)
def _reset_guard_state():
    """Isolate the module-level caches between tests."""
    chat_template_guard._referenced_template_variables.cache_clear()
    chat_template_guard._warn_unused_key.cache_clear()
    yield
    chat_template_guard._referenced_template_variables.cache_clear()
    chat_template_guard._warn_unused_key.cache_clear()


class TestValidateChatTemplateKwargs:
    @pytest.mark.parametrize(
        "template, kwargs",
        [
            ("{{ range(2) | list }}", {"range": range}),
            ("{{ value | custom_filter }}", {"value": "text"}),
            (
                "{% for x in messages %}{% if stop %}{% break %}{% endif %}{% endfor %}",
                {"stop": True},
            ),
            ("{% include 'external.jinja' %}", {"external_control": True}),
        ],
    )
    def test_conservative_analysis_avoids_false_rejections(self, template, kwargs):
        validate_chat_template_kwargs(template, kwargs)

    def test_referenced_kwarg_accepted(self):
        validate_chat_template_kwargs(TEMPLATE_WITH_TOGGLE, {"enable_thinking": True})

    def test_unreferenced_kwarg_rejected_and_named(self):
        with pytest.raises(ValueError, match="disable_reasoning"):
            validate_chat_template_kwargs(TEMPLATE_WITH_TOGGLE, {"disable_reasoning": True})

    def test_mixed_kwargs_rejected_naming_only_the_unknown(self):
        with pytest.raises(ValueError) as excinfo:
            validate_chat_template_kwargs(
                TEMPLATE_WITH_TOGGLE,
                {
                    "enable_thinking": True,
                    "disable_reasoning": True,
                },
            )
        assert "disable_reasoning" in str(excinfo.value)
        assert "'enable_thinking'" not in str(excinfo.value)

    def test_self_defaulted_variable_counts_as_referenced(self):
        validate_chat_template_kwargs(TEMPLATE_WITH_SELF_DEFAULT, {"disable_reasoning": True})

    def test_standard_injected_names_never_rejected(self):
        # None of these appear in TEMPLATE_PLAIN, yet transformers always
        # consumes them (render context, special tokens, or named parameters
        # of apply_chat_template), so they must pass.
        kwargs = {name: None for name in ALWAYS_ALLOWED_CHAT_TEMPLATE_KWARGS}
        validate_chat_template_kwargs(TEMPLATE_PLAIN, kwargs)

    def test_empty_kwargs_and_missing_template_are_noops(self):
        validate_chat_template_kwargs(TEMPLATE_PLAIN, {})
        validate_chat_template_kwargs(TEMPLATE_PLAIN, None)
        validate_chat_template_kwargs(None, {"disable_reasoning": True})

    def test_generation_tag_template_is_analyzed(self):
        # transformers registers the `{% generation %}` tag, which core Jinja
        # rejects; the guard used to fail OPEN on exactly such templates, so
        # every unknown kwarg passed silently on them. The guard's parse
        # environment now understands the tag: referenced kwargs pass and
        # unknown kwargs are rejected, same as for any other template.
        validate_chat_template_kwargs(TEMPLATE_WITH_GENERATION_TAG, {"reasoning_effort": "high"})
        with pytest.raises(ValueError, match="disable_reasoning"):
            validate_chat_template_kwargs(TEMPLATE_WITH_GENERATION_TAG, {"disable_reasoning": True})

    def test_unparseable_template_preserves_renderer_behavior(self):
        validate_chat_template_kwargs(TEMPLATE_UNPARSEABLE, {"disable_reasoning": True})

    def test_unparseable_template_still_accepts_always_allowed_kwargs(self):
        kwargs = {name: None for name in ALWAYS_ALLOWED_CHAT_TEMPLATE_KWARGS}
        validate_chat_template_kwargs(TEMPLATE_UNPARSEABLE, kwargs)

    def test_unparseable_template_respects_escape_hatch(self, monkeypatch):
        monkeypatch.setenv(ALLOW_UNUSED_CHAT_TEMPLATE_KWARGS_ENV_VAR, "1")
        with patch.object(chat_template_guard, "logger") as mock_logger:
            validate_chat_template_kwargs(TEMPLATE_UNPARSEABLE, {"disable_reasoning": True})
            mock_logger.warning.assert_not_called()

    def test_escape_hatch_downgrades_to_warning(self, monkeypatch):
        monkeypatch.setenv(ALLOW_UNUSED_CHAT_TEMPLATE_KWARGS_ENV_VAR, "1")
        with patch.object(chat_template_guard, "logger") as mock_logger:
            validate_chat_template_kwargs(TEMPLATE_WITH_TOGGLE, {"disable_reasoning": True})
            assert mock_logger.warning.call_count == 1
            assert "disable_reasoning" in mock_logger.warning.call_args[0][0]
            # Warned once per (template, key), not once per request.
            validate_chat_template_kwargs(TEMPLATE_WITH_TOGGLE, {"disable_reasoning": True})
            assert mock_logger.warning.call_count == 1

    def test_strict_argument_overrides_env(self, monkeypatch):
        monkeypatch.setenv(ALLOW_UNUSED_CHAT_TEMPLATE_KWARGS_ENV_VAR, "1")
        with pytest.raises(ValueError, match="disable_reasoning"):
            validate_chat_template_kwargs(
                TEMPLATE_WITH_TOGGLE, {"disable_reasoning": True}, strict=True
            )

    def test_template_swap_reanalyzes(self):
        # The cache keys on the template source: the same kwarg must be
        # rejected under one template and accepted under another.
        with pytest.raises(ValueError, match="disable_reasoning"):
            validate_chat_template_kwargs(TEMPLATE_WITH_TOGGLE, {"disable_reasoning": True})
        validate_chat_template_kwargs(TEMPLATE_WITH_SELF_DEFAULT, {"disable_reasoning": True})

    def test_rejection_is_a_distinct_value_error(self):
        # Serving layers map this specific failure to HTTP 400; it must stay a
        # ValueError so existing `except ValueError` handlers keep working.
        with pytest.raises(UnusedChatTemplateKwargsError) as excinfo:
            validate_chat_template_kwargs(TEMPLATE_PLAIN, {"disable_reasoning": True})
        assert isinstance(excinfo.value, ValueError)

    def test_parser_consumed_thinking_keys_are_always_accepted(self):
        # The reasoning parsers read `enable_thinking` / `thinking` from
        # chat_template_kwargs after generation, and tests and servers set
        # them for every model. A template that ignores them is not a no-op.
        validate_chat_template_kwargs(TEMPLATE_PLAIN, {"enable_thinking": False, "thinking": True})

    def test_injected_keys_are_exempt(self):
        # The Anthropic adapter sends `clear_thinking` (GLM) and `drop_thinking`
        # (DeepSeek-V4) together because it cannot know which one the template
        # reads. Against a GLM-style template only one is referenced.
        kwargs = {"clear_thinking": False, "drop_thinking": False}
        with pytest.raises(ValueError, match="drop_thinking"):
            validate_chat_template_kwargs(TEMPLATE_GLM_RETENTION, dict(kwargs))
        validate_chat_template_kwargs(
            TEMPLATE_GLM_RETENTION, dict(kwargs), injected_keys=set(kwargs)
        )

    def test_injected_keys_do_not_shield_caller_keys(self):
        with pytest.raises(ValueError) as excinfo:
            validate_chat_template_kwargs(
                TEMPLATE_GLM_RETENTION,
                {"clear_thinking": False, "drop_thinking": False, "disable_reasoning": True},
                injected_keys={"clear_thinking", "drop_thinking"},
            )
        assert "disable_reasoning" in str(excinfo.value)
        assert "drop_thinking" not in str(excinfo.value)


class _FakeJinjaTokenizer:
    """Tokenizer stub exercising the HF-Jinja apply_chat_template path.

    No real model or tokenizer files are loaded.
    """

    def __init__(self, chat_template: str):
        self.chat_template = chat_template

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or self.chat_template

    def apply_chat_template(self, conversation=None, **kwargs):
        return "rendered"


class TestApplyChatTemplateWiring:
    """The guard must run inside the serving render path."""

    _MESSAGES = [{"role": "user", "content": "hi"}]

    def _apply(self, template: str, chat_template_kwargs: dict):
        return apply_chat_template(
            model_type="fake_model_type_for_guard_test",
            tokenizer=_FakeJinjaTokenizer(template),
            processor=None,
            conversation=list(self._MESSAGES),
            add_generation_prompt=True,
            mm_placeholder_counts=[{}],
            chat_template_kwargs=chat_template_kwargs,
        )

    def test_referenced_kwarg_renders(self):
        assert self._apply(TEMPLATE_WITH_TOGGLE, {"enable_thinking": True}) == "rendered"

    def test_unreferenced_kwarg_raises_value_error(self):
        # ValueError is what the OpenAI server maps to a structured 400.
        with pytest.raises(ValueError, match="disable_reasoning"):
            self._apply(TEMPLATE_WITH_TOGGLE, {"disable_reasoning": True})

    def test_generation_tag_template_guarded_on_server_path(self):
        # The regression this pins: an unknown kwarg on a generation-tag
        # template used to pass silently through this entry point.
        with pytest.raises(ValueError, match="disable_reasoning"):
            self._apply(TEMPLATE_WITH_GENERATION_TAG, {"disable_reasoning": True})
        assert self._apply(TEMPLATE_WITH_GENERATION_TAG, {"reasoning_effort": "high"}) == "rendered"


class _FakeRouterTokenizer:
    """Tokenizer stub for the router tokenization entry point."""

    def __init__(self, chat_template: str):
        self.chat_template = chat_template
        self.applied_kwargs = None

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or self.chat_template

    def apply_chat_template(self, conversation, **kwargs):
        self.applied_kwargs = kwargs
        return "rendered"


class TestRouterPathWiring:
    """The guard must also run on the router tokenization entry point.

    ``serve/chat_tokenization.render_chat_request_for_tokenizer`` (KV-aware
    routing, P-D disaggregation) renders the same requests as the server path
    but used to skip kwarg validation entirely, so even a plain-Jinja
    template's unknown kwargs passed silently there.
    """

    _MESSAGES = [{"role": "user", "content": "hi"}]

    def _render(self, template: str, chat_template_kwargs: dict):
        from tensorrt_llm.serve.chat_tokenization import render_chat_request_for_tokenizer
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="test-model",
            messages=list(self._MESSAGES),
            chat_template_kwargs=chat_template_kwargs,
        )
        tokenizer = _FakeRouterTokenizer(template)
        return render_chat_request_for_tokenizer(request, tokenizer)

    def test_unknown_kwarg_rejected_on_plain_template(self):
        # Independent of the generation-tag parse fix: the router path must
        # validate even templates core Jinja parses fine.
        with pytest.raises(ValueError, match="disable_reasoning"):
            self._render(TEMPLATE_WITH_TOGGLE, {"disable_reasoning": True})

    def test_unknown_kwarg_rejected_on_generation_tag_template(self):
        with pytest.raises(ValueError, match="disable_reasoning"):
            self._render(TEMPLATE_WITH_GENERATION_TAG, {"disable_reasoning": True})

    def test_referenced_kwarg_renders_on_generation_tag_template(self):
        assert (
            self._render(TEMPLATE_WITH_GENERATION_TAG, {"reasoning_effort": "high"}) == "rendered"
        )

    def test_legacy_template_referenced_kwarg_still_renders(self):
        # No regression: a legacy (pre-generation-tag) template accepting its
        # own kwargs keeps rendering through the router path.
        assert self._render(TEMPLATE_WITH_TOGGLE, {"enable_thinking": True}) == "rendered"

    @pytest.mark.parametrize(
        "unused_kwargs",
        [
            {"totally_unknown_kwarg": 1},
            {"disable_reasoning": True},
            {"reasoning": "on"},
        ],
        ids=["unknown", "disable_reasoning", "reasoning"],
    )
    def test_unused_kwargs_rejected_on_both_entry_points(self, unused_kwargs):
        # On a generation-tag template these used to render successfully while
        # being dropped on the floor. Both entry points must reject them, so
        # the router and the server agree on what a request means.
        with pytest.raises(ValueError):
            self._render(TEMPLATE_WITH_GENERATION_TAG, dict(unused_kwargs))
        with pytest.raises(ValueError):
            apply_chat_template(
                model_type="fake_model_type_for_guard_test",
                tokenizer=_FakeJinjaTokenizer(TEMPLATE_WITH_GENERATION_TAG),
                processor=None,
                conversation=list(self._MESSAGES),
                add_generation_prompt=True,
                mm_placeholder_counts=[{}],
                chat_template_kwargs=dict(unused_kwargs),
            )

    def test_validation_does_not_rewrite_render_inputs(self):
        # The guard validates; it must not change what reaches the template.
        from tensorrt_llm.serve.chat_tokenization import render_chat_request_for_tokenizer
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="test-model",
            messages=list(self._MESSAGES),
            chat_template_kwargs={"reasoning_effort": "high"},
        )
        tokenizer = _FakeRouterTokenizer(TEMPLATE_WITH_GENERATION_TAG)
        render_chat_request_for_tokenizer(request, tokenizer)
        assert tokenizer.applied_kwargs["reasoning_effort"] == "high"


class TestServerInjectedControls:
    """Controls the server derives from API-level fields must not be rejected.

    The guard exists to catch a caller's chat_template_kwargs the template
    never reads. Keys the server itself adds (Anthropic `thinking` /
    `context_management`, Responses `reasoning.effort`) are not the caller's
    to remove, so the injection sites exempt them; caller keys stay strict.
    """

    _MESSAGES = [{"role": "user", "content": "hi"}]

    @staticmethod
    def _anthropic_chat_request():
        from tensorrt_llm.serve.anthropic_adapter import convert_anthropic_request
        from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest

        return convert_anthropic_request(
            AnthropicMessagesRequest(
                model="test-model",
                max_tokens=4096,
                messages=[{"role": "user", "content": "hi"}],
                thinking={"type": "enabled", "budget_tokens": 1024},
                context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
            )
        )

    def test_anthropic_adapter_marks_every_derived_key_as_injected(self):
        chat_request = self._anthropic_chat_request()
        assert chat_request.chat_template_kwargs == {
            "enable_thinking": True,
            "clear_thinking": False,
            "drop_thinking": False,
        }
        assert chat_request.injected_chat_template_kwargs == sorted(
            chat_request.chat_template_kwargs
        )

    def test_anthropic_retention_pair_renders_on_router_path(self):
        # Regression: a GLM-style template reads `clear_thinking` only, and
        # the adapter's `drop_thinking` companion used to be rejected as an
        # unused caller control.
        from tensorrt_llm.serve.chat_tokenization import render_chat_request_for_tokenizer

        chat_request = self._anthropic_chat_request()
        tokenizer = _FakeRouterTokenizer(TEMPLATE_GLM_RETENTION)
        assert render_chat_request_for_tokenizer(chat_request, tokenizer) == "rendered"
        # Exempt, not pruned: the template ignores the key it does not know.
        assert tokenizer.applied_kwargs["drop_thinking"] is False
        assert tokenizer.applied_kwargs["clear_thinking"] is False

    def test_anthropic_retention_pair_renders_on_server_path(self):
        chat_request = self._anthropic_chat_request()
        rendered = apply_chat_template(
            model_type="fake_model_type_for_guard_test",
            tokenizer=_FakeJinjaTokenizer(TEMPLATE_GLM_RETENTION),
            processor=None,
            conversation=list(self._MESSAGES),
            add_generation_prompt=True,
            mm_placeholder_counts=[{}],
            chat_template_kwargs=chat_request.chat_template_kwargs,
            injected_chat_template_kwargs=chat_request.injected_chat_template_kwargs,
        )
        assert rendered == "rendered"

    def test_anthropic_retention_pair_is_rejected_without_the_exemption(self):
        # Pins that the exemption, not a widened allow-list, is what lets the
        # pair through: the same kwargs as a plain caller control still fail.
        from tensorrt_llm.serve.chat_tokenization import render_chat_request_for_tokenizer
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        chat_request = self._anthropic_chat_request()
        caller_request = ChatCompletionRequest(
            model="test-model",
            messages=list(self._MESSAGES),
            chat_template_kwargs=dict(chat_request.chat_template_kwargs),
        )
        with pytest.raises(UnusedChatTemplateKwargsError, match="drop_thinking"):
            render_chat_request_for_tokenizer(
                caller_request, _FakeRouterTokenizer(TEMPLATE_GLM_RETENTION)
            )

    def test_responses_reasoning_effort_renders_on_template_without_it(self):
        # `_test_openai_responses.py::test_reasoning_effort` against Qwen3:
        # the client only asked for reasoning.effort; the server derived
        # `reasoning_effort` and `thinking`, which a Qwen3 template (reads
        # `enable_thinking`) never references.
        from types import SimpleNamespace

        from tensorrt_llm.serve.responses_utils import (
            reasoning_chat_template_kwargs,
            reasoning_injected_chat_template_keys,
        )

        request = SimpleNamespace(
            chat_template_kwargs=None, reasoning=SimpleNamespace(effort="high")
        )
        kwargs = reasoning_chat_template_kwargs(request)
        assert kwargs == {"reasoning_effort": "high", "thinking": True}
        assert reasoning_injected_chat_template_keys(request) == {"reasoning_effort", "thinking"}
        rendered = apply_chat_template(
            model_type="fake_model_type_for_guard_test",
            tokenizer=_FakeJinjaTokenizer(TEMPLATE_WITH_TOGGLE),
            processor=None,
            conversation=list(self._MESSAGES),
            add_generation_prompt=True,
            mm_placeholder_counts=[{}],
            chat_template_kwargs=kwargs,
            injected_chat_template_kwargs=reasoning_injected_chat_template_keys(request),
        )
        assert rendered == "rendered"

    def test_responses_caller_supplied_effort_stays_strict(self):
        # A client that passes `reasoning_effort` itself picked that key, so
        # the guard still judges it against the template.
        from types import SimpleNamespace

        from tensorrt_llm.serve.responses_utils import (
            reasoning_chat_template_kwargs,
            reasoning_injected_chat_template_keys,
        )

        request = SimpleNamespace(
            chat_template_kwargs={"reasoning_effort": "high"},
            reasoning=SimpleNamespace(effort="high"),
        )
        injected = reasoning_injected_chat_template_keys(request)
        assert injected == {"thinking"}
        with pytest.raises(UnusedChatTemplateKwargsError, match="reasoning_effort"):
            apply_chat_template(
                model_type="fake_model_type_for_guard_test",
                tokenizer=_FakeJinjaTokenizer(TEMPLATE_WITH_TOGGLE),
                processor=None,
                conversation=list(self._MESSAGES),
                add_generation_prompt=True,
                mm_placeholder_counts=[{}],
                chat_template_kwargs=reasoning_chat_template_kwargs(request),
                injected_chat_template_kwargs=injected,
            )


class TestParserConsumedControls:
    """Controls a reasoning parser reads after generation must not be rejected.

    `force_nonempty_content` is consumed by NemotronV3ReasoningParser
    (llmapi/reasoning_parser.py), which reads it from chat_template_kwargs to
    decide whether an empty-content response gets its reasoning swapped into
    content. The standard Nemotron template never references the key, so a
    guard that only accepts template-referenced kwargs rejected previously
    valid requests.
    """

    _MESSAGES = [{"role": "user", "content": "hi"}]
    # What a Nemotron caller sends against a Nemotron-style template:
    # TEMPLATE_WITH_TOGGLE reads `enable_thinking` but not
    # `force_nonempty_content`, matching the real template.
    _KWARGS = {"enable_thinking": True, "force_nonempty_content": True}

    def test_force_nonempty_content_renders_on_server_path(self):
        rendered = apply_chat_template(
            model_type="fake_model_type_for_guard_test",
            tokenizer=_FakeJinjaTokenizer(TEMPLATE_WITH_TOGGLE),
            processor=None,
            conversation=list(self._MESSAGES),
            add_generation_prompt=True,
            mm_placeholder_counts=[{}],
            chat_template_kwargs=dict(self._KWARGS),
        )
        assert rendered == "rendered"

    def test_force_nonempty_content_renders_on_router_path(self):
        from tensorrt_llm.serve.chat_tokenization import render_chat_request_for_tokenizer
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="test-model",
            messages=list(self._MESSAGES),
            chat_template_kwargs=dict(self._KWARGS),
        )
        tokenizer = _FakeRouterTokenizer(TEMPLATE_WITH_TOGGLE)
        assert render_chat_request_for_tokenizer(request, tokenizer) == "rendered"
        # Accepted, not pruned: the parser reads the key from the same dict
        # after generation, so it must still reach the renderer untouched.
        assert tokenizer.applied_kwargs["force_nonempty_content"] is True

    def test_same_kwargs_drive_the_nemotron_parser(self):
        # The exemption exists because this consumer does: with the kwargs the
        # guard just accepted, the parser swaps an all-reasoning response
        # (no closing think tag) into content instead of leaving it empty.
        from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

        parser = ReasoningParserFactory.create_reasoning_parser("nemotron-v3", dict(self._KWARGS))
        result = parser.parse("a b")
        assert result.content == "a b"
        assert result.reasoning_content == ""


def _chat_request_as_the_server_sees_it(body: dict):
    """Build a ChatCompletionRequest the way ``openai_server.openai_chat`` ends up with it.

    The server validates the body strictly first. Pydantic validates the
    OpenAI message TypedDicts lazily, so a tool call whose ``function.arguments``
    is a JSON object (tau2-bench sends these) passes construction and only
    fails when ``tool_calls`` is materialized. The server catches that
    ``ValidationError`` and re-parses the raw JSON body instead
    (``openai_server.py``, the ``raw_request.json()`` fallback), which hands
    ``_parse_fallback_tool_calls`` plain dicts to normalize. Mirror both steps
    so the fixture covers each representation on the path that really serves it.
    """
    from pydantic import ValidationError

    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    try:
        probe = ChatCompletionRequest(**body)
        for message in probe.messages:
            list(dict(message).get("tool_calls") or [])
    except ValidationError:
        return ChatCompletionRequest.model_construct(**deepcopy(body))
    # The probe consumed the single-use lazy iterators; hand back a fresh one.
    return ChatCompletionRequest(**body)


@pytest.fixture
def synthetic_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "Paris": 1, "hello": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


@pytest.mark.parametrize(
    ("arguments", "served_via_fallback"),
    [
        pytest.param('{"city": "Paris"}', False, id="string_arguments_strict_path"),
        pytest.param({"city": "Paris"}, True, id="object_arguments_fallback_path"),
    ],
)
def test_router_and_server_tool_call_tokenization_match(
    synthetic_tokenizer, arguments, served_via_fallback
):
    # Both representations of `function.arguments` reach the server: the
    # OpenAI shape is a JSON string; tau2-bench sends a JSON object, which the
    # strict request model rejects (lazily) and the server re-parses from the
    # raw body. The router must tokenize each to the same ids the server does.
    from tensorrt_llm.inputs.utils import MultimodalDataTracker
    from tensorrt_llm.serve.chat_tokenization import tokenize_chat_request_for_serving
    from tensorrt_llm.serve.chat_utils import parse_chat_message_content

    template = "{{ messages[0].tool_calls[0].function.arguments.city }}"
    synthetic_tokenizer.chat_template = template
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": arguments},
                }
            ],
        }
    ]
    original = deepcopy(messages)
    request = _chat_request_as_the_server_sees_it({"model": "test-model", "messages": messages})
    # The strict model keeps `tool_calls` as a lazy pydantic iterator; the raw
    # fallback hands the plain list through. Pin which path each shape took.
    assert isinstance(dict(request.messages[0])["tool_calls"], list) is served_via_fallback
    router_tokens = tokenize_chat_request_for_serving(
        request,
        lambda: synthetic_tokenizer,
        lambda text, tokenizer: tokenizer.encode(text, add_special_tokens=False),
        use_harmony=False,
    )
    conversation = [parse_chat_message_content(deepcopy(original[0]), MultimodalDataTracker(""))]
    server_text = apply_chat_template(
        model_type="test-model",
        tokenizer=synthetic_tokenizer,
        processor=None,
        conversation=conversation,
        add_generation_prompt=True,
        mm_placeholder_counts=[{}],
    )
    assert server_text == "Paris"
    assert router_tokens == synthetic_tokenizer.encode(server_text, add_special_tokens=False) == [1]
    assert messages == original
    assert request.prompt_token_ids == router_tokens


def test_router_preserves_content_parts_and_assistant_fields(synthetic_tokenizer):
    from tensorrt_llm.serve.chat_tokenization import render_chat_request_for_tokenizer
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    synthetic_tokenizer.chat_template = (
        "{{ messages[0].content[0].text }} {{ messages[1].reasoning_content }} "
        "{{ messages[1].name }}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
        },
        {"role": "assistant", "content": "", "reasoning": "analysis", "name": "helper"},
    ]
    request = ChatCompletionRequest(model="test-model", messages=messages)
    assert (
        render_chat_request_for_tokenizer(request, synthetic_tokenizer) == "hello analysis helper"
    )


def test_native_renderer_keeps_raw_arguments():
    from tensorrt_llm.serve.chat_tokenization import render_chat_request_for_tokenizer
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    class NativeTokenizer:
        def get_chat_template(self, *args, **kwargs):
            return None

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["tool_calls"][0]["function"]["arguments"]

    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "native-format"},
                    }
                ],
            }
        ],
        chat_template_kwargs={"native_option": True},
    )
    assert render_chat_request_for_tokenizer(request, NativeTokenizer()) == "native-format"


def test_malformed_template_error_comes_from_renderer(synthetic_tokenizer):
    from tensorrt_llm.serve.chat_tokenization import render_chat_request_for_tokenizer
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    synthetic_tokenizer.chat_template = TEMPLATE_UNPARSEABLE
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        chat_template_kwargs={"custom_control": True},
    )
    with pytest.raises(jinja2.TemplateSyntaxError):
        render_chat_request_for_tokenizer(request, synthetic_tokenizer)


@pytest.mark.parametrize("selection_in_kwargs", [False, True])
def test_named_template_selection_validates_the_rendered_template(
    synthetic_tokenizer, selection_in_kwargs
):
    from tensorrt_llm.serve.chat_tokenization import render_chat_request_for_tokenizer
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    synthetic_tokenizer.chat_template = {"default": TEMPLATE_PLAIN, "controlled": "{{ control }}"}
    kwargs = {"control": "hello"}
    if selection_in_kwargs:
        kwargs["chat_template"] = "controlled"
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        chat_template=None if selection_in_kwargs else "controlled",
        chat_template_kwargs=kwargs,
    )
    assert render_chat_request_for_tokenizer(request, synthetic_tokenizer) == "hello"
