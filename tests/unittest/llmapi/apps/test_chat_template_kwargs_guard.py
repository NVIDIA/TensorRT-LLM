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
    validate_chat_template_kwargs,
)
from tensorrt_llm.inputs.utils import apply_chat_template

pytestmark = pytest.mark.cpu_only

# A minimal template that reads `enable_thinking` but nothing else unusual.
TEMPLATE_WITH_TOGGLE = (
    "{%- if enable_thinking %}<think>{%- endif %}"
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


@pytest.fixture
def synthetic_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "Paris": 1, "hello": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


@pytest.mark.parametrize("arguments", ['{"city": "Paris"}', {"city": "Paris"}])
def test_router_and_server_tool_call_tokenization_match(synthetic_tokenizer, arguments):
    from tensorrt_llm.inputs.utils import MultimodalDataTracker
    from tensorrt_llm.serve.chat_tokenization import tokenize_chat_request_for_serving
    from tensorrt_llm.serve.chat_utils import parse_chat_message_content
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

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
    request = ChatCompletionRequest(model="test-model", messages=messages)
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
