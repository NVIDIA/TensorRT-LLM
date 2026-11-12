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
"""Per-model extension hooks for the OpenAI-compatible serving layer.

A model whose serving behavior goes beyond the generic request path registers
a :class:`ServingExtension` here instead of adding name-conditioned branches
to ``openai_protocol.py`` / ``openai_server.py``. Lookups are keyed two ways:

- by the checkpoint's top-level ``model_type`` for chat-request preprocessing
  and the other per-model hooks (:func:`apply_model_chat_extensions`,
  :func:`get_serving_extension`), and
- by reasoning-parser name for structured-output placement
  (:func:`structured_output_format_for`).

The Responses API (``/v1/responses``) consults the same extension through its
own hooks: :meth:`ServingExtension.apply_responses_extensions` preprocesses the
request, :meth:`ServingExtension.replay_stored_turn` decides how a stored turn
of the conversation is replayed, and
:meth:`ServingExtension.prepare_responses_history` adapts the converted chat
messages before the chat template renders them.

Built-in extensions live in :mod:`tensorrt_llm.serve.extensions`; importing
that package registers them. The hook lookup functions import it on first use so
the registry is populated wherever it is consulted (``openai_protocol`` reaches
it from ``to_sampling_params`` without going through ``openai_server``). The
import is deferred to call time because the built-ins import
``openai_protocol``, which imports this module.
"""

import enum
import importlib
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional

if TYPE_CHECKING:
    from tensorrt_llm.serve.openai_protocol import (
        ChatCompletionMessageParam,
        ChatCompletionRequest,
        ResponsesRequest,
    )


RequiredToolChoiceMode = Literal["template", "grammar"]


class OutputMode(enum.Enum):
    """How a model's generated output is turned into an API response."""

    # Detokenized text goes through the tool and reasoning parsers.
    TEXT = "text"
    # Raw token ids go through the Harmony adapter (gpt-oss).
    HARMONY_TOKENS = "harmony_tokens"


class ServingExtension:
    """Per-model serving behavior consulted by the OpenAI-compatible server.

    Subclasses override the hooks they need; the defaults are no-ops that
    match the generic serving path.
    """

    # Part of the rendering fingerprint: bump it when a revision of the extension changes
    # the prompt it renders, so prepared requests from the older revision are refused.
    render_version: int = 1

    def render_prompt(
        self, request: "ChatCompletionRequest", res: Any = None
    ) -> Optional[List[int]]:
        """Render ``request`` into prompt token ids, or ``None`` for the default.

        ``None`` means "use the generic chat-template rendering". A model whose
        prompt is not a chat-template string (gpt-oss/Harmony) returns its own
        token ids here. ``res`` is the render resources handle of the caller.
        """
        return None

    def output_mode(self) -> OutputMode:
        """How this model's output is consumed; see :class:`OutputMode`."""
        return OutputMode.TEXT

    def allows_required_tool_choice(self) -> bool:
        """Whether ``tool_choice="required"`` is honored for this model.

        Models that cannot enforce it reject the request instead of silently
        degrading to ``"auto"``.
        """
        return False

    def serialize_tool(self, tool) -> Dict[str, Any]:
        """Dump one request tool into the dict handed to the chat template."""
        return tool.model_dump()

    def dynamic_tools(self, messages) -> List[Dict[str, Any]]:
        """Tool declarations carried on messages rather than on the request."""
        return []

    def prompt_tokens_excluded_from_usage(
        self,
        *,
        add_generation_prompt: bool,
        pre_tokenized: bool,
        request_chat_template: Optional[str],
    ) -> int:
        """Number of trailing rendered prompt tokens not reported as prompt usage.

        Routes ask through ``RenderResources.prompt_tokens_excluded_from_usage``,
        which answers zero without asking when the server has a ``--chat_template``.

        Args:
            add_generation_prompt: Whether the prompt ends in the template's
                generation prompt.
            pre_tokenized: Whether the client supplied the prompt token ids, so
                that no prompt is rendered.
            request_chat_template: The request-level chat template, if any.
        """
        return 0

    def apply_chat_extensions(self, request) -> None:
        """Preprocess a ``ChatCompletionRequest`` before template rendering.

        Mutates ``request`` in place (typically deriving chat-template kwargs
        from request-level fields). Called only for requests whose resolved
        ``model_type`` this extension is registered for.
        """

    def apply_responses_extensions(self, request: "ResponsesRequest") -> None:
        """Preprocess a ``ResponsesRequest`` before its sampling parameters and prompt are built.

        The Responses counterpart of :meth:`apply_chat_extensions`: mutates
        ``request`` in place, typically deriving chat-template kwargs or
        sampling fields from request-level fields. Called for every Responses
        request of a model this extension is registered for, relayed
        (pre-tokenized) prompts included, before ``to_sampling_params`` reads
        the request. Any exception raised here rejects the request with a 400
        ``BadRequestError`` carrying its message, as the route answers every
        preprocessing error.
        """

    def replay_stored_turn(
        self, message: "ChatCompletionMessageParam"
    ) -> Optional["ChatCompletionMessageParam"]:
        """The chat message that replays one stored message of a Responses conversation.

        A ``previous_response_id`` follow-up replays the chat messages an
        earlier turn stored in front of its own input; this decides what each
        one becomes, ``None`` leaving it out. The default replays a message as
        stored unless it is an assistant message carrying ``reasoning``: the
        reasoning is not replayed, only the tool calls the message carries,
        which the tool results that follow answer.
        """
        if "reasoning" not in message:
            return message
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return None
        return {"role": "assistant", "content": None, "tool_calls": tool_calls}

    def prepare_responses_history(
        self, messages: List["ChatCompletionMessageParam"]
    ) -> List["ChatCompletionMessageParam"]:
        """Adapt the chat messages converted from a Responses request to the model's renderer.

        ``messages`` is the whole conversation the chat template is about to
        render: the instructions, the replayed history and the request's
        input, in order. The returned list is what is rendered and stored;
        the default returns ``messages`` unchanged. An override that changes
        the conversation returns a new list rather than mutating this one.
        """
        return messages

    def required_tool_choice_mode(self) -> Optional[RequiredToolChoiceMode]:
        """Return the required-tool policy; boolean permission selects template enforcement."""
        return "template" if self.allows_required_tool_choice() else None

    def normalize_reasoning_effort(self, effort: str) -> Optional[str]:
        """Map a protocol-level effort onto what this model's template accepts.

        Applied to efforts the serve itself derives from API fields (a
        top-level `reasoning_effort`, Anthropic `output_config.effort`,
        Responses `reasoning.effort`), never to a `chat_template_kwargs`
        value the client wrote. Returning `None` drops the kwarg so the
        template default applies. The default forwards the value unchanged.
        """
        return effort

    def structured_output_format(
        self, content: Dict[str, Any], chat_template_kwargs: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Structural-tag ``format`` dict wrapping ``content``.

        ``content`` is the guided-decoding constraint (json_schema / regex /
        grammar) to be placed relative to the model's reasoning markup.
        Returning ``None`` applies the raw grammar from the first generated
        token instead of a structural tag.
        """
        return None


_BY_MODEL_TYPE: Dict[str, ServingExtension] = {}
_BY_REASONING_PARSER: Dict[str, ServingExtension] = {}
_BUILTINS_PACKAGE = "tensorrt_llm.serve.extensions"
_builtins_loaded = False


def load_builtin_extensions() -> None:
    """Import the built-in extension package once so it self-registers.

    Servers call this at startup so a broken extension module fails the server
    start instead of the first request that needs it.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    importlib.import_module(_BUILTINS_PACKAGE)
    _builtins_loaded = True


_load_builtin_extensions = load_builtin_extensions


def register_serving_extension(
    *, model_types: Iterable[str] = (), reasoning_parsers: Iterable[str] = ()
):
    """Class decorator registering one instance under the given keys.

    Example::

        @register_serving_extension(model_types=("my_model",), reasoning_parsers=("my_parser",))
        class MyServingExtension(ServingExtension): ...
    """

    def decorator(cls):
        instance = cls()
        for key in model_types:
            _BY_MODEL_TYPE[key] = instance
        for key in reasoning_parsers:
            _BY_REASONING_PARSER[key] = instance
        return cls

    return decorator


_DEFAULT_EXTENSION = ServingExtension()


def get_serving_extension(model_type: Optional[str]) -> ServingExtension:
    """Extension registered for ``model_type``, or the all-defaults extension.

    The default extension matches the generic serving path, so callers can
    consult the hooks unconditionally instead of branching on a model name.
    """
    _load_builtin_extensions()
    extension = _BY_MODEL_TYPE.get(model_type) if model_type else None
    return _DEFAULT_EXTENSION if extension is None else extension


def apply_model_chat_extensions(request, model_type: Optional[str]) -> None:
    """Run the registered chat-request preprocessing hook, if any."""
    _load_builtin_extensions()
    extension = _BY_MODEL_TYPE.get(model_type) if model_type else None
    if extension is not None:
        extension.apply_chat_extensions(request)


def normalize_reasoning_effort(effort: str, model_type: Optional[str]) -> Optional[str]:
    """Run the effort normalizer registered for `model_type`; identity if none."""
    _load_builtin_extensions()
    extension = _BY_MODEL_TYPE.get(model_type) if model_type else None
    return effort if extension is None else extension.normalize_reasoning_effort(effort)


def structured_output_format_for(
    reasoning_parser: Optional[str],
) -> Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]]:
    """Structured-output hook registered for ``reasoning_parser``, or None."""
    _load_builtin_extensions()
    extension = _BY_REASONING_PARSER.get(reasoning_parser) if reasoning_parser else None
    return None if extension is None else extension.structured_output_format


def required_tool_choice_mode(model_type: Optional[str]) -> Optional[RequiredToolChoiceMode]:
    """Return required-tool handling for native chat, or None if unsupported."""
    _load_builtin_extensions()
    extension = _BY_MODEL_TYPE.get(model_type) if model_type else None
    return None if extension is None else extension.required_tool_choice_mode()
