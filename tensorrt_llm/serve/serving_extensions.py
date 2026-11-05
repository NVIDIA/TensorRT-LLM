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

Built-in extensions live in :mod:`tensorrt_llm.serve.extensions`; importing
that package registers them. The hook lookup functions import it on first use so
the registry is populated wherever it is consulted (``openai_protocol`` reaches
it from ``to_sampling_params`` without going through ``openai_server``). The
import is deferred to call time because the built-ins import
``openai_protocol``, which imports this module.

Two token-level hooks, :meth:`ServingExtension.extra_stop_token_ids` and
:meth:`ServingExtension.unsupported_features`, let a model whose output is
consumed as raw tokens (gpt-oss) declare the stop tokens every request needs
and the request features its serving path does not provide. The chat handler
reads them, together with ``render_prompt`` and ``output_mode``, through
``RenderResources.harmony_hooks``, which answers with the generic defaults when
the harmony switch is off.

``cli_aliases`` only extend the ``--tool_parser`` / ``--reasoning_parser``
choices of ``trtllm-serve`` and the startup checks of those options. An alias
stands for the model types it is registered with and never selects an
extension: lookups stay keyed by model type.
"""

import enum
import importlib
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest


RequiredToolChoiceMode = Literal["template", "grammar"]


class OutputMode(enum.Enum):
    """How a model's generated output is turned into an API response."""

    # Detokenized text goes through the tool and reasoning parsers.
    TEXT = "text"
    # Raw token ids go through the Harmony adapter (gpt-oss).
    HARMONY_TOKENS = "harmony_tokens"


class UnsupportedFeature(str, enum.Enum):
    """Request features a model's serving path does not provide.

    Returned by :meth:`ServingExtension.unsupported_features`; the chat handler
    checks the rejected ones in the order of :data:`UNSUPPORTED_FEATURE_ORDER`.
    The members are strings, so a declared set matches whether it holds members
    or their values.
    """

    # ``tool_choice={"type": "function", ...}``: rejected.
    NAMED_TOOL_CHOICE = "named_tool_choice"
    # ``thinking_token_budget`` set: rejected.
    THINKING_TOKEN_BUDGET = "thinking_token_budget"
    # ``logprobs`` with ``top_logprobs > 0``: rejected; ``logprobs`` alone is
    # accepted and not requested from the engine.
    TOP_LOGPROBS = "top_logprobs"
    # Per-request speculative-decoding stats: not opted in, never rejected.
    SPEC_DECODE_STATS = "spec_decode_stats"


# The order the harmony chat path has always rejected them in: the named
# tool_choice before rendering, top_logprobs while the sampling parameters are
# built, the thinking budget right after them.
UNSUPPORTED_FEATURE_ORDER: Tuple[UnsupportedFeature, ...] = (
    UnsupportedFeature.NAMED_TOOL_CHOICE,
    UnsupportedFeature.TOP_LOGPROBS,
    UnsupportedFeature.THINKING_TOKEN_BUDGET,
)

# One rejection message per rejected feature, as the harmony chat path has
# always worded them. They name gpt-oss because it is the only model that
# declares these features; a second model that declares one generalizes them.
UNSUPPORTED_FEATURE_MESSAGES: Mapping[str, str] = MappingProxyType(
    {
        UnsupportedFeature.NAMED_TOOL_CHOICE: (
            "tool_choice with a named function is not yet supported for harmony / GPT-OSS "
            "models. Tracking follow-up: harmony sub-task of TRTLLM-12758."
        ),
        UnsupportedFeature.THINKING_TOKEN_BUDGET: (
            "thinking_token_budget is not supported by the Harmony GPT-OSS serving path; "
            "use reasoning_effort instead"
        ),
        UnsupportedFeature.TOP_LOGPROBS: (
            "`gather_generation_logits` must be `True` to use `top_logprobs`"
        ),
    }
)


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

    def required_tool_choice_mode(self) -> Optional[RequiredToolChoiceMode]:
        """Return the required-tool policy; boolean permission selects template enforcement."""
        return "template" if self.allows_required_tool_choice() else None

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

    def extra_stop_token_ids(self) -> List[int]:
        """Stop token ids this model needs on every request.

        The chat handler appends them to ``request.stop_token_ids``. Default:
        none, as on the generic path.
        """
        return []

    def unsupported_features(self) -> FrozenSet[str]:
        """:class:`UnsupportedFeature` values this model's serving path does not provide.

        Default: none, as on the generic path.
        """
        return frozenset()


_BY_MODEL_TYPE: Dict[str, ServingExtension] = {}
_BY_REASONING_PARSER: Dict[str, ServingExtension] = {}
# CLI alias -> the model types it stands for. Aliases never select an extension.
_CLI_ALIASES: Dict[str, Tuple[str, ...]] = {}
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
    *,
    model_types: Iterable[str] = (),
    reasoning_parsers: Iterable[str] = (),
    cli_aliases: Iterable[str] = (),
):
    """Class decorator registering one instance under the given keys.

    ``cli_aliases`` are accepted as ``--tool_parser`` / ``--reasoning_parser``
    values of ``trtllm-serve`` for the model types of this extension; they do
    not register the extension under another lookup key.

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
        for alias in cli_aliases:
            _CLI_ALIASES[alias] = tuple(model_types)
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


def serving_extension_cli_aliases() -> List[str]:
    """Sorted CLI aliases of the registered extensions.

    ``trtllm-serve`` adds them to the ``--tool_parser`` and ``--reasoning_parser``
    choices. Loads the built-in extensions.
    """
    _load_builtin_extensions()
    return sorted(_CLI_ALIASES)


def cli_alias_model_types(alias: str) -> Tuple[str, ...]:
    """Model types ``alias`` stands for; ``()`` for a name that is not an alias."""
    _load_builtin_extensions()
    return _CLI_ALIASES.get(alias, ())


def split_cli_parser_alias(name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split a ``--tool_parser`` / ``--reasoning_parser`` value into (parser name, alias).

    ``("qwen3", None)`` for a parser name, ``(None, "gpt_oss")`` for an alias
    and ``(None, None)`` for ``None``. The alias must not reach the parser
    registries or ``LlmArgs.reasoning_parser``, which would treat it as a
    parser name.
    """
    if name is None:
        return None, None
    if cli_alias_model_types(name):
        return None, name
    return name, None
