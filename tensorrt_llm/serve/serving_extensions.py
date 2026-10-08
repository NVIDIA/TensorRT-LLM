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
that package registers them. Both lookup functions import it on first use so
the registry is populated wherever it is consulted (``openai_protocol`` reaches
it from ``to_sampling_params`` without going through ``openai_server``). The
import is deferred to call time because the built-ins import
``openai_protocol``, which imports this module.
"""

import enum
import importlib
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest


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

    def prompt_tokens_excluded_from_usage(self, request: "ChatCompletionRequest") -> int:
        """Number of trailing rendered prompt tokens not reported as prompt usage."""
        return 0

    def apply_chat_extensions(self, request) -> None:
        """Preprocess a ``ChatCompletionRequest`` before template rendering.

        Mutates ``request`` in place (typically deriving chat-template kwargs
        from request-level fields). Called only for requests whose resolved
        ``model_type`` this extension is registered for.
        """

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


def structured_output_format_for(
    reasoning_parser: Optional[str],
) -> Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]]:
    """Structured-output hook registered for ``reasoning_parser``, or None."""
    _load_builtin_extensions()
    extension = _BY_REASONING_PARSER.get(reasoning_parser) if reasoning_parser else None
    return None if extension is None else extension.structured_output_format
