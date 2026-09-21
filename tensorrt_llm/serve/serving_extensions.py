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
  (:func:`apply_model_chat_extensions`), and
- by reasoning-parser name for structured-output placement
  (:func:`structured_output_format_for`).
"""

from typing import Any, Callable, Dict, Iterable, Optional


class ServingExtension:
    """Per-model serving behavior consulted by the OpenAI-compatible server.

    Subclasses override the hooks they need; the defaults are no-ops that
    match the generic serving path.
    """

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


def apply_model_chat_extensions(request, model_type: Optional[str]) -> None:
    """Run the registered chat-request preprocessing hook, if any."""
    extension = _BY_MODEL_TYPE.get(model_type) if model_type else None
    if extension is not None:
        extension.apply_chat_extensions(request)


def structured_output_format_for(
    reasoning_parser: Optional[str],
) -> Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]]:
    """Structured-output hook registered for ``reasoning_parser``, or None."""
    extension = _BY_REASONING_PARSER.get(reasoning_parser) if reasoning_parser else None
    return None if extension is None else extension.structured_output_format
