# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""model_type -> architecture module, for families that exist only in modeling_v2.

``_router_index`` answers "which target builds this deployment", and is
consulted when the model is about to be built. The engine needs some facts
about a model earlier than that, and for a family of the built-in zoo they
come from the zoo: the config class (read before anything else), the input
processor (read by the driver process, which never builds the model) and the
serving behavior (the OpenAI server's serving extension and parsers, chosen
by model_type). A family that lives only here states those facts in
``models/<family>/architecture.py`` and is listed below. The engine reaches
them through the functions in this file and through nothing else, so a zoo
family and a modeling_v2-only family never need to know about each other.

Keyed by the checkpoint's ``model_type``, the one identity a raw
``config.json`` carries before any config class exists. A module is imported
only when a config of its model_type is in hand, so loading any other model
never imports a modeling_v2 family.

An architecture module may define any of:

* ``CONFIG_CLASSES``: ``{model_type: PretrainedConfig subclass}``, registered
  with ``transformers.AutoConfig`` once the family's config is loaded, so a
  later ``AutoConfig`` lookup in the process resolves the model_type without
  the checkpoint's remote code.
* ``load_config(config_dict, **kwargs) -> PretrainedConfig``: the config the
  engine serves, built from the raw ``config.json`` dict in place of
  ``AutoConfig`` (and of any remote code the checkpoint ships).
* ``input_processor(config) -> type | None``: the input processor class;
  None keeps the default tokenizer path.
* ``serving_extension() -> ServingExtension``: the server's per-model hooks
  (``tensorrt_llm.serve.serving_extensions``) for the model_type.
* ``TOOL_PARSER`` / ``REASONING_PARSER``: the parser names ``trtllm-serve``
  uses for the model_type when ``--tool_parser`` / ``--reasoning_parser`` is
  ``auto`` or not given (``none`` keeps a parser off).
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Optional

_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"

#: model_type -> architecture module, relative to this package.
MODELING_V2_ARCHITECTURES = {
    "kimi_k31": "models.kimi_k31.architecture",
}


def architecture_module(model_type: Optional[str]) -> Optional[ModuleType]:
    """The architecture module for ``model_type``, or None."""
    name = MODELING_V2_ARCHITECTURES.get(model_type) if model_type else None
    if name is None:
        return None
    return import_module(f"{_PACKAGE}.{name}")


def load_config(config_dict: dict, **kwargs: Any) -> Optional[Any]:
    """The config a modeling_v2-only family serves for this ``config.json``, or None."""
    module = architecture_module(config_dict.get("model_type"))
    if module is None:
        return None
    from transformers import AutoConfig

    for model_type, config_class in getattr(module, "CONFIG_CLASSES", {}).items():
        try:
            AutoConfig.register(model_type, config_class)
        except ValueError:
            pass  # registered by an earlier load in this process
    return module.load_config(config_dict, **kwargs)


def _module_for(config: Any) -> Optional[ModuleType]:
    return architecture_module(getattr(config, "model_type", None))


def input_processor_class(config: Any) -> Optional[type]:
    """The input processor class a modeling_v2-only family names for ``config``, or None."""
    module = _module_for(config)
    fn = getattr(module, "input_processor", None) if module is not None else None
    return fn(config) if fn is not None else None


def serving_extension(model_type: Optional[str]) -> Optional[Any]:
    """The serving extension a modeling_v2-only family states for ``model_type``, or None."""
    module = architecture_module(model_type)
    fn = getattr(module, "serving_extension", None) if module is not None else None
    return fn() if fn is not None else None


def serve_parser(model_type: Optional[str], kind: str) -> Optional[str]:
    """The ``kind`` ("tool" or "reasoning") parser a modeling_v2-only family states for ``model_type``, or None."""
    if kind not in ("tool", "reasoning"):
        raise ValueError(f"parser kind must be 'tool' or 'reasoning', got {kind!r}")
    module = architecture_module(model_type)
    return getattr(module, f"{kind.upper()}_PARSER", None) if module is not None else None
