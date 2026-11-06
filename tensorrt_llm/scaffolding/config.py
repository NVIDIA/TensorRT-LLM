# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Build scaffolding controllers from a declarative config file.

Controllers compose recursively: ``MajorityVoteController`` wraps a generation
controller, and ``BestOfNController`` wraps a generation controller and a reward
controller. A config is therefore a tree. Any object carrying a ``"type"`` key is
a controller node, and its ``"args"`` are passed straight to the constructor;
nested controller arguments are built first, depth first.

Example:
    ```python
    from tensorrt_llm.scaffolding import load_controller_config

    controller = load_controller_config("majority_vote.json")
    ```

    where ``majority_vote.json`` is:

    ```json
    {
      "controller": {
        "type": "MajorityVoteController",
        "args": {
          "default_sample_num": 3,
          "generation_controller": {
            "type": "NativeGenerationController",
            "args": {"sampling_params": {"max_tokens": 1024, "temperature": 0.9}}
          }
        }
      }
    }
    ```

Note:
    ``"type"`` is reserved. Any mapping carrying that key is read as a nested controller
    node, so a controller argument that is itself a plain dict cannot contain a literal
    ``"type"`` entry -- it would be routed into :func:`build_controller` and rejected as an
    unknown controller type. No controller in ``CONTROLLER_REGISTRY`` takes such an argument
    today. The rejection is deliberate: mistyping a controller name is far more likely than
    needing a ``"type"`` key in a data argument, and failing loudly on the typo is worth more
    than supporting the rarer case. Arguments that must carry arbitrary keys should be nested
    one level deeper, where they are passed through untouched.
"""

from typing import Any, Dict, Mapping, Optional, Type

import yaml

from .controller import (
    BestOfNController,
    ChatWithMCPController,
    Controller,
    MajorityVoteController,
    NativeChatController,
    NativeGenerationController,
    NativeRewardController,
    PRMController,
)

# Explicit name -> class mapping. Deliberately a literal table rather than a
# module scan, so that reading a config never triggers an import side effect and
# the set of constructible controllers stays obvious.
CONTROLLER_REGISTRY: Dict[str, Type[Controller]] = {
    "NativeGenerationController": NativeGenerationController,
    "NativeChatController": NativeChatController,
    "NativeRewardController": NativeRewardController,
    "PRMController": PRMController,
    "MajorityVoteController": MajorityVoteController,
    "BestOfNController": BestOfNController,
    "ChatWithMCPController": ChatWithMCPController,
}

# Key marking a controller node.
TYPE_KEY = "type"
# Key holding the constructor arguments of a controller node.
ARGS_KEY = "args"
# Key marking a reference to a caller-supplied live object.
REF_KEY = "$ref"


def register_controller(name: str, controller_cls: Type[Controller]) -> None:
    """Make a controller constructible by name from a config file.

    Contributed controllers under ``scaffolding/contrib`` can call this at import
    time so that they become available without the core registry importing them.

    Args:
        name: Name used as the ``"type"`` value in configs.
        controller_cls: The controller class to construct.

    Raises:
        TypeError: If ``controller_cls`` is not a ``Controller`` subclass.
        ValueError: If ``name`` is already registered to a different class.
    """
    if not (isinstance(controller_cls, type) and issubclass(controller_cls, Controller)):
        raise TypeError(
            f"{name!r} must be registered to a Controller subclass, got {controller_cls!r}"
        )

    existing = CONTROLLER_REGISTRY.get(name)
    if existing is not None and existing is not controller_cls:
        raise ValueError(f"Controller name {name!r} is already registered to {existing.__name__}")

    CONTROLLER_REGISTRY[name] = controller_cls


def _is_controller_spec(value: Any) -> bool:
    """Return True if the value describes a nested controller."""
    return isinstance(value, Mapping) and TYPE_KEY in value


def _resolve_value(value: Any, objects: Mapping[str, Any], path: str) -> Any:
    """Resolve a single constructor argument.

    Nested controller specs are built recursively, ``$ref`` entries are looked up
    in ``objects``, and lists are resolved element-wise. Everything else is passed
    through unchanged, so plain dicts such as ``sampling_params`` stay dicts.
    """
    if _is_controller_spec(value):
        return build_controller(value, objects=objects, _path=path)

    if isinstance(value, Mapping) and REF_KEY in value:
        ref = value[REF_KEY]
        if ref not in objects:
            available = ", ".join(sorted(objects)) or "<none>"
            raise KeyError(f"{path}: unknown object reference {ref!r}. Available: {available}")
        return objects[ref]

    if isinstance(value, list):
        return [_resolve_value(item, objects, f"{path}[{i}]") for i, item in enumerate(value)]

    return value


def build_controller(
    spec: Mapping[str, Any],
    objects: Optional[Mapping[str, Any]] = None,
    _path: str = "controller",
) -> Controller:
    """Construct a controller from an in-memory config node.

    Args:
        spec: Mapping with a ``"type"`` key naming a registered controller and an
            optional ``"args"`` mapping of constructor arguments.
        objects: Live objects that cannot be expressed in a config file, such as a
            tokenizer for ``PRMController`` or the tool list for
            ``ChatWithMCPController``. Reference them from the config with
            ``{"$ref": "<name>"}``.

    Returns:
        The constructed controller, with nested controllers already built.

    Raises:
        TypeError: If ``spec`` is not a mapping, or the arguments do not match the
            controller's signature.
        ValueError: If ``"type"`` is missing or names an unregistered controller.
    """
    objects = objects or {}

    if not isinstance(spec, Mapping):
        raise TypeError(
            f"{_path}: expected a mapping describing a controller, got {type(spec).__name__}"
        )

    type_name = spec.get(TYPE_KEY)
    if type_name is None:
        raise ValueError(f"{_path}: missing required {TYPE_KEY!r} key")

    controller_cls = CONTROLLER_REGISTRY.get(type_name)
    if controller_cls is None:
        known = ", ".join(sorted(CONTROLLER_REGISTRY))
        raise ValueError(
            f"{_path}: unknown controller type {type_name!r}. Registered types: {known}"
        )

    unexpected = set(spec) - {TYPE_KEY, ARGS_KEY}
    if unexpected:
        raise ValueError(
            f"{_path}: unexpected key(s) {sorted(unexpected)}; "
            f"constructor arguments belong under {ARGS_KEY!r}"
        )

    raw_args = spec.get(ARGS_KEY) or {}
    if not isinstance(raw_args, Mapping):
        raise TypeError(f"{_path}.{ARGS_KEY}: expected a mapping, got {type(raw_args).__name__}")

    args = {
        key: _resolve_value(value, objects, f"{_path}.{ARGS_KEY}.{key}")
        for key, value in raw_args.items()
    }

    try:
        return controller_cls(**args)
    except TypeError as e:
        raise TypeError(f"{_path}: cannot construct {type_name} with {sorted(args)}: {e}") from e


def load_controller_config(
    path: str,
    objects: Optional[Mapping[str, Any]] = None,
) -> Controller:
    """Build a controller from a JSON or YAML config file.

    The file must contain a top-level ``"controller"`` key. Parsing goes through
    ``yaml.safe_load``, which accepts JSON as well, so both formats work.

    Args:
        path: Path to the config file.
        objects: Live objects referenced from the config via ``{"$ref": "<name>"}``.

    Returns:
        The constructed controller.

    Raises:
        ValueError: If the file is empty or has no top-level ``"controller"`` key.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, Mapping):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(config).__name__}"
        )

    if "controller" not in config:
        raise ValueError(f"{path}: missing required top-level 'controller' key")

    return build_controller(config["controller"], objects=objects)
