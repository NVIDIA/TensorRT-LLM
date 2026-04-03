# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Transform-time registry for backend-native attention mask providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from torch.fx import GraphModule, Node

from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils._graph import _NO_VAL, add_graph_input
from .interface import SharedConfig

AttentionMaskProviderFn = Callable[["AttentionMaskProviderContext", Node], Optional[Node]]


def infer_model_type(factory: Optional[ModelFactory]) -> Optional[str]:
    """Best-effort inference of the source model type for provider lookup."""
    if factory is None:
        return None

    model_type = getattr(factory, "model_type", None)
    if isinstance(model_type, str):
        return model_type

    get_model_type = getattr(factory, "get_model_type", None)
    if callable(get_model_type):
        inferred = get_model_type()
        if isinstance(inferred, str):
            return inferred

    get_model_config = getattr(factory, "_get_model_config", None)
    if callable(get_model_config):
        try:
            model_config, _unused_kwargs = get_model_config()
        except Exception:
            return None
        inferred = getattr(model_config, "model_type", None)
        if isinstance(inferred, str):
            return inferred

    return None


@dataclass
class AttentionMaskProviderContext:
    """Context object shared across provider invocations within one transform pass."""

    gm: GraphModule
    cm: Optional[CachedSequenceInterface]
    factory: Optional[ModelFactory]
    shared_config: SharedConfig
    model_type: str
    backend: str
    cache: Dict[str, Node] = field(default_factory=dict)

    def add_or_retrieve_input(
        self,
        name: str,
        *,
        activate_arg: bool = True,
        val: Any = _NO_VAL,
    ) -> Node:
        """Add or retrieve a graph placeholder for a provider input."""
        input_nodes = self.gm.graph.find_nodes(op="placeholder", target=name)
        if len(input_nodes) == 1:
            return input_nodes[0]
        if len(input_nodes) > 1:
            raise ValueError(f"Expected exactly one input node for {name=}, got {input_nodes=}")

        if activate_arg:
            if self.cm is None:
                raise ValueError(
                    f"Cannot activate managed arg {name!r} without CachedSequenceInterface."
                )
            self.cm.info.activate_arg(name)

        return add_graph_input(self.gm, name=name, val=val)

    def get_or_create_cached_node(self, key: str, builder: Callable[[], Node]) -> Node:
        """Memoize provider-created nodes so shared masks are built once per forward."""
        if key not in self.cache:
            self.cache[key] = builder()
        return self.cache[key]


class AttentionMaskProviderRegistry:
    """Registry for backend-native attention mask providers."""

    _registry: Dict[Tuple[str, str], AttentionMaskProviderFn] = {}

    @classmethod
    def register(
        cls, model_type: str, backend: str
    ) -> Callable[[AttentionMaskProviderFn], AttentionMaskProviderFn]:
        """Register a provider for a specific ``(model_type, backend)`` pair."""

        def decorator(provider: AttentionMaskProviderFn) -> AttentionMaskProviderFn:
            cls._registry[(model_type, backend)] = provider
            return provider

        return decorator

    @classmethod
    def get(
        cls, model_type: Optional[str], backend: Optional[str]
    ) -> Optional[AttentionMaskProviderFn]:
        """Return the provider registered for ``(model_type, backend)``."""
        if model_type is None or backend is None:
            return None
        return cls._registry.get((model_type, backend))

    @classmethod
    def has(cls, model_type: str, backend: str) -> bool:
        """Return whether a provider exists for ``(model_type, backend)``."""
        return (model_type, backend) in cls._registry
