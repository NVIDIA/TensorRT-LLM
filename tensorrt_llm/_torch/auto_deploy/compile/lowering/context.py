# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from torch.fx import Node
from torch.fx.node import map_arg

from .arguments import OpArgumentResolver
from .datamodel import BoundaryValue, ProgramData, RegionSpec, ValueType


@runtime_checkable
class BackendAdapter(Protocol):
    backend_name: str

    def begin_region(self, program: ProgramData, region: RegionSpec) -> None: ...

    def input(self, boundary: BoundaryValue) -> Any: ...

    def constant(self, boundary: BoundaryValue, value: object) -> Any: ...

    def output(self, values: Sequence[Any], outputs: Sequence[BoundaryValue]) -> None: ...

    def emit(
        self,
        op_name: str,
        operands: Sequence[Any],
        attrs: Mapping[str, object],
        result_types: Sequence[ValueType],
        *,
        loc: Any | None = None,
    ) -> Any: ...

    def finalize(self) -> Any: ...


@dataclass
class LoweringContext:
    program: ProgramData
    region: RegionSpec
    adapter: BackendAdapter
    args: OpArgumentResolver = field(default_factory=OpArgumentResolver)
    env: dict[Node, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def b(self) -> BackendAdapter:
        return self.adapter

    @property
    def mode(self):
        return self.region.mode

    @property
    def backend(self) -> str:
        return self.adapter.backend_name

    def resolve(self, value: Any) -> Any:
        def resolve_node(node: Node) -> Any:
            if node not in self.env:
                raise KeyError(f"FX node {node.name!r} has not been lowered")
            return self.env[node]

        return map_arg(value, resolve_node)

    def get_args(self, node: Node, *names: str) -> tuple[Any, ...]:
        return tuple(self.resolve(value) for value in self.args.get(node, *names))

    def arg(self, node: Node, name: str) -> Any:
        return self.kw(node, name)

    def kw(self, node: Node, name: str) -> Any:
        return self.resolve(self.args.one(node, name))

    def result_type(self, node: Node) -> ValueType:
        return ValueType.from_node(node)

    def loc(self, node: Node) -> str:
        return node.name
