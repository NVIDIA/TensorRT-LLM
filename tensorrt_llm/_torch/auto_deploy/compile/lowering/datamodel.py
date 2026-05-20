from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from torch.fx import GraphModule, Node


class InputKind(str, Enum):
    USER_INPUT = "user_input"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"
    CACHE_INPUT = "cache_input"
    RUNTIME_INPUT = "runtime_input"
    WORKSPACE = "workspace"
    TOKEN = "token"


class OutputKind(str, Enum):
    USER_OUTPUT = "user_output"
    CACHE_OUTPUT = "cache_output"
    BUFFER_UPDATE = "buffer_update"
    RUNTIME_OUTPUT = "runtime_output"
    TOKEN = "token"


@dataclass(frozen=True)
class ValueType:
    dtype: Any | None = None
    shape: tuple[Any, ...] | None = None
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "ValueType":
        shape = getattr(value, "shape", None)
        return cls(
            dtype=getattr(value, "dtype", None),
            shape=tuple(shape) if shape is not None else None,
            name=type(value).__name__ if value is not None else None,
        )

    @classmethod
    def from_node(cls, node: Node) -> "ValueType":
        meta_value = node.meta.get("val", node.meta.get("tensor_meta"))
        if meta_value is None:
            return cls(name=node.name)
        value_type = cls.from_value(meta_value)
        return cls(
            dtype=value_type.dtype,
            shape=value_type.shape,
            name=node.name,
            metadata={"source": type(meta_value).__name__},
        )


@dataclass(frozen=True)
class ProgramInput:
    node_name: str
    logical_name: str
    kind: InputKind
    type: ValueType = field(default_factory=ValueType)


@dataclass(frozen=True)
class ProgramOutput:
    source_node: str
    logical_name: str
    kind: OutputKind
    type: ValueType = field(default_factory=ValueType)


@dataclass
class ProgramInterface:
    inputs: list[ProgramInput] = field(default_factory=list)
    outputs: list[ProgramOutput] = field(default_factory=list)


@dataclass
class ProgramData:
    graph_module: GraphModule
    interface: ProgramInterface = field(default_factory=ProgramInterface)
    state: dict[str, Any] = field(default_factory=dict)
    buffers: dict[str, Any] = field(default_factory=dict)
    constants: dict[str, Any] = field(default_factory=dict)
    shape_constraints: dict[str, Any] = field(default_factory=dict)
    node_infos: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: Any | None = None

    @property
    def graph(self) -> GraphModule:
        return self.graph_module


@dataclass(frozen=True)
class ModeContext:
    name: str
    phase: str | None = None
    shape_buckets: tuple[Any, ...] = ()
    cache_abi: Any | None = None
    runtime_facts: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.phase is None:
            object.__setattr__(self, "phase", self.name)


class SupportKind(str, Enum):
    SUPPORTED = "supported"
    BARRIER = "barrier"
    EAGER_ONLY = "eager_only"
    ERROR = "error"
    UNSUPPORTED = "barrier"
    EAGER = "eager_only"


@dataclass(frozen=True)
class SupportDecision:
    kind: SupportKind
    reason: str | None = None
    constraints: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def supported(
        cls, reason: str | None = None, *constraints: Any, **metadata: Any
    ) -> "SupportDecision":
        return cls(
            SupportKind.SUPPORTED,
            reason=reason,
            constraints=tuple(constraints),
            metadata=metadata,
        )

    @classmethod
    def barrier(
        cls, reason: str | None = None, *constraints: Any, **metadata: Any
    ) -> "SupportDecision":
        return cls(
            SupportKind.BARRIER,
            reason=reason,
            constraints=tuple(constraints),
            metadata=metadata,
        )

    @classmethod
    def eager_only(
        cls, reason: str | None = None, *constraints: Any, **metadata: Any
    ) -> "SupportDecision":
        return cls(
            SupportKind.EAGER_ONLY,
            reason=reason,
            constraints=tuple(constraints),
            metadata=metadata,
        )

    @classmethod
    def error(
        cls, reason: str | None = None, *constraints: Any, **metadata: Any
    ) -> "SupportDecision":
        return cls(
            SupportKind.ERROR,
            reason=reason,
            constraints=tuple(constraints),
            metadata=metadata,
        )

    @classmethod
    def unsupported(
        cls, reason: str | None = None, *constraints: Any, **metadata: Any
    ) -> "SupportDecision":
        return cls.barrier(reason, *constraints, **metadata)

    @classmethod
    def eager(
        cls, reason: str | None = None, *constraints: Any, **metadata: Any
    ) -> "SupportDecision":
        return cls.eager_only(reason, *constraints, **metadata)

    @property
    def is_supported(self) -> bool:
        return self.kind == SupportKind.SUPPORTED


@dataclass(frozen=True)
class BoundaryValue:
    node: Node
    type: ValueType = field(default_factory=ValueType)
    logical_name: str | None = None

    @property
    def node_name(self) -> str:
        return self.node.name

    @property
    def name(self) -> str:
        return self.logical_name or self.node.name


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    mode: ModeContext
    source_nodes: tuple[Node, ...]
    boundary_inputs: tuple[BoundaryValue, ...] = ()
    boundary_outputs: tuple[BoundaryValue, ...] = ()
    constraints: tuple[Any, ...] = ()
    debug_name: str | None = None
    support: Mapping[str, SupportDecision] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def nodes(self) -> tuple[Node, ...]:
        return self.source_nodes

    @property
    def inputs(self) -> tuple[BoundaryValue, ...]:
        return self.boundary_inputs

    @property
    def outputs(self) -> tuple[BoundaryValue, ...]:
        return self.boundary_outputs

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.source_nodes)

    @property
    def source_node_names(self) -> tuple[str, ...]:
        return self.node_names

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(value.node_name for value in self.boundary_inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(value.node_name for value in self.boundary_outputs)


@dataclass(frozen=True)
class CoverageManifest:
    plan_id: str = ""
    mode_name: str = ""
    region_to_source_nodes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source_node_to_region: Mapping[str, str] = field(default_factory=dict)
    eager_nodes: tuple[str, ...] = ()
    graph_outputs: tuple[BoundaryValue, ...] = ()
    regions: tuple[RegionSpec, ...] = ()
    eager_source_nodes: tuple[Node, ...] = ()

    @classmethod
    def from_regions(
        cls,
        regions: tuple[RegionSpec, ...] | list[RegionSpec],
        eager_nodes: tuple[Node, ...] | list[Node],
        *,
        plan_id: str = "",
        mode: ModeContext | str | None = None,
        graph_outputs: tuple[BoundaryValue, ...] | list[BoundaryValue] = (),
    ) -> "CoverageManifest":
        region_tuple = tuple(regions)
        eager_tuple = tuple(eager_nodes)
        region_to_source_nodes = {
            region.region_id: tuple(node.name for node in region.source_nodes)
            for region in region_tuple
        }
        source_node_to_region = {
            node_name: region_id
            for region_id, node_names in region_to_source_nodes.items()
            for node_name in node_names
        }
        mode_name = mode.name if isinstance(mode, ModeContext) else str(mode or "")
        return cls(
            plan_id=plan_id,
            mode_name=mode_name,
            region_to_source_nodes=region_to_source_nodes,
            source_node_to_region=source_node_to_region,
            eager_nodes=tuple(node.name for node in eager_tuple),
            graph_outputs=tuple(graph_outputs),
            regions=region_tuple,
            eager_source_nodes=eager_tuple,
        )

    @property
    def region_node_names(self) -> Mapping[str, tuple[str, ...]]:
        return self.region_to_source_nodes

    @property
    def node_to_region(self) -> Mapping[str, str]:
        return self.source_node_to_region

    @property
    def eager_node_names(self) -> tuple[str, ...]:
        return self.eager_nodes

    def region_for_node(self, node: Node | str) -> str | None:
        node_name = node if isinstance(node, str) else node.name
        return self.source_node_to_region.get(node_name)

    def is_eager_node(self, node: Node | str) -> bool:
        node_name = node if isinstance(node, str) else node.name
        return node_name in self.eager_nodes


@dataclass(frozen=True)
class CompiledPlan:
    name: str
    mode: ModeContext
    backend_name: str = ""
    module: Any | None = None
    regions: tuple[RegionSpec, ...] = ()
    selection_guards: Any | None = None
    coverage: CoverageManifest = field(default_factory=CoverageManifest)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    region_modules: Mapping[str, Any] = field(default_factory=dict)
    lowered_regions: Mapping[str, Any] = field(default_factory=dict)
    debug_info: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
