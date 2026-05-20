# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node

from tensorrt_llm._torch.auto_deploy.compile.lowering import (
    CompiledPlan,
    ModeContext,
    MultiModePlanner,
    OpArgumentResolver,
    PlanDispatcher,
    ProgramData,
    RegionSpec,
    SupportDecision,
    SupportKind,
    ValueType,
    analyze_region_boundaries,
    lower_region,
)


class _SchemaTarget:
    def __init__(self, schema: str) -> None:
        self._schema = torch._C.parse_schema(schema)
        self.__name__ = schema.split("::", 1)[1].split("(", 1)[0].replace(".", "_")
        self.__qualname__ = self.__name__
        self.__module__ = __name__

    def __call__(self, *args, **kwargs):
        return args[0] if args else kwargs.get("input")


def _rmsnorm_target():
    try:
        return torch.ops.auto_deploy.torch_rmsnorm.default
    except AttributeError:
        return _SchemaTarget(
            "auto_deploy::torch_rmsnorm(Tensor input, Tensor weight, float eps) -> Tensor"
        )


def _mode(name: str = "static") -> ModeContext:
    return ModeContext(
        name=name,
        phase=name,
        shape_buckets=(),
        cache_abi=None,
        runtime_facts={},
    )


def test_op_argument_resolver_reads_positional_kwargs_and_schema_defaults():
    resolver = OpArgumentResolver()
    target = _rmsnorm_target()

    graph = Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    positional = graph.call_function(target, args=(x, weight, 1e-5))
    kwargs = graph.call_function(
        target,
        kwargs={"input": x, "weight": weight, "eps": 2e-5},
    )

    assert resolver.get(positional, "input", "weight", "eps") == (x, weight, 1e-5)
    assert resolver.get(kwargs, "input", "weight", "eps") == (x, weight, 2e-5)

    default_target = _SchemaTarget(
        "auto_deploy::torch_rmsnorm(Tensor input, Tensor weight, float eps=0.001) -> Tensor"
    )
    defaulted = graph.call_function(default_target, args=(x, weight))
    assert resolver.one(defaulted, "eps") == 0.001


def test_support_decision_and_mode_context_use_design_fields():
    mode = ModeContext(
        name="decode",
        phase="decode",
        shape_buckets=("bucket",),
        cache_abi="cache-v1",
        runtime_facts={"tokens": 1},
    )

    assert mode.phase == "decode"
    assert mode.shape_buckets == ("bucket",)
    assert mode.cache_abi == "cache-v1"
    assert mode.runtime_facts == {"tokens": 1}
    assert SupportKind.UNSUPPORTED is SupportKind.BARRIER
    assert SupportKind.EAGER is SupportKind.EAGER_ONLY
    assert SupportDecision.supported().is_supported
    assert not SupportDecision.barrier().is_supported
    assert not SupportDecision.eager_only().is_supported
    assert not SupportDecision.error().is_supported


def _supported_op(x, weight):
    return x


def _eager_op(x):
    return x + 1


@dataclass
class _BoundaryGraph:
    graph_module: GraphModule
    x: Node
    weight: Node
    supported_a: Node
    supported_b: Node
    eager_consumer: Node


def _make_boundary_graph() -> _BoundaryGraph:
    graph = Graph()
    x = graph.placeholder("x")
    weight = graph.get_attr("weight")
    supported_a = graph.call_function(_supported_op, args=(x, weight), name="supported_a")
    supported_b = graph.call_function(_supported_op, args=(x, weight), name="supported_b")
    eager_consumer = graph.call_function(_eager_op, args=(supported_a,), name="eager_consumer")
    graph.output((supported_b, eager_consumer))

    root = nn.Module()
    root.register_buffer("weight", torch.ones(1))
    return _BoundaryGraph(
        graph_module=GraphModule(root, graph),
        x=x,
        weight=weight,
        supported_a=supported_a,
        supported_b=supported_b,
        eager_consumer=eager_consumer,
    )


def test_region_boundary_analysis_for_external_inputs_outputs_and_eager_consumer():
    bundle = _make_boundary_graph()

    region = analyze_region_boundaries(
        bundle.graph_module,
        [bundle.supported_b, bundle.supported_a],
        mode=_mode(),
        region_id="region",
    )

    assert region.source_node_names == ("supported_a", "supported_b")
    assert region.input_names == ("x", "weight")
    assert region.output_names == ("supported_a", "supported_b")


class _FakePlanModule(nn.Module):
    def __init__(self, artifact: Any) -> None:
        super().__init__()
        self.artifact = artifact

    def forward(self, value):
        return value


class _FakePlugin:
    backend_name = "fake"

    def __init__(self) -> None:
        self.extend_checks: list[tuple[tuple[str, ...], str]] = []

    def mode_contexts(self, program: ProgramData):
        return [_mode()]

    def classify_node(
        self,
        node: Node,
        mode: ModeContext,
        program: ProgramData,
        args: OpArgumentResolver,
    ):
        if node.target is _supported_op:
            return SupportDecision.supported()
            return SupportDecision.eager_only("fake plugin only lowers _supported_op")

    def can_extend_region(
        self,
        region: RegionSpec,
        candidate: Node,
        mode: ModeContext,
        program: ProgramData,
    ):
        self.extend_checks.append((region.source_node_names, candidate.name))
        return True

    def lower_region(self, region: RegionSpec, program: ProgramData, args: OpArgumentResolver):
        return {"backend": self.backend_name, "nodes": region.source_node_names}

    def make_region_module(self, region: RegionSpec, artifact: Any):
        return _FakePlanModule(artifact)


def test_coverage_manifest_exact_region_and_eager_mapping_from_planner():
    bundle = _make_boundary_graph()
    plugin = _FakePlugin()
    planner = MultiModePlanner(plugin)

    (plan,) = planner.plan(ProgramData(bundle.graph_module))

    assert plugin.extend_checks == [(("supported_a",), "supported_b")]
    assert plan.name == "fake:static"
    assert plan.coverage.plan_id == "fake:static"
    assert plan.coverage.mode_name == "static"
    assert plan.coverage.region_to_source_nodes == {
        "fake:static:region_0": ("supported_a", "supported_b")
    }
    assert plan.coverage.source_node_to_region == {
        "supported_a": "fake:static:region_0",
        "supported_b": "fake:static:region_0",
    }
    assert plan.coverage.eager_nodes == ("eager_consumer",)
    assert plan.artifacts == {
        "fake:static:region_0": {"backend": "fake", "nodes": ("supported_a", "supported_b")}
    }
    assert plan.region_modules["fake:static:region_0"].artifact == {
        "backend": "fake",
        "nodes": ("supported_a", "supported_b"),
    }


class _FakeAdapter:
    backend_name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def begin_region(self, program: ProgramData, region: RegionSpec) -> None:
        self.calls.append(("begin_region", region.region_id))

    def input(self, boundary):
        value = f"%{boundary.node_name}"
        self.calls.append(("input", boundary.node_name, value))
        return value

    def constant(self, boundary, value):
        lowered = f"const:{boundary.node_name}"
        self.calls.append(("constant", boundary.node_name, value, lowered))
        return lowered

    def output(self, values, outputs) -> None:
        self.calls.append(("output", tuple(values), tuple(output.node_name for output in outputs)))

    def emit(self, op_name, operands, attrs, result_types, *, loc=None):
        self.calls.append(("emit", op_name, tuple(operands), dict(attrs), tuple(result_types), loc))
        return f"%{op_name}"

    def finalize(self):
        self.calls.append(("finalize",))
        return {"calls": self.calls}


def _make_rmsnorm_graph(target) -> tuple[GraphModule, Node]:
    graph = Graph()
    x = graph.placeholder("x")
    weight = graph.get_attr("weight")
    rms = graph.call_function(target, args=(x, weight, 1e-5), name="rms")
    graph.output(rms)

    root = nn.Module()
    root.register_parameter("weight", nn.Parameter(torch.ones(1)))
    return GraphModule(root, graph), rms


class _FakeRmsNormLowering:
    def __call__(self, ctx, node):
        x, weight, eps = ctx.args.get(node, "input", "weight", "eps")
        return ctx.adapter.emit(
            "fake.rms_norm",
            (ctx.resolve(x), ctx.resolve(weight)),
            {"eps": float(eps)},
            (ctx.result_type(node),),
            loc=ctx.loc(node),
        )


def test_lower_region_invokes_callable_op_lowering():
    target = _rmsnorm_target()
    gm, rms = _make_rmsnorm_graph(target)
    region = analyze_region_boundaries(gm, [rms], mode=_mode(), region_id="r0")
    adapter = _FakeAdapter()

    artifact = lower_region(
        ProgramData(gm),
        region,
        adapter,
        OpArgumentResolver(),
        lowerings={target: _FakeRmsNormLowering()},
    )

    assert artifact is not None
    assert adapter.calls[0] == ("begin_region", "r0")
    assert adapter.calls[1] == ("input", "x", "%x")
    constant_call = adapter.calls[2]
    assert constant_call[:2] == ("constant", "weight")
    assert torch.equal(constant_call[2], torch.ones(1))
    assert not isinstance(constant_call[2], nn.Parameter)
    assert constant_call[3] == "const:weight"
    assert adapter.calls[3] == (
        "emit",
        "fake.rms_norm",
        ("%x", "const:weight"),
        {"eps": 1e-5},
        (ValueType(name="rms"),),
        "rms",
    )
    assert adapter.calls[4] == ("output", ("%fake.rms_norm",), ("rms",))
    assert adapter.calls[5] == ("finalize",)


class _AddModule(nn.Module):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: int) -> int:
        return value + self.amount


class _StaticPolicy:
    def __init__(self, choice) -> None:
        self.choice = choice

    def select(self, plans, args, kwargs):
        return self.choice


def test_plan_dispatcher_routes_eager_and_selected_plan():
    eager = _AddModule(1)
    plan = CompiledPlan(
        name="plan",
        backend_name="fake",
        mode=_mode(),
        module=_AddModule(10),
    )

    eager_executor = PlanDispatcher(eager, [plan], _StaticPolicy(None))
    plan_executor = PlanDispatcher(eager, [plan], _StaticPolicy("plan"))

    assert eager_executor(5) == 6
    assert plan_executor(5) == 15


class _VisibleRegionModule(nn.Module):
    def forward(self, x):
        return x + 10


class _HybridPlugin(_FakePlugin):
    def make_region_module(self, region: RegionSpec, artifact: Any):
        return _VisibleRegionModule()


def _make_hybrid_graph() -> GraphModule:
    graph = Graph()
    x = graph.placeholder("x")
    weight = graph.get_attr("weight")
    supported = graph.call_function(_supported_op, args=(x, weight), name="supported")
    eager = graph.call_function(_eager_op, args=(supported,), name="eager")
    graph.output((eager, supported))

    root = nn.Module()
    root.register_buffer("weight", torch.tensor(2))
    return GraphModule(root, graph)


def test_hybrid_plan_module_runs_region_then_eager_continuation_and_preserves_tuple_output():
    gm = _make_hybrid_graph()
    plugin = _HybridPlugin()
    (plan,) = MultiModePlanner(plugin).plan(ProgramData(gm))

    executor = PlanDispatcher(gm, [plan], _StaticPolicy(plan.name))

    assert plan.regions[0].input_names == ("x", "weight")
    assert executor(torch.tensor(5)) == (torch.tensor(16), torch.tensor(15))
    assert tuple(plan.region_modules) == ("fake:static:region_0",)
