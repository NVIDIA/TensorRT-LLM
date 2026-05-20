from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import torch.nn as nn
from torch.fx import Node

from .arguments import OpArgumentResolver
from .boundaries import analyze_region_boundaries, executable_nodes, graph_order
from .datamodel import (
    CompiledPlan,
    CoverageManifest,
    ModeContext,
    ProgramData,
    RegionSpec,
    SupportDecision,
    SupportKind,
)
from .hybrid import HybridGraphBuilder


class BackendPlugin(Protocol):
    backend_name: str

    def mode_contexts(self, program: ProgramData) -> Iterable[ModeContext | str]: ...

    def classify_node(
        self,
        node: Node,
        mode: ModeContext,
        program: ProgramData,
        args: OpArgumentResolver,
    ) -> SupportDecision | bool: ...

    def can_extend_region(
        self,
        region: RegionSpec,
        candidate: Node,
        mode: ModeContext,
        program: ProgramData,
    ) -> bool: ...

    def lower_region(
        self,
        region: RegionSpec,
        program: ProgramData,
        args: OpArgumentResolver,
    ) -> Any: ...

    def make_region_module(self, region: RegionSpec, artifact: Any) -> nn.Module: ...


def default_contiguous_supported_groups(
    program: ProgramData, decisions: Mapping[Node, SupportDecision]
) -> tuple[tuple[Node, ...], ...]:
    groups: list[tuple[Node, ...]] = []
    current: list[Node] = []
    for node in executable_nodes(program.graph_module):
        if decisions.get(node, SupportDecision.barrier()).is_supported:
            current.append(node)
            continue
        if current:
            groups.append(tuple(current))
            current = []
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _as_mode_context(mode: ModeContext | str) -> ModeContext:
    return mode if isinstance(mode, ModeContext) else ModeContext(str(mode), phase=str(mode))


def _as_support_decision(decision: SupportDecision | bool) -> SupportDecision:
    if isinstance(decision, SupportDecision):
        return decision
    return SupportDecision.supported() if decision else SupportDecision.barrier()


class MultiModePlanner:
    def __init__(
        self,
        plugin: BackendPlugin,
        *,
        argument_resolver: OpArgumentResolver | None = None,
    ) -> None:
        self.plugin = plugin
        self.argument_resolver = argument_resolver or OpArgumentResolver()

    def plan(self, program: ProgramData) -> list[CompiledPlan]:
        modes = tuple(_as_mode_context(mode) for mode in self._mode_contexts(program))
        return [self._plan_mode(program, mode) for mode in modes]

    def build(self, program: ProgramData) -> list[CompiledPlan]:
        return self.plan(program)

    def _mode_contexts(self, program: ProgramData) -> Iterable[ModeContext | str]:
        if hasattr(self.plugin, "mode_contexts"):
            return self.plugin.mode_contexts(program)
        return self.plugin.modes(program)

    def _plan_mode(self, program: ProgramData, mode: ModeContext) -> CompiledPlan:
        backend_name = getattr(self.plugin, "backend_name", "")
        plan_name = f"{backend_name}:{mode.name}" if backend_name else mode.name
        decisions = {
            node: _as_support_decision(
                self.plugin.classify_node(node, mode, program, self.argument_resolver)
            )
            for node in executable_nodes(program.graph_module)
        }
        error_nodes = [
            f"{node.name}: {decision.reason or 'unsupported'}"
            for node, decision in decisions.items()
            if decision.kind == SupportKind.ERROR
        ]
        if error_nodes:
            raise RuntimeError(f"Backend planning failed for {plan_name}: {error_nodes}")

        groups, eager_nodes = self._build_region_groups(program, mode, decisions, plan_name)
        regions = [
            analyze_region_boundaries(
                program.graph_module,
                group,
                mode=mode,
                region_id=f"{plan_name}:region_{index}",
                support={node.name: decisions[node] for node in group},
            )
            for index, group in enumerate(groups)
        ]

        artifacts = {
            region.region_id: self.plugin.lower_region(region, program, self.argument_resolver)
            for region in regions
        }
        region_modules = {
            region.region_id: self.plugin.make_region_module(region, artifacts[region.region_id])
            for region in regions
        }
        coverage = CoverageManifest.from_regions(
            regions,
            eager_nodes,
            plan_id=plan_name,
            mode=mode,
        )
        return CompiledPlan(
            name=plan_name,
            backend_name=backend_name,
            mode=mode,
            module=HybridGraphBuilder(program).build(regions, region_modules)
            if regions
            else program.graph_module,
            regions=tuple(regions),
            coverage=coverage,
            artifacts=artifacts,
            region_modules=region_modules,
            lowered_regions=artifacts,
        )

    def _build_region_groups(
        self,
        program: ProgramData,
        mode: ModeContext,
        decisions: Mapping[Node, SupportDecision],
        plan_name: str,
    ) -> tuple[tuple[Node, ...], tuple[Node, ...]]:
        order = graph_order(program.graph_module)
        groups: list[tuple[Node, ...]] = []
        eager_nodes: list[Node] = []
        current: list[Node] = []

        def flush_current() -> None:
            nonlocal current
            if current:
                groups.append(tuple(sorted(current, key=lambda node: order[node])))
                current = []

        for node in executable_nodes(program.graph_module):
            decision = decisions[node]
            if not decision.is_supported:
                flush_current()
                eager_nodes.append(node)
                continue

            if not current:
                current.append(node)
                continue

            provisional = RegionSpec(
                region_id=f"{plan_name}:pending",
                mode=mode,
                source_nodes=tuple(current),
            )
            if self.plugin.can_extend_region(provisional, node, mode, program):
                current.append(node)
            else:
                flush_current()
                current.append(node)

        flush_current()
        return tuple(groups), tuple(eager_nodes)
