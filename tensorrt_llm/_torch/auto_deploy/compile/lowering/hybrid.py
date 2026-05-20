from __future__ import annotations

import operator
import re
from collections.abc import Mapping

import torch.nn as nn
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg

from .datamodel import ProgramData, RegionSpec

HybridPlanModule = GraphModule


def _copy_meta(src: Node, dst: Node) -> None:
    dst.meta.update(src.meta)


class HybridGraphBuilder:
    """Substitute lowered regions in the original FX graph with call_module nodes."""

    def __init__(self, program: ProgramData) -> None:
        self.program = program

    def build(
        self,
        regions: tuple[RegionSpec, ...] | list[RegionSpec],
        region_modules: Mapping[str, nn.Module],
    ) -> GraphModule:
        region_tuple = tuple(regions)
        if not region_tuple:
            return self.program.graph_module

        first_node_to_region = {
            region.source_nodes[0]: region for region in region_tuple if region.source_nodes
        }
        source_node_to_region = {
            node: region for region in region_tuple for node in region.source_nodes
        }
        module_targets = self._install_region_modules(region_tuple, region_modules)

        graph = Graph()
        env: dict[Node, Node] = {}

        for node in self.program.graph_module.graph.nodes:
            if node.op == "output":
                graph.output(map_arg(node.args[0], lambda arg: env[arg]))
                continue

            region = first_node_to_region.get(node)
            if region is not None:
                self._emit_region_node(graph, env, region, module_targets[region.region_id])
                continue

            if node in source_node_to_region:
                continue

            copied = graph.node_copy(node, lambda arg: env[arg])
            _copy_meta(node, copied)
            env[node] = copied

        graph.lint()
        hybrid = GraphModule(self.program.graph_module, graph)
        hybrid.graph.lint()
        hybrid.recompile()
        return hybrid

    def _install_region_modules(
        self,
        regions: tuple[RegionSpec, ...],
        region_modules: Mapping[str, nn.Module],
    ) -> dict[str, str]:
        container_name = "_lowered_regions"
        container = getattr(self.program.graph_module, container_name, None)
        if container is None:
            container = nn.ModuleDict()
            self.program.graph_module.add_module(container_name, container)
        if not isinstance(container, nn.ModuleDict):
            raise RuntimeError(
                f"Expected {container_name!r} on hybrid root to be an nn.ModuleDict, "
                f"got {type(container).__name__}"
            )

        targets: dict[str, str] = {}
        for index, region in enumerate(regions):
            module = region_modules[region.region_id]
            name = self._safe_region_module_name(region.region_id, index)
            container[name] = module
            targets[region.region_id] = f"{container_name}.{name}"
        return targets

    def _emit_region_node(
        self,
        graph: Graph,
        env: dict[Node, Node],
        region: RegionSpec,
        module_target: str,
    ) -> None:
        args = tuple(
            env[boundary.node]
            for boundary in region.boundary_inputs
            if boundary.node.op != "get_attr"
        )
        region_node = graph.call_module(module_target, args=args)
        region_node.name = self._safe_fx_node_name(region.region_id)

        if len(region.boundary_outputs) == 1:
            boundary = region.boundary_outputs[0]
            _copy_meta(boundary.node, region_node)
            env[boundary.node] = region_node
            return

        for index, boundary in enumerate(region.boundary_outputs):
            getitem = graph.call_function(operator.getitem, args=(region_node, index))
            getitem.name = f"{region_node.name}_{index}"
            _copy_meta(boundary.node, getitem)
            env[boundary.node] = getitem

    @staticmethod
    def _safe_region_module_name(region_id: str, index: int) -> str:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", region_id).strip("_")
        return f"region_{index}_{safe or 'lowered'}"

    @staticmethod
    def _safe_fx_node_name(region_id: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", region_id).strip("_")
        return safe or "lowered_region"
