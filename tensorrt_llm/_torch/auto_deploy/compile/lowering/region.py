from __future__ import annotations

import torch

from .arguments import OpArgumentResolver
from .context import BackendAdapter, LoweringContext
from .datamodel import BoundaryValue, ProgramData, RegionSpec
from .registry import LOWERINGS, LoweringFn, _get_mapping_lowering


def resolve_get_attr_value(program: ProgramData, target: str):
    value = program.graph_module
    for name in target.split("."):
        value = getattr(value, name)
    return value


def _bind_boundary_input(ctx: LoweringContext, boundary: BoundaryValue):
    node = boundary.node
    if node.op != "get_attr":
        return ctx.adapter.input(boundary)

    if not isinstance(node.target, str):
        raise TypeError(f"get_attr node {node.name!r} has non-string target {node.target!r}")

    value = resolve_get_attr_value(ctx.program, node.target)
    if isinstance(value, torch.nn.Parameter):
        value = value.detach()
    return ctx.adapter.constant(boundary, value)


def lower_region(
    program: ProgramData,
    region: RegionSpec,
    adapter: BackendAdapter,
    args: OpArgumentResolver | None = None,
    *,
    lowerings: dict[object, LoweringFn] | None = None,
):
    args = args or OpArgumentResolver()
    ctx = LoweringContext(program=program, region=region, adapter=adapter, args=args, env={})

    adapter.begin_region(program, region)
    for boundary in region.boundary_inputs:
        ctx.env[boundary.node] = _bind_boundary_input(ctx, boundary)

    lowering_map = LOWERINGS if lowerings is None else lowerings
    for node in region.source_nodes:
        rule = _get_mapping_lowering(lowering_map, node.target)
        if rule is None:
            raise KeyError(f"No lowering registered for source_op={node.target!r}")
        ctx.env[node] = rule(ctx, node)

    values = [ctx.resolve(output.node) for output in region.boundary_outputs]
    adapter.output(values, region.boundary_outputs)
    return adapter.finalize()
