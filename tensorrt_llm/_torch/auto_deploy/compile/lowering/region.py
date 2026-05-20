# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping

import torch

from .arguments import OpArgumentResolver
from .context import BackendAdapter, LoweringContext
from .datamodel import BoundaryValue, ProgramData, RegionSpec
from .registry import LOWERINGS, LoweringRule


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
    lowerings: Mapping[object, LoweringRule] | None = None,
):
    args = args or OpArgumentResolver()
    ctx = LoweringContext(program=program, region=region, adapter=adapter, args=args, env={})

    adapter.begin_region(program, region)
    for boundary in region.boundary_inputs:
        ctx.env[boundary.node] = _bind_boundary_input(ctx, boundary)

    lowering_map = LOWERINGS if lowerings is None else lowerings
    for node in region.source_nodes:
        rule = lowering_map.get(node.target)
        if rule is None:
            raise KeyError(f"No lowering registered for source_op={node.target!r}")
        ctx.env[node] = rule(ctx, node)

    values = [ctx.resolve(output.node) for output in region.boundary_outputs]
    adapter.output(values, region.boundary_outputs)
    return adapter.finalize()
