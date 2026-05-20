# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .arguments import OpArgumentResolver
from .boundaries import analyze_region_boundaries, executable_nodes, graph_order
from .context import BackendAdapter, LoweringContext
from .datamodel import (
    BoundaryValue,
    CompiledPlan,
    CoverageManifest,
    InputKind,
    ModeContext,
    OutputKind,
    ProgramData,
    ProgramInput,
    ProgramInterface,
    ProgramOutput,
    RegionSpec,
    SupportDecision,
    SupportKind,
    ValueType,
)
from .executor import (
    DispatchChoice,
    DispatchPolicy,
    EagerDispatchPolicy,
    FirstPlanDispatchPolicy,
    PlanDispatcher,
)
from .hybrid import HybridGraphBuilder, HybridPlanModule
from .planning import BackendPlugin, MultiModePlanner, default_contiguous_supported_groups
from .region import lower_region, resolve_get_attr_value
from .registry import LOWERINGS, LoweringRule

__all__ = [
    "BackendAdapter",
    "BackendPlugin",
    "BoundaryValue",
    "CompiledPlan",
    "CoverageManifest",
    "DispatchChoice",
    "DispatchPolicy",
    "EagerDispatchPolicy",
    "FirstPlanDispatchPolicy",
    "InputKind",
    "HybridGraphBuilder",
    "HybridPlanModule",
    "LoweringContext",
    "LoweringRule",
    "LOWERINGS",
    "ModeContext",
    "MultiModePlanner",
    "OpArgumentResolver",
    "OutputKind",
    "ProgramData",
    "ProgramInput",
    "ProgramInterface",
    "ProgramOutput",
    "RegionSpec",
    "SupportDecision",
    "SupportKind",
    "ValueType",
    "analyze_region_boundaries",
    "default_contiguous_supported_groups",
    "executable_nodes",
    "graph_order",
    "lower_region",
    "resolve_get_attr_value",
    "PlanDispatcher",
]
