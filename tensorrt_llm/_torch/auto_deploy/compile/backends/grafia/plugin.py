# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grafia backend plugin for the generic lowering core."""

from __future__ import annotations

from typing import Any

from torch.fx import Node

from ...lowering import (
    LOWERINGS,
    ModeContext,
    OpArgumentResolver,
    ProgramData,
    RegionSpec,
    SupportDecision,
    lower_region,
    register_builtin_lowerings,
)
from .adapter import GrafiaBackendContext, GrafiaCTMBackendAdapter
from .constants import BACKEND_NAME, GRAFIA_MODES
from .errors import GrafiaUnsupportedError
from .ops import rmsnorm
from .runtime import GrafiaLoweredArtifact, GrafiaRegionModule


class GrafiaBackendPlugin:
    """Mode-aware Grafia plugin for planner-facing backend hooks."""

    backend_name = BACKEND_NAME

    def __init__(
        self,
        compiler_kwargs: dict[str, Any] | None = None,
        *,
        backend_context: GrafiaBackendContext | None = None,
        compile_artifacts: bool = True,
    ) -> None:
        register_builtin_lowerings()
        self.compiler_kwargs = dict(compiler_kwargs or {})
        self.backend_context = backend_context
        self.compile_artifacts = compile_artifacts

    def mode_contexts(self, program: ProgramData):
        return [
            ModeContext(
                name="rmsnorm",
                phase="rmsnorm",
                shape_buckets=(),
                cache_abi=None,
                runtime_facts={"modes": GRAFIA_MODES},
            )
        ]

    def classify_node(
        self,
        node: Node,
        mode: ModeContext,
        program: ProgramData,
        args: OpArgumentResolver,
    ) -> SupportDecision:
        if rmsnorm.is_source_node(node):
            return rmsnorm.classify_node(node, mode, program, args)
        return SupportDecision.eager_only(
            f"unsupported Grafia op remains eager: op={node.op}, target={node.target}"
        )

    def can_extend_region(
        self,
        region: RegionSpec,
        candidate: Node,
        mode: ModeContext,
        program: ProgramData,
    ) -> bool:
        return False

    def lower_region(
        self,
        region: RegionSpec,
        program: ProgramData,
        args: OpArgumentResolver,
    ) -> GrafiaLoweredArtifact:
        if len(region.source_nodes) != 1:
            raise GrafiaUnsupportedError(
                f"{region.region_id}: Grafia V1 supports only single-node regions, "
                f"got {region.source_node_names}"
            )
        context = self._context(program)
        adapter = GrafiaCTMBackendAdapter(context)
        return lower_region(
            program,
            region,
            adapter,
            args,
            lowerings=LOWERINGS,
        )

    def make_region_module(
        self,
        region: RegionSpec,
        artifact: GrafiaLoweredArtifact,
    ):
        return GrafiaRegionModule(artifact)

    def _context(self, program: ProgramData) -> GrafiaBackendContext:
        if self.backend_context is None:
            self.backend_context = GrafiaBackendContext(
                program,
                self.compiler_kwargs,
                compile_artifacts=self.compile_artifacts,
            )
        return self.backend_context
