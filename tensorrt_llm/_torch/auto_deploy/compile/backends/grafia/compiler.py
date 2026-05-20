# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiler entry point for the Grafia lowering backend."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch.fx import GraphModule

from ...compiler import CompilerBackend
from ...lowering import FirstPlanDispatchPolicy, MultiModePlanner, ProgramData
from .errors import GrafiaUnsupportedError
from .plugin import GrafiaBackendPlugin
from .runtime import GrafiaEagerDispatchPolicy, GrafiaExecutor


class GrafiaCompiler(CompilerBackend):
    """AutoDeploy-native Grafia backend using the generic lowering planner."""

    def __init__(self, model: nn.Module, **compiler_kwargs):
        super().__init__(model, **compiler_kwargs)
        self.compiler_kwargs = dict(compiler_kwargs)

    def compile(self) -> nn.Module:
        if not isinstance(self.model, GraphModule):
            raise GrafiaUnsupportedError(
                "compile_backend='grafia' requires an AutoDeploy FX GraphModule; "
                f"got {type(self.model).__name__}"
            )

        program = ProgramData(self.model)
        plugin = GrafiaBackendPlugin(self.compiler_kwargs)
        planner = MultiModePlanner(plugin)
        try:
            plans = planner.plan(program)
        except RuntimeError as exc:
            if "Backend planning failed" in str(exc):
                raise GrafiaUnsupportedError(str(exc)) from exc
            raise

        if not self._dispatch_policy_requested():
            selected = next((plan for plan in plans if plan.regions), None)
            if selected is None:
                return self.model
            self._attach_plan_metadata(selected.module, plans)
            selected.module.eval()
            return selected.module

        executor = GrafiaExecutor(
            eager_module=self.model,
            plans=plans,
            dispatch_policy=self._dispatch_policy(),
        )
        executor.eval()
        return executor

    def _dispatch_policy_requested(self) -> bool:
        return (
            "grafia_dispatch_policy" in self.compiler_kwargs
            or "dispatch_policy" in self.compiler_kwargs
        )

    def _dispatch_policy(self) -> Any:
        policy = self.compiler_kwargs.get(
            "grafia_dispatch_policy", self.compiler_kwargs.get("dispatch_policy")
        )
        if policy is None or policy == "eager":
            return GrafiaEagerDispatchPolicy()
        if policy == "first":
            return FirstPlanDispatchPolicy()
        return policy

    @staticmethod
    def _attach_plan_metadata(module: nn.Module, plans) -> None:
        module.grafia_plans = tuple(plans)
        module.grafia_region_modules = {
            region_id: region_module
            for plan in plans
            for region_id, region_module in plan.region_modules.items()
        }
        module.lowered_op_kinds = [
            op_kind
            for region_module in module.grafia_region_modules.values()
            for op_kind in getattr(region_module, "lowered_op_kinds", ())
        ]
