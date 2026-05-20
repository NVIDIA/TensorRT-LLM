# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict runtime wrappers and dispatch policy for Grafia regions."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils._pytree import tree_unflatten

from ....utils.logger import ad_logger
from ...lowering import CompiledPlan, PlanDispatcher, ProgramData, RegionSpec
from .errors import GrafiaCompileError
from .metadata import _ctm_dtype_to_torch


@dataclass(frozen=True)
class GrafiaLoweredArtifact:
    program: ProgramData
    region: RegionSpec
    spec: Any
    backend: Any | None
    artifact: Any | None
    input_tensors: tuple[Any, ...]
    output_tree_spec: Any
    op_kinds: tuple[str, ...]
    constant_names: tuple[str, ...]
    spec_cache_key: Any


class GrafiaRegionModule(nn.Module):
    """Strict PyTorch-facing wrapper for a selected Grafia CTM region."""

    def __init__(self, lowered: GrafiaLoweredArtifact) -> None:
        super().__init__()
        object.__setattr__(self, "program", lowered.program)
        self.region = lowered.region
        self.region_id = lowered.region.region_id
        self.mode_name = lowered.region.mode.name
        self.source_node_names = lowered.region.source_node_names
        self.ctm_backend = lowered.backend
        self.artifact = lowered.artifact
        self.input_tensors = list(lowered.input_tensors)
        self.input_names = [tensor.name for tensor in self.input_tensors]
        self.output_tree_spec = lowered.output_tree_spec
        self.lowered_op_kinds = list(lowered.op_kinds)
        self.constant_names = list(lowered.constant_names)
        self._signature = inspect.signature(lowered.program.graph_module.forward)

    def _flatten_inputs(self, *args, **kwargs) -> tuple[Any, ...]:
        if kwargs:
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                return tuple(bound.arguments[name] for name in self.input_names)
            except KeyError as exc:
                raise GrafiaCompileError(
                    f"{self.region_id}: missing runtime input {exc.args[0]!r} "
                    "for compile_backend='grafia'"
                ) from exc
        if len(args) != len(self.input_tensors):
            raise GrafiaCompileError(
                f"{self.region_id}: compile_backend='grafia' expected "
                f"{len(self.input_tensors)} runtime input(s) {self.input_names}, "
                f"got {len(args)}"
            )
        return tuple(args)

    def _validate_runtime_inputs(self, args: tuple[Any, ...]) -> None:
        expected_device = None
        for idx, (arg, spec) in enumerate(zip(args, self.input_tensors, strict=True)):
            if not isinstance(arg, torch.Tensor):
                raise GrafiaCompileError(
                    f"{self.region_id}: input {idx} must be a tensor, got {type(arg).__name__}"
                )
            if not arg.is_cuda:
                raise GrafiaCompileError(
                    f"{self.region_id}: input {idx} must be CUDA, got {arg.device}"
                )
            if expected_device is None:
                expected_device = arg.device
            elif arg.device != expected_device:
                raise GrafiaCompileError(
                    f"{self.region_id}: compile_backend='grafia' requires all "
                    f"runtime inputs on the same CUDA device; got {expected_device} "
                    f"and {arg.device}"
                )
            expected_shape = tuple(int(d) for d in spec.spec.shape)
            if tuple(arg.shape) != expected_shape:
                raise GrafiaCompileError(
                    f"{self.region_id}: input {idx} shape mismatch: "
                    f"expected {expected_shape}, got {tuple(arg.shape)}"
                )
            expected_dtype = _ctm_dtype_to_torch(spec.spec.dtype)
            if arg.dtype is not expected_dtype:
                raise GrafiaCompileError(
                    f"{self.region_id}: input {idx} dtype mismatch: "
                    f"expected {expected_dtype}, got {arg.dtype}"
                )
            if not arg.is_contiguous():
                raise GrafiaCompileError(f"{self.region_id}: input {idx} must be contiguous")

    def forward(self, *args, **kwargs):
        if self.ctm_backend is None or self.artifact is None:
            raise GrafiaCompileError(
                f"{self.region_id}: Grafia region has no compiled CTM artifact"
            )
        flat_args = self._flatten_inputs(*args, **kwargs)
        self._validate_runtime_inputs(flat_args)
        outputs = self.ctm_backend.launch(self.artifact, *flat_args)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        return tree_unflatten(list(outputs), self.output_tree_spec)


class GrafiaEagerDispatchPolicy:
    """Default outer policy: route eager while still exposing planned regions."""

    def __init__(self) -> None:
        self._logged = False

    def select(
        self,
        plans: Sequence[CompiledPlan],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        if not self._logged:
            if not plans:
                reason = "no compiled plans were produced"
            elif not any(plan.regions for plan in plans):
                reason = "all Grafia plans have zero backend regions"
            else:
                reason = "default eager route selected before Grafia backend execution"
            ad_logger.debug(
                "Grafia PlanDispatcher routing eager: "
                f"{reason}; plans={[plan.name for plan in plans]}"
            )
            self._logged = True
        return None


class GrafiaExecutor(PlanDispatcher):
    """Plan dispatcher with Grafia inspection conveniences."""

    def __init__(self, eager_module, plans=(), dispatch_policy=None) -> None:
        super().__init__(eager_module, plans, dispatch_policy)

    @property
    def grafia_region_modules(self) -> dict[str, GrafiaRegionModule]:
        modules: dict[str, GrafiaRegionModule] = {}
        for plan in self.plans:
            for region_id, module in plan.region_modules.items():
                if isinstance(module, GrafiaRegionModule):
                    modules[region_id] = module
        return modules

    @property
    def lowered_op_kinds(self) -> list[str]:
        op_kinds: list[str] = []
        for module in self.grafia_region_modules.values():
            op_kinds.extend(module.lowered_op_kinds)
        return op_kinds

    @property
    def input_names(self) -> list[str]:
        modules = list(self.grafia_region_modules.values())
        return modules[0].input_names if modules else []


GrafiaCompiledGraph = GrafiaRegionModule
