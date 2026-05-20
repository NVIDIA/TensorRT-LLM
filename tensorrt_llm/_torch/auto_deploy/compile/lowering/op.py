# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.fx import Node

from .arguments import OpArgumentResolver
from .context import LoweringContext
from .datamodel import ModeContext, ProgramData, SupportDecision


class BackendOpLowering(ABC):
    """Backend op lowering with classification and emission in one unit."""

    def __call__(self, ctx: LoweringContext, node: Node) -> Any:
        return self.lower(ctx, node)

    @property
    @abstractmethod
    def source_ops(self) -> tuple[Any, ...]:
        """FX targets accepted by this lowering."""

    @abstractmethod
    def classify_node(
        self,
        node: Node,
        mode: ModeContext,
        program: ProgramData,
        args: OpArgumentResolver,
    ) -> SupportDecision:
        """Return whether this backend lowering supports the FX node."""

    @abstractmethod
    def lower(self, ctx: LoweringContext, node: Node) -> Any:
        """Emit backend IR for this FX node."""


def build_op_lowering_map(*op_lowerings: BackendOpLowering) -> dict[Any, BackendOpLowering]:
    lowerings_by_target: dict[Any, BackendOpLowering] = {}
    for op_lowering in op_lowerings:
        if not op_lowering.source_ops:
            raise ValueError(f"{type(op_lowering).__name__} must define at least one source op")
        for source_op in op_lowering.source_ops:
            for key in _source_op_keys(source_op):
                existing = lowerings_by_target.get(key)
                if existing is not None and existing is not op_lowering:
                    raise ValueError(
                        f"duplicate lowering for source op {key!r}: "
                        f"{type(existing).__name__} and {type(op_lowering).__name__}"
                    )
                lowerings_by_target[key] = op_lowering
    return lowerings_by_target


def _source_op_keys(source_op: Any) -> tuple[Any, ...]:
    overload_packet = getattr(source_op, "overloadpacket", None)
    if overload_packet is None:
        return (source_op,)
    return (source_op, overload_packet)
