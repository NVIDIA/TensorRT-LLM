# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared interfaces for Grafia op lowerings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.fx import Node

from ....lowering import (
    LoweringContext,
    ModeContext,
    OpArgumentResolver,
    ProgramData,
    SupportDecision,
)


class GrafiaOpLowering(ABC):
    """Backend op lowering with classification and lowering in one unit."""

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
        """Return whether this Grafia lowering supports the FX node."""

    @abstractmethod
    def lower(self, ctx: LoweringContext, node: Node) -> Any:
        """Emit backend IR for this FX node."""


def build_op_lowering_map(*op_lowerings: GrafiaOpLowering) -> dict[Any, GrafiaOpLowering]:
    lowerings_by_target: dict[Any, GrafiaOpLowering] = {}
    for op_lowering in op_lowerings:
        if not op_lowering.source_ops:
            raise ValueError(f"{type(op_lowering).__name__} must define at least one source op")
        for source_op in op_lowering.source_ops:
            for key in _source_op_keys(source_op):
                existing = lowerings_by_target.get(key)
                if existing is not None and existing is not op_lowering:
                    raise ValueError(
                        f"duplicate Grafia lowering for source op {key!r}: "
                        f"{type(existing).__name__} and {type(op_lowering).__name__}"
                    )
                lowerings_by_target[key] = op_lowering
    return lowerings_by_target


def _source_op_keys(source_op: Any) -> tuple[Any, ...]:
    overload_packet = getattr(source_op, "overloadpacket", None)
    if overload_packet is None:
        return (source_op,)
    return (source_op, overload_packet)
