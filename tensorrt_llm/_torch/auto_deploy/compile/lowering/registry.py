# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.fx import Node

    from .context import LoweringContext

LoweringFn = Callable[..., Any]

LOWERINGS: dict[Any, LoweringFn] = {}


def _source_op_variants(source_op: Any) -> tuple[Any, ...]:
    variants = [source_op]
    overload_packet = getattr(source_op, "overloadpacket", None)
    if overload_packet is not None:
        variants.append(overload_packet)
    return tuple(variants)


def _get_mapping_lowering(
    lowerings: Mapping[Any, LoweringFn],
    source_op: Any,
) -> LoweringFn | None:
    for source_op_key in _source_op_variants(source_op):
        fn = lowerings.get(source_op_key)
        if fn is not None:
            return fn
    return None


def lower_rms_norm(ctx: LoweringContext, node: Node) -> Any:
    x, weight, eps = ctx.args.get(node, "input", "weight", "eps")
    return ctx.adapter.emit_rms_norm(
        x=ctx.resolve(x),
        weight=ctx.resolve(weight),
        eps=float(eps),
        result_meta=ctx.result_type(node),
        loc=ctx.loc(node),
    )


def register_builtin_lowerings() -> None:
    from ...custom_ops.normalization import rms_norm as _rms_norm_ops  # noqa: F401

    LOWERINGS.setdefault(torch.ops.auto_deploy.torch_rmsnorm.default, lower_rms_norm)
