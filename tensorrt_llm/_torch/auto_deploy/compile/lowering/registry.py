# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch.fx import Node

from .context import LoweringContext

LoweringRule = Callable[[LoweringContext, Node], Any]

LOWERINGS: dict[Any, LoweringRule] = {}


def _source_op_variants(source_op: Any) -> tuple[Any, ...]:
    variants = [source_op]
    overload_packet = getattr(source_op, "overloadpacket", None)
    if overload_packet is not None:
        variants.append(overload_packet)
    return tuple(variants)


def _get_mapping_lowering(
    lowerings: Mapping[Any, LoweringRule],
    source_op: Any,
) -> LoweringRule | None:
    for source_op_key in _source_op_variants(source_op):
        fn = lowerings.get(source_op_key)
        if fn is not None:
            return fn
    return None
