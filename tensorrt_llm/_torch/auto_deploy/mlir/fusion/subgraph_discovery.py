# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Greedy discovery of maximal fusible subgraphs."""

from dataclasses import dataclass, field
from typing import List

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue

from ..dialect import (
    AdAdd,
    AdCast,
    AdGelu,
    AdMul,
    AdNeg,
    AdPow,
    AdReduceMean,
    AdReduceSum,
    AdRelu,
    AdRsqrt,
    AdSilu,
    AdSplat,
    AdSqrt,
    AdSub,
    AdTanh,
)

FUSIBLE_OPS = frozenset(
    {
        AdAdd,
        AdMul,
        AdSub,
        AdNeg,
        AdPow,
        AdRsqrt,
        AdSqrt,
        AdSilu,
        AdGelu,
        AdRelu,
        AdTanh,
        AdReduceSum,
        AdReduceMean,
        AdCast,
        AdSplat,
    }
)


@dataclass
class FusibleSubgraph:
    """A connected subgraph of fusible primitive ops.

    Attributes:
        ops: Operations in topological order.
        inputs: SSA values produced outside the subgraph and consumed inside.
        outputs: SSA values produced inside the subgraph and consumed outside.
    """

    ops: List[Operation] = field(default_factory=list)
    inputs: List[SSAValue] = field(default_factory=list)
    outputs: List[SSAValue] = field(default_factory=list)


def _is_fusible(op: Operation) -> bool:
    """Return True if *op* is a fusible primitive."""
    return type(op) in FUSIBLE_OPS


def discover_fusible_subgraphs(mlir_module: ModuleOp) -> List[FusibleSubgraph]:
    """Find maximal connected subgraphs of fusible primitives.

    Algorithm: Walk ops in topological order, merging fusible ops into groups
    when connected by data flow. Returns subgraphs with >=2 ops only.
    """
    block = mlir_module.body.block

    op_to_group: dict[Operation, int] = {}
    groups: dict[int, list[Operation]] = {}
    next_id = 0

    for op in block.ops:
        if not _is_fusible(op):
            continue

        # Check if any operand comes from an op already in a group
        merged_group = None
        for operand in op.operands:
            producer = operand.owner
            if producer in op_to_group:
                merged_group = op_to_group[producer]
                break

        if merged_group is not None:
            op_to_group[op] = merged_group
            groups[merged_group].append(op)
        else:
            op_to_group[op] = next_id
            groups[next_id] = [op]
            next_id += 1

    # Build FusibleSubgraph objects for groups with >=2 ops
    result = []
    for gid, ops in groups.items():
        if len(ops) < 2:
            continue
        sg = FusibleSubgraph(ops=ops)

        op_set = set(ops)
        seen_inputs: set = set()
        for op in ops:
            for operand in op.operands:
                if operand.owner not in op_set and id(operand) not in seen_inputs:
                    sg.inputs.append(operand)
                    seen_inputs.add(id(operand))

        for op in ops:
            for res in op.results:
                for use in res.uses:
                    if use.operation not in op_set:
                        sg.outputs.append(res)
                        break

        result.append(sg)
    return result
