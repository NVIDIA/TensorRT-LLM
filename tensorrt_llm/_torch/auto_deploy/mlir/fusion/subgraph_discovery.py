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

        # Check if any operand comes from an op already in a group.
        # If operands come from *multiple* groups, merge them all.
        producer_groups: set[int] = set()
        for operand in op.operands:
            producer = operand.owner
            if producer in op_to_group:
                producer_groups.add(op_to_group[producer])

        if producer_groups:
            # Pick the lowest-numbered group as the merge target
            merged_group = min(producer_groups)
            # Merge all other groups into the target
            for other_gid in producer_groups:
                if other_gid != merged_group:
                    for other_op in groups[other_gid]:
                        op_to_group[other_op] = merged_group
                    groups[merged_group].extend(groups.pop(other_gid))
            op_to_group[op] = merged_group
            groups[merged_group].append(op)
        else:
            op_to_group[op] = next_id
            groups[next_id] = [op]
            next_id += 1

    # Second pass: pull in zero-operand fusible ops (e.g., AdSplat constants)
    # whose outputs are consumed exclusively by ops in a single larger group.
    for op in block.ops:
        if not _is_fusible(op) or len(op.operands) > 0:
            continue
        # Skip if already in a multi-op group
        current_group = op_to_group.get(op)
        if current_group is not None and len(groups[current_group]) > 1:
            continue
        # Check if all users of this op's results are in a single group
        target_group = None
        all_in_one_group = True
        for res in op.results:
            for use in res.uses:
                consumer = use.operation
                if consumer not in op_to_group:
                    all_in_one_group = False
                    break
                g = op_to_group[consumer]
                if target_group is None:
                    target_group = g
                elif g != target_group:
                    all_in_one_group = False
                    break
            if not all_in_one_group:
                break
        if all_in_one_group and target_group is not None and target_group != current_group:
            # Remove from old single-op group if applicable
            if current_group is not None:
                groups[current_group].remove(op)
            op_to_group[op] = target_group
            # Insert at the position before its first consumer in the group
            group_ops = groups[target_group]
            insert_idx = len(group_ops)
            for i, group_op in enumerate(group_ops):
                for operand in group_op.operands:
                    if operand.owner is op:
                        insert_idx = min(insert_idx, i)
            group_ops.insert(insert_idx, op)

    # Build FusibleSubgraph objects for groups with >=2 ops.
    # Re-sort ops in topological order (block iteration order) since group
    # merging may have disrupted ordering.
    topo_order = {op: i for i, op in enumerate(block.ops)}
    result = []
    for gid, ops in groups.items():
        if len(ops) < 2:
            continue
        ops.sort(key=lambda o: topo_order.get(o, 0))
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
