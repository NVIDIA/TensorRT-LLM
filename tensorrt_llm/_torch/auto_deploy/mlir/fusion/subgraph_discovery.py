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

import logging
from dataclasses import dataclass, field
from typing import List

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue

from ..dialect import (
    AdAdd,
    AdCast,
    AdEq,
    AdExp,
    AdFloorDiv,
    AdGelu,
    AdMul,
    AdNeg,
    AdPow,
    AdReduceMean,
    AdReduceSum,
    AdRelu,
    AdRsqrt,
    AdSigmoid,
    AdSilu,
    AdSoftplus,
    AdSplat,
    AdSqrt,
    AdSub,
    AdTanh,
)

logger = logging.getLogger(__name__)

# PyTorch's torch.library.custom_op imposes a hard limit on the number of
# arguments in a function schema (currently 64). Fused subgraphs whose external
# input count exceeds this limit must be split into smaller partitions.
MAX_CUSTOM_OP_INPUTS = 64

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
        AdSigmoid,
        AdExp,
        AdSoftplus,
        AdReduceSum,
        AdReduceMean,
        AdCast,
        AdFloorDiv,
        AdEq,
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

    def refresh_inputs(self) -> None:
        """Recompute external inputs from the current state of ops' operands.

        When earlier subgraphs are replaced with fused ops (``ad.opaque``),
        ``SSAValue.replace_by()`` redirects downstream operands to the new
        fused outputs.  This invalidates the ``inputs`` list computed at
        discovery time.  Call this before kernel generation to pick up the
        redirected operands.
        """
        op_set = set(self.ops)
        self.inputs.clear()
        seen: set = set()
        for op in self.ops:
            for operand in op.operands:
                if operand.owner not in op_set and id(operand) not in seen:
                    self.inputs.append(operand)
                    seen.add(id(operand))


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
    # Re-sort ops in topological order within the subgraph. We use the block's
    # op iteration order as the canonical topological ordering; every subgraph
    # op must appear in the block.
    topo_order = {op: i for i, op in enumerate(block.ops)}
    result = []
    for gid, ops in groups.items():
        if len(ops) < 2:
            continue
        ops = _topo_sort_subgraph(ops, topo_order)
        sg = _build_subgraph(ops)

        if len(sg.inputs) > MAX_CUSTOM_OP_INPUTS or _has_placement_conflict(sg, topo_order):
            partitions = _split_subgraph(sg, topo_order)
            result.extend(partitions)
        else:
            result.append(sg)
    return result


def _has_placement_conflict(sg: FusibleSubgraph, topo_order: dict[Operation, int]) -> bool:
    """Return True if the subgraph cannot be replaced by a single fused op.

    A fused op must be placed after all its input producers and before all its
    output consumers.  If the latest input producer appears at or after the
    earliest output consumer in the block, no valid insertion point exists.
    """
    op_set = set(sg.ops)

    max_input_pos = -1
    for inp in sg.inputs:
        producer = inp.owner
        if producer in topo_order:
            max_input_pos = max(max_input_pos, topo_order[producer])

    min_consumer_pos = len(topo_order) + 1
    for out in sg.outputs:
        for use in out.uses:
            consumer = use.operation
            if consumer not in op_set and consumer in topo_order:
                min_consumer_pos = min(min_consumer_pos, topo_order[consumer])

    return max_input_pos >= min_consumer_pos


def _topo_sort_subgraph(ops: List[Operation], topo_order: dict[Operation, int]) -> List[Operation]:
    """Topologically sort ops within a subgraph using dependency edges.

    Uses Kahn's algorithm: start with ops whose in-subgraph dependencies are
    all satisfied, then repeatedly emit ops whose predecessors are done.
    Ties are broken by the block-level topological order to produce a
    deterministic, stable result.
    """
    op_set = set(ops)

    # Build in-degree counts (only counting edges from ops within the subgraph)
    in_degree: dict[Operation, int] = {op: 0 for op in ops}
    # Map: producer op -> list of consumer ops (within the subgraph)
    dependents: dict[Operation, list[Operation]] = {op: [] for op in ops}

    for op in ops:
        seen_producers: set[Operation] = set()
        for operand in op.operands:
            producer = operand.owner
            if producer in op_set and producer is not op and producer not in seen_producers:
                in_degree[op] += 1
                dependents[producer].append(op)
                seen_producers.add(producer)

    # Kahn's algorithm with topo_order tie-breaking
    import heapq

    # Use (block_position, op) as heap entries; compare only on block_position
    ready = []
    for op in ops:
        if in_degree[op] == 0:
            heapq.heappush(ready, (topo_order.get(op, 0), id(op), op))

    sorted_ops: List[Operation] = []
    while ready:
        _, _, op = heapq.heappop(ready)
        sorted_ops.append(op)
        for dep in dependents[op]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                heapq.heappush(ready, (topo_order.get(dep, 0), id(dep), dep))

    if len(sorted_ops) != len(ops):
        logger.warning(
            "Topological sort incomplete: %d/%d ops emitted (cycle?), falling back to block order",
            len(sorted_ops),
            len(ops),
        )
        ops.sort(key=lambda o: topo_order.get(o, 0))
        return ops

    return sorted_ops


def _build_subgraph(ops: List[Operation]) -> FusibleSubgraph:
    """Build a FusibleSubgraph from a topologically sorted list of ops."""
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
    return sg


def _split_subgraph(sg: FusibleSubgraph, topo_order: dict[Operation, int]) -> List[FusibleSubgraph]:
    """Split a subgraph that exceeds MAX_CUSTOM_OP_INPUTS into partitions.

    Walks ops in topological order, greedily assigning each op to the current
    partition.  A new partition is started when adding the next op would either:

    1. Push the partition's external input count past ``MAX_CUSTOM_OP_INPUTS``.
    2. Create an invalid placement: the fused op must be inserted *after* all
       its input producers and *before* all its output consumers in the block.
       If a new input's block position is at or after the earliest consumer of
       a current output, placement is impossible and we must split.
    """
    sg_op_set = set(sg.ops)

    def _earliest_consumer_pos(op: Operation) -> int:
        """Return the earliest block position of any external consumer of op's results."""
        earliest = len(topo_order) + 1  # sentinel: after all ops
        for res in op.results:
            for use in res.uses:
                consumer = use.operation
                if consumer not in sg_op_set and consumer in topo_order:
                    earliest = min(earliest, topo_order[consumer])
        return earliest

    partitions: List[List[Operation]] = []
    current_ops: List[Operation] = []
    current_produced: set = set()  # ids of values produced by current_ops
    current_external_inputs: set = set()  # ids of external inputs
    # Track the latest input producer position and earliest consumer position
    # in the MLIR block to ensure valid fused op placement.
    current_max_input_pos = -1
    current_min_consumer_pos = len(topo_order) + 1

    def _flush():
        nonlocal current_ops, current_produced, current_external_inputs
        nonlocal current_max_input_pos, current_min_consumer_pos
        if current_ops:
            partitions.append(current_ops)
        current_ops = []
        current_produced = set()
        current_external_inputs = set()
        current_max_input_pos = -1
        current_min_consumer_pos = len(topo_order) + 1

    for op in sg.ops:
        # Compute new external inputs and the max input position they'd add
        new_external = set()
        new_max_input_pos = current_max_input_pos
        for operand in op.operands:
            oid = id(operand)
            if oid not in current_produced and oid not in current_external_inputs:
                new_external.add(oid)
                producer = operand.owner
                if producer in topo_order:
                    new_max_input_pos = max(new_max_input_pos, topo_order[producer])

        # Compute the earliest consumer position this op's outputs would add
        new_min_consumer_pos = min(current_min_consumer_pos, _earliest_consumer_pos(op))

        # Check both constraints
        input_overflow = (
            current_ops and len(current_external_inputs) + len(new_external) > MAX_CUSTOM_OP_INPUTS
        )
        placement_invalid = current_ops and new_max_input_pos >= new_min_consumer_pos

        if input_overflow or placement_invalid:
            _flush()
            # Recompute for this op alone
            new_external = set()
            new_max_input_pos = -1
            for operand in op.operands:
                new_external.add(id(operand))
                producer = operand.owner
                if producer in topo_order:
                    new_max_input_pos = max(new_max_input_pos, topo_order[producer])
            new_min_consumer_pos = _earliest_consumer_pos(op)

        current_ops.append(op)
        current_external_inputs.update(new_external)
        current_max_input_pos = new_max_input_pos
        current_min_consumer_pos = new_min_consumer_pos
        for r in op.results:
            current_produced.add(id(r))

    _flush()

    # Build FusibleSubgraph for each partition (only keep partitions with >=2 ops)
    result = []
    for part_ops in partitions:
        if len(part_ops) >= 2:
            result.append(_build_subgraph(part_ops))
    logger.info(
        "Split subgraph with %d ops and %d inputs into %d partitions (%d fusible)",
        len(sg.ops),
        len(sg.inputs),
        len(partitions),
        len(result),
    )
    return result
