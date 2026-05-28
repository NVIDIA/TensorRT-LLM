# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Utilities for piecewise CUDA graph: dynamic op registry, classification, and graph splitting.

This module provides the logic to:
1. Identify dynamic (uncapturable) custom ops in the FX graph.
2. Classify dynamic submodules by their piecewise output handling policy.
3. Split the FX GraphModule at dynamic op boundaries using torch.fx.passes.split_module.
4. Return the split GraphModule and metadata about which submodules are dynamic vs static.
"""

import operator
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch.nn as nn
from torch.fx import GraphModule, Node
from torch.fx.passes.split_module import split_module

from ..utils.logger import ad_logger
from ..utils.node_utils import (
    DynamicOpPolicy,
    get_all_piecewise_dynamic_op_names,
    get_piecewise_dynamic_op_policy,
)
from ..utils.node_utils import piecewise_dynamic_op as piecewise_dynamic_op
from ..utils.node_utils import register_piecewise_dynamic_op as register_piecewise_dynamic_op

# ---------------------------------------------------------------------------
# Dynamic ops are registered from their custom op definition modules with
# ``piecewise_dynamic_op``. Each registered op is a split boundary for piecewise
# CUDA graph. The policy describes the output contract of the dynamic submodule
# after it has been split out of the captured static graph.
# ---------------------------------------------------------------------------


def _get_all_dynamic_op_names() -> Set[str]:
    """Return the full set of dynamic op qualified names."""
    return get_all_piecewise_dynamic_op_names()


def _get_dynamic_op_policy(node: Node) -> Optional[DynamicOpPolicy]:
    """Return the piecewise policy for a dynamic custom op node, if any."""
    if node.op != "call_function":
        return None

    return get_piecewise_dynamic_op_policy(node.target)


def is_dynamic_cached_op(node: Node) -> bool:
    """Check if a node is a dynamic op that should split piecewise partitions."""
    return _get_dynamic_op_policy(node) is not None


# ---------------------------------------------------------------------------
# Partition classification helpers for submodules produced by piecewise splitting.
# ---------------------------------------------------------------------------

# Trivial FX ops that are metadata-only or typically no-ops — used to identify
# static partitions with no meaningful GPU compute (e.g., between adjacent dynamic ops).
# NOTE: reshape, contiguous, and to *can* launch kernels in edge cases (non-contiguous
# tensors, dtype/device casts), but in practice these appear only as lightweight
# plumbing in empty partitions.
_TRIVIAL_CALL_FUNCTIONS = {operator.getitem, getattr}
_TRIVIAL_CALL_METHODS = {
    "view",
    "reshape",
    "contiguous",
    "permute",
    "transpose",
    "unsqueeze",
    "squeeze",
    "expand",
    "size",
    "dim",
    "to",
}


def submod_has_cuda_ops(submod: nn.Module) -> bool:
    """Check if a submodule has ops beyond trivial metadata/reshape operations."""
    if not isinstance(submod, GraphModule):
        return True  # Conservative: non-FX modules assumed to have GPU ops

    for node in submod.graph.nodes:
        if node.op == "call_module":
            return True
        if node.op == "call_function":
            if node.target in _TRIVIAL_CALL_FUNCTIONS:
                continue
            return True
        if node.op == "call_method":
            if node.target in _TRIVIAL_CALL_METHODS:
                continue
            return True

    return False


def needs_out_buffer(submod: nn.Module) -> bool:
    """Return True if this dynamic submodule produces a NEW output tensor.

    Inplace ops (mutate input, return None) don't produce new tensors.
    Metadata prep ops are handled by MetadataWrapper (stable output addresses).
    All of these are skipped -- only OUT_BUFFER policy ops need out= buffers.
    """
    if not isinstance(submod, GraphModule):
        return True

    has_dynamic_op = False
    for node in submod.graph.nodes:
        policy = _get_dynamic_op_policy(node)
        if policy is None:
            continue
        has_dynamic_op = True
        if policy == DynamicOpPolicy.OUT_BUFFER:
            return True
    return not has_dynamic_op


def needs_metadata_wrapper(submod: nn.Module) -> bool:
    """Return True if metadata-prep outputs should be stabilized for captured runners."""
    if not isinstance(submod, GraphModule):
        return False

    for node in submod.graph.nodes:
        if _get_dynamic_op_policy(node) == DynamicOpPolicy.METADATA_WRAPPER:
            return True
    return False


# ---------------------------------------------------------------------------
# Graph splitting
# ---------------------------------------------------------------------------


@dataclass
class SplitInfo:
    """Metadata about a split GraphModule."""

    # The split GraphModule with submod_0, submod_1, ... submodules
    split_gm: GraphModule
    # Total number of submodules
    num_submodules: int
    # Indices of dynamic (uncapturable) submodules — these run eagerly
    dynamic_submod_indices: List[int] = field(default_factory=list)
    # Indices of static (capturable) submodules — these get CUDA graph captured
    static_submod_indices: List[int] = field(default_factory=list)


def split_graph_at_dynamic_ops(gm: GraphModule) -> SplitInfo:
    """Split an FX GraphModule at dynamic op boundaries.

    Each dynamic op (attention, SSM, conv, delta) becomes its own submodule.
    Static regions between dynamic ops are grouped into separate submodules.

    The split produces submodules named `submod_0`, `submod_1`, etc.
    Dynamic submodules contain exactly one dynamic op.
    Static submodules contain everything else (norms, linears, MLPs, etc.).

    Args:
        gm: The FX GraphModule to split.

    Returns:
        SplitInfo with the split GraphModule and metadata.
    """
    # Assign partition IDs: each dynamic op gets its own partition,
    # static ops between dynamic ops share a partition.
    partition_counter = [0]  # mutable counter
    node_to_partition: Dict[Node, int] = {}
    dynamic_partitions: Set[int] = set()

    # First pass: identify dynamic nodes and assign them unique partitions
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue

        if is_dynamic_cached_op(node):
            # Dynamic op gets its own partition
            partition_counter[0] += 1
            node_to_partition[node] = partition_counter[0]
            dynamic_partitions.add(partition_counter[0])
            # Next static region gets a new partition
            partition_counter[0] += 1
        else:
            # Static op joins the current static partition
            node_to_partition[node] = partition_counter[0]

    if not dynamic_partitions:
        ad_logger.info("No dynamic ops found in graph — no splitting needed.")
        return SplitInfo(
            split_gm=gm,
            num_submodules=1,
            dynamic_submod_indices=[],
            static_submod_indices=[0],
        )

    # Use torch.fx split_module to perform the actual split
    def partition_fn(node: Node) -> int:
        return node_to_partition.get(node, 0)

    split_gm = split_module(
        gm,
        gm,  # root_module
        partition_fn,
        keep_original_order=True,
    )

    # Analyze the split result to identify dynamic vs static submodules.
    # split_module names submodules "submod_{partition_id}", so the name suffix
    # IS the partition ID.  We classify each submodule directly by checking its
    # partition ID against dynamic_partitions.  This avoids fragile alignment
    # between partition_ids_in_order and submod_names (they can diverge when
    # get_attr-only partitions exist in the original graph but split_module
    # does not create submodules for them).
    submod_names = []
    for name, _ in split_gm.named_children():
        if name.startswith("submod_"):
            submod_names.append(name)

    submod_names.sort(key=lambda n: int(n.split("_")[1]))

    dynamic_indices = []
    static_indices = []
    for name in submod_names:
        pid = int(name.split("_")[1])
        if pid in dynamic_partitions:
            dynamic_indices.append(pid)
        else:
            static_indices.append(pid)

    ad_logger.info(
        f"Piecewise split: {len(submod_names)} submodules "
        f"({len(static_indices)} static, {len(dynamic_indices)} dynamic)"
    )

    return SplitInfo(
        split_gm=split_gm,
        num_submodules=len(submod_names),
        dynamic_submod_indices=dynamic_indices,
        static_submod_indices=static_indices,
    )
