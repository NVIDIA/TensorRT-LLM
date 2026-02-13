"""Utilities for piecewise CUDA graph: graph splitting at dynamic op boundaries.

This module provides the logic to:
1. Identify dynamic (uncapturable) custom ops in the FX graph (attention, SSM, conv, delta).
2. Split the FX GraphModule at those boundaries using torch.fx.passes.split_module.
3. Return the split GraphModule and metadata about which submodules are dynamic vs static.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

from torch.fx import GraphModule, Node
from torch.fx.passes.split_module import split_module

from ..utils.logger import ad_logger

# ---------------------------------------------------------------------------
# Dynamic ops registry: these ops cannot be captured in CUDA graphs for
# mixed/prefill batches because they have data-dependent control flow or
# dynamic kernel configurations.
# ---------------------------------------------------------------------------

# Cached attention ops (grid depends on per-sequence lengths)
_CACHED_ATTENTION_OPS = [
    "auto_deploy::flashinfer_attention_mha_with_cache",
    "auto_deploy::triton_attention_flattened_mha_with_cache",
    "auto_deploy::torch_cached_attention_with_cache",
]

# Cached SSM ops (Python-level branching on batch_info_host)
_CACHED_SSM_OPS = [
    "auto_deploy::triton_cached_ssm",
    "auto_deploy::torch_cached_ssm",
    "auto_deploy::flashinfer_cached_ssm",
]

# Cached causal conv ops (branching on prefill vs decode)
_CACHED_CONV_OPS = [
    "auto_deploy::triton_cached_causal_conv1d",
    "auto_deploy::cuda_cached_causal_conv1d",
]

# Cached delta rule ops (branching on prefill vs decode)
_CACHED_DELTA_OPS = [
    "auto_deploy::fla_cached_delta_rule",
]

# Metadata preparation ops (branch on batch_info_host, do CPU math on CUDA tensors)
_METADATA_PREP_OPS = [
    "auto_deploy::flashinfer_attention_prepare_metadata",
    "auto_deploy::mamba_ssm_prepare_metadata",
]

# Logits gather ops (CPU branching on host tensor + shape-dependent logic)
_LOGITS_GATHER_OPS = [
    "auto_deploy::gather_logits_before_lm_head",
]


def _get_all_dynamic_op_names() -> Set[str]:
    """Return the full set of dynamic op qualified names."""
    return set(
        _CACHED_ATTENTION_OPS
        + _CACHED_SSM_OPS
        + _CACHED_CONV_OPS
        + _CACHED_DELTA_OPS
        + _METADATA_PREP_OPS
        + _LOGITS_GATHER_OPS
    )


def is_dynamic_cached_op(node: Node) -> bool:
    """Check if a node is a dynamic (uncapturable) cached op.

    These are ops that cannot be captured inside a CUDA graph for mixed/prefill
    batches due to data-dependent control flow or dynamic kernel grids.
    """
    if node.op != "call_function":
        return False

    target = node.target
    # Handle OpOverload: get the qualified name
    if hasattr(target, "name"):
        # torch._ops.OpOverload has .name() method
        op_name = target.name()
    elif hasattr(target, "__qualname__"):
        op_name = target.__qualname__
    else:
        op_name = str(target)

    # Strip the ".default" suffix if present for matching
    dynamic_ops = _get_all_dynamic_op_names()
    # Check with namespace::name format
    for dyn_op in dynamic_ops:
        if dyn_op in op_name:
            return True

    return False


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

    # Analyze the split result to identify dynamic vs static submodules
    submod_names = []
    for name, _ in split_gm.named_children():
        if name.startswith("submod_"):
            submod_names.append(name)

    # Sort by index
    submod_names.sort(key=lambda n: int(n.split("_")[1]))

    # Build a mapping from partition ID to submod index
    # The split_module assigns submod_N names in order of first-seen partition IDs
    partition_ids_in_order = []
    seen = set()
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        pid = node_to_partition.get(node, 0)
        if pid not in seen:
            seen.add(pid)
            partition_ids_in_order.append(pid)

    dynamic_indices = []
    static_indices = []
    for idx, pid in enumerate(partition_ids_in_order):
        if idx >= len(submod_names):
            break
        if pid in dynamic_partitions:
            dynamic_indices.append(idx)
        else:
            static_indices.append(idx)

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
