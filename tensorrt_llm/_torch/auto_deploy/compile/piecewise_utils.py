"""Utilities for piecewise CUDA graph: dynamic op registry, classification, and graph splitting.

This module provides the logic to:
1. Identify dynamic (uncapturable) custom ops in the FX graph (attention, SSM, conv, delta).
2. Classify dynamic submodules by behaviour (inplace, metadata-prep, needs out= buffer).
3. Split the FX GraphModule at dynamic op boundaries using torch.fx.passes.split_module.
4. Return the split GraphModule and metadata about which submodules are dynamic vs static.
"""

import operator
from dataclasses import dataclass, field
from typing import Dict, List, Set

import torch.nn as nn
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
    "auto_deploy::trtllm_attention_mha_with_cache",
    # MLA attention variants
    "auto_deploy::flashinfer_mla_with_cache",
    "auto_deploy::torch_cached_mla_with_cache",
    "auto_deploy::trtllm_mla_with_cache",
    "auto_deploy::trtllm_mla_fused_rope_with_cache",
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
    "auto_deploy::fla_cached_gated_delta_rule",
]

# Metadata preparation ops (branch on batch_info_host, do CPU math on CUDA tensors)
_METADATA_PREP_OPS = [
    "auto_deploy::flashinfer_attention_prepare_metadata",
    "auto_deploy::flashinfer_mla_prepare_metadata",
    "auto_deploy::trtllm_mla_prepare_metadata",
    "auto_deploy::mamba_ssm_prepare_metadata",
]

# Logits gather ops (CPU branching on host tensor + shape-dependent logic)
_LOGITS_GATHER_OPS = [
    "auto_deploy::gather_tokens",
]

# Persistent-buffer dynamic ops: these ops produce outputs in pre-allocated
# persistent buffers with stable addresses.  They must run eagerly (cannot be
# captured in CUDA graphs due to CPU control flow / dynamic kernel grids),
# but they do NOT need MetadataWrapper (addresses are already stable) or
# DynamicOpWrapper (no fresh allocation, returns persistent buffer directly).
_PERSISTENT_BUFFER_OPS = [
    "auto_deploy::trtllm_attention_prepare_metadata",
]

# Inplace dynamic ops: these ops mutate their input tensor and return None,
# so they do NOT produce a new output tensor and do NOT need an out= buffer.
# This is a semantic property separate from _CACHED_CONV_OPS (which classifies
# ops as dynamic/uncapturable).  Keep this list in sync when adding new inplace ops.
_INPLACE_DYNAMIC_OPS = [
    "auto_deploy::triton_cached_causal_conv1d",
    "auto_deploy::cuda_cached_causal_conv1d",
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
        + _PERSISTENT_BUFFER_OPS
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
    # Check with namespace::name format AND base name (for wrapper functions
    for dyn_op in dynamic_ops:
        if dyn_op in op_name:
            return True
        # Also check by base op name without namespace prefix
        base_name = dyn_op.split("::")[-1] if "::" in dyn_op else dyn_op
        if base_name in op_name:
            return True

    return False


# ---------------------------------------------------------------------------
# Partition classification: these constants and functions classify submodules
# produced by piecewise splitting (trivial vs non-trivial, inplace, metadata-prep,
# needs out= buffer).
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


_SKIP_OUT_DYNAMIC_OPS: Set[str] = (
    set(_INPLACE_DYNAMIC_OPS)
    | set(_METADATA_PREP_OPS)
    | set(_LOGITS_GATHER_OPS)
    | set(_PERSISTENT_BUFFER_OPS)
)


def needs_out_buffer(submod: nn.Module) -> bool:
    """Return True if this dynamic submodule produces a NEW output tensor.

    Inplace ops (mutate input, return None) don't produce new tensors.
    Metadata prep ops are handled by MetadataWrapper (stable output addresses).
    Both are skipped — only attention/SSM/delta/logits ops need out= buffers.
    """
    if not isinstance(submod, GraphModule):
        return True

    for node in submod.graph.nodes:
        if node.op == "call_function" and is_dynamic_cached_op(node):
            op_name = node.target.name() if hasattr(node.target, "name") else str(node.target)
            for skip_op in _SKIP_OUT_DYNAMIC_OPS:
                if skip_op in op_name:
                    return False
                base_name = skip_op.split("::")[-1] if "::" in skip_op else skip_op
                if base_name in op_name:
                    return False
    return True


def is_metadata_prep(submod: nn.Module) -> bool:
    """Return True if this dynamic submodule contains a metadata-prep op."""
    if not isinstance(submod, GraphModule):
        return False

    for node in submod.graph.nodes:
        if node.op == "call_function" and is_dynamic_cached_op(node):
            op_name = node.target.name() if hasattr(node.target, "name") else str(node.target)
            for prep_op in _METADATA_PREP_OPS:
                if prep_op in op_name:
                    return True
                base_name = prep_op.split("::")[-1] if "::" in prep_op else prep_op
                if base_name in op_name:
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
