"""Contains transformations that add perf related (nvtx) annotations to the graph."""

import torch
from torch.fx import GraphModule

from ...custom_ops.nvtx import *  # noqa
from ...utils.logger import ad_logger


def add_nvtx_annotations(gm: GraphModule, module_name: str) -> GraphModule:
    """Add NVTX profiling ranges around linear operations in the graph.

    This transformation:
    1. Traverses the graph looking for linear operations
    2. For each linear operation, adds:
       - start_range before the operation
       - end_range after the operation
    3. Uses the node's name as the range identifier

    Args:
        gm: The graph module to transform

    Returns:
        The transformed graph module with NVTX annotations
    """
    ad_logger.info("Adding NVTX annotations around linear operations")
    ad_logger.debug("Before adding NVTX annotations: " + str(gm))

    graph = gm.graph
    regions = []

    # First pass: identify nodes to annotate based on module name
    first_node = None
    last_node = None

    # Find first and last nodes matching module_name
    for node in graph.nodes:
        if node.meta.get("nn_module_stack"):
            current_module = list(node.meta["nn_module_stack"].keys())[-1]
            if module_name in current_module:
                if first_node is None:
                    first_node = node
                last_node = node

    if first_node is not None:
        regions.append((first_node, last_node))

    # Second pass: add annotations
    for region in regions:
        # Create a unique name for this range based on the node's name
        start_node, end_node = region
        range_name = list(start_node.meta["nn_module_stack"].keys())[-1]

        # Add start_range before the node
        with graph.inserting_before(start_node):
            start_node = graph.call_function(
                torch.ops.nvtx_ops.start_range, args=(range_name,), kwargs={}
            )

        # Add end_range after the node
        with graph.inserting_after(end_node):
            end_node = graph.call_function(
                torch.ops.nvtx_ops.end_range, args=(range_name,), kwargs={}
            )

        # Preserve metadata
        start_node.meta = start_node.meta.copy()
        end_node.meta = end_node.meta.copy()
    # Clean up the graph
    gm.recompile()
    ad_logger.debug("After adding NVTX annotations: " + str(gm))

    return gm
