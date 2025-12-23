"""
Utility to apply NCCL_SYMMETRIC patterns directly to models without full torch.compile.

This module reuses the existing Backend infrastructure instead of reinventing it.
"""

from typing import List

import torch
import torch.nn as nn
from torch.fx import GraphModule

from tensorrt_llm import logger
from tensorrt_llm.mapping import Mapping

from .backend import Backend


def apply_nccl_symmetric_patterns_to_model(model: nn.Module, mapping: Mapping) -> nn.Module:
    """
    Apply NCCL_SYMMETRIC patterns to a model using the existing Backend infrastructure.

    This reuses the existing Backend class and its pattern matching infrastructure,
    avoiding code duplication. The Backend already includes NCCL_SYMMETRIC patterns
    via register_ar_fusions() -> register_nccl_symmetric_patterns().

    Args:
        model: The PyTorch model to optimize
        mapping: The mapping configuration (needed for tp_group)

    Returns:
        The model with patterns applied (wrapped by torch.compile for tracing only)
    """

    # Reuse existing Backend infrastructure - it already has NCCL_SYMMETRIC patterns
    # registered via register_ar_fusions() -> register_nccl_symmetric_patterns()
    # We create a minimal backend that only applies patterns (no inductor, no CUDA graphs)
    class PatternOnlyBackend(Backend):
        """Backend that only applies patterns, reusing existing Backend infrastructure."""

        def __init__(self, mapping: Mapping):
            # Initialize Backend with minimal settings - just for pattern matching
            # enable_userbuffers=False because we're only doing NCCL_SYMMETRIC patterns
            # NCCL_SYMMETRIC patterns are registered regardless of enable_userbuffers
            super().__init__(
                enable_inductor=False,  # No inductor compilation
                enable_userbuffers=False,  # Not using UB, just NCCL_SYMMETRIC
                enable_piecewise_cuda_graph=False,  # No CUDA graphs
                capture_num_tokens=None,
                max_num_streams=1,
                mapping=mapping,
            )

        def __call__(self, gm: GraphModule, example_inputs: List[torch.Tensor]) -> callable:
            """
            Apply patterns using existing Backend.optimize() logic.

            This reuses the exact same pattern application code that Backend uses,
            ensuring consistency with the rest of TRT-LLM.
            """
            # Debug: Print graph structure to understand why patterns aren't matching
            logger.debug(
                f"[NCCL_SYMMETRIC] PatternOnlyBackend: Graph has {len(list(gm.graph.nodes))} nodes"
            )
            # Log all allreduce calls in the graph
            allreduce_nodes = []
            for n in gm.graph.nodes:
                if n.op == "call_function":
                    target_str = str(n.target)
                    if "allreduce" in target_str.lower():
                        allreduce_nodes.append(n)

            logger.debug(
                f"[NCCL_SYMMETRIC] PatternOnlyBackend: Found {len(allreduce_nodes)} allreduce nodes in graph"
            )
            for i, node in enumerate(allreduce_nodes):
                # Extract strategy from args (it's typically the 7th positional arg)
                strategy_val = None
                if len(node.args) > 7:
                    strategy_val = node.args[7]
                elif "strategy" in node.kwargs:
                    strategy_val = node.kwargs["strategy"]

                logger.debug(
                    f"[NCCL_SYMMETRIC] PatternOnlyBackend: AllReduce node {i}: "
                    f"target={node.target}, args={len(node.args)}, "
                    f"strategy={strategy_val} (type={type(strategy_val)})"
                )

            # Use the existing optimize() method which already handles:
            # - recover_pass()
            # - Pattern application via custom_passes
            # - eliminate_dead_code()
            # - remove_copy_for_mutates_args()
            # - recompile()
            optimized_gm = self.optimize(gm, example_inputs)
            # Return graph module as callable (no further compilation)
            return optimized_gm

    backend = PatternOnlyBackend(mapping)

    # Use torch.compile only for tracing - patterns are applied via existing Backend infrastructure
    compiled_model = torch.compile(model, backend=backend, fullgraph=False)

    logger.info(
        "[NCCL_SYMMETRIC] Applied pattern matching to model using existing Backend infrastructure"
    )

    return compiled_model
