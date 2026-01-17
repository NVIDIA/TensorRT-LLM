"""Custom op for debug markers that survive torch.export.

This custom op is used to mark tensor values at module boundaries so they can be
identified and compared during debugging. Unlike nn.Module-based markers, custom ops
survive torch.export as distinct call_function nodes.

Usage:
    from tensorrt_llm._torch.auto_deploy.custom_ops.debug_mark import debug_mark
    marked_tensor = debug_mark(tensor, "model.layers.0.self_attn.out0")
"""

import torch

# The tag is stored as a string attribute on each call site.
# We use a global registry to track tag -> node name mappings after export.
_DEBUG_MARK_NAMESPACE = "auto_deploy"
_DEBUG_MARK_OP_NAME = "debug_mark"
_DEBUG_MARK_FULL_NAME = f"{_DEBUG_MARK_NAMESPACE}::{_DEBUG_MARK_OP_NAME}"


@torch.library.custom_op(f"{_DEBUG_MARK_NAMESPACE}::{_DEBUG_MARK_OP_NAME}", mutates_args=())
def debug_mark(x: torch.Tensor, tag: str) -> torch.Tensor:
    """Mark a tensor with a debug tag. Returns the tensor unchanged.

    The tag is embedded in the graph as a string constant, making the node
    identifiable for comparison purposes.

    Args:
        x: The tensor to mark.
        tag: A unique string tag identifying this marker (e.g., "model.layers.0.self_attn.out0").

    Returns:
        The same tensor, unchanged.
    """
    # Identity operation - just return the tensor.
    # We use clone() to ensure the op has a side effect and isn't optimized away.
    return x.clone()


@debug_mark.register_fake
def _debug_mark_fake(x: torch.Tensor, tag: str) -> torch.Tensor:
    """Fake implementation for torch.compile / export shape inference."""
    return torch.empty_like(x)


def get_debug_mark_op_name() -> str:
    """Return the full qualified name of the debug_mark op for graph scanning."""
    return _DEBUG_MARK_FULL_NAME
