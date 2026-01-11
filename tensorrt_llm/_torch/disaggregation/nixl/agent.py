from tensorrt_llm import logger

"""NIXL Transfer Agent implementations.

This module provides two implementations:
1. BindingsNixlTransferAgent - Uses the standalone nixl_bindings C++ module with GIL release support
2. NixlTransferAgent - Uses the Python nixl library directly (fallback)

The standalone nixl_bindings module is separate from the main trtllm bindings,
so trtllm can still function normally even without NIXL dependencies.
"""

try:
    # Try to import the standalone tensorrt_llm_transfer_agent_binding module
    # Located at tensorrt_llm/ (same level as bindings.so)
    from ._agent_cpp import NixlTransferAgent, NixlTransferStatus

    logger.info("Using C++ NIXL TransferAgent")

except ImportError:
    from ._agent_python import NixlTransferAgent, NixlTransferStatus

    logger.info("Using Python NIXL TransferAgent")

__all__ = ["NixlTransferStatus", "NixlTransferAgent"]
