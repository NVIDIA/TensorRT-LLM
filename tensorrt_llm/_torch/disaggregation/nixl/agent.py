from tensorrt_llm.logger import logger

from ..base.agent import _force_py_nixl_kv_transfer

"""NIXL Transfer Agent implementations.

This module provides two implementations:
1. BindingsNixlTransferAgent - Uses the standalone nixl_bindings C++ module with GIL release support
2. NixlTransferAgent - Uses the Python nixl library directly (fallback)

The standalone nixl_bindings module is separate from the main trtllm bindings,
so trtllm can still function normally even without NIXL dependencies.
"""


def _load_agent(module_name, required_attributes):
    try:
        module = __import__(module_name, fromlist=required_attributes, level=0)
        if all(hasattr(module, attr) for attr in required_attributes):
            return module
    except ImportError:
        logger.info("Failed to import module: %s", module_name)
    return None


_py_agent = _load_agent(
    module_name="tensorrt_llm._torch.disaggregation.nixl._agent_py",
    required_attributes=["NixlTransferAgent", "NixlTransferStatus"],
)

_cpp_agent = _load_agent(
    module_name="tensorrt_llm._torch.disaggregation.nixl._agent_cpp",
    required_attributes=["BindingsNixlTransferAgent", "BindingsNixlTransferStatus"],
)

# Determine which Transfer Agent implementation to use
if _cpp_agent and not _force_py_nixl_kv_transfer():
    NixlTransferStatus = _cpp_agent.BindingsNixlTransferStatus
    NixlTransferAgent = _cpp_agent.BindingsNixlTransferAgent
    logger.info("Using C++ NIXL Transfer Agent implementation.")
elif _py_agent:
    NixlTransferStatus = _py_agent.NixlTransferStatus
    NixlTransferAgent = _py_agent.NixlTransferAgent
    logger.info("Using Python NIXL Transfer Agent implementation.")
else:
    raise ImportError("Both C++ and Python NIXL Transfer Agents failed to load.")
