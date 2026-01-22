from tensorrt_llm.logger import logger

from ..base.agent import use_pure_python_transfer_agent

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
    except ImportError as e:
        logger.info("Failed to import module: %s. Error: %s", module_name, str(e))
    return None


NixlTransferStatus, NixlTransferAgent = None, None

if use_pure_python_transfer_agent():
    _py_agent = _load_agent(
        module_name="tensorrt_llm._torch.disaggregation.nixl._agent_py",
        required_attributes=["NixlTransferAgent", "NixlTransferStatus"],
    )
    assert _py_agent is not None, "Failed to load pure Python NIXL Transfer Agent."
    NixlTransferStatus = _py_agent.NixlTransferStatus
    NixlTransferAgent = _py_agent.NixlTransferAgent
else:
    _cpp_agent = _load_agent(
        module_name="tensorrt_llm._torch.disaggregation.nixl._agent_cpp",
        required_attributes=["BindingsNixlTransferAgent", "BindingsNixlTransferStatus"],
    )
    assert _cpp_agent is not None, "Failed to load C++ NIXL Transfer Agent bindings."
    NixlTransferStatus = _cpp_agent.BindingsNixlTransferStatus
    NixlTransferAgent = _cpp_agent.BindingsNixlTransferAgent
