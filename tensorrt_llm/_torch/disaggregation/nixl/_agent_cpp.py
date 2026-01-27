from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (  # noqa: E402
    AgentDesc,
    BaseAgentConfig,
    MemoryDescs,
    MemoryType,
    TransferState,
)
from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
    NixlTransferAgent as CppNixlTransferAgent,
)

from ..base.agent import BaseTransferAgent, RegMemoryDescs, TransferRequest, TransferStatus


class BindingsNixlTransferStatus(TransferStatus):
    """TransferStatus wrapper using C++ bindings with GIL release."""

    def __init__(self, cpp_status):
        self._cpp_status = cpp_status

    def is_completed(self) -> bool:
        """Check if transfer is completed (releases GIL)."""
        return self._cpp_status.is_completed()

    @nvtx_range("BindingsNixlTransferStatus.wait")
    def wait(self, timeout_ms=None) -> bool:
        """Wait for transfer to complete (releases GIL)."""
        if timeout_ms is None:
            timeout_ms = -1
        return self._cpp_status.wait(timeout_ms) == TransferState.SUCCESS


class BindingsNixlTransferAgent(BaseTransferAgent):
    """NixlTransferAgent using C++ bindings with GIL release support.

    This implementation uses the standalone nixl_bindings C++ module which releases
    the GIL during blocking operations like wait().

    The nixl_bindings module is independent from the main trtllm bindings,
    so trtllm can function normally even without NIXL.
    """

    def __init__(
        self,
        name: str,
        use_prog_thread: bool = True,
        num_threads: int = 1,
        enable_telemetry: bool = False,
        **kwargs,
    ):
        backend_params = kwargs
        for key, value in backend_params.items():
            backend_params[key] = str(value)
        backend_params["num_threads"] = str(num_threads)

        config = BaseAgentConfig(
            name,
            use_prog_thread,
            multi_thread=False,
            use_listen_thread=False,
            enable_telemetry=enable_telemetry,
            backend_params=backend_params,
        )
        self._cpp_agent = CppNixlTransferAgent(config)
        self.name = name

    def register_memory(self, descs: RegMemoryDescs):
        """Register memory regions."""
        cpp_descs = self._convert_reg_memory_descs(descs)
        self._cpp_agent.register_memory(cpp_descs)

    def deregister_memory(self, descs: RegMemoryDescs):
        """Deregister memory regions."""
        cpp_descs = self._convert_reg_memory_descs(descs)
        self._cpp_agent.deregister_memory(cpp_descs)

    def load_remote_agent(self, name: str, agent_desc: bytes):
        """Load a remote agent by its descriptor (bytes)."""
        # AgentDesc expects std::string which can hold binary data
        desc_str = agent_desc if isinstance(agent_desc, bytes) else agent_desc.encode()
        cpp_desc = AgentDesc(desc_str)
        self._cpp_agent.load_remote_agent(name, cpp_desc)

    def load_remote_agent_by_connection(self, name: str, connection_info: str):
        """Load a remote agent by connection info."""
        self._cpp_agent.load_remote_agent_by_connection(name, connection_info)

    def get_local_agent_desc(self) -> bytes:
        """Get the local agent descriptor as bytes."""
        agent_desc = self._cpp_agent.get_local_agent_desc()
        return agent_desc.backend_agent_desc  # Returns bytes

    def get_local_connection_info(self) -> str:
        """Get the local connection info."""
        return self._cpp_agent.get_local_connection_info()

    def invalidate_remote_agent(self, name: str):
        """Invalidate a remote agent."""
        self._cpp_agent.invalidate_remote_agent(name)

    def check_remote_descs(self, name: str, memory_descs: MemoryDescs) -> bool:
        """Check if remote descriptors are available.

        memory_descs should be C++ MemoryDescs type.
        """
        return self._cpp_agent.check_remote_descs(name, memory_descs)

    def notify_sync_message(self, name: str, sync_message: str):
        """Send a sync message to a remote agent."""
        self._cpp_agent.notify_sync_message(name, sync_message)

    def get_notified_sync_messages(self):
        """Get notified sync messages."""
        return self._cpp_agent.get_notified_sync_messages()

    @nvtx_range("BindingsNixlTransferAgent.submit_transfer_requests")
    def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus:
        """Submit transfer requests and return status.

        request should be a C++ TransferRequest (from tensorrt_llm_transfer_agent_binding).
        """
        cpp_status = self._cpp_agent.submit_transfer_requests(request)
        return BindingsNixlTransferStatus(cpp_status)

    def _convert_reg_memory_descs(self, descs: RegMemoryDescs) -> "MemoryDescs":
        """Convert Python RegMemoryDescs to C++ MemoryDescs.

        RegMemoryDescs.descs is List[Tuple[int, int, int, str]] = (ptr, size, device_id, name)
        Extract first 3 elements for C++ batch constructor.
        """
        mem_type = self._convert_memory_type(descs.type)
        # Extract (ptr, size, device_id) from 4-tuple, discard name
        tuples = [(d[0], d[1], d[2]) for d in descs.descs]
        return MemoryDescs(mem_type, tuples)

    def _convert_memory_type(self, py_type: str) -> "MemoryType":
        """Convert Python memory type string to C++ MemoryType."""
        type_map = {
            "DRAM": MemoryType.DRAM,
            "VRAM": MemoryType.VRAM,
            "GPU": MemoryType.VRAM,
            "BLK": MemoryType.BLK,
            "OBJ": MemoryType.OBJ,
            "FILE": MemoryType.FILE,
        }
        return type_map.get(py_type.upper(), MemoryType.VRAM)
