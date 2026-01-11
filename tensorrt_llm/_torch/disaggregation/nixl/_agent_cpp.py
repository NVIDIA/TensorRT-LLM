from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
    AgentDesc,
    BaseAgentConfig,
    MemoryDescs,
    MemoryType,
    TransferState,
)
from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
    NixlTransferAgent as CppNixlTransferAgent,
)
from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
    NixlTransferStatus as CppNixlTransferStatus,
)

from ..base.agent import BaseTransferAgent, BaseTransferStatus, RegMemoryDescs, TransferRequest


class NixlTransferStatus(BaseTransferStatus):
    def __init__(self, cpp_status: CppNixlTransferStatus):
        self._cpp_status = cpp_status

    def is_completed(self) -> bool:
        """Check if transfer is completed (releases GIL)."""
        return self._cpp_status.is_completed()

    @nvtx_range("NixlTransferStatus.wait")
    def wait(self, timeout: float = None) -> bool:
        """Wait for transfer to complete (releases GIL)."""
        return self._cpp_status.wait() == TransferState.SUCCESS


class NixlTransferAgent(BaseTransferAgent):
    """NixlTransferAgent using C++ bindings with GIL release support.

    This implementation uses the standalone nixl_bindings C++ module which releases
    the GIL during blocking operations like wait().

    The nixl_bindings module is independent from the main trtllm bindings,
    so trtllm can function normally even without NIXL.
    """

    def __init__(self, name: str, use_prog_thread: bool = True, num_workers: int = 1):
        config = BaseAgentConfig(
            name=name,
            use_prog_thread=use_prog_thread,
            multi_thread=False,
            use_listen_thread=False,
            num_workers=num_workers,
        )
        self._cpp_agent = CppNixlTransferAgent(config)
        self.name = name

    def register_memory(self, descs: RegMemoryDescs):
        cpp_descs = self._convert_reg_memory_descs(descs)
        self._cpp_agent.register_memory(cpp_descs)

    def deregister_memory(self, descs: RegMemoryDescs):
        cpp_descs = self._convert_reg_memory_descs(descs)
        self._cpp_agent.deregister_memory(cpp_descs)

    def load_remote_agent(self, name: str, agent_desc: bytes):
        desc_str = agent_desc if isinstance(agent_desc, bytes) else agent_desc.encode()
        cpp_desc = AgentDesc(desc_str)
        self._cpp_agent.load_remote_agent(name, cpp_desc)

    def load_remote_agent_by_connection(self, name: str, connection_info: str):
        self._cpp_agent.load_remote_agent_by_connection(name, connection_info)

    def get_local_agent_desc(self) -> bytes:
        agent_desc = self._cpp_agent.get_local_agent_desc()
        return agent_desc.backend_agent_desc

    def get_local_connection_info(self) -> str:
        return self._cpp_agent.get_local_connection_info()

    def invalidate_remote_agent(self, name: str):
        self._cpp_agent.invalidate_remote_agent(name)

    def check_remote_descs(self, name: str, memory_descs: MemoryDescs) -> bool:
        return self._cpp_agent.check_remote_descs(name, memory_descs)

    def notify_sync_message(self, name: str, sync_message: str):
        self._cpp_agent.notify_sync_message(name, sync_message)

    def get_notified_sync_messages(self):
        return self._cpp_agent.get_notified_sync_messages()

    @nvtx_range("BindingsNixlTransferAgent.submit_transfer_requests")
    def submit_transfer_requests(self, request: TransferRequest) -> BaseTransferStatus:
        cpp_status = self._cpp_agent.submit_transfer_requests(request)
        return NixlTransferStatus(cpp_status)

    def _convert_reg_memory_descs(self, descs: RegMemoryDescs) -> "MemoryDescs":
        mem_type = self._convert_memory_type(descs.type)
        tuples = [(d[0], d[1], d[2]) for d in descs.descs]  # Extract (ptr, size, device_id)
        return MemoryDescs(mem_type, tuples)

    def _convert_memory_type(self, py_type: str) -> "MemoryType":
        type_map = {
            "DRAM": MemoryType.DRAM,
            "VRAM": MemoryType.VRAM,
            "GPU": MemoryType.VRAM,
            "BLK": MemoryType.BLK,
            "OBJ": MemoryType.OBJ,
            "FILE": MemoryType.FILE,
        }
        return type_map.get(py_type.upper(), MemoryType.VRAM)
