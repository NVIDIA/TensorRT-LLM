from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union

# Try to import C++ bindings for zero-copy performance
_CPP_BINDING_AVAILABLE = False
try:
    import tensorrt_llm.tensorrt_llm_transfer_agent_binding as _cpp_binding

    _CPP_BINDING_AVAILABLE = True
    # Use C++ types directly when available
    MemoryType = _cpp_binding.MemoryType
    TransferOp = _cpp_binding.TransferOp
    MemoryDesc = _cpp_binding.MemoryDesc
    MemoryDescs = _cpp_binding.MemoryDescs
    TransferRequest = _cpp_binding.TransferRequest
    TransferStatus = _cpp_binding.TransferStatus
    BaseTransferAgent = _cpp_binding.BaseTransferAgent
except ImportError:
    _CPP_BINDING_AVAILABLE = False


def is_cpp_binding_available() -> bool:
    """Check if C++ transfer agent bindings are available."""
    return _CPP_BINDING_AVAILABLE


# Fallback Python implementations when C++ bindings not available
if not _CPP_BINDING_AVAILABLE:

    class TransferOp:
        READ = "READ"
        WRITE = "WRITE"

    class MemoryType:
        DRAM = "DRAM"
        VRAM = "VRAM"
        BLK = "BLK"
        OBJ = "OBJ"
        FILE = "FILE"

    @dataclass
    class MemoryDesc:
        ptr: int
        size: int
        device_id: int

    @dataclass
    class MemoryDescs:
        type: str
        descs: List[Union[Tuple[int, int, int], MemoryDesc]]

    @dataclass
    class TransferRequest:
        op: TransferOp
        src_descs: MemoryDescs
        dst_descs: MemoryDescs
        remote_name: str
        sync_message: str

    class TransferStatus(ABC):
        @abstractmethod
        def is_completed(self) -> bool: ...

        @abstractmethod
        def wait(self, timeout: float | None = None) -> None: ...

    class BaseTransferAgent(ABC):
        @abstractmethod
        def register_memory(self, descs: MemoryDescs) -> None: ...

        @abstractmethod
        def deregister_memory(self, descs: MemoryDescs) -> None: ...

        @abstractmethod
        def load_remote_agent(self, name: str, agent_desc: str) -> None: ...

        @abstractmethod
        def get_local_agent_desc(self) -> str: ...

        @abstractmethod
        def invalidate_remote_agent(self, name: str) -> None: ...

        @abstractmethod
        def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus: ...

        @abstractmethod
        def notify_sync_message(self, name: str, sync_message: str) -> None: ...

        @abstractmethod
        def check_remote_descs(self, name: str, memory_descs: List[int]) -> bool: ...


# RegMemoryDescs is Python-only (used for registration with name field)
@dataclass
class RegMemoryDescs:
    type: str
    descs: List[Tuple[int, int, int, str]]
