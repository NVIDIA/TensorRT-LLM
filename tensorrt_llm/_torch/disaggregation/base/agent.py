from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Enum, List, Tuple, Union

from tensorrt_llm.logger import logger

# Try to import C++ bindings for zero-copy performance
try:
    from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
        BaseTransferAgent,
        MemoryDesc,
        MemoryDescs,
        MemoryType,
        TransferOp,
        TransferRequest,
        TransferStatus,
    )

    _CPP_BINDING_AVAILABLE = True
except ImportError:
    _CPP_BINDING_AVAILABLE = False
    logger.warning(
        "C++ transfer agent bindings not available. "
        "Falling back to Python implementations which may have lower performance."
    )


def is_cpp_binding_available() -> bool:
    """Check if C++ transfer agent bindings are available."""
    return _CPP_BINDING_AVAILABLE


# Fallback Python implementations when C++ bindings not available
if not _CPP_BINDING_AVAILABLE:

    class TransferOp(Enum):
        READ = "READ"
        WRITE = "WRITE"

    class MemoryType(Enum):
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
        def register_memory(self, descs: MemoryDescs) -> None:
            """Register a set of memory descriptors on the agent."""
            ...

        @abstractmethod
        def deregister_memory(self, descs: MemoryDescs) -> None:
            """De-register a set of memory descriptors on the agent."""
            ...

        @abstractmethod
        def load_remote_agent(self, name: str, agent_desc: str) -> None:
            """
            Load information about a remote agent specified by name.

            Args:
                name (str): The remote agent's identifier.
                agent_desc (str): A serialized description of the agent.
            """
            ...

        @abstractmethod
        def get_local_agent_desc(self) -> str:
            """Return the serialized description of this agent."""
            ...

        @abstractmethod
        def invalidate_remote_agent(self, name: str) -> None:
            """Invalidate any cached information about the specified remote agent."""
            ...

        @abstractmethod
        def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus:
            """Submit transfer tasks to the agent based on a request."""
            ...

        @abstractmethod
        def notify_sync_message(self, name: str, sync_message: str) -> None:
            """Send a synchronization message to the specified remote agent."""
            ...

        @abstractmethod
        def check_remote_descs(self, name: str, memory_descs: List[int]) -> bool:
            """
            Verify the remote agent's memory descriptors.
            """
            ...


# RegMemoryDescs is Python-only (used for registration with name field)
@dataclass
class RegMemoryDescs:
    type: str
    descs: List[Tuple[int, int, int, str]]
