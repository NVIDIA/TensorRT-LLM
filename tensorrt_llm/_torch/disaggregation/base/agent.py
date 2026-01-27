import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

from tensorrt_llm import logger


# We deliberately use a non-enum data structure here. This choice ensures that
# members are directly equivalent to the plain strings.
class TransferOp:
    READ = "READ"
    WRITE = "WRITE"


class MemoryType:
    DRAM = "DRAM"
    VRAM = "VRAM"
    BLK = "BLK"
    OBJ = "OBJ"
    FILE = "FILE"


class MemoryDesc(NamedTuple):
    ptr: int
    size: int
    device_id: int
    name: Optional[str] = None


@dataclass
class MemoryDescs:
    type: str
    descs: List[MemoryDesc]


@dataclass
class TransferRequest:
    op: TransferOp
    src_descs: MemoryDescs
    dst_descs: MemoryDescs
    remote_name: str
    sync_message: Optional[str] = None


@dataclass
class RegMemoryDescs:
    type: str
    descs: List[MemoryDesc]


class TransferStatus(ABC):
    @abstractmethod
    def is_completed(self) -> bool: ...

    @abstractmethod
    def wait(self, timeout_ms: int | None = None) -> bool: ...


class BaseTransferAgent(ABC):
    @abstractmethod
    def register_memory(self, descs: RegMemoryDescs) -> None: ...

    @abstractmethod
    def deregister_memory(self, descs: RegMemoryDescs) -> None: ...

    @abstractmethod
    def load_remote_agent(self, name: str, agent_desc: bytes) -> None: ...

    @abstractmethod
    def get_local_agent_desc(self) -> bytes: ...

    @abstractmethod
    def invalidate_remote_agent(self, name: str) -> None: ...

    @abstractmethod
    def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus: ...

    @abstractmethod
    def notify_sync_message(self, name: str, sync_message: str) -> None: ...

    @abstractmethod
    def check_remote_descs(self, name: str, memory_descs: MemoryDescs) -> bool: ...


def _force_py_nixl_kv_transfer() -> bool:
    env_value = os.getenv("TRTLLM_USE_PY_NIXL_KVCACHE", "0")
    if env_value not in {"0", "1"}:
        logger.warning(
            f"Invalid value for TRTLLM_USE_PY_NIXL_KVCACHE: {env_value}. Expected '0' or '1'. Defaulting to '0'."
        )
        return False
    if env_value == "1":
        logger.info("Forcing use of pure Python NIXL KV Transfer Agent implementation.")
        return True
    return False


def _try_load_cpp_binding():
    try:
        import tensorrt_llm.tensorrt_llm_transfer_agent_binding as _cpp_binding

        required_attributes = [
            "MemoryType",
            "TransferOp",
            "MemoryDesc",
            "MemoryDescs",
            "TransferRequest",
            "TransferStatus",
            "BaseTransferAgent",
        ]
        if all(hasattr(_cpp_binding, attr) for attr in required_attributes):
            return _cpp_binding
    except ImportError:
        logger.info("tensorrt_llm_transfer_agent_binding module not found.")
    return None


_use_pure_python_transfer_agent = None

# The current implementation still implicitly depends on cpp_bindings.
# We should remove this dependency.
_cpp_binding = _try_load_cpp_binding()

if _force_py_nixl_kv_transfer():
    logger.info("Using pure Python transfer agent (forced by TRTLLM_USE_PY_NIXL_KVCACHE)")
    _use_pure_python_transfer_agent = True
else:
    if _cpp_binding:
        MemoryType = _cpp_binding.MemoryType
        TransferOp = _cpp_binding.TransferOp
        MemoryDesc = _cpp_binding.MemoryDesc
        MemoryDescs = _cpp_binding.MemoryDescs
        TransferRequest = _cpp_binding.TransferRequest
        TransferStatus = _cpp_binding.TransferStatus
        BaseTransferAgent = _cpp_binding.BaseTransferAgent
        logger.info("Using C++ transfer agent binding")
        _use_pure_python_transfer_agent = False
    else:
        logger.info("C++ transfer agent binding unavailable, using pure Python implementation")
        _use_pure_python_transfer_agent = True


def use_pure_python_transfer_agent() -> bool:
    return _use_pure_python_transfer_agent
