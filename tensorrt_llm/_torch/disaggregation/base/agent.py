import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union

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
    descs: List[Tuple[int, int, int, str]]  # (ptr, size, device_id, name)


def _force_py_nixl_kv_transfer() -> bool:
    res = os.getenv("TRTLLM_USE_PY_NIXL_KVCACHE", "0") == "1"
    if res:
        logger.info("Forcing use of pure Python NIXL KV Transfer Agent implementation.")
    return res


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


_cpp_binding = _try_load_cpp_binding()

if _cpp_binding and not _force_py_nixl_kv_transfer():
    MemoryType = _cpp_binding.MemoryType
    TransferOp = _cpp_binding.TransferOp
    MemoryDesc = _cpp_binding.MemoryDesc
    MemoryDescs = _cpp_binding.MemoryDescs
    TransferRequest = _cpp_binding.TransferRequest
    TransferStatus = _cpp_binding.TransferStatus
    BaseTransferAgent = _cpp_binding.BaseTransferAgent
    logger.info("Using Pybind transfer agent binding for Transfer Agent implementation.")
else:
    logger.warning(
        "Failed to import Pybind transfer agent binding, using pure Python implementation."
    )
