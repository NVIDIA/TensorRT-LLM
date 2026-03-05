from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


@dataclass
class TokenRange:
    """Range of tokens in the sequence dimension."""

    start: int
    end: int  # exclusive

    def __post_init__(self):
        if self.start < 0 or self.end < 0:
            raise ValueError("Token indices must be non-negative")
        if self.start >= self.end:
            raise ValueError(f"Invalid range: [{self.start}, {self.end})")


@dataclass
class LayerRange:
    """Range of layers to transfer."""

    start: int
    end: int  # exclusive

    def __post_init__(self):
        if self.start < 0 or self.end < 0:
            raise ValueError("Layer indices must be non-negative")
        if self.start >= self.end:
            raise ValueError(f"Invalid range: [{self.start}, {self.end})")


@dataclass
class KVSlice:
    """
    Specifies which portion of KV cache to transfer.
    """

    token_range: Optional[TokenRange] = None
    layer_range: Optional[LayerRange] = None
    block_ids: List[int] = field(default_factory=list)  # Physical block IDs
    is_last_slice: bool = False


class SessionStatus(Enum):
    """Status of a transfer session.

    Represents the various stages/statuses that a file transfer session can go through:

    - INIT: The session has been initialized but not yet ready.
    - READY: The session is ready to start transferring.
    - TRANSFERRING: The session is in progress, currently transferring data.
    - TRANSFERRED: The primary transfer has completed successfully.
    - AUX_TRANSFERRED: The auxiliary part (such as tokens) of the transfer has completed successfully.
    - COMPLETED: The entire session process, including all transfers, has been successfully completed.
    - CANCELED: The session has been canceled by the user or system.
    - ERROR: An error occurred during the session. The session could not complete successfully.
    """

    INIT = "INIT"
    READY = "READY"
    TRANSFERRING = "TRANSFERRING"
    TRANSFERRED = "TRANSFERRED"
    AUX_TRANSFERRED = "AUX_TRANSFERRED"
    COMPLETED = "COMPLETED"
    CANCELED = "CANCELED"
    ERROR = "ERROR"


TaskIdType = int


@dataclass
class AuxBufferMeta:
    ptrs: list[int]
    size: list[int]
    item_sizes: list[int] = field(default_factory=list)
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ptrs": self.ptrs,
            "size": self.size,
            "item_sizes": self.item_sizes,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuxBufferMeta":
        return cls(
            ptrs=data["ptrs"],
            size=data["size"],
            item_sizes=data.get("item_sizes", []),
            device=data.get("device", "cpu"),
        )


AuxSlot = namedtuple("AuxSlot", ["id", "buffer"])


class AuxBufferBase(ABC):
    """
    Abstract base class defining the interface for auxiliary buffer management.
    """

    @abstractmethod
    def alloc_slot(self) -> AuxSlot:
        """
        Allocate a free slot and return its index.
        """
        ...

    @abstractmethod
    def free_slot(self, slot: int) -> None:
        """
        Release the specified slot.
        """
        ...

    @property
    @abstractmethod
    def meta(self) -> AuxBufferMeta:
        """
        Retrieve meta-information about the underlying buffer(s).
        Returns buffer info (e.g., pointers, sizes, device).
        """
        ...

    @abstractmethod
    def fill_slot(self, slot: int, request: LlmRequest) -> None:
        """
        Fill/overwrite the contents of the given slot with data from the request.
        """
        ...

    @abstractmethod
    def get_slot_tokens(self, slot: int) -> tuple[list[int], list[int]]:
        """
        Get the token data (e.g., first/draft tokens) from the specified slot.
        """
        ...


@dataclass
class SessionState:
    """State of a transfer session."""

    status: SessionStatus
    finished_tasks: List[TaskIdType]


class SenderBase(ABC):
    """Base class for sending KV cache data."""

    ...


class ReceiverBase(ABC):
    """Base class for receiving KV cache data."""

    ...


def get_unique_rid(request: LlmRequest) -> Optional[int]:
    return (
        request.py_disaggregated_params.disagg_request_id
        if request.py_disaggregated_params
        else None
    )


class SessionBase(ABC):
    def __init__(self, request: Optional[LlmRequest], unique_rid: Optional[int] = None):
        self._request = request
        self._unique_rid: int = get_unique_rid(request) if request else unique_rid
        self._state = SessionState(status=SessionStatus.INIT, finished_tasks=[])
        self._exception: Optional[Exception] = None

    @property
    def unique_rid(self) -> Optional[int]:
        # readonly
        return self._unique_rid

    @property
    def disagg_params(self) -> DisaggregatedParams:
        return self._request.py_disaggregated_params if self._request else None

    @property
    def request(self) -> Optional[LlmRequest]:
        return self._request

    @request.setter
    def set_request(self, request: LlmRequest):
        """
        Set the request for the session. The request must have the same unique_rid as the session.
        :param request: The request to set.
        """
        req_uid = get_unique_rid(request)
        assert req_uid == self.unique_rid, f"request_id mismatch: {req_uid} != {self.unique_rid}"
        self._request = request

    @property
    def state(self) -> SessionState:
        """
        Returns the current state of the session.
        """
        return self._state

    @state.setter
    def state(self, state: SessionState):
        """
        Set the state of the session.
        :param state: The state to set.
        """
        self._state = state

    @abstractmethod
    def poll_task(self, task_id: TaskIdType) -> SessionStatus:
        """
        Polls the status of a specific task by its ID.
        :param task_id: The task ID to poll.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Closes the session and releases any resources.
        """
        ...

    @property
    def exception(self) -> Optional[Exception]:
        """
        Returns any exception that occurred during the session.
        """
        return self._exception


class TxSessionBase(SessionBase):
    def __init__(
        self, sender: SenderBase, request: Optional[LlmRequest], unique_rid: Optional[int] = None
    ):
        """
        Initializes the transmission session.
        :param sender: The sender instance responsible for sending data.
        :param args: The session arguments.
        """
        self._sender = sender
        super().__init__(request, unique_rid)

    @abstractmethod
    def send(self, slice: KVSlice) -> TaskIdType:
        """
        Sends a slice of KV cache data and returns the task ID.
        :param slice: The KV slice to send.
        """


class RxSessionBase(SessionBase):
    def __init__(self, receiver: ReceiverBase, request: Optional[LlmRequest]):
        """
        Initializes the reception session.
        :param receiver: The receiver instance responsible for receiving data.
        """
        super().__init__(request)
        self._receiver = receiver

    @abstractmethod
    def receive(self, slice: KVSlice) -> TaskIdType:
        """
        Receives a slice of KV cache data and returns the task ID.
        :param slice: The KV slice to receive.
        """
        ...
