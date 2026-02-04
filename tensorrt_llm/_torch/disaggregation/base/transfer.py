from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from tensorrt_llm import DisaggregatedParams


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
class SessionState:
    """State of a transfer session."""

    status: SessionStatus
    finished_tasks: List[TaskIdType]


@dataclass
class SessionArgsBase:
    """Base arguments for transfer sessions."""

    params: DisaggregatedParams


class SenderBase(ABC):
    """Base class for sending KV cache data."""

    ...


class ReceiverBase(ABC):
    """Base class for receiving KV cache data."""

    ...


class TxSessionBase(ABC):
    def __init__(self, sender: SenderBase, args: SessionArgsBase):
        """
        Initializes the transmission session.
        :param sender: The sender instance responsible for sending data.
        :param args: The session arguments.
        """
        self._sender = sender
        self._base_args = args

    @property
    @abstractmethod
    def state(self) -> SessionState:
        """
        Returns the current state of the session.
        """
        ...

    @abstractmethod
    def poll_task(self, id: TaskIdType) -> SessionStatus:
        """
        Polls the status of a specific task by its ID.
        :param id: The task ID to poll.
        """
        ...

    @abstractmethod
    def send(self, slice: KVSlice) -> TaskIdType:
        """
        Sends a slice of KV cache data and returns the task ID.
        :param slice: The KV slice to send.
        """
        ...

    @property
    @abstractmethod
    def exception(self) -> Optional[Exception]:
        """
        Returns any exception that occurred during the session.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Closes the session and releases any resources.
        """
        ...


class RxSessionBase(ABC):
    def __init__(self, receiver: ReceiverBase, args: SessionArgsBase):
        """
        Initializes the reception session.
        :param receiver: The receiver instance responsible for receiving data.
        """
        self._receiver = receiver
        self._base_args = args

    @property
    @abstractmethod
    def state(self) -> SessionState:
        """
        Returns the current state of the session.
        """
        ...

    @abstractmethod
    def poll_task(self, task_id: TaskIdType) -> SessionStatus:
        """
        Polls the status of a specific task by its ID.
        :param task_id: The task ID to poll.
        """
        ...

    @abstractmethod
    def receive(self, slice: KVSlice) -> TaskIdType:
        """
        Receives a slice of KV cache data and returns the task ID.
        :param slice: The KV slice to receive.
        """
        ...

    @property
    @abstractmethod
    def exception(self) -> Optional[Exception]:
        """Returns any exception that occurred during the session."""
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Closes the session and releases any resources.
        """
        ...
