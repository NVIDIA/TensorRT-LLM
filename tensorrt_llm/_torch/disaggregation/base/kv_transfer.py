from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from tensorrt_llm import DisaggregatedParams


@dataclass
class KVSlice:
    """Supports transmitting only part of the request cache, e.g, chunks or layers."""

    start_token_idx: Optional[int] = None
    end_token_idx: Optional[int] = None
    start_layer: Optional[int] = None
    end_layer: Optional[int] = None
    blocks: List[int] = field(default_factory=list)
    is_last_slice: bool = False


class SessionStatus(Enum):
    """Status of a transfer session."""

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


class RxSessionBase(ABC):
    def __init__(self, receiver: ReceiverBase, args: SessionArgsBase):
        """
        Initializes the reception session.
        :param receiver: The receiver instance responsible for receiving data.
        """
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
