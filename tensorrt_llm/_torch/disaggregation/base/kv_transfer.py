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
    status: SessionStatus
    finished_tasks: List[TaskIdType]


@dataclass
class SessionArgsBase:
    request_id: int
    params: DisaggregatedParams


class SenderBase(ABC): ...


class ReceiverBase(ABC): ...


class TxSessionBase(ABC):
    def __init__(self, sender: SenderBase, args: SessionArgsBase):
        self._base_args = args

    @property
    @abstractmethod
    def state(self) -> SessionState: ...

    @abstractmethod
    def poll_task(self, id: TaskIdType) -> SessionStatus: ...

    @abstractmethod
    def send(self, slice: KVSlice) -> TaskIdType: ...

    """
    Async send slice to the peer. return the task id. Task state can be polled by poll_task().
    """

    @property
    @abstractmethod
    def exception(self) -> Optional[Exception]: ...


class RxSessionBase(ABC):
    def __init__(self, receiver: ReceiverBase, args: SessionArgsBase):
        self._base_args = args

    @property
    @abstractmethod
    def state(self) -> SessionState: ...

    @abstractmethod
    def poll_task(self, id: TaskIdType) -> SessionStatus: ...

    @abstractmethod
    def receive(self, slice: KVSlice) -> TaskIdType: ...

    """
    Async receive slice from the peer. return the task id. Task state can be polled by poll_task().
    """

    @property
    @abstractmethod
    def exception(self) -> Optional[Exception]: ...
