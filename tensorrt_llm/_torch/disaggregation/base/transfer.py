from __future__ import annotations

import concurrent.futures
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

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
    block_ids_per_layer_groups: List[List[int]] = field(
        default_factory=list
    )  # Physical block IDs per layer group
    is_last_slice: bool = False


class SessionStatus(Enum):
    """Status of a transfer session.

    Represents the lifecycle stages of a KV cache transfer session:

    - INIT: Session initialized; waiting for the remote peer to become ready.
    - READY: Peer is ready; transfer can begin.
    - TRANSFERRING: KV cache transfer is in progress.
    - KV_TRANSFERRED: KV cache transfer completed; auxiliary data transfer may still be pending.
    - FULLY_TRANSFERRED: Both KV cache and auxiliary data (e.g. tokens) transferred successfully.
    - ERROR: A transfer error occurred; the session cannot complete.
    """

    INIT = "INIT"
    READY = "READY"
    TRANSFERRING = "TRANSFERRING"
    KV_TRANSFERRED = "KV_TRANSFERRED"
    FULLY_TRANSFERRED = "FULLY_TRANSFERRED"
    ERROR = "ERROR"


class WaitResult(Enum):
    """Result of waiting for a transfer session to complete."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass
class SessionArgsBase:
    """Base arguments for transfer sessions."""

    params: DisaggregatedParams


def get_unique_rid(request: LlmRequest) -> Optional[int]:
    return (
        request.py_disaggregated_params.disagg_request_id
        if request.py_disaggregated_params
        else request.request_id
    )


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
    def disagg_request_id(self) -> int:
        return self._base_args.params.disagg_request_id

    @abstractmethod
    def send(self, slice: KVSlice) -> concurrent.futures.Future:
        """
        Sends a slice of KV cache data and returns a Future for the transfer.
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
    def disagg_request_id(self) -> int:
        return self._base_args.params.disagg_request_id

    @abstractmethod
    def receive(self, slice: KVSlice) -> concurrent.futures.Future:
        """
        Receives a slice of KV cache data and returns a Future for the transfer.
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
