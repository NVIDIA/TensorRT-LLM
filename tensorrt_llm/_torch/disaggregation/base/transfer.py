from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, cast

import numpy as np

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


@dataclass
class TokenRange:
    """Half-open token range [start, end) within one request.

    ``KVSlice`` ranges are block-aligned. Empty ranges are valid.
    """

    start: int
    end: int  # exclusive

    def __post_init__(self):
        if self.start < 0 or self.end < 0:
            raise ValueError("Token indices must be non-negative")
        if self.start > self.end:
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
    """KV-cache blocks for one request transfer slice.

    Monolithic transfers omit ``token_range`` and cover ``prompt_len``.
    Pipelined transfers use a block-aligned ``token_range`` and mark only the
    final chunk with ``is_last_slice``. Block lists may omit cached or evicted
    prefixes.
    """

    layer_range: Optional[LayerRange] = None
    block_ids_per_layer_groups: List[np.ndarray] = field(
        default_factory=list
    )  # Physical block IDs per layer group, each np.ndarray(dtype=np.int64)
    is_last_slice: bool = False
    token_range: Optional[TokenRange] = None


class SessionStatus(Enum):
    """Status of a transfer session.

    Represents the lifecycle stages of a KV cache transfer session:

    - INIT: Session initialized; waiting for the remote peer to become ready.
    - READY: Peer is ready; transfer can begin.
    - TRANSFERRING: KV cache transfer is in progress.
    - KV_TRANSFERRED: KV cache transfer completed; auxiliary data transfer may still be pending.
    - FULLY_TRANSFERRED: Both KV cache and auxiliary data (e.g. tokens) transferred successfully.
    - ERROR: A transfer error occurred; the session cannot complete.
    - CANCELLED: The session was explicitly cancelled before or during transfer.
    """

    INIT = "INIT"
    READY = "READY"
    TRANSFERRING = "TRANSFERRING"
    KV_TRANSFERRED = "KV_TRANSFERRED"
    FULLY_TRANSFERRED = "FULLY_TRANSFERRED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class WaitResult(Enum):
    """Result of waiting for a transfer session to complete."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass
class SessionArgsBase:
    """Base arguments for transfer sessions."""

    params: DisaggregatedParams
    # Captured from LlmRequest.prompt_len; needed for SWA stale_end derivation.
    prompt_len: int
    beam_width: int = 1


def resolve_transfer_rid(params, fallback: Optional[int] = None) -> Optional[int]:
    """The transfer key both sides of a request must agree on.

    ``ctx_request_id`` first: an orchestrator retry regenerates
    ``disagg_request_id`` for the generation request only, leaving it different
    from the id the context session registered under.
    """
    if params is not None:
        if params.ctx_request_id is not None:
            return params.ctx_request_id
        if params.disagg_request_id is not None:
            return params.disagg_request_id
    return fallback


def get_unique_rid(request: LlmRequest) -> Optional[int]:
    return resolve_transfer_rid(request.py_disaggregated_params, request.request_id)


class SenderBase(ABC):
    """Base class for sending KV cache data."""

    ...


class ReceiverBase(ABC):
    """Base class for receiving KV cache data."""

    ...


class _SessionBase(ABC):
    """Shared base for Tx/Rx sessions."""

    def __init__(self, args: SessionArgsBase):
        self._base_args = args

    @property
    def disagg_request_id(self) -> int:
        return cast(int, resolve_transfer_rid(self._base_args.params))

    @abstractmethod
    def is_completed(self) -> bool: ...

    @abstractmethod
    def wait_complete(self, blocking: bool = False) -> Optional[WaitResult]: ...

    @property
    @abstractmethod
    def exception(self) -> Optional[Exception]: ...

    @abstractmethod
    def close(self) -> None: ...


class TxSessionBase(_SessionBase):
    def __init__(self, sender: SenderBase, args: SessionArgsBase):
        super().__init__(args)
        self._sender = sender

    @abstractmethod
    def send(self, slice: KVSlice) -> None:
        """Send a KV slice.

        Args:
            slice: The KV slice describing which source blocks to send.
                For pipelined chunks, ``token_range`` is the shared sender-side
                chunk cursor; each layer group projects it into its own
                resident/windowed source and destination block ranges.
        """
        ...

    @abstractmethod
    def wait_complete(self, blocking: bool = True) -> Optional[WaitResult]: ...


class RxSessionBase(_SessionBase):
    def __init__(self, receiver: ReceiverBase, args: SessionArgsBase):
        super().__init__(args)
        self._receiver = receiver

    @abstractmethod
    def receive(self, slice: KVSlice) -> None: ...

    @abstractmethod
    def wait_complete(self, blocking: bool = False) -> Optional[WaitResult]: ...
