from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, cast

import numpy as np

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


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

    Each layer group's block list is anchored explicitly: ``first_ordinals[i]``
    is the block ordinal (0-based sequence block index) of element 0 of group
    ``i``'s beam-0 list, so positions never have to be inferred from list
    lengths. The anchor describes only the beam-0 prefix; packed beam tails
    (``beam_width > 1`` appends per-beam tail blocks after beam-0) stay outside
    the anchored region. STATE groups and empty lists use anchor 0.

    Monolithic transfers cover ``prompt_len``. Pipelined transfers send one
    slice per prefill chunk (anchored at the chunk's first block ordinal) and
    mark only the final chunk with ``is_last_slice``.
    """

    layer_range: Optional[LayerRange] = None
    block_ids_per_layer_groups: List[np.ndarray] = field(
        default_factory=list
    )  # Physical block IDs per layer group, each np.ndarray(dtype=np.int64)
    is_last_slice: bool = False
    # Block ordinal of element 0 of each group's beam-0 list; parallel to
    # block_ids_per_layer_groups.
    first_ordinals: List[int] = field(default_factory=list)


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
    # Captured from LlmRequest.prompt_len; bounds the beam-0 span
    # (ceil(prompt_len / tokens_per_block) block ordinals).
    prompt_len: int
    beam_width: int = 1


def get_unique_rid(request: LlmRequest) -> Optional[int]:
    if request.py_disaggregated_params:
        rid = request.py_disaggregated_params.disagg_request_id
        if rid is not None:
            return rid
    return request.request_id


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
        return cast(int, self._base_args.params.disagg_request_id)

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
                For pipelined chunks, each layer group's list is anchored at
                the chunk's first block ordinal via ``first_ordinals``.
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
