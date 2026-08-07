from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, cast

import numpy as np

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


def project_blocks_to_global_chunk(
    block_ids: np.ndarray,
    chunk_block_offset: int,
    chunk_block_count: int,
    resident_block_end: int,
) -> np.ndarray:
    """Project a global block chunk into a suffix-resident block list.

    ``block_ids`` represents the resident suffix of the logical range
    ``[0, resident_block_end)``. ``chunk_block_offset`` and
    ``chunk_block_count`` describe a chunk in that global coordinate space.
    """
    if chunk_block_count <= 0 or len(block_ids) == 0:
        return block_ids[:0]

    resident_start = max(0, resident_block_end - len(block_ids))
    resident_end = resident_block_end
    chunk_start = chunk_block_offset
    chunk_end = chunk_start + chunk_block_count

    overlap_start = max(chunk_start, resident_start)
    overlap_end = min(chunk_end, resident_end)
    if overlap_start >= overlap_end:
        return block_ids[:0]

    local_start = overlap_start - resident_start
    local_end = overlap_end - resident_start
    return block_ids[local_start:local_end]


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


def derive_chunk_block_coords(
    token_range: Optional[TokenRange],
    tokens_per_block: int,
) -> tuple[int, int]:
    """Derive global chunk block offset and count from a block-aligned token_range."""
    if token_range is None:
        return 0, 0
    if tokens_per_block <= 0:
        raise ValueError("tokens_per_block must be positive")
    if token_range.start % tokens_per_block != 0 or token_range.end % tokens_per_block != 0:
        raise ValueError(
            f"token_range [{token_range.start}, {token_range.end}) must be "
            f"block-aligned with tokens_per_block={tokens_per_block}"
        )
    chunk_offset = token_range.start // tokens_per_block
    chunk_block_count = (token_range.end - token_range.start) // tokens_per_block
    return chunk_offset, chunk_block_count


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
    """A KV cache slice covering token_range = [start, end) of one request.

    Single-slice transfer uses [0, prompt_len) with is_last_slice=True;
    multi-slice (pipelined) transfers split token_range and mark the last slice.
    For pipelined chunks, token_range is block-aligned and encodes the global
    chunk position; derive block offset/count via derive_chunk_block_coords().

    Per-layer token starts are NOT encoded in token_range — they are derived
    from block count by the sender:
        total_blocks    = ceil(token_range.end / tpb)
        token_start_i   = (total_blocks - len(block_ids_per_layer_groups[i])) * tpb
    Cached prefix (full-attn or per-layer SWA) shows up only by shrinking the
    block list. Beam search keeps this field 1-D: beam 0's blocks first,
    followed by the final unshared block from each remaining beam.

    SWA stale_end uses the request prompt_len (on the session), not
    token_range.end — they differ for non-final slices.
    """

    token_range: Optional[TokenRange] = None
    layer_range: Optional[LayerRange] = None
    block_ids_per_layer_groups: List[np.ndarray] = field(
        default_factory=list
    )  # Physical block IDs per layer group, each np.ndarray(dtype=np.int64)
    is_last_slice: bool = False
    mamba_state_index: Optional[int] = None
    total_blocks: Optional[int] = None


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
