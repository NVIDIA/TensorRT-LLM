# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

from .backend import Chunk


class SessionStatus(Enum):
    """Status of a transfer session.

    Represents the lifecycle stages of a KV cache transfer session:

    - INIT: Session initialized; waiting for the remote peer to become ready.
    - READY: Peer is ready; transfer can begin.
    - TRANSFERRING: Cache or auxiliary data is still moving.
    - TRANSFERRED: Everything this session owes has landed.
    - ERROR: A transfer error occurred; the session cannot complete.
    - CANCELLED: The session was explicitly cancelled before or during transfer.

    READY is a send-side state only. A sender has nowhere to write until the receiver has asked,
    so a receive session never reports it and goes from INIT straight to TRANSFERRING.

    A generation-first schedule owes auxiliary data as well as cache, and a session whose cache has
    landed while that is outstanding still reports TRANSFERRING.

    TODO: Retirement reads this rather than the handles, so the two ways of asking "is it done"
    still coexist.
    """

    INIT = "INIT"
    READY = "READY"
    TRANSFERRING = "TRANSFERRING"
    TRANSFERRED = "TRANSFERRED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class WaitResult(Enum):
    """Result of waiting for a transfer session to complete.

    A wait that has reached no conclusion answers ``None`` rather than a member.

    COMPLETED and FAILED are terminal; TIMEOUT is not. The transfer keeps running through one, so
    the session and its pages must be kept and the wait repeated. Only the send side reports
    TIMEOUT -- a receive that runs out of time reports FAILED.
    """

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass
class SessionArgsBase:
    """Base arguments for transfer sessions."""

    # Carries the ids both sides key their session tables by, and the schedule style that decides
    # whether a session still owes auxiliary data once its KV cache has landed.
    params: DisaggregatedParams
    # Captured from LlmRequest.prompt_len; needed for SWA stale_end derivation.
    prompt_len: int


def get_unique_rid(request: LlmRequest) -> Optional[int]:
    """This rank's own handle on a request, which has to equal ``py_request_id``.

    ``ctx_request_id`` is deliberately not in this chain. A session's own id does consult it, and
    answers a different question: this one keys local tables, so it has to agree with
    ``py_request_id``; a session's has to be the name the peer can resolve. A request carrying a
    context id but no disaggregated one therefore has two ids, both correct.

    TODO: Six expressions in this tree derive a request id, and only two of them are these.
    """
    if request.py_disaggregated_params:
        rid = request.py_disaggregated_params.disagg_request_id
        if rid is not None:
            return rid
    return request.request_id


class SenderBase(ABC):
    """Base class for sending KV cache data.

    Deliberately empty. It names the role so a session can be typed against its sender without this
    package importing a backend; what a session may ask of one is settled per backend.
    """

    ...


class ReceiverBase(ABC):
    """Base class for receiving KV cache data.

    Empty for the same reason as ``SenderBase``: a name to type against, with the protocol between
    a session and its receiver left to each backend.
    """

    ...


class _SessionBase(ABC):
    """Shared base for Tx/Rx sessions."""

    def __init__(self, args: SessionArgsBase):
        self._base_args = args

    @property
    def disagg_request_id(self) -> int:
        """The id the peer holds this session under too, so it is the one that goes on the wire.

        The cast assumes the field was set. A backend that admits requests without it has to
        override this with a fallback of its own.
        """
        return cast(int, self._base_args.params.disagg_request_id)

    @abstractmethod
    def is_completed(self) -> bool:
        """Non-blocking, and true only on success: a failed or cancelled session answers ``False``
        for good, so this is not the complement of still running."""
        ...

    @abstractmethod
    def wait_complete(self, blocking: bool = False) -> Optional[WaitResult]:
        """``None`` is a fourth answer rather than an error -- no conclusion yet, so the caller has
        to ask again on a later sweep instead of retiring the session."""
        ...

    @property
    @abstractmethod
    def exception(self) -> Optional[Exception]:
        """The error recorded against the session, if one was. ``None`` does not mean healthy: a
        cancelled session carries none, and one that failed through a task may not either."""
        ...

    @abstractmethod
    def close(self) -> bool:
        """Release what the session holds -- its registration with the sender or receiver, and any
        auxiliary slot. Not a cancel.

        ``True`` means it is closed; ``False`` means the implementation refused for now -- a peer
        is still writing -- and the caller is expected to come back. Callers compare against
        ``False`` rather than testing truthiness, so an implementation answering ``None`` would
        read as accepted every time.
        """
        ...


class TxSessionBase(_SessionBase):
    """The send side of one request. It exists before there is anywhere to write; the receiver has
    to ask first, which is what moves it out of INIT."""

    def __init__(self, sender: SenderBase, args: SessionArgsBase):
        super().__init__(args)
        self._sender = sender

    @abstractmethod
    def send(self, chunk: Chunk) -> None:
        """Send one piece.

        Args:
            chunk: which source blocks to send. ``token_range`` is the shared
                sender-side cursor; each layer group projects it into its own
                resident/windowed source and destination block ranges.
        """
        ...

    @abstractmethod
    def wait_complete(self, blocking: bool = True) -> Optional[WaitResult]:
        """Blocking by default: a send is waited on, where a receive is polled."""
        ...


class RxSessionBase(_SessionBase):
    """The receive side of one request. It is the side that asks: nothing moves until this session
    has told the peer where to write."""

    def __init__(self, receiver: ReceiverBase, args: SessionArgsBase):
        super().__init__(args)
        self._receiver = receiver

    @abstractmethod
    def receive(self, chunk: Chunk) -> None:
        """Post where one piece is to land. ``chunk`` carries destination blocks here, the mirror
        of the source blocks ``send`` takes.

        Pieces are addressed by the order they are posted in rather than by anything inside the
        chunk, so they cannot be posted out of order or skipped.
        """
        ...

    @abstractmethod
    def wait_complete(self, blocking: bool = False) -> Optional[WaitResult]:
        """Non-blocking by default, opposite the send side."""
        ...
