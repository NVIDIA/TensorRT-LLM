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
"""The session protocol between the coordinator and rank 0.

A session is one generation whose model state stays alive on the worker while it
produces results one advance at a time. This module holds what both sides share:
the commands and replies that cross the sockets and the errors a caller can see.
The worker's own bookkeeping is ``session_state.py``; the pipeline declares what
it accepts and produces through ``SessionCapabilities`` in ``pipeline.py`` and
never sees any of this.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest

# --------------------------------------------------------------------------- errors


class SessionError(RuntimeError):
    """Base class for session failures raised to the client."""


class SessionQueueFull(SessionError):
    """A queued input has no room; retry after the next advance."""


class SessionEnded(SessionError):
    """The session has a terminal result or is closing; no more input is admitted."""


class SessionLeaseExpired(SessionError):
    """A result's storage was returned to the worker; the view is no longer valid."""


# --------------------------------------------------------------------------- wire messages
#
# A delivery slot buffer crosses to the client as the handle SharedTensorContainer
# produces (``dump_to_dict``); the client maps it back with ``from_dict``.
SharedTensorHandle = Dict[str, Any]
SlotHandles = Dict[str, SharedTensorHandle]  # one buffer per stream name
#
# Commands travel from the coordinator to rank 0: SessionOpen and SessionSubmit on
# the request socket (they carry payloads), SessionControl on the control socket.
# Replies travel back on the response socket. Every message names its session.


@dataclass
class SessionCommand:
    """A message from the coordinator to rank 0 about one session; every subclass
    answers ``session_id``."""


@dataclass
class SessionOpen(SessionCommand):
    """Open a session; ``request.request_id`` is the session id."""

    request: DiffusionRequest

    @property
    def session_id(self) -> int:
        return self.request.request_id


@dataclass
class SessionSubmit(SessionCommand):
    """One value for the named input. A session has at most one unanswered submit."""

    session_id: int
    name: str
    value: object  # whatever the pipeline's validator accepts; opaque to the protocol


class ControlType(Enum):
    """The three verbs of the control socket."""

    SLOT_RETURN = "slot_return"  # the client is done with slot_id; the worker may reuse it
    KEEP_ALIVE = "keep_alive"  # the client is still there; restarts the worker's idle clock
    CLOSE = "close"  # close the session; answered by SessionClosed


@dataclass
class SessionControl(SessionCommand):
    """A control-socket command: small, payload-free, serviced ahead of input."""

    session_id: int
    type: ControlType
    slot_id: Optional[int] = None  # SLOT_RETURN only


# ---- replies


@dataclass
class SessionReply:
    """A message from rank 0 about one session; the client's router hands it to
    that session's channel."""

    session_id: int


@dataclass
class SessionOpened(SessionReply):
    """The answer to ``SessionOpen``: the declared capabilities and one shared-memory
    handle per slot per stream, or ``error`` with the executor-wide failure class
    (``"client"``, ``"capacity"`` or ``None``, as on ``DiffusionResponse``)."""

    capabilities: Optional[Dict[str, Any]] = None
    slots: Optional[List[SlotHandles]] = None  # indexed by slot id
    error: Optional[str] = None
    error_type: Optional[str] = None


class AdmissionFailure(Enum):
    """Why a submit was refused; each maps to one exception on the client."""

    INVALID = "invalid"  # the value failed the input's schema: ValueError
    QUEUE_FULL = "queue_full"  # no room until the next advance: SessionQueueFull
    ENDED = "ended"  # terminal reached, closing, or no such session: SessionEnded


@dataclass
class SessionAdmitted(SessionReply):
    """The answer to the one submit a session has in flight: the advance index the
    value was assigned, or ``error`` with its ``failure``."""

    advance_index: Optional[int] = None
    error: Optional[str] = None
    failure: Optional[AdmissionFailure] = None


class ResultType(Enum):
    """What an entry of a session's result stream is. The worker publishes
    ``MEDIA`` and ``TERMINAL``; ``ENDED`` and ``CLOSED`` exist only on the client,
    as the entry left at the head of its queue once the stream is over."""

    MEDIA = "media"  # a delivery slot holds the streams
    TERMINAL = "terminal"  # the session is over; ``error`` says why, ``None`` for normal completion
    ENDED = "ended"  # the terminal was taken: nothing more to read
    CLOSED = "closed"  # the client closed: the stream is discarded


@dataclass
class SessionResultMessage(SessionReply):
    """One entry of the result stream. For ``MEDIA``, ``streams`` maps a stream
    name to the ``(shape, dtype)`` written in ``slots[slot_id][name]`` and
    ``position`` maps it to ``(start, count)``. ``TERMINAL`` has ``slot_id=None``
    and no streams. The client-local kinds carry no sequence."""

    sequence: Optional[int]
    advance_index: Optional[int]
    slot_id: Optional[int]
    streams: Dict[str, Tuple[Tuple[int, ...], torch.dtype]]
    position: Dict[str, Tuple[int, int]]
    type: ResultType = ResultType.MEDIA
    error: Optional[str] = None

    @classmethod
    def end(cls, session_id: int, end_type: ResultType) -> SessionResultMessage:
        """The client-local entry that ends a stream: ``ENDED`` or ``CLOSED``."""
        if end_type not in (ResultType.ENDED, ResultType.CLOSED):
            raise ValueError(f"{end_type} is published by the worker, not an end of stream")
        return cls(session_id, None, None, None, {}, {}, type=end_type)


@dataclass
class SessionClosed(SessionReply):
    """The worker released the session: on the client's close, on the idle
    timeout, or after a terminal result."""
