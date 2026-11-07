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
"""Session state owned by the executor's worker loop on rank 0: the input queues
and live values, the advance condition, the stream positions and the terminal
result. Built from the pipeline's ``SessionCapabilities``; speaks to the client
only through the messages of ``session_protocol.py``.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import torch

from .pipeline import SessionCapabilities
from .session_protocol import ResultType, SessionEnded, SessionQueueFull, SessionResultMessage


class SessionState:
    """Executor-side bookkeeping for one open session on rank 0.

    Holds the input buffers, the counters, the stream positions and whether it has ended, and
    answers the advance condition. The worker loop runs one advance at a time on
    one thread, so "in flight" and "initialized" are positions in that loop, not
    state kept here. It does not touch slots: the caller passes the
    free-slot count in, because the pool is a separate object with its own lifetime.
    """

    def __init__(self, session_id: int, capabilities: SessionCapabilities) -> None:
        self.session_id = session_id
        self.capabilities = capabilities
        # Per input, the values waiting for an advance, oldest first. A queue input
        # holds up to its queue depth and each advance consumes one; a live input
        # always holds exactly one value, the default until the first submit, which
        # later submits replace and every advance reads. A value is whatever the
        # pipeline's validator accepted: opaque here, only carried to advance().
        self.inputs: Dict[str, Deque[object]] = {
            s.name: deque([s.default]) if s.policy == "live" else deque()
            for s in capabilities.inputs
        }
        # Advances dispatched so far; the next one gets this index.
        self.next_advance = 0
        # Results published so far, the terminal included; the next one gets this number.
        self.next_sequence = 0
        # Per stream: where its next result starts, in that stream's units (frames, samples).
        self.stream_positions: Dict[str, int] = {s.name: 0 for s in capabilities.streams}
        # True once the terminal result is published: no more input or advances;
        # slot returns are still served until the client closes.
        self.ended = False

    # ---- admission

    def admit(self, name: str, value: object) -> int:
        """Validate and admit one value. Returns the advance index: for a queued input
        the advance that will consume the value; for a live input the next advance to
        be dispatched. Raises ``ValueError`` (the validator),
        ``SessionQueueFull`` or ``SessionEnded``; a rejection changes nothing."""
        if self.ended:
            raise SessionEnded(f"session {self.session_id}: a terminal result exists")
        spec = self.capabilities.input(name)
        if spec.validator is not None:
            spec.validator(value)
        buffer = self.inputs[name]
        if spec.policy == "live":
            buffer[0] = value
            return self.next_advance
        length = self.capabilities.length
        index = self.admitted_count(name)
        if length is not None and index >= length:
            raise SessionEnded(
                f"input {name!r}: all {length} values of a finite session are already queued"
            )
        if len(buffer) >= spec.queue_depth:
            raise SessionQueueFull(
                f"input {name!r}: queue holds {spec.queue_depth} values; retry after the next advance"
            )
        buffer.append(value)
        return index

    def admitted_count(self, name: str) -> int:
        """Values of a queued input admitted so far: every advance consumed one, the
        rest are waiting. Also the advance index the next value gets."""
        return self.next_advance + len(self.inputs[name])

    # ---- advance

    def can_advance(self, free_slots: int) -> bool:
        """The advance condition: the session has not ended and is not past a
        declared length, a delivery slot is free, and every queue input has a value."""
        if self.ended:
            return False
        if free_slots <= 0:
            return False
        length = self.capabilities.length
        if length is not None and self.next_advance >= length:
            return False
        return all(self.inputs[s.name] for s in self.capabilities.queue_inputs)

    def complete(self) -> bool:
        """A finite session has dispatched every advance it declared; the terminal
        result follows."""
        length = self.capabilities.length
        return length is not None and self.next_advance >= length

    def dispatch(self) -> Tuple[int, Dict[str, object]]:
        """Start the next advance: its index, and its inputs, one value consumed from
        every queue input and the current value of every live input."""
        index = self.next_advance
        self.next_advance = index + 1
        inputs = {
            s.name: self.inputs[s.name][0] if s.policy == "live" else self.inputs[s.name].popleft()
            for s in self.capabilities.inputs
        }
        return index, inputs

    # ---- results

    def media_result(
        self,
        advance_index: Optional[int],
        slot_id: int,
        written: Dict[str, Tuple[Tuple[int, ...], torch.dtype]],
    ) -> SessionResultMessage:
        """Build the message for the views the pipeline wrote (stream name to shape
        and dtype, shapes already checked by the buffer that handed them out).
        Each stream's start is its position, which then moves by the extent along
        the stream axis; an unknown stream name raises before anything moves."""
        caps = self.capabilities
        counts = {name: shape[caps.stream(name).axis] for name, (shape, _) in written.items()}
        position: Dict[str, Tuple[int, int]] = {}
        for name, count in counts.items():
            start = self.stream_positions[name]
            position[name] = (start, count)
            self.stream_positions[name] = start + count
        message = SessionResultMessage(
            session_id=self.session_id,
            sequence=self.next_sequence,
            advance_index=advance_index,
            slot_id=slot_id,
            streams=dict(written),
            position=position,
        )
        self.next_sequence += 1
        return message

    def terminal_result(self, error: Optional[str] = None) -> SessionResultMessage:
        """End the session: the one terminal result, ``error`` saying why if it did
        not complete normally. Published once; a second call is a bug."""
        if self.ended:
            raise RuntimeError(f"session {self.session_id}: the terminal result was already built")
        self.ended = True
        message = SessionResultMessage(
            session_id=self.session_id,
            sequence=self.next_sequence,
            advance_index=None,
            slot_id=None,
            streams={},
            position={},
            type=ResultType.TERMINAL,
            error=error,
        )
        self.next_sequence += 1
        return message

    def discard_inputs(self) -> None:
        """Drop every queued value; the close path, nothing will consume them."""
        for spec in self.capabilities.queue_inputs:
            self.inputs[spec.name].clear()
