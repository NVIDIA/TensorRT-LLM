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
"""Streaming sessions for VisualGen.

A session is a generation whose model state stays alive on the worker while it
delivers results one advance at a time and takes input while it runs.

Open one with :meth:`tensorrt_llm.visual_gen.VisualGen.open_session`. The session
delivers :class:`SessionResult` objects through :meth:`VisualGenSession.results`
or, for an interactive pipeline, through :meth:`VisualGenSession.advance`. Each
result is a lease on worker-owned storage that expires when the next result is
requested, when the iterator closes, or when the session closes. Call
:meth:`SessionResult.copy` before then to keep media.

A session is driven from one thread or one asyncio task. The methods are not
reentrant: a second submit while one waits, or a second result request while one
waits, raises :class:`SessionError` instead of blocking.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.visual_gen.session_protocol import (
    ResultType,
    SessionEnded,
    SessionError,
    SessionLeaseExpired,
    SessionQueueFull,
    SessionResultMessage,
)
from tensorrt_llm._torch.visual_gen.slot_pool import make_view
from tensorrt_llm.llmapi.utils import set_api_status

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRemoteClient, SessionChannel

__all__ = [
    "SessionEnded",
    "SessionError",
    "SessionLeaseExpired",
    "SessionQueueFull",
    "SessionResult",
    "SessionResultsIterator",
    "VisualGenSession",
]


@dataclass
class _SlotHold:
    """The session's hold on the delivery slot one result views.

    The session that issued the result owns the hold: it sets ``released`` and
    returns the slot to the worker when the next result is requested. The result
    only reads it.
    """

    channel: SessionChannel  # asked at access time whether the session is gone
    slot_id: int
    released: bool = False
    # The CUDA device the result's device-placed views live on, if it has any.
    device: Optional[torch.device] = None
    # The streams copy() queued its clones on.
    copy_streams: List[torch.cuda.Stream] = field(default_factory=list)


@set_api_status("prototype")
class SessionResult:
    """One result of a session.

    ``outputs`` maps each stream the pipeline produced to a tensor that views the
    worker's delivery slot; it is valid until the next result is requested.
    Requesting the next result never waits for the device: reads of a device
    tensor that are queued on the CUDA stream current at that moment run before
    the slot is written again. Reads queued on any other stream must have
    finished by then.

    A terminal result (``done=True``) carries no outputs and ``error`` is ``None``
    for normal completion or a description. ``advance_index`` is ``None`` for an
    initial result and for the terminal result.
    """

    def __init__(
        self,
        request_id: int,
        sequence: int,
        advance_index: Optional[int],
        outputs: Dict[str, torch.Tensor],
        position: Dict[str, Tuple[int, int]],
        done: bool,
        error: Optional[str],
        *,
        slot_hold: Optional[_SlotHold] = None,
    ):
        self.request_id: int = request_id  # the session
        self.sequence: int = sequence  # delivery order within the session
        self.advance_index: Optional[int] = advance_index  # the submit that produced it
        self.position: Dict[str, Tuple[int, int]] = dict(position)  # per stream: start, count
        self.done: bool = done
        self.error: Optional[str] = error
        # Views over the worker's slot, or the caller's own memory after copy().
        self._outputs: Dict[str, torch.Tensor] = dict(outputs)
        # The session's hold on the delivery slot the media views; reading outputs
        # raises once the session released it. None when outputs is the caller's
        # own memory (a copy) or empty (a terminal).
        self._slot_hold: Optional[_SlotHold] = slot_hold

    @property
    def outputs(self) -> Dict[str, torch.Tensor]:
        hold = self._slot_hold
        if hold is not None:
            if hold.released:
                raise SessionLeaseExpired(
                    f"result {self.sequence} of session {self.request_id} expired; "
                    "call copy() before requesting the next result to keep its media"
                )
            if hold.channel.lease_lost is not None:
                raise SessionLeaseExpired(
                    f"result {self.sequence} of session {self.request_id} expired: "
                    f"{hold.channel.lease_lost}"
                )
        return self._outputs

    @property
    def video(self) -> Optional[torch.Tensor]:
        """``outputs["video"]`` when that stream exists."""
        return self.outputs.get("video")

    def copy(self) -> SessionResult:
        """Return this result with its media in memory the caller owns.

        Does not wait for the device: a device stream is cloned on the caller's
        current CUDA stream like any tensor operation, and the slot is not
        written again until that clone has run. Does not expire the view it was
        copied from.
        """
        outputs = self.outputs
        hold = self._slot_hold
        if hold is not None and hold.device is not None:
            stream = torch.cuda.current_stream(hold.device)
            if stream not in hold.copy_streams:
                hold.copy_streams.append(stream)
        copies = {name: tensor.clone() for name, tensor in outputs.items()}
        return SessionResult(
            self.request_id,
            self.sequence,
            self.advance_index,
            copies,
            self.position,
            self.done,
            self.error,
        )

    def __repr__(self) -> str:
        streams = {name: tuple(t.shape) for name, t in self._outputs.items()}
        return (
            f"SessionResult(sequence={self.sequence}, advance_index={self.advance_index}, "
            f"streams={streams}, position={self.position}, done={self.done}, error={self.error!r})"
        )


@set_api_status("prototype")
class SessionResultsIterator:
    """The session's results, as a sync or async iterator.

    Use ``for r in it`` or ``async for r in it``. Requesting the next result expires
    the previous one and returns its slot to the worker. The terminal result is
    yielded once; the request after it ends the iteration. Closing the iterator (or
    leaving its context) releases the held result without closing the session; a
    bare ``break`` does not.
    """

    def __init__(self, session: VisualGenSession):
        self._session = session
        self._closed = False

    def __iter__(self) -> SessionResultsIterator:
        return self

    def __next__(self) -> SessionResult:
        self._check_open()
        result = self._session._next_result()
        if result is None:
            raise StopIteration
        return result

    def __aiter__(self) -> SessionResultsIterator:
        return self

    async def __anext__(self) -> SessionResult:
        self._check_open()
        result = await self._session._next_result_async()
        if result is None:
            raise StopAsyncIteration
        return result

    def _check_open(self) -> None:
        if self._closed:
            raise SessionError("results iterator is closed")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._session._release_hold()

    def __enter__(self) -> SessionResultsIterator:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    async def __aenter__(self) -> SessionResultsIterator:
        return self

    async def __aexit__(self, *exc) -> bool:
        self.close()
        return False


@set_api_status("prototype")
class VisualGenSession:
    """One open session on a model instance.

    Inputs are submitted by name. A queued input supplies one value per advance
    and the worker waits for it; a live input is a state value the worker reads
    at each advance. Results arrive in order through :meth:`results`; an
    interactive pipeline also offers :meth:`advance`, which submits one value and
    returns that advance's result. The session holds one result at a time: asking
    for the next one expires the previous. ``close`` ends the session and is also
    the context manager's exit.

    A session is driven from one thread or one asyncio task. One submit and one
    result request can be in progress at a time; a second raises
    :class:`SessionError` instead of waiting.
    """

    def __init__(self, client: DiffusionRemoteClient, channel: SessionChannel):
        self._client = client  # sends the session's commands to the worker
        self._channel = channel  # this session's record on the client: replies land here
        # The hold on the slot of the one result the client is reading; requesting
        # the next result, or closing, releases it and returns the slot.
        self._hold: Optional[_SlotHold] = None

    # ---- introspection

    @property
    def request_id(self) -> int:
        return self._channel.session_id

    @property
    def capabilities(self) -> Dict[str, Any]:
        """What the pipeline declared for this session.

        ``inputs`` (name, policy, default, queue_depth), ``streams``
        (name, kind, dtype, max_shape, axis, rate, placement), ``initial_result``,
        ``length``, ``lockstep``.
        """
        return self._channel.capabilities or {}

    @property
    def closed(self) -> bool:
        """Whether the session is over or ending.

        Closed by ``close()``, closed by the worker (idle timeout), or lost with
        the worker group.
        """
        return self._channel.closing() is not None

    # ---- inputs

    def submit(self, name: str, value: Any) -> int:
        """Admit one value and return an advance index.

        For a queued input the index is the advance that will consume the value;
        for a live input it is the next advance not yet dispatched when the value
        was admitted, so every advance from that index on reads this value or a
        later one for the same input. Raises ``ValueError`` (invalid input),
        ``SessionQueueFull`` (retry after the next advance) or ``SessionEnded``.
        """
        fut = self._client.session_submit(self._channel, name, value)
        try:
            return fut.result()
        finally:
            self._channel.release_submit(fut)

    async def submit_async(self, name: str, value: Any) -> int:
        fut = self._client.session_submit(self._channel, name, value)
        try:
            return await asyncio.wrap_future(fut)
        finally:
            self._channel.release_submit(fut)

    # ---- results

    def results(self) -> SessionResultsIterator:
        """An iterator over the results not yet requested, in order."""
        return SessionResultsIterator(self)

    def advance(self, value: Any) -> SessionResult:
        """Submit ``value`` to the queued input and return that advance's result.

        Interactive sessions only: the pipeline declared one queued input and one
        result per advance. The result request is reserved for the whole call,
        admission included, so no other result request runs meanwhile. Refuses
        while a declared initial result or the media of an earlier queued value
        has not been taken, even if it has not arrived yet. If the session has
        already ended, returns the terminal result once without submitting; if
        it ends before the submitted advance produces media, returns the
        terminal result instead. Later calls raise ``SessionEnded``.
        """
        name = self._advance_input()
        self._channel.reserve_read()
        try:
            self._keep_alive()
            fut = self._submit_for_advance(name, value)
            index = None
            if fut is not None:
                try:
                    index = fut.result()
                except SessionEnded:
                    index = None
                finally:
                    self._channel.release_submit(fut)
            return self._advance_result(index)
        finally:
            self._channel.release_read()

    async def advance_async(self, value: Any) -> SessionResult:
        name = self._advance_input()
        self._channel.reserve_read()
        try:
            self._keep_alive()
            fut = self._submit_for_advance(name, value)
            index = None
            if fut is not None:
                try:
                    index = await asyncio.wrap_future(fut)
                except SessionEnded:
                    index = None
                finally:
                    self._channel.release_submit(fut)
            return await self._advance_result_async(index)
        finally:
            self._channel.release_read()

    def _submit_for_advance(self, name: str, value: Any) -> Optional[Future]:
        """Submit for an advance.

        Returns the admission future, or ``None`` when nothing was submitted: a
        terminal is already next, or the session ended between the reservation
        and the publication (the reservation was withdrawn). Either way the
        result read that follows settles the call: the retained terminal once, or
        ``SessionEnded`` after an explicit close.
        """
        try:
            return self._client.session_submit(self._channel, name, value, for_advance=True)
        except SessionEnded:
            return None

    def _advance_input(self) -> str:
        caps = self.capabilities
        if not caps.get("lockstep"):
            raise SessionError(
                "advance() needs an interactive pipeline; use submit() and results()"
            )
        for spec in caps["inputs"]:
            if spec["policy"] == "queued":
                return spec["name"]
        raise SessionError("interactive session without a queued input")

    def _advance_result(self, index: Optional[int]) -> SessionResult:
        result = self._take_result(expected_index=index)
        if result is None:
            raise SessionEnded(f"session {self.request_id} has ended")
        return result

    async def _advance_result_async(self, index: Optional[int]) -> SessionResult:
        result = await self._take_result_async(expected_index=index)
        if result is None:
            raise SessionEnded(f"session {self.request_id} has ended")
        return result

    def _next_result(self) -> Optional[SessionResult]:
        """One read under the reservation.

        Returns the next result, or ``None`` at the end of the stream.
        """
        self._channel.reserve_read()
        try:
            self._keep_alive()
            return self._take_result()
        finally:
            self._channel.release_read()

    async def _next_result_async(self) -> Optional[SessionResult]:
        self._channel.reserve_read()
        try:
            self._keep_alive()
            return await self._take_result_async()
        finally:
            self._channel.release_read()

    def _take_result(self, expected_index: Optional[int] = None) -> Optional[SessionResult]:
        while True:
            taken = self._channel.take(self._build_result, expected_index)
            if taken is ResultType.ENDED:
                return None
            if taken is not None:
                return taken
            self._channel.notification().result()

    async def _take_result_async(
        self, expected_index: Optional[int] = None
    ) -> Optional[SessionResult]:
        while True:
            taken = self._channel.take(self._build_result, expected_index)
            if taken is ResultType.ENDED:
                return None
            if taken is not None:
                return taken
            await asyncio.wrap_future(self._channel.notification())

    # ---- close

    def close(self) -> None:
        """Cancel the session.

        Queued inputs and undelivered results are discarded, in-flight work runs to
        its boundary, state is released. Idempotent.
        """
        hold = self._expire_hold()
        fut = self._client.session_close(self._channel)
        self._return_slot(hold)
        try:
            fut.result()
        except RuntimeError:
            # The worker is gone; the client-side state is released regardless.
            pass

    async def close_async(self) -> None:
        hold = self._expire_hold()
        fut = self._client.session_close(self._channel)
        self._return_slot(hold)
        closing = asyncio.wrap_future(fut)
        try:
            await asyncio.shield(closing)
        except RuntimeError:
            pass
        except asyncio.CancelledError:
            closing.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            raise

    def __enter__(self) -> VisualGenSession:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    async def __aenter__(self) -> VisualGenSession:
        return self

    async def __aexit__(self, *exc) -> bool:
        await self.close_async()
        return False

    # ---- internals

    def _keep_alive(self) -> None:
        """Reset the worker's idle clock with the held slot's return, or with a bare keep-alive."""
        if self._hold is None:
            self._client.session_keep_alive(self._channel)
        else:
            self._release_hold()

    def _release_hold(self) -> None:
        """Expire the held result and give its slot back to the worker."""
        self._return_slot(self._expire_hold())

    def _expire_hold(self) -> Optional[_SlotHold]:
        """Expire the held result and make its slot safe to return.

        The device reads the caller queued on the slot run before the worker
        writes it again. Returns the hold whose slot is still to be returned,
        if any.
        """
        hold, self._hold = self._hold, None
        if hold is None:
            return None
        hold.released = True
        if self._channel.lease_lost is not None:
            return None
        event = self._channel.return_event(hold.slot_id)
        if event is not None and hold.device is not None:
            stream = torch.cuda.current_stream(hold.device)
            for other in hold.copy_streams:
                if other != stream:
                    stream.wait_stream(other)
            event.record(stream)
        return hold

    def _return_slot(self, hold: Optional[_SlotHold]) -> None:
        if hold is not None:
            self._client.session_return_slot(self._channel, hold.slot_id)

    def _build_result(self, message: SessionResultMessage) -> SessionResult:
        outputs: Dict[str, torch.Tensor] = {}
        if message.slot_id is not None:
            slot = self._channel.slots[message.slot_id]
            for name, (shape, dtype) in message.streams.items():
                outputs[name] = make_view(slot[name], tuple(shape), dtype)
        # The channel takes the result it has built, so the slot is held from here.
        slot_hold = None if message.slot_id is None else _SlotHold(self._channel, message.slot_id)
        if slot_hold is not None:
            slot_hold.device = next((t.device for t in outputs.values() if t.is_cuda), None)
        self._hold = slot_hold
        return SessionResult(
            self.request_id,
            message.sequence,
            message.advance_index,
            outputs,
            message.position,
            message.type is ResultType.TERMINAL,
            message.error,
            slot_hold=slot_hold,
        )
