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
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

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


@set_api_status("prototype")
class SessionResult:
    """One result of a session.

    ``outputs`` maps each stream the pipeline produced to a tensor that views the
    worker's delivery slot; it is valid until the next result is requested. A
    terminal result (``done=True``) carries no outputs and ``error`` is ``None`` for
    normal completion or a description. ``advance_index`` is ``None`` for an initial
    result and for the terminal result.
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
        lease: Optional[Tuple[SessionChannel, int]] = None,
    ):
        self.request_id: int = request_id  # the session
        self.sequence: int = sequence  # delivery order within the session
        self.advance_index: Optional[int] = advance_index  # the submit that produced it
        self.position: Dict[str, Tuple[int, int]] = dict(position)  # per stream: start, count
        self.done: bool = done
        self.error: Optional[str] = error
        # Views over the worker's slot, or the caller's own memory after copy().
        # None once the slot went back to the worker: reading outputs then raises.
        self._outputs: Optional[Dict[str, torch.Tensor]] = dict(outputs)
        # Set while the media views a worker slot: the session's channel, asked at
        # access time whether the session is gone, and the slot to return. None
        # for a copy or a terminal, which own nothing on the worker.
        self._lease: Optional[Tuple[SessionChannel, int]] = lease

    @property
    def outputs(self) -> Dict[str, torch.Tensor]:
        if self._outputs is None:
            raise SessionLeaseExpired(
                f"result {self.sequence} of session {self.request_id} expired; "
                "call copy() before requesting the next result to keep its media"
            )
        if self._lease is not None and self._lease[0].lease_lost is not None:
            raise SessionLeaseExpired(
                f"result {self.sequence} of session {self.request_id} expired: "
                f"{self._lease[0].lease_lost}"
            )
        return self._outputs

    @property
    def video(self) -> Optional[torch.Tensor]:
        """``outputs["video"]`` when that stream exists."""
        return self.outputs.get("video")

    def copy(self) -> SessionResult:
        """Return this result with its media in memory the caller owns.

        The copies are complete on return, so the slot may be reused as soon as
        the next result is requested. Does not expire the view it was copied from.
        """
        outputs = self.outputs
        copies: Dict[str, torch.Tensor] = {}
        try:
            for name, tensor in outputs.items():
                copies[name] = tensor.clone()
        finally:
            # A device clone reads the slot asynchronously on the caller's current
            # stream; the worker reuses the slot with no view of that stream, so
            # every issued read must finish before the slot can be returned, on
            # success and on a failure partway through the streams alike.
            for device in {t.device for t in copies.values() if t.is_cuda}:
                torch.cuda.current_stream(device).synchronize()
        return SessionResult(
            self.request_id,
            self.sequence,
            self.advance_index,
            copies,
            self.position,
            self.done,
            self.error,
        )

    def release(self) -> Optional[int]:
        """Give up the lease.

        The views die; the result is the slot to return to the worker, or ``None``
        for a copy or a terminal, which hold no slot.
        """
        lease, self._lease = self._lease, None
        if lease is None:
            return None
        self._outputs = None
        return lease[1]

    def __repr__(self) -> str:
        streams = {name: tuple(t.shape) for name, t in (self._outputs or {}).items()}
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
            self._session._release_held()

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
        # The one result whose slot the client is holding; requesting the next
        # result, or closing, expires it and returns the slot.
        self._held: Optional[SessionResult] = None

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
                self._held = taken
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
                self._held = taken
                return taken
            await asyncio.wrap_future(self._channel.notification())

    # ---- close

    def close(self) -> None:
        """Cancel the session.

        Queued inputs and undelivered results are discarded, in-flight work runs to
        its boundary, state is released. Idempotent.
        """
        fut = self._client.session_close(self._channel)
        self._release_held()
        try:
            fut.result()
        except RuntimeError:
            # The worker is gone; the client-side state is released regardless.
            pass

    async def close_async(self) -> None:
        fut = self._client.session_close(self._channel)
        self._release_held()
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
        if not self._release_held():
            self._client.session_keep_alive(self._channel)

    def _release_held(self) -> bool:
        """Expire the held result and give its slot back to the worker.

        Returns whether a slot was returned.
        """
        held, self._held = self._held, None
        if held is None:
            return False
        slot_id = held.release()
        if slot_id is None or self._channel.lease_lost is not None:
            return False
        self._client.session_return_slot(self._channel, slot_id)
        return True

    def _build_result(self, message: SessionResultMessage) -> SessionResult:
        outputs: Dict[str, torch.Tensor] = {}
        if message.slot_id is not None:
            slot = self._channel.slots[message.slot_id]
            for name, (shape, dtype) in message.streams.items():
                outputs[name] = make_view(slot[name], tuple(shape), dtype)
        lease = None if message.slot_id is None else (self._channel, message.slot_id)
        return SessionResult(
            self.request_id,
            message.sequence,
            message.advance_index,
            outputs,
            message.position,
            message.type is ResultType.TERMINAL,
            message.error,
            lease=lease,
        )
