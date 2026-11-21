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
"""The session protocol end to end in one process: a rank-0 ``DiffusionExecutor``
over fake sockets, a ``DiffusionRemoteClient`` fed by them, the public
``VisualGenSession`` on top, and a synthetic pipeline that counts advances. No GPU,
no worker processes, no model."""

import asyncio
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import pytest
import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.executor import (
    DiffusionExecutor,
    DiffusionRemoteClient,
    DiffusionRequest,
    DiffusionResponse,
    FatalRankError,
    SessionChannel,
    _SessionExit,
)
from tensorrt_llm._torch.visual_gen.pipeline import (
    AdvanceOutcome,
    SessionCapabilities,
    SessionInputSpec,
    SessionStreamSpec,
)
from tensorrt_llm._torch.visual_gen.session_protocol import (
    ControlType,
    ResultType,
    SessionAdmitted,
    SessionClosed,
    SessionControl,
    SessionEnded,
    SessionError,
    SessionLeaseExpired,
    SessionOpen,
    SessionOpened,
    SessionQueueFull,
    SessionReply,
    SessionResultMessage,
    SessionSubmit,
)
from tensorrt_llm.visual_gen.session import VisualGenSession

pytestmark = pytest.mark.cpu_only

FRAMES, H, W = 4, 2, 3


class CountingPipeline:
    """Lockstep when ``queued`` is set: one queued input ``controls`` and a live
    input ``camera``; each advance writes a frame block whose pixels encode the
    advance index, the control value and the camera value. With ``queued`` unset it
    is autonomous and ends after ``length`` advances."""

    supports_sessions = True
    default_generation_params: Dict[str, Any] = {}
    extra_param_specs: Dict[str, Any] = {}

    def __init__(self, *, queued=True, initial=True, length=None, fail_at=None, open_fails=False):
        self.queued = queued
        # When set, every advance waits on it before writing (tests gate the worker).
        self.advance_gate: Optional[threading.Event] = None
        self.initial = initial
        self.length = length
        self.fail_at = fail_at
        self.open_fails = open_fails
        self.opened = 0
        self.closed = 0
        self.advances = []

    def prepare_request(self, req):
        pass

    def classify_request_failure(self, exc):
        return "client" if isinstance(exc, ValueError) else None

    def session_capabilities(self, req):
        inputs = []
        if self.queued:
            inputs.append(SessionInputSpec(name="controls", validator=_non_negative, queue_depth=2))
        inputs.append(SessionInputSpec(name="camera", policy="live", default=0))
        return SessionCapabilities(
            inputs=tuple(inputs),
            streams=(
                SessionStreamSpec(
                    name="video", kind="video", dtype=torch.uint8, max_shape=(1, FRAMES, H, W, 3)
                ),
            ),
            initial_result=self.initial,
            length=self.length,
            lockstep=self.queued,
        )

    def open_session(self, req, initial_slot):
        if self.open_fails:
            raise ValueError("bad prompt")
        self.opened += 1
        state = SimpleNamespace(prompt=req.prompt[0], count=0)
        written = {}
        if initial_slot is not None:
            view = initial_slot["video"].view((1, 1, H, W, 3), torch.uint8)
            view.fill_(255)
            written["video"] = view
        return state, AdvanceOutcome(written=written)

    def validate_input(self, state, name, value):
        pass

    def advance(self, state, inputs, slot):
        if self.advance_gate is not None:
            self.advance_gate.wait()
        index = state.count
        state.count += 1
        if self.fail_at is not None and index == self.fail_at:
            raise RuntimeError("kernel exploded")
        self.advances.append(dict(inputs))
        control = inputs.get("controls", 0)
        camera = inputs.get("camera", 0)
        written = {}
        if slot is not None:
            view = slot["video"].view((1, FRAMES, H, W, 3), torch.uint8)
            view[..., 0] = index
            view[..., 1] = control
            view[..., 2] = camera
            written["video"] = view
        done = self.length is not None and state.count >= self.length
        return AdvanceOutcome(done=done, written=written)

    def close_session(self, state):
        self.closed += 1


def _non_negative(value):
    if not isinstance(value, int) or value < 0:
        raise ValueError("controls must be a non-negative int")


class FakeRequestSocket:
    """What the worker sees as ``requests_ipc``: ``get(timeout)`` and ``drain()``."""

    def __init__(self):
        self.q: "queue.Queue[Any]" = queue.Queue()

    def put_nowait(self, item):
        self.q.put(item)

    def get(self, timeout: Optional[float] = None):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            raise queue.Empty from None

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                return items


def _make_executor(pipeline, session_config=None, *, rank=0, world_size=1, in_client_process=True):
    ex = DiffusionExecutor.__new__(DiffusionExecutor)
    ex.pipeline = pipeline
    ex.rank = rank
    ex.world_size = world_size
    ex.device_id = rank
    ex.in_client_process = in_client_process  # True: same process as the client, local handles
    ex.response_queue = queue.Queue()
    ex.requests_ipc = FakeRequestSocket()
    ex.controls_ipc = FakeRequestSocket()
    ex.session_config = session_config
    return ex


def _make_client():
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._sessions = {}
    client._sessions_lock = threading.Lock()
    client.pending_requests = queue.Queue()
    client._controls = OrderedDict()
    client._controls_lock = threading.Lock()
    client._request_to_send = None
    client._worker_failure = None
    return client


class Harness:
    """Wires the client's outgoing queue to the executor's fake socket and the
    executor's responses to the client's router, each on its own thread, and runs
    the executor's session loop on a third."""

    def __init__(self, pipeline, session_config=None, *, forward_controls=True):
        self.pipeline = pipeline
        self.executor = _make_executor(pipeline, session_config)
        self.forward_controls = forward_controls  # False: the test drives the control path
        self.client = _make_client()
        self._stop = threading.Event()
        self.worker_error: Optional[BaseException] = None
        # Reply gates: while a kind is held, its messages queue up here in order
        # and are routed when released; other kinds keep flowing. Lets a test
        # observe "accepted but not yet delivered" deterministically.
        self._held_kinds: set = set()
        self.responses: list = []  # ordinary-request answers the worker sent
        self._held_replies: list = []
        self._hold_lock = threading.Lock()
        # Cleared by a test to keep the client's commands in pending_requests.
        self.send_gate = threading.Event()
        self.send_gate.set()
        # Cleared by a test to model a request socket that no longer accepts
        # submits: they are held here (never forwarded) while controls flow.
        self.payload_gate = threading.Event()
        self.payload_gate.set()
        self.held_payloads: list = []
        self.threads = [
            threading.Thread(target=self._forward_commands, daemon=True),
            threading.Thread(target=self._forward_controls, daemon=True),
            threading.Thread(target=self._forward_replies, daemon=True),
            threading.Thread(target=self._worker, daemon=True),
        ]
        for t in self.threads:
            t.start()

    def _forward_commands(self):
        while not self._stop.is_set():
            try:
                item = self.client.pending_requests.get(timeout=0.02)
            except queue.Empty:
                continue
            while not self.send_gate.wait(timeout=0.02):  # a test holds commands unsent
                if self._stop.is_set():
                    return
            if isinstance(item, SessionSubmit) and not self.payload_gate.is_set():
                self.held_payloads.append(item)  # the request socket is not taking payloads
                continue
            self.executor.requests_ipc.q.put(item)

    def _forward_controls(self):
        """What the client's control dispatcher does on a writable socket: send
        the first outbox entry, then retire it by key."""
        while not self._stop.is_set():
            if not self.forward_controls:
                time.sleep(0.02)
                continue
            with self.client._controls_lock:
                entry = next(iter(self.client._controls.items()), None)
            if entry is None:
                time.sleep(0.005)
                continue
            key, item = entry
            self.executor.controls_ipc.q.put(item)
            with self.client._controls_lock:
                if self.client._controls.get(key) is item:
                    del self.client._controls[key]

    def _forward_replies(self):
        while not self._stop.is_set():
            try:
                item = self.executor.response_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            with self._hold_lock:
                if self._held_kinds and isinstance(item, tuple(self._held_kinds)):
                    self._held_replies.append(item)  # released later, in order
                    continue
            if isinstance(item, SessionReply):
                self.client._route_session_reply(item)
            else:
                self.responses.append(item)  # an ordinary request's answer: kept for the test

    def hold(self, *kinds):
        with self._hold_lock:
            self._held_kinds.update(kinds)

    def release(self):
        with self._hold_lock:
            self._held_kinds.clear()
            held, self._held_replies = self._held_replies, []
        for item in held:
            self.client._route_session_reply(item)

    def held(self, kind) -> bool:
        with self._hold_lock:
            return any(isinstance(m, kind) for m in self._held_replies)

    def _worker(self):
        try:
            # serve_forever, reduced to the session path: one open, served to its close.
            while not self._stop.is_set():
                try:
                    message = self.executor.requests_ipc.get(timeout=0.02)
                except queue.Empty:
                    continue
                if message is None:
                    return
                if isinstance(message, SessionOpen):
                    if self.executor._serve_session(message.request) is _SessionExit.SHUTDOWN:
                        return
                else:
                    self.executor._answer_stray_command(message)
        except BaseException as e:  # surfaced by the test, not swallowed
            self.worker_error = e

    def open(self, prompt="p", timeout=10.0) -> VisualGenSession:
        req = DiffusionRequest(request_id=1, prompt=[prompt], params=SimpleNamespace())
        channel = self.client.open_session(req)
        channel.wait_open(timeout=timeout)
        return VisualGenSession(self.client, channel)

    def stop(self):
        gate = self.pipeline.advance_gate
        if gate is not None:
            gate.set()  # never leave the worker parked inside an advance
        self.send_gate.set()
        self.payload_gate.set()
        self.release()
        self._stop.set()
        # A worker still inside a session (a test that failed the channel without
        # closing the session) waits on its request socket and never sees the
        # stop flag; the shutdown sentinel is what ends its session loop.
        self.executor.requests_ipc.put_nowait(None)
        for t in self.threads:
            t.join(timeout=5.0)
        if self.worker_error is not None:
            raise self.worker_error
        alive = [t.name for t in self.threads if t.is_alive()]
        assert not alive, f"harness threads still running: {alive}"

    def __enter__(self) -> "Harness":
        return self

    def __exit__(self, *exc) -> bool:
        # Threads end inside the test body: the thread-leak check runs before
        # fixture teardown.
        self.stop()
        return False


@pytest.fixture
def harness():
    return Harness


def _pixels(result) -> Tuple[int, int, int]:
    video = result.outputs["video"]
    return tuple(int(x) for x in video[0, 0, 0, 0])


# --------------------------------------------------------------------------- lockstep


def test_lockstep_session_delivers_one_result_per_advance(harness):
    pipe = CountingPipeline()
    with harness(pipe) as h:
        with h.open() as s:
            caps = s.capabilities
            assert caps["lockstep"] and caps["initial_result"] and caps["length"] is None
            with s.results() as results:
                first = next(results)  # the declared initial result
                assert first.advance_index is None and first.sequence == 0
                assert first.outputs["video"].shape == (1, 1, H, W, 3)
                assert int(first.outputs["video"][0, 0, 0, 0, 0]) == 255
                assert first.position == {"video": (0, 1)}
            for value in (5, 6, 7):
                r = s.advance(value)
                assert r.advance_index == value - 5
                assert r.position["video"] == (1 + FRAMES * (value - 5), FRAMES)
                assert _pixels(r) == (value - 5, value, 0)
            assert [a["controls"] for a in pipe.advances] == [5, 6, 7]
        assert pipe.opened == 1 and pipe.closed == 1


def test_submit_returns_indices_and_results_carry_them(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        with h.open() as s, s.results() as results:
            assert s.submit("controls", 1) == 0
            assert s.submit("controls", 2) == 1
            r0 = next(results)
            assert r0.advance_index == 0 and _pixels(r0) == (0, 1, 0)
            r1 = next(results)
            assert r1.advance_index == 1 and _pixels(r1) == (1, 2, 0)


def test_live_input_index_is_the_next_dispatch_and_newest_wins(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        with h.open() as s, s.results() as results:
            first_boundary = s.submit("camera", 3)
            second_boundary = s.submit("camera", 4)  # overwrites 3 before any dispatch
            assert first_boundary == second_boundary == 0
            s.submit("controls", 1)
            r = next(results)
            assert r.advance_index == 0 and _pixels(r) == (0, 1, 4)
            later = s.submit("camera", 9)
            assert later >= 1
            s.submit("controls", 2)
            r = next(results)
            assert _pixels(r) == (1, 2, 9)


def test_queue_full_and_invalid_input_errors(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        with h.open() as s:
            # The consumer never asks for a result, so after the first advance fills the
            # only free slot (K=2 with one taken by nothing: two advances run), the
            # queue of depth 2 fills up.
            s.submit("controls", 1)
            s.submit("controls", 2)
            time.sleep(0.2)  # let the worker consume what it can with 2 slots
            s.submit("controls", 3)
            s.submit("controls", 4)
            with pytest.raises(SessionQueueFull):
                s.submit("controls", 5)
            with pytest.raises(ValueError):
                s.submit("controls", -1)
            with pytest.raises(ValueError):
                s.submit("unknown", 1)


def test_backpressure_pauses_the_worker_until_a_slot_returns(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe, SimpleNamespace(delivery_slots=1, idle_timeout=None)) as h:
        with h.open() as s, s.results() as results:
            s.submit("controls", 1)
            s.submit("controls", 2)
            time.sleep(0.2)
            assert len(pipe.advances) == 1  # the single slot is published and unread
            r0 = next(results)
            assert r0.advance_index == 0
            time.sleep(0.2)
            assert len(pipe.advances) == 1  # r0 still holds the slot
            r1 = next(results)  # returns slot 0, advance 1 runs
            assert r1.advance_index == 1
            with pytest.raises(SessionLeaseExpired):
                r0.outputs


def test_copy_survives_the_next_request(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open() as s, s.results() as results:
            s.submit("controls", 1)
            s.submit("controls", 2)
            r0 = next(results)
            keep = r0.copy()
            next(results)
            assert _pixels(keep) == (0, 1, 0)
            with pytest.raises(SessionLeaseExpired):
                r0.video


# --------------------------------------------------------------------------- autonomous


def test_autonomous_session_runs_to_its_terminal(harness):
    pipe = CountingPipeline(queued=False, initial=False, length=3)
    with harness(pipe) as h:
        with h.open() as s:
            assert not s.capabilities["lockstep"]
            with pytest.raises(SessionError):
                s.advance(1)
            seen = []
            with s.results() as results:
                for r in results:
                    if r.done:
                        assert r.error is None
                        seen.append("done")
                        break
                    seen.append(r.advance_index)
            assert seen == [0, 1, 2, "done"]
            with pytest.raises(SessionEnded):
                s.submit("camera", 1)


def test_iterator_ends_after_the_terminal_result(harness):
    with harness(CountingPipeline(queued=False, initial=False, length=1)) as h:
        with h.open() as s:
            results = s.results()
            media = next(results)
            terminal = next(results)
            assert not media.done and terminal.done
            with pytest.raises(StopIteration):
                next(results)
            results.close()


def test_advance_returns_the_terminal_once_then_raises(harness):
    with harness(CountingPipeline(initial=False, length=1)) as h:
        with h.open() as s:
            r = s.advance(1)
            assert r.advance_index == 0
            terminal = s.advance(2)  # session already ended: terminal, nothing submitted
            assert terminal.done and terminal.error is None
            with pytest.raises(SessionEnded):
                s.advance(3)


# --------------------------------------------------------------------------- errors


def test_open_failure_raises_with_the_failure_class(harness):
    with harness(CountingPipeline(open_fails=True)) as h:
        with pytest.raises(ValueError, match="bad prompt"):
            h.open()


def test_unsupported_pipeline_refuses_a_session_as_a_client_error(harness):
    class NoSessions(CountingPipeline):
        supports_sessions = False

        def classify_request_failure(self, exc):
            return None  # the base pipeline leaves every failure unclassified

    pipe = NoSessions()
    with harness(pipe) as h:
        with pytest.raises(ValueError, match="does not support sessions"):
            h.open()
    assert pipe.opened == 0


def test_execution_error_ends_the_session_with_a_terminal_error(harness):
    pipe = CountingPipeline(initial=False, fail_at=1)
    with harness(pipe) as h:
        with h.open() as s, s.results() as results:
            s.submit("controls", 1)
            s.submit("controls", 2)
            r0 = next(results)
            terminal = next(results)
            assert r0.advance_index == 0
            assert terminal.done and "kernel exploded" in terminal.error
            with pytest.raises(SessionEnded):
                s.submit("controls", 3)
        assert pipe.closed == 1


def test_idle_timeout_ends_an_abandoned_session(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe, SimpleNamespace(delivery_slots=2, idle_timeout=0.2)) as h:
        s = h.open()
        results = s.results()
        terminal = next(results)  # nothing submitted: the clock runs, the timeout fires
        assert terminal.done and "idle timeout" in terminal.error
        deadline = time.monotonic() + 5
        while pipe.closed == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipe.closed == 1  # model instance released without a close from the client
        s.close()


def test_client_send_loop_forwards_session_commands():
    """Session commands travel through the same send loop as requests; the loop
    must not assume every outgoing object is a ``DiffusionRequest``."""
    client = _make_client()
    client._request_to_send = None
    client.requests_ipc = FakeRequestSocket()
    req = DiffusionRequest(request_id=7, prompt=["p"], params=SimpleNamespace())
    client.pending_requests.put(SessionOpen(req))
    client.pending_requests.put(SessionSubmit(7, "controls", 1))
    client._process_requests()
    client._process_requests()
    sent = client.requests_ipc.drain()
    assert [type(m) for m in sent] == [SessionOpen, SessionSubmit]
    assert client._request_to_send is None


def test_one_result_request_and_one_submit_at_a_time(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open() as s:
            results = s.results()
            errors = []

            def wait_for_result():
                try:
                    next(results)
                except SessionError as exc:
                    errors.append(exc)

            # Nothing is submitted, so the first request blocks inside the channel;
            # a second request from another thread must be refused, not queued.
            t = threading.Thread(target=wait_for_result)
            t.start()
            time.sleep(0.1)
            with pytest.raises(SessionError, match="already in progress"):
                next(results)
            with pytest.raises(SessionError, match="already in progress"):
                s.advance(1)
            s.submit("controls", 1)
            t.join(timeout=10)
            assert not errors
            # advance() and the iterator share the one held result.
            r = s.advance(2)
            assert int(r.video[0, 0, 0, 0, 1]) == 2
            results.close()
            with pytest.raises(SessionLeaseExpired):
                r.outputs


def test_advance_refuses_while_the_initial_result_is_unread(harness):
    with harness(CountingPipeline(initial=True)) as h:
        with h.open() as s:
            time.sleep(0.1)
            with pytest.raises(SessionError, match="unread"):
                s.advance(1)


def test_close_discards_queued_inputs_and_second_session_opens(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        s = h.open()
        s.submit("controls", 1)
        s.close()
        s.close()  # idempotent
        with pytest.raises(SessionError):
            s.submit("controls", 2)
        deadline = time.monotonic() + 5
        while pipe.closed == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipe.closed == 1
        # The model instance is free again.
        req = DiffusionRequest(request_id=2, prompt=["q"], params=SimpleNamespace())
        channel = h.client.open_session(req)
        channel.wait_open(timeout=10)
        VisualGenSession(h.client, channel).close()
        assert pipe.opened == 2


# --------------------------------------------------------------------------- async


def test_async_forms(harness):
    pipe = CountingPipeline(initial=False, length=2)
    with harness(pipe) as h:

        async def run():
            req = DiffusionRequest(request_id=1, prompt=["p"], params=SimpleNamespace())
            channel = h.client.open_session(req)
            await channel.wait_open_async()
            s = VisualGenSession(h.client, channel)
            async with s, s.results() as results:
                assert await s.submit_async("controls", 1) == 0
                assert await s.submit_async("controls", 2) == 1
                seen = []
                async for r in results:
                    if r.done:
                        seen.append("done")
                        break
                    seen.append(_pixels(r))
                assert seen == [(0, 1, 0), (1, 2, 0), "done"]

        asyncio.run(run())
        assert pipe.closed == 1


# --------------------------------------------------------------------------- worker failure


def test_worker_failure_raises_to_the_client_instead_of_a_terminal_result(harness):
    """A dead worker group is not a session outcome: the parked result request,
    the submit awaiting admission and every later call raise the failure; unread
    results are dropped and the held view expires."""
    with harness(CountingPipeline(initial=True)) as h:
        s = h.open()
        results = s.results()
        time.sleep(0.1)
        held = next(results)  # the initial result, a view over the worker's slot
        assert held.video is not None
        s.submit("controls", 1)  # produces a result nobody asks for
        time.sleep(0.2)
        channel = h.client._sessions[s.request_id]
        assert channel.unread_media()
        caught = []

        def wait_for_result():
            try:
                next(results)
            except BaseException as exc:  # noqa: BLE001  (the test inspects the type)
                caught.append(exc)

        channel.fail(
            RuntimeError("DiffusionClient: local worker processes exited: pid=1, exitcode=1")
        )
        with pytest.raises(SessionLeaseExpired, match="exited"):
            held.outputs  # the held view expired with its producer
        assert not channel.unread_media()  # the unread result was dropped, not served
        t = threading.Thread(target=wait_for_result)
        t.start()
        t.join(timeout=10)
        assert len(caught) == 1 and isinstance(caught[0], RuntimeError)
        assert not isinstance(caught[0], SessionError)
        assert "exited" in str(caught[0])
        with pytest.raises(RuntimeError, match="exited"):
            s.submit("controls", 2)
        with pytest.raises(RuntimeError, match="exited"):
            s.advance(2)
        s.close()  # the close future carries the failure; close() swallows it
        assert s.closed


def test_worker_failure_wakes_a_parked_request(harness):
    with harness(CountingPipeline(initial=False)) as h:
        s = h.open()
        results = s.results()
        caught = []

        def wait_for_result():
            try:
                next(results)
            except BaseException as exc:  # noqa: BLE001
                caught.append(exc)

        t = threading.Thread(target=wait_for_result)
        t.start()
        time.sleep(0.1)  # nothing submitted: the request is parked in the channel
        h.client._sessions[s.request_id].fail(RuntimeError("worker died"))
        t.join(timeout=10)
        assert len(caught) == 1 and isinstance(caught[0], RuntimeError)
        s.close()


def test_liveness_check_sees_a_dead_in_process_worker_thread():
    """External launch runs rank 0 as a thread in the client's process; its end
    must be reported like a dead worker process so the abort path runs."""
    client = _make_client()
    client._monitor_worker_liveness = True
    client._shutdown_started = False
    client.worker_processes = []
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    client._ext_worker_thread = t
    failure = client._check_worker_liveness()
    assert failure is not None and "external-launch worker thread exited" in failure


# --------------------------------------------------------------------------- two ranks, one throws
#
# Two ranks on the gloo backend with a short collective timeout. The pipeline's open
# and advance each enter an all-reduce; a failing rank raises either before it (a
# pipeline-call failure, phases ``open`` / ``advance``) or after it, from the
# completion event in ``_finalize_advance`` (an asynchronous device error, phases
# ``open_event`` / ``advance_event``). Either way the other rank is inside a
# collective the failing rank will never join. Required: the failing rank enters no
# further collective (no status all-reduce after a pipeline failure, no CLOSE
# broadcast ever) and ends its worker at once; as a thread in the client's process
# it raises FatalRankError, as a dedicated process it exits 1. The surviving rank
# hits the collective timeout, which is the same rank-local exception, and takes
# the same path.

COLLECTIVE_TIMEOUT_S = 3.0


class _FailingEvent:
    """Stands in for the CUDA event of an advance whose device work failed."""

    def __init__(self, events, phase):
        self.events = events
        self.phase = phase

    def synchronize(self):
        self.events.append(f"raise:{self.phase}")
        raise RuntimeError(f"device error surfaced at the {self.phase} event")


class CollectivePipeline(CountingPipeline):
    def __init__(self, rank, fail_rank, fail_phase, events):
        super().__init__(initial=False)
        self.rank = rank
        self.fail_rank = fail_rank
        self.fail_phase = fail_phase
        self.events = events

    def _collective(self, phase):
        if self.rank == self.fail_rank and self.fail_phase == phase:
            self.events.append(f"raise:{phase}")
            raise RuntimeError(f"rank {self.rank} failed in {phase}")
        self.events.append(f"enter:{phase}")
        dist.all_reduce(torch.zeros(1))
        self.events.append(f"leave:{phase}")

    def _outcome(self, phase, outcome):
        if self.rank == self.fail_rank and self.fail_phase == f"{phase}_event":
            outcome.event = _FailingEvent(self.events, f"{phase}_event")
        return outcome

    def open_session(self, req, initial_slot):
        self._collective("open")
        return SimpleNamespace(count=0), self._outcome("open", AdvanceOutcome())

    def advance(self, state, inputs, slot):
        self._collective("advance")
        return self._outcome("advance", super().advance(state, inputs, slot))


class _EventLog(list):
    """Appends go to a file too, so the parent can read a dead child's events."""

    def __init__(self, path=None):
        super().__init__()
        self.path = path

    def append(self, item):
        entry = (time.time(), item)
        super().append(entry)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()


def _instrument(ex, events):
    """Record every status all-reduce and tag broadcast the rank performs."""
    finalize, send, recv = ex._finalize_advance, ex._send_tag, ex._recv_tag

    def _finalize(outcome):
        events.append("status")
        return finalize(outcome)

    def _send(tag, payload=None):
        events.append("tag:" + tag.value)
        return send(tag, payload)

    def _recv():
        events.append("tag:recv")
        return recv()

    ex._finalize_advance = _finalize
    ex._send_tag = _send
    ex._recv_tag = _recv


def _session_request():
    return DiffusionRequest(request_id=1, prompt=["p"], params=SimpleNamespace())


IMPORT_WAIT_S = 600.0  # a cold import of the package from network storage takes minutes


def _wait_for(path, timeout_s=IMPORT_WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{path} never appeared")
        time.sleep(0.02)


def _run_rank(rank, port, fail_rank, fail_phase, events, in_client_process, holder, timeout_s):
    """One rank's whole life for the test: join the group, run the session. The
    executor is handed out through ``holder`` before the run so its published
    messages can be inspected after it raised. Callers line the ranks up before
    this (see ``_RankProcess.ready``) so the short collective timeout never
    covers a peer that is still importing."""
    # Join with a generous timeout (the peer may still be setting up), confirm
    # both are in with a barrier, then arm the short timeout the test is about.
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=IMPORT_WAIT_S),
    )
    dist.barrier()
    torch.distributed.distributed_c10d._set_pg_timeout(
        timedelta(seconds=timeout_s), dist.group.WORLD
    )
    pipe = CollectivePipeline(rank, fail_rank, fail_phase, events)
    ex = _make_executor(pipe, rank=rank, world_size=2, in_client_process=in_client_process)
    _instrument(ex, events)
    holder["executor"] = ex
    req = _session_request()
    if rank == 0:
        ex.requests_ipc.put_nowait(SessionSubmit(1, "controls", 1))
    ex._serve_session(req)


def _child_main(argv):
    """Entry of a rank process: ``python -c`` loads this file by path. Reports
    that its imports are done, then waits for the parent's go before joining the
    process group."""
    rank, port, fail_rank, fail_phase, events_path, timeout_s = argv
    with open(events_path + ".ready", "w"):
        pass
    _wait_for(events_path + ".go")
    events = _EventLog(events_path)
    _run_rank(int(rank), int(port), int(fail_rank), fail_phase, events, False, {}, float(timeout_s))
    events.append("returned")  # only a coordinated path gets here


_CHILD_CODE = """
import importlib.util, os, sys
boot = os.environ.get("SESSION_TEST_BOOTSTRAP")
if boot:
    exec(compile(open(boot).read(), boot, "exec"))
spec = importlib.util.spec_from_file_location("session_rank_child", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod._child_main(sys.argv[2:])
"""


class _RankProcess:
    """A rank in its own python process, with the ``multiprocessing.Process``
    surface the coordinator's liveness check and abort path use."""

    def __init__(self, rank, port, fail_rank, fail_phase, tmp_path, timeout_s):
        self.events_path = tmp_path / f"rank{rank}.events"
        self.stderr_path = tmp_path / f"rank{rank}.stderr"
        with open(self.stderr_path, "w") as stderr:
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _CHILD_CODE,
                    __file__,
                    str(rank),
                    str(port),
                    str(fail_rank),
                    fail_phase,
                    str(self.events_path),
                    str(timeout_s),
                ],
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )
        self.exited_at = None
        self._watch = threading.Thread(target=self._wait, daemon=True)
        self._watch.start()

    def _wait(self):
        self._proc.wait()
        self.exited_at = time.time()

    def ready(self):
        """Block until the process has imported everything; raises if it died."""
        deadline = time.monotonic() + IMPORT_WAIT_S
        while not os.path.exists(f"{self.events_path}.ready"):
            assert self.is_alive(), self.stderr()
            assert time.monotonic() < deadline, "rank process never got through its imports"
            time.sleep(0.05)

    def go(self):
        with open(f"{self.events_path}.go", "w"):
            pass

    @property
    def pid(self):
        return self._proc.pid

    @property
    def exitcode(self):
        return self._proc.poll()

    def is_alive(self):
        return self._proc.poll() is None

    def join(self, timeout=None):
        self._watch.join(timeout)

    def terminate(self):
        self._proc.terminate()

    def kill(self):
        self._proc.kill()

    def events(self):
        if not os.path.exists(self.events_path):
            return []
        return [tuple(json.loads(line)) for line in open(self.events_path) if line.strip()]

    def stderr(self):
        return self.stderr_path.read_text()[-2000:]


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _names(events):
    return [name for _, name in events]


def _after(events, marker):
    names = _names(events)
    return names[names.index(marker) + 1 :] if marker in names else names


def _published(ex):
    out = []
    while not ex.response_queue.empty():
        out.append(ex.response_queue.get_nowait())
    return out


@pytest.mark.parametrize("fail_phase", ["open", "advance", "open_event", "advance_event"])
@pytest.mark.parametrize("fail_rank", [0, 1])
def test_rank_local_exception_is_fatal_not_a_hung_collective(tmp_path, fail_rank, fail_phase):
    """Rank 0 as a thread in this process (the external-launch shape), rank 1 as a
    dedicated process."""
    port = _free_port()
    child = _RankProcess(1, port, fail_rank, fail_phase, tmp_path, COLLECTIVE_TIMEOUT_S)
    child.ready()
    rank0_events = _EventLog()
    outcome = {}

    def rank0():
        child.go()
        try:
            _run_rank(
                0, port, fail_rank, fail_phase, rank0_events, True, outcome, COLLECTIVE_TIMEOUT_S
            )
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc
        outcome["ended"] = time.time()

    t = threading.Thread(target=rank0, daemon=True)
    t.start()
    try:
        t.join(timeout=120)
        assert not t.is_alive(), "rank 0 hung"
        child.join(timeout=60)
        assert not child.is_alive(), "rank 1 hung"
    finally:
        if child.is_alive():
            child.kill()
        if dist.is_initialized():
            dist.destroy_process_group()
    rank1_events = child.events()

    # The in-process rank 0 raised and the caller (this test) is alive; the
    # dedicated rank-1 process exited with status 1.
    assert isinstance(outcome.get("error"), FatalRankError), outcome
    assert child.exitcode == 1, child.stderr()

    # The failing rank entered no further collective after raising ...
    failing = rank0_events if fail_rank == 0 else rank1_events
    assert f"raise:{fail_phase}" in _names(failing)
    assert _after(failing, f"raise:{fail_phase}") == []
    # ... and left well inside the collective timeout, so it waited in none.
    raise_ts = next(ts for ts, n in failing if n == f"raise:{fail_phase}")
    left_ts = outcome["ended"] if fail_rank == 0 else child.exited_at
    assert left_ts - raise_ts < COLLECTIVE_TIMEOUT_S * 0.75

    # The surviving rank was inside a collective (the pipeline's, or the status
    # barrier after an event failure), got the timeout there and took the same
    # fatal path: nothing after it, in particular no CLOSE.
    surviving = rank1_events if fail_rank == 0 else rank0_events
    base = fail_phase.removesuffix("_event")
    last = "status" if fail_phase.endswith("_event") else f"enter:{base}"
    assert _names(surviving)[-1] == last, _names(surviving)
    assert "tag:close" not in _names(surviving)

    # Nothing coordinated was published by rank 0: no SessionClosed, no terminal.
    published = _published(outcome["executor"])
    assert not any(isinstance(m, SessionClosed) for m in published)
    assert not any(
        isinstance(m, SessionResultMessage) and m.type is ResultType.TERMINAL for m in published
    )
    if fail_phase.startswith("advance"):
        assert any(isinstance(m, SessionOpened) for m in published)


def test_coordinator_contains_the_group_when_a_rank_dies(tmp_path):
    """Both ranks as dedicated processes under a client whose liveness check runs
    the way the background thread runs it. Rank 1 dies in its advance; the
    coordinator notices, fails the public session (the parked waiter raises) and
    kills rank 0, which is inside the pipeline collective, long before that
    collective's timeout. Only liveness is wired to the workers here; the
    session channel is the client's own, so the public failure comes from the
    abort path and nothing else."""
    port = _free_port()
    timeout_s = 20.0
    rank0 = _RankProcess(0, port, 1, "advance", tmp_path, timeout_s)
    rank1 = _RankProcess(1, port, 1, "advance", tmp_path, timeout_s)
    rank0.ready()
    rank1.ready()
    rank0.go()
    rank1.go()

    client = _make_client()
    client._monitor_worker_liveness = True
    client._shutdown_started = False
    client._ext_worker_thread = None
    client._request_to_send = None
    client.shutdown_event = threading.Event()
    client.response_event = threading.Event()
    client.worker_processes = [rank0, rank1]
    channel = SessionChannel(1)
    client._sessions[1] = channel
    channel.open_succeeded(CountingPipeline().session_capabilities(None).public(), [])
    session = VisualGenSession(client, channel)
    results = session.results()
    waiter = {}

    def wait_for_result():
        try:
            next(results)
        except BaseException as exc:  # noqa: BLE001
            waiter["error"] = exc
        waiter["ended"] = time.time()

    def poll_liveness():
        while not client.shutdown_event.is_set():
            failure = client._check_worker_liveness()
            if failure is not None:
                client._abort_worker_group(failure)
                return
            time.sleep(0.01)

    threading.Thread(target=wait_for_result, daemon=True).start()
    poller = threading.Thread(target=poll_liveness, daemon=True)
    poller.start()
    try:
        poller.join(timeout=120)
        assert not poller.is_alive(), "the coordinator never saw a dead worker"
        rank0.join(timeout=60)
        assert not rank0.is_alive(), "the surviving rank was not killed"
    finally:
        for proc in (rank0, rank1):
            if proc.is_alive():
                proc.kill()

    assert rank1.exitcode == 1, rank1.stderr()
    # The survivor ends one of two ways, both inside the bound: killed by the
    # coordinator (negative exit), or, where the transport reports the dead peer
    # at once (Linux gloo), by its own fatal exit from the collective (exit 1).
    # Never a normal return.
    assert rank0.exitcode != 0, rank0.stderr()
    raise_ts = next(ts for ts, n in rank1.events() if n == "raise:advance")
    assert rank0.exited_at - raise_ts < timeout_s * 0.5  # contained before any collective timeout
    assert _names(rank0.events())[-1] == "enter:advance"  # ended inside the collective
    assert "exited" in client._worker_failure
    assert isinstance(waiter.get("error"), RuntimeError) and "exited" in str(waiter["error"])
    assert waiter["ended"] - raise_ts < timeout_s * 0.5
    with pytest.raises(RuntimeError, match="exited"):
        session.submit("controls", 2)


# --------------------------------------------------------------------------- close and idle timeout


def _closed(channel) -> bool:
    """The worker confirmed the close (or the group failed)."""
    close = channel.closing()
    return close is not None and close.done()


def _terminal_next(channel) -> bool:
    head = channel._results[0] if channel._results else None
    return head is not None and head.type is ResultType.TERMINAL


def _wait_until(predicate, timeout=10.0, what="condition"):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def test_close_settles_a_parked_reader(harness):
    with harness(CountingPipeline(initial=False)) as h:
        s = h.open()
        results = s.results()
        caught = []

        def read():
            try:
                next(results)
            except BaseException as exc:  # noqa: BLE001
                caught.append(exc)

        t = threading.Thread(target=read)
        t.start()
        time.sleep(0.1)  # nothing submitted: parked
        s.close()
        t.join(timeout=10)
        assert not t.is_alive()
        assert len(caught) == 1 and isinstance(caught[0], SessionEnded)
        assert s.closed


def test_close_async_settles_a_parked_reader(harness):
    with harness(CountingPipeline(initial=False)) as h:

        async def run():
            req = DiffusionRequest(request_id=1, prompt=["p"], params=SimpleNamespace())
            channel = h.client.open_session(req)
            await channel.wait_open_async()
            s = VisualGenSession(h.client, channel)
            results = s.results()
            reader = asyncio.ensure_future(results.__anext__())
            await asyncio.sleep(0.1)
            assert not reader.done()
            await s.close_async()
            with pytest.raises(SessionEnded):
                await asyncio.wait_for(reader, 10)
            assert s.closed

        asyncio.run(run())


def test_idle_timeout_expires_held_and_unread_media_and_keeps_its_terminal_once(harness):
    pipe = CountingPipeline(initial=True)
    with harness(pipe, SimpleNamespace(delivery_slots=2, idle_timeout=0.2)) as h:
        s = h.open()
        results = s.results()
        held = next(results)  # initial result, a view over slot storage
        assert held.video is not None
        s.submit("controls", 1)  # one media result nobody reads
        channel = h.client._sessions[s.request_id]
        _wait_until(channel.unread_media, what="the unread result")
        _wait_until(lambda: _closed(channel), what="the idle close")
        assert s.closed
        with pytest.raises(SessionLeaseExpired, match="closed"):
            held.outputs
        assert not channel.unread_media()  # dropped with its mapping
        terminal = next(results)  # the timeout terminal survives the close
        assert terminal.done and "idle timeout" in terminal.error
        with pytest.raises(StopIteration):
            next(results)
        with pytest.raises(SessionEnded):
            s.submit("controls", 2)  # prompt, no round trip to a closed session
        with pytest.raises(SessionEnded):
            s.advance(2)
        s.close()  # idempotent after the worker's own close
        _wait_until(lambda: pipe.closed == 1, what="pipeline close")


def test_idle_timeout_keeps_an_existing_terminal_and_adds_none(harness):
    pipe = CountingPipeline(initial=False, length=1)
    with harness(pipe, SimpleNamespace(delivery_slots=2, idle_timeout=0.2)) as h:
        s = h.open()
        s.submit("controls", 1)  # the one advance: media, then the normal terminal
        channel = h.client._sessions[s.request_id]
        _wait_until(lambda: _closed(channel), timeout=10, what="the idle close")
        results = s.results()
        terminal = next(results)  # media dropped, the normal terminal kept
        assert terminal.done and terminal.error is None
        with pytest.raises(StopIteration):
            next(results)


def test_first_result_request_restarts_the_idle_clock(harness):
    """A request that returns no slot still sends a keep-alive: the first request
    of a session, or one after the iterator was closed."""
    idle = 0.6
    with harness(
        CountingPipeline(initial=True), SimpleNamespace(delivery_slots=2, idle_timeout=idle)
    ) as h:
        s = h.open()
        opened = time.monotonic()
        channel = h.client._sessions[s.request_id]
        time.sleep(idle * 0.6)
        first = next(s.results())  # first request: no slot to return, so a ping
        requested = time.monotonic()
        assert first.advance_index is None
        time.sleep(idle * 0.6)  # past the original deadline, inside the restarted one
        assert not _closed(channel), "the request did not restart the clock"
        _wait_until(lambda: _closed(channel), timeout=idle * 3, what="the idle close")
        assert time.monotonic() - requested >= idle * 0.9
        assert time.monotonic() - opened >= idle * 1.5


# --------------------------------------------------------------------------- close racing a handed-out result
#
# The response thread may complete a reader's future and the close may land before
# the reader resumes (an await, a preempted thread). The driving below is done by
# hand on the channel, so every ordering is deterministic and nothing sleeps.


def _timeout_terminal(session_id, sequence):
    return SessionResultMessage(
        session_id,
        sequence,
        None,
        None,
        {},
        {},
        type=ResultType.TERMINAL,
        error="session idle timeout (0s) expired",
    )


def test_terminals_keep_empty_outputs_after_the_close(harness):
    pipe = CountingPipeline(initial=False, length=1)
    with harness(pipe, SimpleNamespace(delivery_slots=2, idle_timeout=0.2)) as h:
        s = h.open()
        s.submit("controls", 1)  # media, then the normal terminal
        channel = h.client._sessions[s.request_id]
        _wait_until(lambda: _closed(channel), what="the idle close")
        terminal = next(s.results())
        assert terminal.done and terminal.error is None
        assert terminal.outputs == {} and terminal.video is None
        assert terminal.copy().outputs == {}
        with pytest.raises(StopIteration):
            next(s.results())


# --------------------------------------------------------------------------- explicit close discards terminals


def _wait_for_terminal(channel):
    """The worker publishes media then the normal terminal; wait for the latter."""
    _wait_until(
        lambda: any(m.type is ResultType.TERMINAL for m in channel._results),
        what="the queued terminal",
    )


def test_worker_initiated_close_still_keeps_its_terminal(harness):
    """The timeout path is unchanged by the cancellation rule."""
    pipe = CountingPipeline(initial=False, length=1)
    with harness(pipe, SimpleNamespace(delivery_slots=2, idle_timeout=0.2)) as h:
        s = h.open()
        s.submit("controls", 1)
        channel = h.client._sessions[s.request_id]
        _wait_until(lambda: _closed(channel), what="the idle close")
        terminal = next(s.results())
        assert terminal.done and terminal.error is None


def test_explicit_close_after_a_timeout_close_discards_the_retained_terminal(harness):
    with harness(
        CountingPipeline(initial=True), SimpleNamespace(delivery_slots=2, idle_timeout=0.15)
    ) as h:
        s = h.open()
        channel = h.client._sessions[s.request_id]
        _wait_until(lambda: _closed(channel), what="the idle close")
        assert any(m.type is ResultType.TERMINAL for m in channel._results)  # terminal retained
        s.close()  # explicit cancellation after the worker close: discard everything
        with pytest.raises(SessionEnded):
            next(s.results())


def test_timeout_close_without_explicit_close_still_returns_its_terminal(harness):
    with harness(
        CountingPipeline(initial=True), SimpleNamespace(delivery_slots=2, idle_timeout=0.15)
    ) as h:
        s = h.open()
        channel = h.client._sessions[s.request_id]
        _wait_until(lambda: _closed(channel), what="the idle close")
        results = s.results()
        terminal = next(results)  # unread initial media dropped, the retained terminal built
        assert terminal.done and "idle timeout" in terminal.error and terminal.outputs == {}
        with pytest.raises(StopIteration):
            next(results)


# --------------------------------------------------------------------------- advance(): input to result
#
# advance(value) must return the media of the value it admitted. The hazards are
# results that are owed but not yet in the client's hands: the declared initial
# result still in transit behind SessionOpened, an earlier queued value still
# computing or its media still in transit. Gating the worker (advance_gate) and
# the reply forwarding (Harness.hold) makes each ordering deterministic.


def test_advance_refuses_while_the_initial_result_is_in_transit(harness):
    with harness(CountingPipeline(initial=True)) as h:
        h.hold(SessionResultMessage)  # SessionOpened passes, the initial media does not
        s = h.open()
        _wait_until(lambda: h.held(SessionResultMessage), what="the held initial result")
        assert not h.client._sessions[s.request_id].unread_media()  # nothing arrived yet
        with pytest.raises(SessionError, match="unread"):
            s.advance(1)
        assert h.pipeline.advances == []  # nothing was submitted
        h.release()
        first = next(s.results())
        assert first.advance_index is None
        r = s.advance(1)
        assert r.advance_index == 0 and _pixels(r) == (0, 1, 0)


def test_advance_refuses_while_an_earlier_input_is_still_computing(harness):
    pipe = CountingPipeline(initial=False)
    pipe.advance_gate = threading.Event()
    with harness(pipe) as h:
        s = h.open()
        assert s.submit("controls", 1) == 0
        _wait_until(lambda: h.executor.pipeline is pipe and pipe.advance_gate is not None)
        with pytest.raises(SessionError, match="unread"):
            s.advance(2)  # advance 0 is computing: nothing in the deque, still refused
        pipe.advance_gate.set()
        with s.results() as results:
            r0 = next(results)
            assert r0.advance_index == 0 and _pixels(r0) == (0, 1, 0)
        r1 = s.advance(2)
        assert r1.advance_index == 1 and _pixels(r1) == (1, 2, 0)
        assert [a["controls"] for a in pipe.advances] == [1, 2]


def test_advance_refuses_while_an_earlier_result_is_in_transit(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        s = h.open()
        h.hold(SessionResultMessage)
        s.submit("controls", 1)
        _wait_until(lambda: h.held(SessionResultMessage), what="the held media")
        with pytest.raises(SessionError, match="unread"):
            s.advance(2)
        h.release()
        assert next(s.results()).advance_index == 0
        assert s.advance(2).advance_index == 1


def test_iterator_request_is_refused_while_advance_awaits_admission(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        s = h.open()
        results = s.results()
        h.hold(SessionAdmitted)  # advance() will park in its submit
        outcome = {}

        def run_advance():
            try:
                outcome["result"] = s.advance(1)
            except BaseException as exc:  # noqa: BLE001
                outcome["error"] = exc

        t = threading.Thread(target=run_advance)
        t.start()
        _wait_until(lambda: h.held(SessionAdmitted), what="the held admission")
        with pytest.raises(SessionError, match="already in progress"):
            next(results)  # the reservation covers the admission wait
        with pytest.raises(SessionError, match="already in progress"):
            s.advance(2)
        h.release()
        t.join(timeout=10)
        assert "error" not in outcome, outcome
        assert outcome["result"].advance_index == 0 and _pixels(outcome["result"]) == (0, 1, 0)
        assert s._channel._reader is None
        assert s.advance(2).advance_index == 1


def test_iterator_request_is_refused_while_advance_async_awaits_admission(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:

        async def run():
            req = DiffusionRequest(request_id=1, prompt=["p"], params=SimpleNamespace())
            channel = h.client.open_session(req)
            await channel.wait_open_async()
            s = VisualGenSession(h.client, channel)
            results = s.results()
            h.hold(SessionAdmitted)
            task = asyncio.ensure_future(s.advance_async(1))
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: _wait_until(lambda: h.held(SessionAdmitted), what="admission")
            )
            with pytest.raises(SessionError, match="already in progress"):
                await results.__anext__()
            h.release()
            r = await asyncio.wait_for(task, 10)
            assert r.advance_index == 0
            assert (await s.advance_async(2)).advance_index == 1
            await s.close_async()

        asyncio.run(run())


def test_advance_returns_its_admitted_index_in_sequence(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open() as s:
            for value in (5, 6, 7):
                r = s.advance(value)
                assert r.advance_index == value - 5 and _pixels(r) == (value - 5, value, 0)


def test_advance_after_the_session_ended_returns_the_terminal_once_without_submitting(harness):
    pipe = CountingPipeline(initial=False, length=1)
    with harness(pipe) as h:
        with h.open() as s:
            r = s.advance(1)
            assert r.advance_index == 0
            channel = h.client._sessions[s.request_id]
            _wait_until(
                lambda: any(m.type is ResultType.TERMINAL for m in channel._results),
                what="the terminal",
            )
            sent = []
            original = h.client.pending_requests.put
            h.client.pending_requests.put = lambda m: (sent.append(m), original(m))[1]
            terminal = s.advance(2)  # the terminal is already here: nothing is submitted
            h.client.pending_requests.put = original
            assert terminal.done and terminal.error is None
            assert sent == [] and [a["controls"] for a in pipe.advances] == [1]
            with pytest.raises(SessionEnded):
                s.advance(3)


def test_advance_whose_input_fails_returns_the_error_terminal(harness):
    pipe = CountingPipeline(initial=False, fail_at=1)
    with harness(pipe) as h:
        with h.open() as s:
            assert s.advance(1).advance_index == 0
            terminal = s.advance(2)  # admitted as 1, the advance raised: terminal instead
            assert terminal.done and "kernel exploded" in terminal.error
            with pytest.raises(SessionEnded):
                s.advance(3)


def test_rejected_input_leaves_advance_and_results_usable(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open() as s:
            with pytest.raises(ValueError):
                s.advance(-1)  # the validator rejects it
            assert s._channel._reader is None and s._channel._submit is None
            assert s.advance(1).advance_index == 0  # holds one of the two slots
            h.hold(SessionResultMessage)  # results stay in transit: no slot comes back
            for value in (2, 3, 4):  # 2 takes the last free slot, 3 and 4 fill the queue
                s.submit("controls", value)
            with pytest.raises(SessionQueueFull):
                s.submit("controls", 5)
            assert s._channel._submit is None
            with pytest.raises(SessionError, match="unread"):
                s.advance(6)  # three results owed
            h.release()
            with s.results() as results:
                assert [next(results).advance_index for _ in range(3)] == [1, 2, 3]
            assert s.advance(6).advance_index == 4


@pytest.mark.parametrize("slots", [1, 2])
def test_advance_releases_its_lease_before_waiting(harness, slots):
    """With one slot the worker cannot produce the next result until the held one
    is returned; advance() returns it at entry, before admission."""
    with harness(
        CountingPipeline(initial=False), SimpleNamespace(delivery_slots=slots, idle_timeout=None)
    ) as h:
        with h.open() as s:
            previous = None
            for value in (1, 2, 3):
                r = s.advance(value)
                assert r.advance_index == value - 1
                if previous is not None:
                    with pytest.raises(SessionLeaseExpired):
                        previous.outputs
                previous = r


# --------------------------------------------------------------------------- advance(): what is owed
#
# The owed-media bookkeeping lives in the channel and is kept by the background
# thread, so a cancelled submit does not lose an admission, and a terminal that
# has arrived supersedes admissions that ended without media.


def test_advance_async_does_not_submit_when_the_terminal_is_known(harness):
    pipe = CountingPipeline(initial=False, length=1)
    with harness(pipe) as h:

        async def run():
            req = DiffusionRequest(request_id=1, prompt=["p"], params=SimpleNamespace())
            channel = h.client.open_session(req)
            await channel.wait_open_async()
            s = VisualGenSession(h.client, channel)
            assert (await s.advance_async(1)).advance_index == 0
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: _wait_until(lambda: _terminal_next(channel), what="terminal")
            )
            sent = []
            original = h.client.pending_requests.put
            h.client.pending_requests.put = lambda m: (sent.append(m), original(m))[1]
            terminal = await s.advance_async(2)
            h.client.pending_requests.put = original
            assert terminal.done and sent == []
            with pytest.raises(SessionEnded):
                await s.advance_async(3)
            await s.close_async()

        asyncio.run(run())


def test_advance_takes_the_terminal_when_the_earlier_input_died_without_media(harness):
    pipe = CountingPipeline(initial=False, fail_at=0)
    with harness(pipe) as h:
        with h.open() as s:
            assert s.submit("controls", 1) == 0  # admitted, then the advance raises
            channel = h.client._sessions[s.request_id]
            _wait_until(lambda: _terminal_next(channel), what="the error terminal")
            assert channel.lockstep_due() > 0  # index 0 is still recorded ...
            terminal = s.advance(2)  # ... but the terminal supersedes it; nothing submitted
            assert terminal.done and "kernel exploded" in terminal.error
            assert [a["controls"] for a in pipe.advances] == []
            with pytest.raises(SessionEnded):
                s.advance(3)


def test_advance_takes_a_retained_timeout_terminal_after_the_worker_closed(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe, SimpleNamespace(delivery_slots=2, idle_timeout=0.15)) as h:
        s = h.open()
        channel = h.client._sessions[s.request_id]
        _wait_until(lambda: _closed(channel), what="the idle close")
        terminal = s.advance(1)  # closed by the worker, terminal retained: delivered, not refused
        assert terminal.done and "idle timeout" in terminal.error
        with pytest.raises(SessionEnded):
            s.advance(2)


def test_advance_still_refuses_while_real_media_precedes_the_terminal(harness):
    pipe = CountingPipeline(initial=False, length=1)
    with harness(pipe) as h:
        with h.open() as s:
            s.submit("controls", 1)  # media 0, then the normal terminal
            channel = h.client._sessions[s.request_id]
            _wait_until(
                lambda: any(m.type is ResultType.TERMINAL for m in channel._results),
                what="terminal",
            )
            assert channel.unread_media()
            with pytest.raises(SessionError, match="unread"):
                s.advance(2)
            with s.results() as results:
                assert next(results).advance_index == 0
            assert s.advance(2).done


def test_cancelled_submit_keeps_its_admission_and_its_media(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:

        async def run():
            req = DiffusionRequest(request_id=1, prompt=["p"], params=SimpleNamespace())
            channel = h.client.open_session(req)
            await channel.wait_open_async()
            s = VisualGenSession(h.client, channel)
            h.hold(SessionAdmitted, SessionResultMessage)  # released later in wire order
            task = asyncio.ensure_future(s.submit_async("controls", 1))
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: _wait_until(lambda: h.held(SessionAdmitted), what="admission")
            )
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            pending = s._channel._submit
            assert pending is not None and pending.admission.cancelled()  # kept until the reply
            h.release()
            await loop.run_in_executor(
                None, lambda: _wait_until(channel.unread_media, what="media")
            )
            assert channel.lockstep_due() > 0 and channel._submit is None  # owed; record retired
            with pytest.raises(SessionError, match="unread"):
                await s.advance_async(2)  # refused: nothing admitted, result 0 not consumed
            assert [a["controls"] for a in pipe.advances] == [1]
            results = s.results()
            r0 = await results.__anext__()  # the old media is still obtainable
            assert r0.advance_index == 0 and _pixels(r0) == (0, 1, 0)
            assert channel.lockstep_due() == 0
            assert (await s.advance_async(2)).advance_index == 1
            await s.close_async()

        asyncio.run(run())


def test_cancelled_advance_async_keeps_its_admission(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:

        async def run():
            req = DiffusionRequest(request_id=1, prompt=["p"], params=SimpleNamespace())
            channel = h.client.open_session(req)
            await channel.wait_open_async()
            s = VisualGenSession(h.client, channel)
            h.hold(SessionAdmitted, SessionResultMessage)
            task = asyncio.ensure_future(s.advance_async(1))
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: _wait_until(lambda: h.held(SessionAdmitted), what="admission")
            )
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert s._channel._reader is None  # the read reservation went with the caller
            h.release()
            await loop.run_in_executor(
                None, lambda: _wait_until(channel.unread_media, what="media")
            )
            with pytest.raises(SessionError, match="unread"):
                await s.advance_async(2)
            r0 = await s.results().__anext__()
            assert r0.advance_index == 0
            await s.close_async()  # closing after a cancellation works
            assert s.closed

        asyncio.run(run())


def _drive_cancelled_submit_then_advance(h, start_call):
    """The reviewer's interleaving: a submit is cancelled while its admission is
    held; then, right after advance's owed check passes, the held replies arrive
    (admission of index 0, media 0) before advance reserves its own admission."""

    async def run():
        req = DiffusionRequest(request_id=1, prompt=["p"], params=SimpleNamespace())
        channel = h.client.open_session(req)
        await channel.wait_open_async()
        s = VisualGenSession(h.client, channel)
        loop = asyncio.get_running_loop()
        h.hold(SessionAdmitted, SessionResultMessage)
        task = asyncio.ensure_future(start_call(s))
        await loop.run_in_executor(
            None, lambda: _wait_until(lambda: h.held(SessionAdmitted), what="admission")
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert s._channel._reader is None  # the submit record waits for its reply
        original = channel.plan_advance

        def replies_arrive_then_plan(name):
            # The old admission and media 0 land right before the reservation.
            h.release()
            _wait_until(channel.unread_media, what="media 0")
            return original(name)

        channel.plan_advance = replies_arrive_then_plan
        with pytest.raises(SessionError, match="unread|in progress"):
            await s.advance_async(2)
        channel.plan_advance = original
        assert [a["controls"] for a in h.pipeline.advances] == [1]  # no second input admitted
        r0 = await s.results().__anext__()
        assert r0.advance_index == 0 and _pixels(r0) == (0, 1, 0)  # media 0 still obtainable
        assert (await s.advance_async(2)).advance_index == 1
        await s.close_async()

    asyncio.run(run())


def test_advance_cannot_slip_between_the_owed_check_and_a_late_admission(harness):
    with harness(CountingPipeline(initial=False)) as h:
        _drive_cancelled_submit_then_advance(h, lambda s: s.submit_async("controls", 1))


def test_advance_cannot_slip_after_a_cancelled_advance_async(harness):
    with harness(CountingPipeline(initial=False)) as h:
        _drive_cancelled_submit_then_advance(h, lambda s: s.advance_async(1))


# --------------------------------------------------------------------------- advance(): the session ends mid-call


def _idle_closes_before_the_submit(s, channel):
    """Wrap the client's submit so the worker's idle close (terminal, SessionClosed)
    has landed by the time advance tries to reserve its admission."""
    client = s._client
    original = client.session_submit

    def wait_then_submit(*args, **kwargs):
        _wait_until(lambda: _closed(channel), what="the idle close")
        return original(*args, **kwargs)

    client.session_submit = wait_then_submit
    return lambda: setattr(client, "session_submit", original)


def test_advance_returns_the_terminal_when_the_idle_close_lands_before_its_submit(harness):
    with harness(
        CountingPipeline(initial=False), SimpleNamespace(delivery_slots=2, idle_timeout=0.15)
    ) as h:
        s = h.open()
        channel = h.client._sessions[s.request_id]
        restore = _idle_closes_before_the_submit(s, channel)
        terminal = s.advance(1)
        restore()
        assert terminal.done and "idle timeout" in terminal.error
        assert s._channel._reader is None and s._channel._submit is None
        with pytest.raises(SessionEnded):
            s.advance(2)


def test_advance_async_returns_the_terminal_when_the_idle_close_lands_before_its_submit(harness):
    with harness(
        CountingPipeline(initial=False), SimpleNamespace(delivery_slots=2, idle_timeout=0.15)
    ) as h:

        async def run():
            req = DiffusionRequest(request_id=1, prompt=["p"], params=SimpleNamespace())
            channel = h.client.open_session(req)
            await channel.wait_open_async()
            s = VisualGenSession(h.client, channel)
            restore = _idle_closes_before_the_submit(s, channel)
            terminal = await s.advance_async(1)
            restore()
            assert terminal.done and "idle timeout" in terminal.error
            assert s._channel._reader is None and s._channel._submit is None
            with pytest.raises(SessionEnded):
                await s.advance_async(2)

        asyncio.run(run())


def test_advance_ends_when_an_explicit_close_lands_before_its_submit(harness):
    with harness(CountingPipeline(initial=False)) as h:
        s = h.open()
        channel = h.client._sessions[s.request_id]
        original = h.client.session_submit

        def close_then_submit(*args, **kwargs):
            h.client.session_close(channel).result(timeout=10)  # the client's own close
            return original(*args, **kwargs)

        h.client.session_submit = close_then_submit
        with pytest.raises(SessionEnded):
            s.advance(1)  # cancelled close: nothing retained, the call ends
        h.client.session_submit = original
        assert s._channel._reader is None and s._channel._submit is None
        s.close()


def test_advance_raises_the_worker_failure_when_it_lands_before_its_submit(harness):
    with harness(CountingPipeline(initial=False)) as h:
        s = h.open()
        channel = h.client._sessions[s.request_id]
        original = h.client.session_submit

        def fail_then_submit(*args, **kwargs):
            channel.fail(RuntimeError("DiffusionClient: local worker processes exited"))
            return original(*args, **kwargs)

        h.client.session_submit = fail_then_submit
        with pytest.raises(RuntimeError, match="exited"):
            s.advance(1)
        h.client.session_submit = original
        assert s._channel._reader is None and s._channel._submit is None
        s.close()


# --------------------------------------------------------------------------- one session per model instance


def _request(request_id):
    return DiffusionRequest(request_id=request_id, prompt=["p"], params=SimpleNamespace())


def test_second_open_is_refused_at_once_and_the_first_session_stays_usable(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open() as s:
            t0 = time.monotonic()
            with pytest.raises(MemoryError, match="session 1 is open"):
                h.client.open_session(_request(2))
            assert time.monotonic() - t0 < 1.0
            assert list(h.client._sessions) == [1]
            assert s.advance(1).advance_index == 0


def test_second_open_async_is_refused_at_once(harness):
    with harness(CountingPipeline(initial=False)) as h:

        async def run():
            channel = h.client.open_session(_request(1))
            await channel.wait_open_async()
            s = VisualGenSession(h.client, channel)
            with pytest.raises(MemoryError):
                h.client.open_session(_request(2))
            assert (await s.advance_async(1)).advance_index == 0
            await s.close_async()

        asyncio.run(run())


def test_second_open_is_refused_while_the_worker_is_inside_an_advance(harness):
    pipe = CountingPipeline(initial=False)
    pipe.advance_gate = threading.Event()
    with harness(pipe) as h:
        s = h.open()
        s.submit("controls", 1)  # the worker is now parked inside advance 0
        t0 = time.monotonic()
        with pytest.raises(MemoryError):
            h.client.open_session(_request(2))
        assert time.monotonic() - t0 < 1.0  # no round trip to the busy worker
        pipe.advance_gate.set()
        assert next(s.results()).advance_index == 0
        s.close()


def test_competing_opens_admit_exactly_one(harness):
    with harness(CountingPipeline(initial=False)) as h:
        outcomes = []
        barrier = threading.Barrier(4)

        def attempt(i):
            barrier.wait()
            try:
                outcomes.append(h.client.open_session(_request(i)))
            except MemoryError as exc:
                outcomes.append(exc)

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(1, 5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        channels = [o for o in outcomes if isinstance(o, SessionChannel)]
        assert len(channels) == 1 and len(outcomes) == 4
        channel = channels[0]
        channel.wait_open(timeout=10)
        VisualGenSession(h.client, channel).close()


def test_open_succeeds_again_after_close_idle_close_and_open_failure(harness):
    pipe = CountingPipeline(initial=False, open_fails=True)
    with harness(pipe, SimpleNamespace(delivery_slots=2, idle_timeout=0.2)) as h:
        channel = h.client.open_session(_request(1))
        with pytest.raises(ValueError):  # the pipeline refuses the open
            channel.wait_open(timeout=10)
        assert h.client._sessions == {}  # the reservation went with the failure
        pipe.open_fails = False
        s = h.open()  # explicit close releases it
        s.close()
        assert h.client._sessions == {}
        s = h.open()  # idle close releases it too
        channel = h.client._sessions[s.request_id]
        _wait_until(lambda: _closed(channel), what="the idle close")
        assert h.client._sessions == {}
        s = h.open()
        assert s.advance(1).advance_index == 0
        s.close()


def test_rejected_open_releases_the_request_payload(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open():
            released = []
            original = h.client._discard_unsent_request
            h.client._discard_unsent_request = lambda req: (released.append(req), original(req))[1]
            req = _request(2)
            with pytest.raises(MemoryError):
                h.client.open_session(req)
            h.client._discard_unsent_request = original
            assert released == [req]


def test_worker_refuses_an_open_and_a_generate_while_a_session_is_open(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open() as s:
            replies = []
            original = h.client._route_session_reply
            h.client._route_session_reply = lambda r: (replies.append(r), original(r))[1]
            # Straight onto the worker's socket: an open that slipped past the client.
            h.executor.requests_ipc.put_nowait(SessionOpen(_request(7)))
            generate = _request(8)
            h.executor.requests_ipc.put_nowait(generate)
            s.submit("controls", 1)  # makes the worker service its socket
            _wait_until(
                lambda: any(isinstance(r, SessionOpened) and r.session_id == 7 for r in replies),
                what="the capacity reply",
            )
            refused = next(r for r in replies if isinstance(r, SessionOpened) and r.session_id == 7)
            assert refused.error_type == "capacity" and "is open" in refused.error
            _wait_until(
                lambda: any(r.request_id == 8 for r in h.responses), what="the generate's refusal"
            )
            answer = next(r for r in h.responses if r.request_id == 8)
            assert isinstance(answer, DiffusionResponse)
            assert answer.error_type == "capacity" and "is open" in answer.error_msg
            h.client._route_session_reply = original
            assert next(s.results()).advance_index == 0
            assert h.pipeline.opened == 1


# --------------------------------------------------------------------------- open racing a worker failure


def _request_with_payload(request_id):
    """A request carrying one image reference, moved to shared memory the way
    VisualGen._session_request does before open_session."""
    ref = SimpleNamespace(content=b"\x07" * 4096)
    req = DiffusionRequest(
        request_id=request_id, prompt=["p"], params=SimpleNamespace(image_reference=[ref])
    )
    req.refs_to_shm()
    assert req.ref_handles and ref.content == b""  # the bytes live in shared memory now
    return req, ref


def _aborting_client():
    client = _make_client()
    client._shutdown_started = False
    client._request_to_send = None
    client.shutdown_event = threading.Event()
    client.response_event = threading.Event()
    client.worker_processes = []
    return client


def test_open_after_a_worker_failure_refuses_and_releases_its_payload():
    client = _aborting_client()
    client._abort_worker_group("review probe: worker exited")
    req, ref = _request_with_payload(3)
    with pytest.raises(RuntimeError, match="worker exited"):
        client.open_session(req)
    assert client._sessions == {} and client.pending_requests.empty()
    assert req.ref_handles is None and ref.content == b"\x07" * 4096  # handle consumed


def test_abort_between_the_failure_check_and_the_reservation_fails_the_opener(monkeypatch):
    """The abort lands after open_session built its channel and before it took
    the lock: the open must then see the failure, not register behind it."""
    import tensorrt_llm._torch.visual_gen.executor as executor_module

    client = _aborting_client()
    real = executor_module.SessionChannel

    def abort_then_build(session_id):
        client._abort_worker_group("review probe: worker exited")
        return real(session_id)

    monkeypatch.setattr(executor_module, "SessionChannel", abort_then_build)
    req, ref = _request_with_payload(3)
    with pytest.raises(RuntimeError, match="worker exited"):
        client.open_session(req)
    assert client._sessions == {} and client.pending_requests.empty()
    assert req.ref_handles is None and ref.content == b"\x07" * 4096


def test_abort_after_the_reservation_fails_the_opener_and_drains_its_message():
    client = _aborting_client()
    req, ref = _request_with_payload(3)
    channel = client.open_session(req)
    assert list(client._sessions) == [3] and client.pending_requests.qsize() == 1
    client._abort_worker_group("review probe: worker exited")
    with pytest.raises(RuntimeError, match="worker exited"):
        channel.wait_open(timeout=1)  # the opener raises promptly
    assert client._sessions == {}  # no reservation remains ...
    assert client.pending_requests.empty()  # ... and the unsent open is gone
    assert req.ref_handles is None and ref.content == b"\x07" * 4096  # payload released
    with pytest.raises(RuntimeError, match="worker exited"):
        client.open_session(_request(4))


def test_rejected_second_open_consumes_its_real_payload(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open():
            req, ref = _request_with_payload(2)
            with pytest.raises(MemoryError):
                h.client.open_session(req)
            assert req.ref_handles is None and ref.content == b"\x07" * 4096


# --------------------------------------------------------------------------- cancelled opening
#
# open_session_async on the client is the code VisualGen.open_session_async runs
# after building the request; the tests drive it directly.


def _run_async(coro_fn):
    asyncio.run(coro_fn())


def test_cancelled_open_relinquishes_the_session_when_the_open_succeeds_later(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:

        async def run():
            h.hold(SessionOpened)  # the open is in progress; its reply is held
            task = asyncio.ensure_future(h.client.open_session_async(_request(1)))
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: _wait_until(lambda: h.held(SessionOpened)))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            channel = h.client._sessions[1]
            assert not channel._opened.cancelled()
            assert list(h.client._sessions) == [1]  # still reserved: the worker holds it
            h.release()  # the late success: no InvalidStateError, closed at once
            await loop.run_in_executor(
                None, lambda: _wait_until(lambda: not h.client._sessions, what="release")
            )
            assert _closed(channel) and channel.lease_lost is not None
            await loop.run_in_executor(None, lambda: _wait_until(lambda: pipe.closed == 1))
            assert pipe.opened == 1
            s = VisualGenSession(h.client, await h.client.open_session_async(_request(2)))
            assert (await s.advance_async(1)).advance_index == 0
            await s.close_async()

        _run_async(run)


def test_cancelled_open_whose_open_fails_later_releases_the_reservation(harness):
    pipe = CountingPipeline(initial=False, open_fails=True)
    with harness(pipe) as h:

        async def run():
            h.hold(SessionOpened)
            task = asyncio.ensure_future(h.client.open_session_async(_request(1)))
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: _wait_until(lambda: h.held(SessionOpened)))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            h.release()  # the late failure: no InvalidStateError
            await loop.run_in_executor(
                None, lambda: _wait_until(lambda: not h.client._sessions, what="release")
            )
            pipe.open_fails = False
            channel = await h.client.open_session_async(_request(2))
            VisualGenSession(h.client, channel).close()

        _run_async(run)


def test_cancellation_after_the_open_completed_but_before_the_handle_is_returned(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:

        async def run():
            channels = []
            original = h.client.open_session
            h.client.open_session = lambda req: (channels.append(original(req)), channels[-1])[1]
            task = asyncio.ensure_future(h.client.open_session_async(_request(1)))
            loop = asyncio.get_running_loop()
            # Let the task start and send its open; block the loop thread until the
            # worker's success is in, so the task cannot have resumed yet.
            await asyncio.sleep(0)
            channels[0].wait_open(timeout=10)
            task.cancel()  # arrives before the task sees its completed future
            with pytest.raises(asyncio.CancelledError):
                await task
            await loop.run_in_executor(
                None, lambda: _wait_until(lambda: not h.client._sessions, what="release")
            )
            await loop.run_in_executor(None, lambda: _wait_until(lambda: pipe.closed == 1))
            h.client.open_session = original
            channel = await h.client.open_session_async(_request(2))
            VisualGenSession(h.client, channel).close()

        _run_async(run)


def test_cancelled_open_still_unsent_does_not_leave_an_ownerless_session(harness):
    """The open is cancelled while its command is still in the client's queue
    (the sender is gated, as when it is busy behind an earlier request); when the
    worker eventually opens it, the session is closed and the reservation freed."""
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:

        async def run():
            h.send_gate.clear()
            task = asyncio.ensure_future(h.client.open_session_async(_request(1)))
            await asyncio.sleep(0)
            assert pipe.opened == 0  # the open is still on the client's side of the gate
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            loop = asyncio.get_running_loop()
            h.send_gate.set()
            await loop.run_in_executor(
                None, lambda: _wait_until(lambda: not h.client._sessions, what="release")
            )
            await loop.run_in_executor(None, lambda: _wait_until(lambda: pipe.closed == 1))
            assert pipe.opened == 1 and pipe.closed == 1

        _run_async(run)


def test_open_reply_routing_tolerates_a_cancelled_lifecycle_future():
    client = _make_client()
    channel = client.open_session(_request(9))
    channel._opened.cancel()  # a caller that bypassed the shield
    client._route_session_reply(
        SessionOpened(9, capabilities={"inputs": [], "streams": []}, slots=[])
    )
    assert list(client._sessions) == [9]  # success recorded, nothing raised
    client._route_session_reply(SessionClosed(9))
    assert client._sessions == {}
    channel = client.open_session(_request(10))
    channel._opened.cancel()
    client._route_session_reply(SessionOpened(10, error="bad prompt", error_type="client"))
    assert client._sessions == {}


# --------------------------------------------------------------------------- result request cancelled during delivery


# --------------------------------------------------------------------------- input byte bound, end to end


def test_repeated_rejected_submits_do_not_grow_staging(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open() as s:
            channel = h.client._sessions[s.request_id]
            for _ in range(50):
                with pytest.raises(ValueError):
                    s.submit("controls", -1)  # the worker's validator refuses it
            assert h.client.pending_requests.empty() and channel._submit is None
            assert s.submit("controls", 1) == 0


# --------------------------------------------------------------------------- a command that cannot be sent


class _FailingSocket:
    """``requests_ipc`` whose send raises what the caller chose."""

    def __init__(self, exc):
        self.exc = exc
        self.sent = []

    def put_nowait(self, obj):
        if self.exc is not None:
            raise self.exc
        self.sent.append(obj)


def _sending_client(exc):
    client = _aborting_client()
    client.requests_ipc = _FailingSocket(exc)
    client.controls_ipc = _FailingSocket(None)
    return client


def _declared(client, session_id):
    channel = client.open_session(_request(session_id))
    channel.open_succeeded(
        {"inputs": [{"name": "x", "policy": "queued"}], "initial_result": False}, []
    )
    return channel


def test_unserializable_submit_fails_that_call_only():
    client = _sending_client(None)
    channel = _declared(client, 1)
    client._process_requests()  # the open goes out
    client.requests_ipc.exc = TypeError("cannot pickle 'probe' object")
    fut = client.session_submit(channel, "x", [1.0])
    client._process_requests()
    with pytest.raises(ValueError, match="cannot be serialized"):
        fut.result(timeout=1)
    channel.release_submit(fut)  # what the public submit does in its finally
    assert channel._submit is None and client._worker_failure is None
    assert client.pending_requests.empty() and client._request_to_send is None
    client.requests_ipc.exc = None
    fut = client.session_submit(channel, "x", [1.0])  # the next submit goes out
    client._process_requests()
    sent = client.requests_ipc.sent[-1]
    assert isinstance(sent, SessionSubmit)
    channel.admitted(0, None)
    assert fut.result(timeout=1) == 0  # no index was consumed by the failed one


def test_unserializable_open_fails_the_opener_and_releases_everything():
    client = _sending_client(TypeError("cannot pickle 'probe' object"))
    req, ref = _request_with_payload(2)
    channel = client.open_session(req)
    client._process_requests()
    with pytest.raises(ValueError, match="cannot be serialized"):
        channel.wait_open(timeout=1)
    assert client._sessions == {} and client._worker_failure is None
    assert req.ref_handles is None and ref.content == b"\x07" * 4096  # references released
    client.requests_ipc.exc = None
    client.open_session(_request(3))  # capacity is free again
    client._process_requests()
    assert isinstance(client.requests_ipc.sent[-1], SessionOpen)


@pytest.mark.parametrize("command", ["submit", "close", "slot_return", "keep_alive", "open"])
def test_transport_failure_on_a_session_command_aborts_the_group(command):
    client = _sending_client(None)
    channel = _declared(client, 1)
    client._process_requests()
    client.requests_ipc.exc = OSError("probe: send failed")
    client.controls_ipc = _FailingSocket(OSError("probe: send failed"))
    waiters = []
    if command == "submit":
        waiters.append(client.session_submit(channel, "x", [1.0]))
    elif command == "close":
        waiters.append(client.session_close(channel))
    elif command == "slot_return":
        client.session_return_slot(channel, 0)
    elif command == "keep_alive":
        client.session_keep_alive(channel)
    else:
        client.requests_ipc.exc = None
        with client._sessions_lock:
            client._sessions.clear()  # make room: a fresh open whose send then fails
        channel = client.open_session(_request(5))
        client.requests_ipc.exc = OSError("probe: send failed")
        waiters.append(channel._opened)
    client._process_controls()
    client._process_requests()
    assert client._worker_failure is not None and "send failed" in client._worker_failure
    for fut in waiters:
        with pytest.raises(RuntimeError, match="send failed"):
            fut.result(timeout=1)
    assert client._sessions == {}
    with pytest.raises(RuntimeError, match="send failed"):
        client.open_session(_request(9))  # sticky: every later call raises


def test_zmq_again_still_retries_without_failing_the_call():
    import zmq

    client = _sending_client(None)
    client._monitor_worker_liveness = False
    channel = _declared(client, 1)
    client._process_requests()
    client.requests_ipc.exc = zmq.Again()
    fut = client.session_submit(channel, "x", [1.0])
    client._process_requests()  # not writable: kept at the head, nothing failed
    assert not fut.done() and client._request_to_send is not None
    client.requests_ipc.exc = None
    client._process_requests()
    assert isinstance(client.requests_ipc.sent[-1], SessionSubmit)


def test_transport_failure_on_an_open_releases_its_references_exactly_once():
    client = _sending_client(OSError("probe: send failed"))
    req, ref = _request_with_payload(2)
    channel = client.open_session(req)
    client._process_requests()
    with pytest.raises(RuntimeError, match="send failed"):
        channel.wait_open(timeout=1)
    assert client._sessions == {} and client._request_to_send is None
    assert req.ref_handles is None and ref.content == b"\x07" * 4096  # consumed by the abort
    req.refs_from_shm()  # a second release is a no-op: nothing left to consume
    assert ref.content == b"\x07" * 4096


def test_transport_failure_releases_the_failing_open_and_the_queued_payload_once():
    """The open in the send slot fails; a generate with its own payload is still
    queued behind it. Both are released by the abort, neither twice."""
    client = _sending_client(None)
    req, ref = _request_with_payload(2)
    channel = client.open_session(req)
    queued, queued_ref = _request_with_payload(3)
    client.pending_requests.put(queued)  # an ordinary request behind the open
    client.requests_ipc.exc = OSError("probe: send failed")
    client._process_requests()
    with pytest.raises(RuntimeError, match="send failed"):
        channel.wait_open(timeout=1)
    assert client.pending_requests.empty()
    assert req.ref_handles is None and ref.content == b"\x07" * 4096
    assert queued.ref_handles is None and queued_ref.content == b"\x07" * 4096
    for r, content in ((req, ref), (queued, queued_ref)):
        r.refs_from_shm()  # idempotent: handles are gone, content stays
        assert content.content == b"\x07" * 4096


# --------------------------------------------------------------------------- the control path
#
# Slot returns, close and keep-alive travel on their own socket with their own
# send slot, so they are serviced however the request socket behaves, and the
# worker watches both sockets while it waits.


def test_close_and_slot_returns_pass_while_the_request_path_is_blocked(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        s = h.open()
        h.payload_gate.clear()  # from now on submits never reach the worker
        results = s.results()
        outcome = {}

        def read():
            try:
                next(results)
            except BaseException as exc:  # noqa: BLE001
                outcome["read"] = exc

        def submit():
            try:
                s.submit("controls", 1)
            except BaseException as exc:  # noqa: BLE001
                outcome["submit"] = exc

        reader, submitter = threading.Thread(target=read), threading.Thread(target=submit)
        reader.start()
        submitter.start()
        _wait_until(lambda: h.held_payloads, what="the stuck submit")
        s.close()  # travels on the control path and completes
        reader.join(timeout=10)
        submitter.join(timeout=10)
        assert not reader.is_alive() and not submitter.is_alive()
        assert isinstance(outcome["read"], SessionEnded)
        assert isinstance(outcome["submit"], SessionEnded)
        assert s.closed and pipe.closed == 1 and pipe.advances == []
        # The next session opens and runs although the old submit is still held.
        s2 = h.open()
        h.payload_gate.set()
        assert s2.advance(5).advance_index == 0 and _pixels(s2.advance(6)) == (1, 6, 0)
        s2.close()
        assert len(h.held_payloads) == 1  # the old submit was never resent


def test_output_blocked_session_is_released_by_a_slot_return_on_the_control_path(harness):
    pipe = CountingPipeline(queued=False, initial=False, length=3)  # autonomous: no inputs
    with harness(pipe, SimpleNamespace(delivery_slots=1, idle_timeout=None)) as h:
        s = h.open()
        h.payload_gate.clear()  # irrelevant to this session, blocked anyway
        results = s.results()
        r0 = next(results)
        assert r0.advance_index == 0
        _wait_until(lambda: len(pipe.advances) == 1, what="the worker to block on its one slot")
        time.sleep(0.1)
        assert len(pipe.advances) == 1  # output-blocked: the single slot is held
        r1 = next(results)  # returns the slot on the control path
        assert r1.advance_index == 1
        r2 = next(results)
        assert r2.advance_index == 2 and next(results).done
        s.close()


def test_closed_session_retires_its_unsent_submits_in_the_dispatcher():
    """Dispatcher level: a submit parked on Again and two more queued; the close
    acknowledgement drops all three, settles their waiters, and the next open is
    what goes out when the socket accepts again."""
    import zmq

    client = _sending_client(None)
    client._monitor_worker_liveness = False
    channel = _declared(client, 1)
    client._process_requests()  # the open goes out
    client.requests_ipc.exc = zmq.Again()
    parked = client.session_submit(channel, "x", [1.0])
    client._process_requests()  # parked in the send slot
    assert isinstance(client._request_to_send, SessionSubmit)
    waiting = client.session_close(channel)  # control path: goes out at once
    client._process_controls()
    assert client.controls_ipc.sent[-1].type is ControlType.CLOSE
    client._route_session_reply(SessionClosed(1))  # the worker confirmed
    assert waiting.result(timeout=1) is None
    with pytest.raises(SessionEnded):
        parked.result(timeout=1)
    assert client._request_to_send is None and client.pending_requests.empty()
    client.requests_ipc.exc = None
    new = client.open_session(_request(2))
    client._process_requests()
    assert isinstance(client.requests_ipc.sent[-1], SessionOpen)  # not the retired submit
    assert all(not isinstance(m, SessionSubmit) for m in client.requests_ipc.sent)
    assert new is client._sessions[2]


def test_keep_alives_are_coalesced_and_close_is_sent_once():
    client = _sending_client(None)
    channel = _declared(client, 1)
    client._process_requests()
    for _ in range(5):
        client.session_keep_alive(channel)
    assert len(client._controls) == 1  # one pending note per session
    client._process_controls()
    assert sum(m.type is ControlType.KEEP_ALIVE for m in client.controls_ipc.sent) == 1
    client.session_keep_alive(channel)  # after it was sent, a new one is accepted
    assert len(client._controls) == 1
    client._process_controls()
    client.session_close(channel)
    client.session_close(channel)
    client.session_close(channel)
    assert len(client._controls) == 1
    client._process_controls()
    assert sum(m.type is ControlType.CLOSE for m in client.controls_ipc.sent) == 1


def test_close_before_the_open_is_acknowledged_waits_for_the_acknowledgement():
    client = _sending_client(None)
    channel = client.open_session(_request(1))
    client._process_requests()
    closed = client.session_close(channel)  # opened is still pending
    client._process_controls()
    assert client.controls_ipc.sent == []  # nothing crossed early
    client._route_session_reply(
        SessionOpened(1, capabilities={"inputs": [], "streams": []}, slots=[])
    )
    client._process_controls()
    assert client.controls_ipc.sent[-1].type is ControlType.CLOSE  # now, after the ack
    client._route_session_reply(SessionClosed(1))
    assert closed.result(timeout=1) is None and client._sessions == {}


def test_stray_close_on_the_control_socket_does_not_touch_a_later_session(harness):
    with harness(CountingPipeline(initial=False)) as h:
        h.executor.controls_ipc.put_nowait(
            SessionControl(41, ControlType.CLOSE)
        )  # a close for no session
        with h.open() as s:
            assert s.advance(1).advance_index == 0  # the stray did not close this one
        assert h.pipeline.closed == 1


def test_control_socket_real_zmq_carries_close_while_the_request_path_is_blocked(harness):
    """The control path over a real ZeroMQ pair (PUSH server on the client, PULL
    client on the worker), HMAC on, driven by the client's real control
    dispatcher, while the request path holds every submit."""
    from tensorrt_llm.executor.ipc import ZeroMqQueue

    port = _free_port()
    key = os.urandom(32)
    client_ctl = ZeroMqQueue(
        (f"tcp://127.0.0.1:{port}", key),
        is_server=True,
        socket_type=zmq_push(),
        use_hmac_encryption=True,
    )
    worker_ctl = ZeroMqQueue(
        (f"tcp://127.0.0.1:{port}", key),
        is_server=False,
        socket_type=zmq_pull(),
        use_hmac_encryption=True,
    )
    pipe = CountingPipeline(initial=False)
    stop = threading.Event()
    dispatcher = None
    try:
        with harness(pipe, forward_controls=False) as h:
            h.executor.controls_ipc = worker_ctl  # the worker thread drains this socket
            h.client.controls_ipc = client_ctl

            def dispatch():
                while not stop.is_set():
                    h.client._process_controls()
                    time.sleep(0.002)

            dispatcher = threading.Thread(target=dispatch, daemon=True)
            dispatcher.start()
            s = h.open()
            h.payload_gate.clear()  # submits never reach the worker from here on
            results = s.results()
            outcome = {}

            def read():
                try:
                    next(results)
                except BaseException as exc:  # noqa: BLE001
                    outcome["read"] = exc

            def submit():
                try:
                    s.submit("controls", 1)
                except BaseException as exc:  # noqa: BLE001
                    outcome["submit"] = exc

            reader, submitter = threading.Thread(target=read), threading.Thread(target=submit)
            reader.start()
            submitter.start()
            _wait_until(lambda: h.held_payloads, what="the stuck submit")
            s.close()  # the close travels on the real control socket
            reader.join(timeout=10)
            submitter.join(timeout=10)
            assert isinstance(outcome.get("read"), SessionEnded)
            assert isinstance(outcome.get("submit"), SessionEnded)
            assert s.closed and pipe.closed == 1 and pipe.advances == []
            stop.set()
            dispatcher.join(timeout=5)
            assert not dispatcher.is_alive()
    finally:
        stop.set()
        if dispatcher is not None:
            dispatcher.join(timeout=5)
        client_ctl.close()
        worker_ctl.close()


def zmq_push():
    import zmq

    return zmq.PUSH


def zmq_pull():
    import zmq

    return zmq.PULL


# --------------------------------------------------------------------------- the state model itself


def test_a_cancelled_reader_loses_no_media_and_order_holds(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:

        async def run():
            channel = h.client.open_session(_request(1))
            await channel.wait_open_async()
            s = VisualGenSession(h.client, channel)
            results = s.results()
            first = asyncio.ensure_future(results.__anext__())
            await asyncio.sleep(0.05)  # parked: nothing published yet
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert channel._reader is None  # the reservation went with the caller
            await s.submit_async("controls", 1)
            await s.submit_async("controls", 2)
            r0 = await results.__anext__()  # nothing was lost, order holds
            assert r0.advance_index == 0 and _pixels(r0) == (0, 1, 0)
            r1 = await results.__anext__()
            assert r1.advance_index == 1 and _pixels(r1) == (1, 2, 0)
            await s.close_async()

        asyncio.run(run())


def test_channel_failure_before_any_call_raises_everywhere():
    channel = SessionChannel(7)
    channel.fail(RuntimeError("gone"))
    with pytest.raises(RuntimeError, match="gone"):
        channel.reserve_submit("x")
    channel.reserve_read()
    with pytest.raises(RuntimeError, match="gone"):
        channel.take(lambda m: m)
    channel.release_read()
    with pytest.raises(RuntimeError, match="gone"):
        channel.wait_open()
    assert channel.lease_lost is not None and "gone" in channel.lease_lost


def test_explicit_close_discards_a_queued_terminal(harness):
    pipe = CountingPipeline(initial=False, length=1)
    with harness(pipe) as h:
        s = h.open()
        channel = h.client._sessions[s.request_id]
        s.submit("controls", 1)  # media 0, then the normal terminal, both unread
        _wait_until(lambda: any(m.type is ResultType.TERMINAL for m in channel._results))
        s.close()
        with pytest.raises(SessionEnded):
            next(s.results())
        assert channel.lockstep_due() == 0


def test_channel_members_are_the_nine_and_the_session_the_three(harness):
    """The state model: one authoritative member per fact, nothing duplicated."""
    with harness(CountingPipeline(initial=False)) as h:
        with h.open() as s:
            channel = h.client._sessions[s.request_id]
            assert set(vars(channel)) == {
                "session_id",
                "_opened",
                "_closed",
                "_lock",
                "_slots",
                "_submit",
                "_reader",
                "_results",
                "_lockstep_due",
            }
            assert set(vars(s)) == {"_client", "_channel", "_held"}


# --------------------------------------------------------------------------- transitions on the nine members


def _media(session_id, sequence, index):
    return SessionResultMessage(
        session_id, sequence, index, 0, {"video": ((1, 1, 2, 3, 3), torch.uint8)}, {}
    )


def _open_channel(session_id=1, lockstep=True):
    channel = SessionChannel(session_id)
    caps = {
        "inputs": [{"name": "controls", "policy": "queued"}],
        "initial_result": False,
        "lockstep": lockstep,
    }
    channel.open_succeeded(caps, [])
    return channel


def test_no_lost_wakeup_between_an_empty_take_and_the_notification():
    identity = lambda m: m  # noqa: E731
    # delivery lands between the empty observation and the wait
    channel = _open_channel()
    channel.reserve_read()
    assert channel.take(identity) is None
    channel.deliver(_media(1, 0, 0))
    channel.notification().result(timeout=1)  # fired: the delivery was not missed
    assert channel.take(identity).advance_index == 0
    channel.release_read()
    # explicit close lands in the window
    channel = _open_channel()
    channel.reserve_read()
    assert channel.take(identity) is None
    channel.request_close()
    channel.notification().result(timeout=1)
    with pytest.raises(SessionEnded):
        channel.take(identity)
    channel.release_read()
    # failure lands in the window
    channel = _open_channel()
    channel.reserve_read()
    assert channel.take(identity) is None
    channel.fail(RuntimeError("gone"))
    channel.notification().result(timeout=1)
    with pytest.raises(RuntimeError, match="gone"):
        channel.take(identity)
    channel.release_read()


def test_public_read_sees_a_delivery_injected_between_observation_and_wait(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open() as s:
            channel = h.client._sessions[s.request_id]
            original = channel.notification

            def deliver_then_wait():
                s.submit("controls", 1)  # the worker delivers while we are about to wait
                _wait_until(channel.unread_media, what="media")
                return original()

            channel.notification = deliver_then_wait
            r = next(s.results())  # returns, instead of waiting for a wakeup that already fired
            channel.notification = original
            assert r.advance_index == 0 and _pixels(r) == (0, 1, 0)


def test_cancelled_admission_never_retains_the_submit_record():
    channel = _open_channel()
    # cancellation before the reply: the reply retires the record
    fut = channel.reserve_submit("controls")
    channel.release_submit(fut)  # the caller left: marks it cancelled, keeps it
    assert channel._submit is not None and fut.cancelled()
    channel.admitted(0, None)
    assert channel._submit is None and channel.lockstep_due() == 1
    # reply first, then the caller's release: the record retires on release
    fut = channel.reserve_submit("controls")
    channel.admitted(1, None)
    assert channel._submit is not None and fut.result(timeout=1) == 1
    channel.release_submit(fut)
    assert channel._submit is None and channel.lockstep_due() == 2
    # and a new submit is possible after both
    fut = channel.reserve_submit("controls")
    channel.admitted(2, None)
    channel.release_submit(fut)
    assert channel.lockstep_due() == 3


def test_worker_failure_during_a_requested_close_stays_a_failure():
    # close requested while opening, then the group dies
    client = _aborting_client()
    channel = client.open_session(_request(1))
    closing = client.session_close(channel)
    client._abort_worker_group("probe: worker exited during opening")
    with pytest.raises(RuntimeError, match="exited"):
        channel.wait_open(timeout=1)
    with pytest.raises(RuntimeError, match="exited"):
        closing.result(timeout=1)
    assert channel.failure is not None and "exited" in channel.lease_lost
    # the group dies first, then a close is requested: the same failed operation
    client = _aborting_client()
    channel = client.open_session(_request(2))
    client._abort_worker_group("probe: worker exited")
    closing = client.session_close(channel)
    with pytest.raises(RuntimeError, match="exited"):
        closing.result(timeout=1)
    session = VisualGenSession(client, channel)
    session.close()  # the public close swallows the worker failure
    assert session.closed


def test_cancelled_opener_keeps_its_cancellation_while_the_failure_reaches_the_rest(harness):
    with harness(CountingPipeline(initial=False)) as h:

        async def run():
            h.hold(SessionOpened)
            task = asyncio.ensure_future(h.client.open_session_async(_request(1)))
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: _wait_until(lambda: h.held(SessionOpened)))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            channel = h.client._sessions[1]
            channel.fail(RuntimeError("worker died"))
            with pytest.raises(RuntimeError, match="worker died"):
                channel.closing().result(timeout=1)
            h.release()  # the late SessionOpened finds a failed channel: nothing changes
            assert channel.failure is not None

        asyncio.run(run())


def test_known_terminal_is_returned_without_looking_at_the_advance_value(harness):
    pipe = CountingPipeline(initial=False, length=1)
    with harness(pipe) as h:
        with h.open() as s:
            assert s.advance(1).advance_index == 0
            channel = h.client._sessions[s.request_id]
            _wait_until(lambda: _terminal_next(channel), what="the terminal")
            terminal = s.advance(object())  # unsupported, oversized: irrelevant, unused
            assert terminal.done and terminal.error is None
            with pytest.raises(SessionEnded):
                s.advance([1] * 100)
    pipe = CountingPipeline(initial=False)
    with harness(pipe, SimpleNamespace(delivery_slots=2, idle_timeout=0.15)) as h:
        s = h.open()
        channel = h.client._sessions[s.request_id]
        _wait_until(lambda: _closed(channel), what="the idle close")
        terminal = s.advance(object())  # retained timeout terminal, value unused
        assert terminal.done and "idle timeout" in terminal.error


def test_rejected_advance_value_withdraws_its_reservation(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        with h.open() as s:
            channel = h.client._sessions[s.request_id]
            with pytest.raises(ValueError):
                s.advance(-1)  # the worker's validator refuses it
            with pytest.raises(ValueError):
                s.advance(object())
            assert (
                channel._submit is None and channel._reader is None and channel.lockstep_due() == 0
            )
            r = s.advance(1)
            assert r.advance_index == 0 and pipe.advances == [{"controls": 1, "camera": 0}]


def test_unexpected_advance_index_leaves_the_result_for_a_normal_read():
    channel = _open_channel()
    channel.reserve_read()
    channel.deliver(_media(1, 0, 5))  # the worker (misbehaving) answers with advance 5
    with pytest.raises(SessionError, match="ordering"):
        channel.take(lambda m: m, expected_index=0)
    assert channel.unread_media()  # the head stayed
    assert channel.take(lambda m: m).advance_index == 5  # a plain read gets it
    channel.release_read()


# --------------------------------------------------------------------------- publication against retirement


def test_close_between_reservation_and_enqueue_leaves_no_stale_submit():
    """The worker's close lands after the submit reserved its record and before
    its command is queued: the command is never queued, the caller gets
    SessionEnded, and the next session is not blocked behind a stale submit."""
    import zmq

    client = _sending_client(None)
    client._monitor_worker_liveness = False
    channel = _declared(client, 1)
    client._process_requests()
    original = channel.reserve_submit

    def reserve_then_close(name, for_advance=False):
        fut = original(name, for_advance)
        client._route_session_reply(SessionClosed(1))  # retirement runs now: nothing queued yet
        return fut

    channel.reserve_submit = reserve_then_close
    with pytest.raises(SessionEnded):
        client.session_submit(channel, "x", [1.0])
    assert client.pending_requests.empty() and client._sessions == {}
    second = client.open_session(_request(2))
    client.requests_ipc.exc = zmq.Again()
    for _ in range(3):
        client._process_requests()  # a stale submit would be retried here forever
    client.requests_ipc.exc = None
    client._process_requests()
    assert isinstance(client.requests_ipc.sent[-1], SessionOpen)
    assert all(not isinstance(m, SessionSubmit) for m in client.requests_ipc.sent)
    assert second is client._sessions[2]


def test_close_between_advance_reservation_and_enqueue_leaves_no_stale_submit():
    client = _sending_client(None)
    client._monitor_worker_liveness = False
    channel = _declared(client, 1)
    client._process_requests()
    original = channel.plan_advance

    def plan_then_close(name):
        fut = original(name)
        client._route_session_reply(SessionClosed(1))
        return fut

    channel.plan_advance = plan_then_close
    with pytest.raises(SessionEnded):
        client.session_submit(channel, "x", [1.0], for_advance=True)
    assert client.pending_requests.empty() and channel._submit is None


def test_abort_between_reservation_and_enqueue_leaves_no_stale_submit():
    client = _sending_client(None)
    channel = _declared(client, 1)
    client._process_requests()
    original = channel.reserve_submit

    def reserve_then_abort(name, for_advance=False):
        fut = original(name, for_advance)
        client._abort_worker_group("probe: worker exited")  # drains and fails under the lock
        return fut

    channel.reserve_submit = reserve_then_abort
    with pytest.raises(RuntimeError, match="exited"):
        client.session_submit(channel, "x", [1.0])
    assert client.pending_requests.empty() and client._sessions == {}
    with pytest.raises(RuntimeError, match="exited"):
        client.open_session(_request(2))


def test_interrupted_advance_validation_releases_both_reservations(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        with h.open() as s:
            channel = h.client._sessions[s.request_id]
            import tensorrt_llm._torch.visual_gen.executor as executor_module

            original = executor_module.SessionSubmit

            def interrupt(*args):  # between the reservation and the publication
                raise KeyboardInterrupt

            executor_module.SessionSubmit = interrupt
            try:
                with pytest.raises(KeyboardInterrupt):
                    s.advance(1)
            finally:
                executor_module.SessionSubmit = original
            assert channel._submit is None and channel._reader is None  # nothing leaked
            r = s.advance(1)
            assert r.advance_index == 0 and pipe.advances == [{"controls": 1, "camera": 0}]


def _worker_closes_after_the_reservation(h, s):
    """Wrap plan_advance so the idle close (terminal retained, SessionClosed
    routed) has landed after the submit was reserved and before it is published."""
    channel = h.client._sessions[s.request_id]
    original = channel.plan_advance

    def reserve_then_wait_for_close(name):
        fut = original(name)
        _wait_until(lambda: _closed(channel), what="the idle close")
        return fut

    channel.plan_advance = reserve_then_wait_for_close
    return channel


def test_advance_returns_the_terminal_when_the_close_lands_between_reservation_and_publication(
    harness,
):
    with harness(
        CountingPipeline(initial=False), SimpleNamespace(delivery_slots=2, idle_timeout=0.15)
    ) as h:
        s = h.open()
        channel = _worker_closes_after_the_reservation(h, s)
        terminal = s.advance(1)
        assert terminal.done and "idle timeout" in terminal.error
        assert channel._submit is None and channel._reader is None
        assert h.pipeline.advances == []  # nothing was sent
        with pytest.raises(SessionEnded):
            s.advance(2)


def test_advance_async_returns_the_terminal_when_the_close_lands_between_reservation_and_publication(
    harness,
):
    with harness(
        CountingPipeline(initial=False), SimpleNamespace(delivery_slots=2, idle_timeout=0.15)
    ) as h:

        async def run():
            channel = h.client.open_session(_request(1))
            await channel.wait_open_async()
            s = VisualGenSession(h.client, channel)
            _worker_closes_after_the_reservation(h, s)
            terminal = await s.advance_async(1)
            assert terminal.done and "idle timeout" in terminal.error
            assert channel._submit is None and channel._reader is None
            assert h.pipeline.advances == []
            with pytest.raises(SessionEnded):
                await s.advance_async(2)

        asyncio.run(run())
