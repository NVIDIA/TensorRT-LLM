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
"""Device-placed media of a session: the caller's queued reads against slot reuse."""

import multiprocessing
from types import SimpleNamespace

import pytest
import torch
from test_session_executor import CountingPipeline, Harness

from tensorrt_llm._torch.visual_gen.pipeline import (
    AdvanceResult,
    SessionCapabilities,
    SessionStreamSpec,
)
from tensorrt_llm._torch.visual_gen.slot_pool import (
    SlotPool,
    make_view,
    map_slots,
    open_return_events,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device media needs CUDA")

SHAPE = (1, 4, 256, 256, 3)
RESULTS = 6
# The caller's own work queued ahead of each read of a slot, in device clock ticks.
PENDING_TICKS = 300_000_000


def _device_caps():
    return SessionCapabilities(
        inputs=(),
        streams=(
            SessionStreamSpec(
                name="video", kind="video", dtype=torch.uint8, max_shape=SHAPE, placement="device"
            ),
        ),
        initial_result=False,
        length=RESULTS,
        lockstep=False,
    )


class DeviceMediaPipeline(CountingPipeline):
    """Autonomous: each advance fills a device-placed slot with its index on the
    worker thread's stream and returns the event recorded behind that write."""

    def __init__(self):
        super().__init__(queued=False, initial=False, length=RESULTS)

    def session_capabilities(self, req):
        return _device_caps()

    def advance(self, state, inputs, slot):
        index = state.count
        state.count += 1
        view = slot["video"].view(SHAPE, torch.uint8)
        view.fill_(index)
        event = torch.cuda.Event()
        event.record()
        return AdvanceResult(written={"video": view}, event=event, done=state.count >= RESULTS)


def _warm_up(*streams):
    """Run once, on each stream, what the tests time against pending device
    work, so no first-use setup on the host outlasts that work."""
    sample = torch.zeros(SHAPE, dtype=torch.uint8, device="cuda")
    for stream in streams or (torch.cuda.current_stream(),):
        with torch.cuda.stream(stream):
            bool((sample.clone() == 0).all())
            torch.cuda._sleep(1000)
            torch.cuda.Event().record()
    if len(streams) > 1:
        streams[0].wait_stream(streams[1])
    torch.cuda.synchronize()


def _read_view(result, _other):
    return result.video == result.advance_index


def _read_copy(result, _other):
    return result.copy().video == result.advance_index


def _read_copy_on_another_stream(result, other):
    with torch.cuda.stream(other):
        torch.cuda._sleep(PENDING_TICKS)
        return result.copy().video == result.advance_index


@pytest.mark.parametrize("read", [_read_view, _read_copy, _read_copy_on_another_stream])
def test_queued_reads_of_a_slot_run_before_the_worker_writes_it_again(read):
    """The caller reads each result on its own stream, behind work of its own, and
    asks for the next one without waiting. With one slot the worker's next advance
    writes the slot the caller's reads are still queued on; every read must see the
    result it was issued for. ``pending`` proves the reads had not run when the
    slot went back."""
    caller, other = torch.cuda.Stream(), torch.cuda.Stream()
    _warm_up(caller, other)
    intact, pending = [], []
    config = SimpleNamespace(delivery_slots=1, idle_timeout=None)
    with Harness(DeviceMediaPipeline(), config) as h, torch.cuda.stream(caller):
        with h.open() as session, session.results() as results:
            while True:
                result = next(results)
                if result.done:
                    assert result.error is None, result.error
                    break
                torch.cuda._sleep(PENDING_TICKS)
                intact.append(read(result, other).all())
                marker = torch.cuda.Event()
                marker.record()
                pending.append(not marker.query())
        torch.cuda.synchronize()
    assert pending == [True] * RESULTS, "the reads were not pending when the slot went back"
    assert [bool(flag) for flag in intact] == [True] * RESULTS


def _pool_owner(conn):
    """The worker's side in a process of its own: a one-slot pool, rewritten with
    each value the test sends, through the pool's own acquire."""
    pool = SlotPool(_device_caps(), 1, torch.device("cuda:0"))
    conn.send((pool.export_handles(), pool.export_return_events()))
    for value in iter(conn.recv, None):
        slot_id, slot = pool.acquire()
        slot["video"].view(SHAPE, torch.uint8).fill_(value)
        torch.cuda.synchronize()
        pool.release(slot_id)
        conn.send(value)


def test_pool_in_another_process_waits_for_the_reads_recorded_on_its_slot():
    """The event handles a pool exports order a write in the pool's process behind
    reads queued in the process that mapped the slot."""
    ctx = multiprocessing.get_context("spawn")
    conn, child_conn = ctx.Pipe()
    owner = ctx.Process(target=_pool_owner, args=(child_conn,), daemon=True)
    owner.start()
    try:
        assert conn.poll(600), "the pool's process did not come up"
        handles, event_handles = conn.recv()
        slots = map_slots(handles)
        event = open_return_events(event_handles, slots)[0]
        view = make_view(slots[0]["video"], SHAPE, torch.uint8)
        _warm_up()
        intact, pending = [], []
        for value in (1, 2, 3, 0):
            conn.send(value)
            assert conn.poll(60), "the pool's process did not write the slot"
            conn.recv()
            if value == 0:
                break  # the write behind the last read
            torch.cuda._sleep(PENDING_TICKS)
            intact.append((view == value).all())
            event.record(torch.cuda.current_stream())
            pending.append(not event.query())
        torch.cuda.synchronize()
        assert pending == [True] * 3, "the reads were not pending when the slot went back"
        assert [bool(flag) for flag in intact] == [True] * 3
        del view, slots, event
        conn.send(None)
        owner.join(timeout=60)
        assert owner.exitcode == 0
    finally:
        if owner.is_alive():
            owner.kill()
            owner.join()


def test_failed_host_registration_leaves_no_error_for_the_next_kernel():
    """The runtime refuses to register a range twice. The refusal must not be
    raised by whatever kernel is launched next."""
    from tensorrt_llm._torch.visual_gen.slot_pool import _page_lock, _page_unlock

    torch.zeros(1, device="cuda")
    storage = torch.empty(1 << 20, dtype=torch.uint8).share_memory_()
    assert _page_lock(storage)
    try:
        assert not _page_lock(storage)
        assert int((torch.ones(4, device="cuda") + 1).sum()) == 8
    finally:
        _page_unlock(storage)
