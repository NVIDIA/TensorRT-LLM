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
"""Session bookkeeping on the worker: admission, the advance condition, result
assembly and the slot pool. No pipeline, no processes."""

import gc

import pytest
import torch

from tensorrt_llm._torch.visual_gen.pipeline import (
    AdvanceOutcome,
    SessionCapabilities,
    SessionInputSpec,
    SessionStreamSpec,
)
from tensorrt_llm._torch.visual_gen.session_protocol import (
    ResultType,
    SessionEnded,
    SessionQueueFull,
)
from tensorrt_llm._torch.visual_gen.session_state import SessionState
from tensorrt_llm._torch.visual_gen.slot_pool import SlotPool, make_view, map_slots

pytestmark = pytest.mark.cpu_only


def _video_stream(frames=8, h=4, w=6):
    return SessionStreamSpec(
        name="video", kind="video", dtype=torch.uint8, max_shape=(1, frames, h, w, 3), rate=24.0
    )


def _positive(value):
    if not isinstance(value, int) or value < 0:
        raise ValueError("controls must be a non-negative int")


def lockstep_caps(initial=True, length=None):
    return SessionCapabilities(
        inputs=(SessionInputSpec(name="controls", validator=_positive, queue_depth=2),),
        streams=(_video_stream(),),
        initial_result=initial,
        length=length,
        lockstep=True,
    )


def steered_caps():
    return SessionCapabilities(
        inputs=(
            SessionInputSpec(name="controls", queue_depth=3),
            SessionInputSpec(name="camera", policy="live", default=(0.0, 0.0)),
        ),
        streams=(
            _video_stream(),
            SessionStreamSpec(
                name="audio", kind="audio", dtype=torch.float32, max_shape=(1, 2, 1024), axis=2
            ),
        ),
    )


# --------------------------------------------------------------------------- capabilities


def test_capabilities_reject_bad_declarations():
    with pytest.raises(ValueError):
        SessionInputSpec(name="x", policy="sometimes")
    with pytest.raises(ValueError):
        SessionInputSpec(name="x", queue_depth=0)
    with pytest.raises(ValueError):
        SessionStreamSpec(
            name="v", kind="video", dtype=torch.uint8, max_shape=(1, 2), placement="pinned"
        )
    with pytest.raises(ValueError):
        SessionStreamSpec(name="v", kind="video", dtype=torch.uint8, max_shape=(0, 1))
    with pytest.raises(ValueError):
        SessionStreamSpec(name="v", kind="video", dtype=torch.uint8, max_shape=(1, 2), axis=5)
    with pytest.raises(ValueError):
        SessionCapabilities(inputs=(), streams=())
    with pytest.raises(ValueError):
        SessionCapabilities(
            inputs=(SessionInputSpec(name="a"), SessionInputSpec(name="a")),
            streams=(_video_stream(),),
        )
    with pytest.raises(ValueError):
        SessionCapabilities(inputs=(), streams=(_video_stream(),), lockstep=True)


def test_public_form_is_picklable_and_callable_free():
    import pickle

    public = lockstep_caps().public()
    pickle.dumps(public)
    assert public["inputs"][0]["policy"] == "queued"
    assert public["streams"][0]["dtype"] == "torch.uint8"
    assert public["lockstep"] is True
    assert public["streams"][0]["placement"] == "host"
    assert "validate" not in public["inputs"][0]


# --------------------------------------------------------------------------- admission


def test_queued_admission_returns_consumption_index_and_bounds_the_queue():
    state = SessionState(7, lockstep_caps())
    assert state.admit("controls", 1) == 0
    assert state.admit("controls", 2) == 1
    with pytest.raises(SessionQueueFull):
        state.admit("controls", 3)
    with pytest.raises(ValueError):
        state.admit("controls", -1)
    with pytest.raises(ValueError):
        state.admit("nope", 1)


def test_live_admission_returns_next_dispatch_and_newest_wins():
    state = SessionState(1, steered_caps())
    assert state.admit("camera", (1.0, 0.0)) == 0
    assert state.admit("camera", (2.0, 0.0)) == 0
    state.admit("controls", "a")
    assert state.can_advance(free_slots=1)
    _, inputs = state.dispatch()
    assert inputs == {"controls": "a", "camera": (2.0, 0.0)}
    # Submitted while advance 0 is in flight: observed from advance 1 on.
    assert state.admit("camera", (3.0, 0.0)) == 1


def test_advance_condition():
    state = SessionState(1, steered_caps())
    assert not state.can_advance(free_slots=1)  # not initialized
    assert not state.can_advance(free_slots=1)  # queue empty
    state.admit("controls", "a")
    assert not state.can_advance(free_slots=0)  # no slot
    assert state.can_advance(free_slots=1)
    state.dispatch()
    state.admit("controls", "b")
    assert state.can_advance(free_slots=1)
    state.terminal_result()
    assert not state.can_advance(free_slots=1)  # ended


def test_finite_session_exhausts_and_refuses_extra_values():
    state = SessionState(1, lockstep_caps(initial=False, length=2))
    assert state.admit("controls", 1) == 0
    assert state.admit("controls", 2) == 1
    with pytest.raises(SessionEnded):
        state.admit("controls", 3)
    for _ in range(2):
        assert state.can_advance(1)
        state.dispatch()
    assert not state.can_advance(1)
    assert state.complete()


def test_admission_refused_after_the_terminal():
    state = SessionState(1, lockstep_caps())
    state.terminal_result()
    with pytest.raises(SessionEnded):
        state.admit("controls", 1)


# --------------------------------------------------------------------------- results


def test_media_result_advances_the_stream_positions():
    state = SessionState(3, steered_caps())
    written = {
        "video": ((1, 8, 4, 6, 3), torch.uint8),
        "audio": ((1, 2, 1000), torch.float32),
    }
    first = state.media_result(0, 0, written)
    assert first.sequence == 0 and first.advance_index == 0 and first.slot_id == 0
    assert first.position == {"video": (0, 8), "audio": (0, 1000)}
    assert first.streams["video"] == ((1, 8, 4, 6, 3), torch.uint8)
    second = state.media_result(1, 1, {"video": written["video"]})
    assert second.sequence == 1
    assert second.position == {"video": (8, 8)}  # audio omitted: its position stays
    with pytest.raises(ValueError):
        state.media_result(2, 0, {"depth": ((1, 8, 4, 6), torch.uint8)})  # not declared
    assert state.stream_positions == {"video": 16, "audio": 1000}  # nothing moved
    terminal = state.terminal_result()
    assert terminal.type is ResultType.TERMINAL and terminal.error is None
    assert terminal.sequence == 2
    with pytest.raises(RuntimeError):
        state.terminal_result()  # exactly one


# --------------------------------------------------------------------------- slot pool


def test_slot_pool_acquire_release_and_views():
    caps = steered_caps()
    pool = SlotPool(caps, num_slots=2, device=torch.device("cpu"))
    assert pool.free_count == 2
    a_id, a = pool.acquire()
    b_id, b = pool.acquire()
    assert (a_id, b_id) == (0, 1)
    with pytest.raises(RuntimeError):
        pool.acquire()  # ask free_count first
    view = a["video"].view((1, 8, 4, 6, 3), torch.uint8)
    view.fill_(7)
    assert a["video"].is_view_of(view) and not b["video"].is_view_of(view)
    with pytest.raises(ValueError):
        a["video"].view((1, 9, 4, 6, 3), torch.uint8)
    with pytest.raises(ValueError):
        a["video"].view((1, 8, 4, 6, 3), torch.float32)
    pool.release(0)
    with pytest.raises(ValueError):
        pool.release(0)
    assert pool.acquire()[0] == 0


def test_client_map_sees_the_worker_write_without_a_copy():
    caps = lockstep_caps()
    pool = SlotPool(caps, num_slots=1, device=torch.device("cpu"))
    handles = pool.export_handles()
    client = map_slots(handles)
    _, slot = pool.acquire()
    worker_view = slot["video"].view((1, 2, 4, 6, 3), torch.uint8)
    worker_view.fill_(0)
    worker_view[0, 1] = 9
    client_view = make_view(client[0]["video"], tuple(worker_view.shape), worker_view.dtype)
    assert client_view.shape == (1, 2, 4, 6, 3)
    assert int(client_view[0, 1, 0, 0, 0]) == 9 and int(client_view[0, 0, 0, 0, 0]) == 0
    assert client_view.data_ptr() == worker_view.data_ptr()  # same process, same bytes


def test_outcome_defaults():
    outcome = AdvanceOutcome()
    assert outcome.event is None and not outcome.done


def test_live_input_must_declare_a_default():
    with pytest.raises(ValueError, match="needs a default"):
        SessionInputSpec(name="camera", policy="live")
    # A queued input reads no default, so it may leave it out.
    assert SessionInputSpec(name="controls").default is None


def _nonnegative(value):
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{value!r} is not a nonnegative int")


def test_declared_default_is_checked_like_a_submitted_value():
    with pytest.raises(ValueError, match="default is refused"):
        SessionInputSpec(name="camera", policy="live", validator=_nonnegative, default=-1)
    with pytest.raises(ValueError, match="default is refused"):
        SessionInputSpec(name="camera", policy="live", validator=_nonnegative, default=None)

    def allows_none(value):
        if value is not None:
            _nonnegative(value)

    spec = SessionInputSpec(name="camera", policy="live", validator=allows_none, default=None)
    assert spec.default is None


def test_valid_live_default_reaches_the_first_advance():
    caps = SessionCapabilities(
        inputs=(
            SessionInputSpec(name="controls"),
            SessionInputSpec(name="camera", policy="live", validator=_nonnegative, default=3),
        ),
        streams=(_video_stream(),),
    )
    state = SessionState(1, caps)
    state.admit("controls", 1)
    assert state.can_advance(1)
    assert state.dispatch() == (0, {"controls": 1, "camera": 3})


def test_queue_full_rejection_leaves_the_next_index_unchanged():
    state = SessionState(1, lockstep_caps(initial=False))
    assert state.admit("controls", 1) == 0
    assert state.admit("controls", 2) == 1
    with pytest.raises(SessionQueueFull):
        state.admit("controls", 3)
    assert state.admitted_count("controls") == 2 and list(state.inputs["controls"]) == [1, 2]
    state.dispatch()
    assert state.admit("controls", 3) == 2  # the next index is the one after the admitted two


def test_supported_input_forms_survive_the_wire_serialization():
    """Every accepted form goes through what the request socket does to it
    (pickle) and comes back equal."""
    import pickle

    import numpy as np

    from tensorrt_llm._torch.visual_gen.session_protocol import SessionSubmit

    forms = [
        1,
        2.5,
        True,
        "text",
        b"bytes",
        bytearray(b"ba"),
        [1.0, 2.0, [3.0]],
        (1, 2),
        {"k": [1, 2]},
        torch.arange(6, dtype=torch.float32).reshape(2, 3)[:, 1:],
        np.arange(6, dtype=np.float32).reshape(2, 3)[:, 1:],
    ]
    for value in forms:
        back = pickle.loads(pickle.dumps(SessionSubmit(1, "x", value))).value
        if isinstance(value, torch.Tensor):
            assert torch.equal(back, value)
        elif isinstance(value, np.ndarray):
            assert np.array_equal(back, value)
        else:
            assert back == value
    with pytest.raises(TypeError):
        pickle.dumps(memoryview(b"abc"))  # which is why it is not a supported form


# --------------------------------------------------------------------------- delivery slots


def _two_stream_caps(n=4):
    return SessionCapabilities(
        streams=(
            SessionStreamSpec(
                name="video", kind="video", dtype=torch.float32, max_shape=(n,), axis=0
            ),
            SessionStreamSpec(
                name="audio", kind="audio", dtype=torch.float32, max_shape=(n,), axis=0
            ),
        ),
    )


def test_buffer_views_fit_or_raise():
    pool = SlotPool(_two_stream_caps(), num_slots=1, device=torch.device("cpu"), local=True)
    _, slot = pool.acquire()
    video = slot["video"]  # 4 float32 by _two_stream_caps: 16 bytes
    assert video.nbytes == 16 and video.storage.is_contiguous()
    with pytest.raises(ValueError, match="bytes"):
        video.view((5,), torch.float32)
    view = video.view((3,), torch.float32)
    assert view.shape == (3,) and video.is_view_of(view)
    assert video.view((2, 2), torch.float32).shape == (2, 2)  # any shape that fits
    assert not video.is_view_of(torch.zeros(3))  # someone else's memory


def test_partial_pool_construction_undoes_every_registration(monkeypatch):
    import tensorrt_llm._torch.visual_gen.slot_pool as pool_module

    registered, unregistered = [], []

    def fake_lock(storage):
        registered.append(storage.data_ptr())
        return True

    def fake_unlock(storage):
        unregistered.append(storage.data_ptr())

    monkeypatch.setattr(pool_module, "_page_lock", fake_lock)
    monkeypatch.setattr(pool_module, "_page_unlock", fake_unlock)
    original_allocate = pool_module._stream_buffer
    calls = {"n": 0}

    def failing_allocate(spec, device, *, local):
        calls["n"] += 1
        if calls["n"] == 3:  # second slot, first stream: mid-way through the pool
            raise MemoryError("probe: out of memory")
        return original_allocate(spec, device, local=local)

    monkeypatch.setattr(pool_module, "_stream_buffer", failing_allocate)
    with pytest.raises(MemoryError, match="probe"):
        SlotPool(_two_stream_caps(), num_slots=2, device=torch.device("cpu"))
    gc.collect()  # the traceback kept the half-built pool alive
    assert len(registered) == 2 and sorted(unregistered) == sorted(registered)  # each undone once
    # and a failure inside one slot's streams behaves the same
    registered.clear()
    unregistered.clear()
    calls["n"] = 0

    def failing_second_stream(spec, device, *, local):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MemoryError("probe: out of memory")
        return original_allocate(spec, device, local=local)

    monkeypatch.setattr(pool_module, "_stream_buffer", failing_second_stream)
    with pytest.raises(MemoryError, match="probe"):
        SlotPool(_two_stream_caps(), num_slots=2, device=torch.device("cpu"))
    gc.collect()
    assert registered and sorted(unregistered) == sorted(registered)


def test_registered_pointer_is_the_exported_storage_pointer(monkeypatch):
    """The handle is exported before the page-lock, so the address the client
    maps is the address that was registered (sharing moves the storage)."""
    import tensorrt_llm._torch.visual_gen.slot_pool as pool_module

    registered = []
    monkeypatch.setattr(
        pool_module,
        "_page_lock",
        lambda storage: bool(registered.append(storage.data_ptr())) or True,
    )
    monkeypatch.setattr(pool_module, "_page_unlock", lambda storage: None)
    pool = SlotPool(_two_stream_caps(), num_slots=1, device=torch.device("cpu"))
    exported = [buf.storage.data_ptr() for slot in pool._slots for buf in slot.values()]
    assert sorted(registered) == sorted(exported)
    client = map_slots(pool.export_handles())
    view = make_view(client[0]["video"], (4,), torch.float32)
    assert view.data_ptr() == pool._slots[0]["video"].storage.data_ptr()
