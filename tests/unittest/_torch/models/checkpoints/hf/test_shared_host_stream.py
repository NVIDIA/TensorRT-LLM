# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bounded SafeTensors shared-host streaming."""

import json
import os
import threading
from dataclasses import replace

import pytest

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BorrowedWeightStorageRetentionError,
    WeightGroup,
)
from tensorrt_llm._torch.models.checkpoints.hf import (
    shared_host_stream as shared_host_stream_module,
)
from tensorrt_llm._torch.models.checkpoints.hf.shared_host_stream import (
    HostMemoryRegistration,
    SharedHostProducerMode,
    SharedHostStreamError,
    SharedHostStreamUnavailableError,
    _allocate_mpi_shared_window,
    _assigned_extents,
    _plan_batches,
    _producer_worker_counts,
    open_shared_host_weight_stream,
    prepare_shared_host_weight_stream,
)


def _write_safetensors(path, entries, payload):
    header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)


def _u8_checkpoint(path, tensors):
    entries = {}
    payload = bytearray()
    for key, value in tensors.items():
        start = len(payload)
        payload.extend(value)
        entries[key] = {
            "dtype": "U8",
            "shape": [len(value)],
            "data_offsets": [start, len(payload)],
        }
    _write_safetensors(path, entries, bytes(payload))


class _FakeCommunicator:
    def __init__(self, *, size=1, allgather_hook=None):
        self._size = size
        self._allgather_hook = allgather_hook
        self.call_threads = []

    def Get_rank(self):
        return 0

    def Get_size(self):
        return self._size

    def bcast(self, value, root):
        assert root == 0
        assert value is not None
        self.call_threads.append(threading.get_ident())
        return value

    def allgather(self, value):
        self.call_threads.append(threading.get_ident())
        if self._allgather_hook is not None:
            return self._allgather_hook(value)
        return [value]

    def Barrier(self):
        self.call_threads.append(threading.get_ident())


class _ThreadedCollectiveGroup:
    def __init__(self, size):
        self._size = size
        self._condition = threading.Condition()
        self._states = {}

    def exchange(self, kind, index, rank, value=None, root=0):
        key = (kind, index)
        with self._condition:
            state = self._states.setdefault(key, {"values": {}, "returned": 0})
            assert rank not in state["values"]
            state["values"][rank] = value
            self._condition.notify_all()
            ready = self._condition.wait_for(lambda: len(state["values"]) == self._size, timeout=10)
            if not ready:
                raise TimeoutError(f"Timed out in fake {kind} collective {index}")
            if kind == "bcast":
                result = state["values"][root]
            elif kind == "allgather":
                result = [state["values"][candidate] for candidate in range(self._size)]
            else:
                result = None
            state["returned"] += 1
            if state["returned"] == self._size:
                del self._states[key]
            self._condition.notify_all()
            return result


class _ThreadedCommunicator:
    def __init__(self, group, rank, size):
        self._group = group
        self._rank = rank
        self._size = size
        self._indices = {"bcast": 0, "allgather": 0, "barrier": 0}

    def _next(self, kind):
        index = self._indices[kind]
        self._indices[kind] += 1
        return index

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def bcast(self, value, root):
        return self._group.exchange("bcast", self._next("bcast"), self._rank, value, root)

    def allgather(self, value):
        return self._group.exchange("allgather", self._next("allgather"), self._rank, value)

    def Barrier(self):
        self._group.exchange("barrier", self._next("barrier"), self._rank)


class _FakeWindow:
    def __init__(self, *, fail_lock=False, fail_unlock=False):
        self.call_threads = []
        self.events = []
        self.fail_lock = fail_lock
        self.fail_unlock = fail_unlock
        self.locked = False
        self.lock_calls = 0
        self.unlock_calls = 0
        self.freed = False

    def Lock_all(self):
        self.call_threads.append(threading.get_ident())
        self.events.append("lock_all")
        self.lock_calls += 1
        if self.fail_lock:
            # Model MPI's unspecified local epoch state after an error.
            self.locked = True
            raise RuntimeError("injected Win.Lock_all failure")
        assert not self.locked
        self.locked = True

    def Sync(self):
        self.call_threads.append(threading.get_ident())
        self.events.append("sync")
        assert self.locked

    def Unlock_all(self):
        self.call_threads.append(threading.get_ident())
        self.events.append("unlock_all")
        self.unlock_calls += 1
        assert self.locked
        if self.fail_unlock:
            raise RuntimeError("injected Win.Unlock_all failure")
        self.locked = False

    def Free(self):
        self.call_threads.append(threading.get_ident())
        self.events.append("free")
        assert not self.locked
        self.freed = True


class _FakeWindowFactory:
    def __init__(self):
        self.window = None
        self.arena = None
        self.total_bytes = None
        self.calls = 0

    def __call__(self, communicator, total_bytes):
        del communicator
        self.calls += 1
        self.total_bytes = total_bytes
        self.window = _FakeWindow()
        self.arena = memoryview(bytearray(total_bytes))
        return self.window, self.arena


class _ExportCheckingWindow(_FakeWindow):
    def __init__(self, storage):
        super().__init__()
        self._storage = storage

    def Free(self):
        # Resizing a bytearray fails while any worker traceback still retains
        # a destination memoryview into the emulated shared window.
        self._storage.extend(b"\0")
        self._storage.pop()
        super().Free()


class _ExportCheckingWindowFactory:
    def __init__(self):
        self.window = None
        self.arena = None
        self.storage = None

    def __call__(self, communicator, total_bytes):
        del communicator
        self.storage = bytearray(total_bytes)
        self.window = _ExportCheckingWindow(self.storage)
        self.arena = memoryview(self.storage)
        return self.window, self.arena


class _ThreadedWindowFactory:
    def __init__(self):
        self._lock = threading.Lock()
        self._storage = None
        self.windows = []

    def __call__(self, communicator, total_bytes):
        del communicator
        with self._lock:
            if self._storage is None:
                self._storage = bytearray(total_bytes)
            assert len(self._storage) == total_bytes
            window = _FakeWindow()
            self.windows.append(window)
            return window, memoryview(self._storage)


class _ThreadedEpochFailureWindowFactory:
    def __init__(self, *, failing_lock_rank=None, failing_unlock_rank=None):
        self._failing_lock_rank = failing_lock_rank
        self._failing_unlock_rank = failing_unlock_rank
        self._lock = threading.Lock()
        self._storage = None
        self.windows = {}

    def __call__(self, communicator, total_bytes):
        rank = communicator.Get_rank()
        with self._lock:
            if self._storage is None:
                self._storage = bytearray(total_bytes)
            assert len(self._storage) == total_bytes
            window = _FakeWindow(
                fail_lock=rank == self._failing_lock_rank,
                fail_unlock=rank == self._failing_unlock_rank,
            )
            self.windows[rank] = window
            return window, memoryview(self._storage)


class _InjectedReleaseArena:
    def __init__(self, storage, *, fail_release):
        self._view = memoryview(storage)
        self._fail_release = fail_release
        self.release_calls = 0

    def __len__(self):
        return len(self._view)

    def release(self):
        self.release_calls += 1
        if self._fail_release:
            raise BufferError("injected exported arena release failure")
        self._view.release()


class _ThreadedReleaseFailureWindowFactory:
    def __init__(self, *, failing_rank):
        self._failing_rank = failing_rank
        self._lock = threading.Lock()
        self._storage = None
        self.windows = []
        self.arenas = {}

    def __call__(self, communicator, total_bytes):
        rank = communicator.Get_rank()
        with self._lock:
            if self._storage is None:
                self._storage = bytearray(total_bytes)
            assert len(self._storage) == total_bytes
            window = _FakeWindow()
            arena = _InjectedReleaseArena(
                self._storage,
                fail_release=rank == self._failing_rank,
            )
            self.windows.append(window)
            self.arenas[rank] = arena
            return window, arena


class _RecordingReader:
    def __init__(self, *, watched_offsets=(), failing_offsets=()):
        self._watched_offsets = set(watched_offsets)
        self._failing_offsets = set(failing_offsets)
        self._lock = threading.Lock()
        self.thread_ids = set()
        self.calls = []
        self.watched = threading.Event()
        self.closed = False

    def read_into(self, source_path, source_offset, destination, cancel_event):
        with self._lock:
            self.thread_ids.add(threading.get_ident())
            self.calls.append((source_path, source_offset, len(destination)))
        if source_offset in self._watched_offsets:
            self.watched.set()
        if source_offset in self._failing_offsets:
            raise OSError("injected producer read failure")
        if cancel_event.is_set():
            raise RuntimeError("cancelled")
        descriptor = os.open(source_path, os.O_RDONLY)
        try:
            data = os.pread(descriptor, len(destination), source_offset)
        finally:
            os.close(descriptor)
        if len(data) != len(destination):
            raise EOFError("short test read")
        destination[:] = data

    def close(self):
        self.closed = True


class _BlockingReader(_RecordingReader):
    def __init__(self, blocked_offset):
        super().__init__()
        self._blocked_offset = blocked_offset
        self.blocked = threading.Event()
        self.release = threading.Event()

    def read_into(self, source_path, source_offset, destination, cancel_event):
        if source_offset == self._blocked_offset:
            self.blocked.set()
            if not self.release.wait(timeout=10):
                raise TimeoutError("Timed out waiting to release blocked checkpoint read")
        super().read_into(source_path, source_offset, destination, cancel_event)


class _CloseFailingReader(_RecordingReader):
    def __init__(self):
        super().__init__()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        raise OSError("injected reader close failure")


class _ForbiddenReader:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def read_into(self, source_path, source_offset, destination, cancel_event):
        del source_path, source_offset, destination, cancel_event
        self.calls += 1
        raise AssertionError("A nonproducer rank attempted checkpoint I/O")

    def close(self):
        self.closed = True


class _RegistrationFactory:
    def __init__(self):
        self.closed = False

    def __call__(self, arena):
        assert arena
        return HostMemoryRegistration(True, "test registration", self._mark_closed)

    def _mark_closed(self):
        self.closed = True


class _FailingRegistrationFactory:
    def __init__(self):
        self.close_calls = 0

    def __call__(self, arena):
        assert arena
        return HostMemoryRegistration(True, "test registration", self._fail_close)

    def _fail_close(self):
        self.close_calls += 1
        raise RuntimeError("injected cudaHostUnregister failure")


def _open_test_stream(
    preflight,
    groups,
    *,
    slot_bytes,
    buffer_budget_bytes=None,
    reader=None,
    world_communicator=None,
    window_factory=None,
    producer_mode=SharedHostProducerMode.SINGLE_PRODUCER,
    start=True,
):
    node_communicator = _FakeCommunicator()
    world_communicator = world_communicator or _FakeCommunicator()
    window_factory = window_factory or _FakeWindowFactory()
    registration_factory = _RegistrationFactory()
    stream = open_shared_host_weight_stream(
        preflight,
        node_communicator,
        world_communicator,
        producer_mode=producer_mode,
        group_manifest=groups,
        slot_bytes=slot_bytes,
        buffer_budget_bytes=(
            2 * slot_bytes if buffer_budget_bytes is None else buffer_budget_bytes
        ),
        read_chunk_bytes=2,
        io_workers=2,
        strict=True,
        strict_host_registration=True,
        window_factory=window_factory,
        reader=reader,
        host_registrar=registration_factory,
    )
    assert stream is not None
    if start:
        stream.start()
    return (stream, node_communicator, world_communicator, window_factory, registration_factory)


def test_slots_adapt_to_largest_packed_atomic_group(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = {"a": bytes(range(12)), "b": bytes(range(20, 25))}
    _u8_checkpoint(checkpoint, expected)
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("fused", ("a", "b")),)
    stream, _, _, window_factory, _ = _open_test_stream(
        preflight,
        groups,
        slot_bytes=8,
        buffer_budget_bytes=64,
    )

    lease = stream.begin_next()
    assert lease is not None
    assert lease.batch.group_complete
    assert len(lease.batch.segments) == 2
    assert all(
        segment.tensor_offset == 0 and segment.nbytes == segment.tensor_nbytes
        for segment in lease.batch.segments
    )
    assert stream.supports_direct_tensor_views
    direct_buffer = lease.borrow_direct_buffer(lease.batch.segments[0])
    assert direct_buffer is not None
    assert bytes(direct_buffer) == expected["a"]
    direct_buffer.release()
    stream.record_materialization(direct=True, nbytes=17)
    telemetry = stream.telemetry
    stream.complete(lease)
    assert stream.begin_next() is None
    stream.finalize()

    assert telemetry.configured_slot_bytes == 8
    assert telemetry.slot_bytes == 17
    assert telemetry.buffer_budget_bytes == 64
    assert telemetry.largest_group_nbytes == 17
    assert telemetry.groups_fitting_single_slot == 1
    assert telemetry.group_count == 1
    assert telemetry.direct_view_groups == 1
    assert telemetry.direct_view_bytes == 17
    assert telemetry.staged_groups == 0
    assert window_factory.total_bytes == 34


def test_slot_adaptation_respects_budget_and_keeps_segmented_fallback(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": bytes(range(12)), "b": b"bcdef"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("fused", ("a", "b")),)
    stream, _, _, _, _ = _open_test_stream(
        preflight,
        groups,
        slot_bytes=8,
        buffer_budget_bytes=16,
    )

    batches = []
    while True:
        lease = stream.begin_next()
        if lease is None:
            break
        batches.append(lease.batch)
        stream.complete(lease)
    telemetry = stream.telemetry
    stream.finalize()

    assert len(batches) == 3
    assert [batch.group_complete for batch in batches] == [False, False, True]
    assert telemetry.slot_bytes == 8
    assert telemetry.largest_group_nbytes == 17
    assert telemetry.groups_fitting_single_slot == 0


def test_group_stream_is_bounded_double_buffered_and_worker_only_reads(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = {"a": bytes(range(12)), "b": bytes(range(20, 25))}
    _u8_checkpoint(checkpoint, expected)
    preflight = prepare_shared_host_weight_stream([checkpoint])
    by_key = {tensor.key: tensor for tensor in preflight.tensors}
    reader = _RecordingReader(watched_offsets=(by_key["a"].source_offset + 8,))
    groups = (WeightGroup("fused", ("a", "b")),)
    (stream, node_communicator, world_communicator, window_factory, registration_factory) = (
        _open_test_stream(preflight, groups, slot_bytes=8, reader=reader)
    )

    main_thread = threading.get_ident()
    reconstructed = {"a": bytearray(12), "b": bytearray(5)}
    batches = []
    lease = stream.begin_next()
    assert lease is not None
    # begin_next publishes N and schedules producer reads for N+1 before
    # returning N to the consumer.
    assert reader.watched.wait(timeout=2)
    while lease is not None:
        batch = lease.batch
        batches.append(batch)
        assert batch.payload_nbytes <= 8
        for segment in batch.segments:
            view = lease.view(segment)
            try:
                start = segment.tensor_offset
                reconstructed[segment.key][start : start + segment.nbytes] = view
            finally:
                view.release()
        stream.complete(lease)
        lease = stream.begin_next()

    telemetry = stream.telemetry
    stream.finalize()
    stream.finalize()

    assert bytes(reconstructed["a"]) == expected["a"]
    assert bytes(reconstructed["b"]) == expected["b"]
    assert [batch.sequence for batch in batches] == [0, 1, 2]
    assert [batch.slot for batch in batches] == [0, 1, 0]
    assert [batch.group_complete for batch in batches] == [False, False, True]
    assert stream.groups == groups
    assert telemetry.is_node_producer
    assert telemetry.batches_published == 3
    assert telemetry.bytes_published == 17
    assert telemetry.host_registered
    assert reader.thread_ids
    assert main_thread not in reader.thread_ids
    assert set(node_communicator.call_threads) == {main_thread}
    assert set(world_communicator.call_threads) == {main_thread}
    assert set(window_factory.window.call_threads) == {main_thread}
    assert window_factory.window.events[0] == "lock_all"
    assert window_factory.window.events[-2:] == ["unlock_all", "free"]
    assert window_factory.window.freed
    assert not window_factory.window.locked
    assert registration_factory.closed


def test_pread_reader_reuses_one_descriptor_for_many_file_extents(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    expected = bytes(range(32))
    _u8_checkpoint(checkpoint, {"a": expected})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    target_path = os.fspath(checkpoint)
    real_open = os.open
    real_close = os.close
    opened_descriptors = []
    closed_descriptors = []
    tracked_descriptors = set()

    def recording_open(path, flags, *args):
        descriptor = real_open(path, flags, *args)
        if os.fspath(path) == target_path:
            opened_descriptors.append(descriptor)
            tracked_descriptors.add(descriptor)
        return descriptor

    def recording_close(descriptor):
        if descriptor in tracked_descriptors:
            closed_descriptors.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "close", recording_close)
    stream, _, _, _, _ = _open_test_stream(
        preflight,
        groups,
        slot_bytes=4,
        buffer_budget_bytes=8,
    )

    reconstructed = bytearray(len(expected))
    while True:
        lease = stream.begin_next()
        if lease is None:
            break
        for segment in lease.batch.segments:
            view = lease.view(segment)
            try:
                start = segment.tensor_offset
                reconstructed[start : start + segment.nbytes] = view
            finally:
                view.release()
        stream.complete(lease)
    stream.finalize()

    assert bytes(reconstructed) == expected
    assert len(opened_descriptors) == 1
    assert closed_descriptors == opened_descriptors


def test_reader_close_failure_is_coordinated_after_safe_window_teardown(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    reader = _CloseFailingReader()
    stream, _, _, window_factory, registration_factory = _open_test_stream(
        preflight,
        groups,
        slot_bytes=4,
        reader=reader,
    )

    lease = stream.begin_next()
    assert lease is not None
    stream.complete(lease)
    assert stream.begin_next() is None
    with pytest.raises(SharedHostStreamError, match="injected reader close failure"):
        stream.finalize()

    assert reader.close_calls == 1
    assert registration_factory.closed
    assert window_factory.window.events[-2:] == ["unlock_all", "free"]
    assert window_factory.window.freed


def test_two_local_ranks_share_producer_bytes_without_consumer_io(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = b"abcdefgh"
    _u8_checkpoint(checkpoint, {"a": expected})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    window_factory = _ThreadedWindowFactory()
    producer_reader = _RecordingReader()
    consumer_reader = _ForbiddenReader()
    readers = (producer_reader, consumer_reader)
    registrations = (_RegistrationFactory(), _RegistrationFactory())
    reconstructed = [None, None]
    telemetry = [None, None]
    errors = [None, None]

    def run_rank(rank):
        try:
            node_communicator = _ThreadedCommunicator(node_group, rank, 2)
            world_communicator = _ThreadedCommunicator(world_group, rank, 2)
            stream = open_shared_host_weight_stream(
                preflight,
                node_communicator,
                world_communicator,
                group_manifest=groups,
                slot_bytes=4,
                read_chunk_bytes=2,
                io_workers=2,
                strict=True,
                strict_host_registration=True,
                window_factory=window_factory,
                reader=readers[rank],
                host_registrar=registrations[rank],
            )
            assert stream is not None
            stream.start()
            result = bytearray(len(expected))
            while True:
                lease = stream.begin_next()
                if lease is None:
                    break
                for segment in lease.batch.segments:
                    view = lease.view(segment)
                    try:
                        start = segment.tensor_offset
                        result[start : start + segment.nbytes] = view
                    finally:
                        view.release()
                stream.complete(lease)
            telemetry[rank] = stream.telemetry
            stream.finalize()
            reconstructed[rank] = bytes(result)
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == [None, None]
    assert reconstructed == [expected, expected]
    assert producer_reader.thread_ids
    assert consumer_reader.calls == 0
    assert telemetry[0].is_node_producer
    assert not telemetry[1].is_node_producer
    assert len(window_factory.windows) == 2
    assert all(window.freed for window in window_factory.windows)
    assert all(registration.closed for registration in registrations)


def test_rank_cooperative_assignment_is_disjoint_complete_and_rotating(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": bytes(range(24))})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    plans = _plan_batches(preflight, groups, slot_bytes=8, read_chunk_bytes=2)

    for plan in plans:
        assignments = [
            _assigned_extents(plan, producer_count=3, producer_rank=rank) for rank in range(3)
        ]
        assigned_extents = [extent for assignment in assignments for extent in assignment]
        assert sorted(assigned_extents, key=lambda extent: extent.payload_offset) == list(
            plan.extents
        )
        assert len({id(extent) for extent in assigned_extents}) == len(plan.extents)
        first_owner = next(
            rank for rank, assignment in enumerate(assignments) if plan.extents[0] in assignment
        )
        assert first_owner == plan.batch.sequence % 3


@pytest.mark.parametrize(
    ("mode", "node_size", "io_workers", "expected"),
    [
        (SharedHostProducerMode.SINGLE_PRODUCER, 4, 7, (7, 0, 0, 0)),
        (SharedHostProducerMode.RANK_COOPERATIVE, 4, 10, (3, 3, 2, 2)),
        (SharedHostProducerMode.RANK_COOPERATIVE, 4, 4, (1, 1, 1, 1)),
        (SharedHostProducerMode.RANK_COOPERATIVE, 4, 2, (1, 1, 0, 0)),
        (SharedHostProducerMode.RANK_COOPERATIVE, 1, 8, (8,)),
    ],
)
def test_producer_worker_budget_is_node_scoped(mode, node_size, io_workers, expected):
    assert _producer_worker_counts(mode, node_size, io_workers) == expected


def test_rank_cooperative_single_rank_degenerates_to_local_stream(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = b"abcdefgh"
    _u8_checkpoint(checkpoint, {"a": expected})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    stream, _, _, _, _ = _open_test_stream(
        preflight,
        groups,
        slot_bytes=4,
        producer_mode=SharedHostProducerMode.RANK_COOPERATIVE,
    )

    reconstructed = bytearray(len(expected))
    while True:
        lease = stream.begin_next()
        if lease is None:
            break
        for segment in lease.batch.segments:
            view = lease.view(segment)
            try:
                start = segment.tensor_offset
                reconstructed[start : start + segment.nbytes] = view
            finally:
                view.release()
        stream.complete(lease)
    telemetry = stream.telemetry
    stream.finalize()

    assert bytes(reconstructed) == expected
    assert telemetry.producer_mode == SharedHostProducerMode.RANK_COOPERATIVE.value
    assert telemetry.producer_count == 1
    assert telemetry.local_io_workers == 2
    assert telemetry.producer_ordinal == 0


def test_source_backing_agreement_is_cooperative_only(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)

    def fail_source_identity(*args):
        del args
        raise OSError("injected source identity failure")

    monkeypatch.setattr(
        shared_host_stream_module,
        "_source_backing_fingerprint",
        fail_source_identity,
    )
    stream, _, _, _, _ = _open_test_stream(
        preflight,
        groups,
        slot_bytes=4,
        producer_mode=SharedHostProducerMode.SINGLE_PRODUCER,
    )
    lease = stream.begin_next()
    assert lease is not None
    stream.complete(lease)
    assert stream.begin_next() is None
    stream.finalize()

    with pytest.raises(SharedHostStreamUnavailableError, match="injected source identity failure"):
        _open_test_stream(
            preflight,
            groups,
            slot_bytes=4,
            producer_mode=SharedHostProducerMode.RANK_COOPERATIVE,
        )


def test_rank_cooperative_source_disagreement_fails_before_allocation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    window_factory = _ThreadedWindowFactory()
    rank_context = threading.local()
    readers = (_ForbiddenReader(), _ForbiddenReader())
    errors = [None, None]

    def rank_dependent_source_identity(*args):
        del args
        return f"source-backing-rank-{rank_context.rank}"

    monkeypatch.setattr(
        shared_host_stream_module,
        "_source_backing_fingerprint",
        rank_dependent_source_identity,
    )

    def run_rank(rank):
        rank_context.rank = rank
        try:
            open_shared_host_weight_stream(
                preflight,
                _ThreadedCommunicator(node_group, rank, 2),
                _ThreadedCommunicator(world_group, rank, 2),
                producer_mode=SharedHostProducerMode.RANK_COOPERATIVE,
                group_manifest=groups,
                slot_bytes=4,
                buffer_budget_bytes=8,
                read_chunk_bytes=2,
                io_workers=2,
                strict=True,
                window_factory=window_factory,
                reader=readers[rank],
            )
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert all(isinstance(error, SharedHostStreamUnavailableError) for error in errors)
    assert str(errors[0]) == str(errors[1])
    assert "source backing differs" in str(errors[0])
    assert not window_factory.windows
    assert all(reader.calls == 0 for reader in readers)


def test_rank_cooperative_producers_collectively_fill_shared_batches(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = bytes(range(24))
    _u8_checkpoint(checkpoint, {"a": expected})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(4)
    world_group = _ThreadedCollectiveGroup(4)
    window_factory = _ThreadedWindowFactory()
    readers = [_RecordingReader() for _ in range(3)] + [_ForbiddenReader()]
    registrations = [_RegistrationFactory() for _ in range(4)]
    reconstructed = [None] * 4
    telemetry = [None] * 4
    owner_threads = [None] * 4
    errors = [None] * 4

    def run_rank(rank):
        owner_threads[rank] = threading.get_ident()
        try:
            stream = open_shared_host_weight_stream(
                preflight,
                _ThreadedCommunicator(node_group, rank, 4),
                _ThreadedCommunicator(world_group, rank, 4),
                producer_mode=SharedHostProducerMode.RANK_COOPERATIVE,
                group_manifest=groups,
                slot_bytes=24,
                buffer_budget_bytes=48,
                read_chunk_bytes=2,
                io_workers=3,
                strict=True,
                strict_host_registration=True,
                window_factory=window_factory,
                reader=readers[rank],
                host_registrar=registrations[rank],
            )
            assert stream is not None
            stream.start()
            lease = stream.begin_next()
            assert lease is not None
            view = lease.view(lease.batch.segments[0])
            try:
                reconstructed[rank] = bytes(view)
            finally:
                view.release()
            stream.complete(lease)
            assert stream.begin_next() is None
            telemetry[rank] = stream.telemetry
            stream.finalize()
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == [None] * 4
    assert reconstructed == [expected] * 4
    assert [item.producer_count for item in telemetry] == [3] * 4
    assert [item.local_io_workers for item in telemetry] == [1, 1, 1, 0]
    assert [item.is_storage_producer for item in telemetry] == [True, True, True, False]
    assert telemetry[0].is_node_producer
    assert all(not item.is_node_producer for item in telemetry[1:])
    assert readers[3].calls == 0

    calls = [call for reader in readers[:3] for call in reader.calls]
    plan = _plan_batches(preflight, groups, slot_bytes=24, read_chunk_bytes=2)[0]
    expected_calls = [
        (extent.source_path, extent.source_offset, extent.nbytes) for extent in plan.extents
    ]
    assert sorted(calls) == sorted(expected_calls)
    assert len(calls) == len(set(calls))
    assert sum(item.assigned_extent_count for item in telemetry) == len(expected_calls)
    assert sum(item.completed_extent_count for item in telemetry) == len(expected_calls)
    assert sum(item.assigned_extent_bytes for item in telemetry) == len(expected)
    assert sum(item.completed_extent_bytes for item in telemetry) == len(expected)
    worker_threads = set().union(*(reader.thread_ids for reader in readers[:3]))
    assert worker_threads
    assert worker_threads.isdisjoint(owner_threads)
    assert all(len(set(window.call_threads)) == 1 for window in window_factory.windows)
    assert all(
        set(window.call_threads).issubset(owner_threads) for window in window_factory.windows
    )
    assert all(window.freed for window in window_factory.windows)
    assert all(registration.closed for registration in registrations)


def test_rank_cooperative_publish_waits_for_slowest_producer(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = b"abcdefgh"
    _u8_checkpoint(checkpoint, {"a": expected})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    tensor = preflight.tensors[0]
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    window_factory = _ThreadedWindowFactory()
    blocking_reader = _BlockingReader(tensor.source_offset + 2)
    readers = (_RecordingReader(), blocking_reader)
    registrations = (_RegistrationFactory(), _RegistrationFactory())
    published = [threading.Event(), threading.Event()]
    errors = [None, None]

    def run_rank(rank):
        stream = None
        try:
            stream = open_shared_host_weight_stream(
                preflight,
                _ThreadedCommunicator(node_group, rank, 2),
                _ThreadedCommunicator(world_group, rank, 2),
                producer_mode=SharedHostProducerMode.RANK_COOPERATIVE,
                group_manifest=groups,
                slot_bytes=8,
                buffer_budget_bytes=16,
                read_chunk_bytes=2,
                io_workers=2,
                strict=True,
                strict_host_registration=True,
                window_factory=window_factory,
                reader=readers[rank],
                host_registrar=registrations[rank],
            )
            assert stream is not None
            stream.start()
            lease = stream.begin_next()
            assert lease is not None
            published[rank].set()
            stream.complete(lease)
            assert stream.begin_next() is None
            stream.finalize()
        except BaseException as error:
            errors[rank] = error
            if stream is not None:
                stream.finalize(error)

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()

    assert blocking_reader.blocked.wait(timeout=5)
    assert not any(event.is_set() for event in published)
    blocking_reader.release.set()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == [None, None]
    assert all(event.is_set() for event in published)


def test_rank_cooperative_overlaps_next_fill_and_gates_slot_reuse(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = b"abcdefghijkl"
    _u8_checkpoint(checkpoint, {"a": expected})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    tensor = preflight.tensors[0]
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    window_factory = _ThreadedWindowFactory()
    # Batch 0 occupies slot 0, batch 1 slot 1, and batch 2 reuses slot 0.
    # With rotating two-producer assignment, rank 1 owns the first extent of
    # batch 1, while rank 0 owns the first extent of batch 2.
    blocking_reader = _BlockingReader(tensor.source_offset + 4)
    slot_reuse_reader = _RecordingReader(watched_offsets=(tensor.source_offset + 8,))
    readers = (slot_reuse_reader, blocking_reader)
    registrations = (_RegistrationFactory(), _RegistrationFactory())
    first_batch_ready = [threading.Event(), threading.Event()]
    allow_first_batch_completion = [threading.Event(), threading.Event()]
    rank_zero_entered_completion = threading.Event()
    reconstructed = [None, None]
    errors = [None, None]

    class _CompletionSignalingCommunicator(_ThreadedCommunicator):
        def allgather(self, value):
            if self.Get_rank() == 0 and getattr(value, "phase", None) == "complete:0":
                rank_zero_entered_completion.set()
            return super().allgather(value)

    def run_rank(rank):
        stream = None
        try:
            stream = open_shared_host_weight_stream(
                preflight,
                _ThreadedCommunicator(node_group, rank, 2),
                _CompletionSignalingCommunicator(world_group, rank, 2),
                producer_mode=SharedHostProducerMode.RANK_COOPERATIVE,
                group_manifest=groups,
                slot_bytes=4,
                buffer_budget_bytes=8,
                read_chunk_bytes=2,
                io_workers=2,
                strict=True,
                strict_host_registration=True,
                window_factory=window_factory,
                reader=readers[rank],
                host_registrar=registrations[rank],
            )
            assert stream is not None
            stream.start()
            result = bytearray(len(expected))
            lease = stream.begin_next()
            assert lease is not None
            first_batch_ready[rank].set()
            while lease is not None:
                for segment in lease.batch.segments:
                    view = lease.view(segment)
                    try:
                        start = segment.tensor_offset
                        result[start : start + segment.nbytes] = view
                    finally:
                        view.release()
                if lease.batch.sequence == 0:
                    assert allow_first_batch_completion[rank].wait(timeout=10)
                stream.complete(lease)
                lease = stream.begin_next()
            reconstructed[rank] = bytes(result)
            stream.finalize()
        except BaseException as error:
            errors[rank] = error
            if stream is not None:
                stream.finalize(error)

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()

    next_batch_read_started = blocking_reader.blocked.wait(timeout=5)
    consumers_received_first_batch = all(event.wait(timeout=5) for event in first_batch_ready)
    blocking_reader.release.set()
    allow_first_batch_completion[0].set()
    rank_zero_waiting_for_peer = rank_zero_entered_completion.wait(timeout=5)
    slot_reused_before_consensus = slot_reuse_reader.watched.is_set()
    allow_first_batch_completion[1].set()
    slot_reused_after_consensus = slot_reuse_reader.watched.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=15)

    assert next_batch_read_started
    assert consumers_received_first_batch
    assert rank_zero_waiting_for_peer
    assert not slot_reused_before_consensus
    assert slot_reused_after_consensus
    assert all(not thread.is_alive() for thread in threads)
    assert errors == [None, None]
    assert reconstructed == [expected, expected]


def test_rank_cooperative_nonroot_read_failure_is_collective(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcdefgh"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    tensor = preflight.tensors[0]
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    window_factory = _ThreadedWindowFactory()
    readers = (
        _RecordingReader(),
        _RecordingReader(failing_offsets=(tensor.source_offset + 2,)),
    )
    registrations = (_RegistrationFactory(), _RegistrationFactory())
    errors = [None, None]
    finalize_errors = [None, None]

    def run_rank(rank):
        stream = None
        try:
            stream = open_shared_host_weight_stream(
                preflight,
                _ThreadedCommunicator(node_group, rank, 2),
                _ThreadedCommunicator(world_group, rank, 2),
                producer_mode=SharedHostProducerMode.RANK_COOPERATIVE,
                group_manifest=groups,
                slot_bytes=8,
                buffer_budget_bytes=16,
                read_chunk_bytes=2,
                io_workers=2,
                strict=True,
                strict_host_registration=True,
                window_factory=window_factory,
                reader=readers[rank],
                host_registrar=registrations[rank],
            )
            assert stream is not None
            stream.start()
            stream.begin_next()
        except BaseException as error:
            errors[rank] = error
        finally:
            if stream is not None:
                try:
                    stream.finalize()
                except BaseException as error:
                    finalize_errors[rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert all(isinstance(error, SharedHostStreamError) for error in errors)
    assert all("failed on world rank 1" in str(error) for error in errors)
    assert finalize_errors == [None, None]
    assert all(window.freed for window in window_factory.windows)
    assert all(registration.closed for registration in registrations)


def test_registration_fallback_is_scoped_to_each_node(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = b"abcdefgh"
    _u8_checkpoint(checkpoint, {"a": expected})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    world_group = _ThreadedCollectiveGroup(4)
    node_groups = (_ThreadedCollectiveGroup(2), _ThreadedCollectiveGroup(2))
    window_factories = (_ThreadedWindowFactory(), _ThreadedWindowFactory())
    readers = (
        _RecordingReader(),
        _ForbiddenReader(),
        _RecordingReader(),
        _ForbiddenReader(),
    )
    registrations = [_RegistrationFactory() for _ in range(4)]
    direct_enabled = [None] * 4
    errors = [None] * 4

    def run_rank(world_rank):
        node_index = world_rank // 2
        node_rank = world_rank % 2
        node_communicator = _ThreadedCommunicator(node_groups[node_index], node_rank, 2)
        world_communicator = _ThreadedCommunicator(world_group, world_rank, 4)

        def register(arena):
            if world_rank == 3:
                return HostMemoryRegistration(False, "injected node-1 fallback")
            return registrations[world_rank](arena)

        try:
            stream = open_shared_host_weight_stream(
                preflight,
                node_communicator,
                world_communicator,
                group_manifest=groups,
                slot_bytes=8,
                buffer_budget_bytes=16,
                strict=True,
                strict_host_registration=False,
                window_factory=window_factories[node_index],
                reader=readers[world_rank],
                host_registrar=register,
            )
            assert stream is not None
            direct_enabled[world_rank] = stream.supports_direct_tensor_views
            stream.start()
            lease = stream.begin_next()
            assert lease is not None
            view = lease.view(lease.batch.segments[0])
            try:
                assert bytes(view) == expected
            finally:
                view.release()
            stream.complete(lease)
            assert stream.begin_next() is None
            stream.finalize()
        except BaseException as error:
            errors[world_rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == [None] * 4
    assert direct_enabled == [True, True, False, False]
    assert readers[0].thread_ids
    assert readers[2].thread_ids
    assert readers[1].calls == 0
    assert readers[3].calls == 0


def test_retention_error_quarantines_shared_resources_on_every_rank(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = b"abcdefgh"
    _u8_checkpoint(checkpoint, {"a": expected})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    window_factory = _ThreadedWindowFactory()
    producer_reader = _RecordingReader()
    consumer_reader = _ForbiddenReader()
    readers = (producer_reader, consumer_reader)
    registrations = (_RegistrationFactory(), _RegistrationFactory())
    errors = [None, None]

    def run_rank(rank):
        try:
            node_communicator = _ThreadedCommunicator(node_group, rank, 2)
            world_communicator = _ThreadedCommunicator(world_group, rank, 2)
            stream = open_shared_host_weight_stream(
                preflight,
                node_communicator,
                world_communicator,
                group_manifest=groups,
                slot_bytes=8,
                buffer_budget_bytes=16,
                strict=True,
                strict_host_registration=True,
                window_factory=window_factory,
                reader=readers[rank],
                host_registrar=registrations[rank],
            )
            assert stream is not None
            stream.start()
            lease = stream.begin_next()
            assert lease is not None
            view = lease.view(lease.batch.segments[0])
            try:
                assert bytes(view) == expected
            finally:
                view.release()
            local_error = (
                BorrowedWeightStorageRetentionError("retained source tensor") if rank == 1 else None
            )
            with pytest.raises(
                BorrowedWeightStorageRetentionError,
                match="world rank 1.*retained source tensor",
            ):
                stream.complete(lease, local_error)
            stream.finalize()
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == [None, None]
    assert len(window_factory.windows) == 2
    assert all(not window.freed for window in window_factory.windows)
    assert all(window.locked for window in window_factory.windows)
    assert all(window.unlock_calls == 0 for window in window_factory.windows)
    assert all(not registration.closed for registration in registrations)


def test_lock_all_failure_quarantines_locked_windows_on_every_node_rank(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    window_factory = _ThreadedEpochFailureWindowFactory(failing_lock_rank=1)
    registrations = (_RegistrationFactory(), _RegistrationFactory())
    errors = [None, None]

    def run_rank(rank):
        node_communicator = _ThreadedCommunicator(node_group, rank, 2)
        world_communicator = _ThreadedCommunicator(world_group, rank, 2)
        try:
            open_shared_host_weight_stream(
                preflight,
                node_communicator,
                world_communicator,
                group_manifest=groups,
                slot_bytes=4,
                buffer_budget_bytes=8,
                strict=True,
                strict_host_registration=True,
                window_factory=window_factory,
                host_registrar=registrations[rank],
            )
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert all(isinstance(error, SharedHostStreamError) for error in errors)
    assert all(
        "node-local rank 1: RuntimeError: injected Win.Lock_all failure" in str(error)
        for error in errors
    )
    assert len(window_factory.windows) == 2
    assert all(window.events == ["lock_all"] for window in window_factory.windows.values())
    assert all(window.locked for window in window_factory.windows.values())
    assert all(not window.freed for window in window_factory.windows.values())
    assert all(not registration.closed for registration in registrations)


def test_one_node_setup_quarantine_hard_fails_world_in_soft_mode(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    world_group = _ThreadedCollectiveGroup(4)
    node_groups = (_ThreadedCollectiveGroup(2), _ThreadedCollectiveGroup(2))
    window_factories = (
        _ThreadedEpochFailureWindowFactory(failing_lock_rank=1),
        _ThreadedEpochFailureWindowFactory(),
    )
    registrations = [_RegistrationFactory() for _ in range(4)]
    returned = [False] * 4
    errors = [None] * 4

    def run_rank(world_rank):
        node_index = world_rank // 2
        node_rank = world_rank % 2
        node_communicator = _ThreadedCommunicator(node_groups[node_index], node_rank, 2)
        world_communicator = _ThreadedCommunicator(world_group, world_rank, 4)
        try:
            open_shared_host_weight_stream(
                preflight,
                node_communicator,
                world_communicator,
                group_manifest=groups,
                slot_bytes=4,
                buffer_budget_bytes=8,
                strict=False,
                strict_host_registration=True,
                window_factory=window_factories[node_index],
                host_registrar=registrations[world_rank],
            )
            returned[world_rank] = True
        except BaseException as error:
            errors[world_rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert not any(returned)
    assert all(isinstance(error, SharedHostStreamError) for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert "epoch setup was quarantined" in str(errors[0])
    assert all(not window.freed for window in window_factories[0].windows.values())
    assert all(window.freed for window in window_factories[1].windows.values())
    assert all(not registration.closed for registration in registrations[:2])
    assert all(registration.closed for registration in registrations[2:])


def test_producer_constructor_failure_is_world_coordinated(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    world_group = _ThreadedCollectiveGroup(4)
    node_groups = (_ThreadedCollectiveGroup(2), _ThreadedCollectiveGroup(2))
    window_factories = (_ThreadedWindowFactory(), _ThreadedWindowFactory())
    registrations = [_RegistrationFactory() for _ in range(4)]
    readers = (
        _RecordingReader(),
        _ForbiddenReader(),
        _RecordingReader(),
        _ForbiddenReader(),
    )
    real_executor = shared_host_stream_module.ThreadPoolExecutor
    executor_lock = threading.Lock()
    executor_calls = 0

    def fail_first_producer(*args, **kwargs):
        nonlocal executor_calls
        with executor_lock:
            call_index = executor_calls
            executor_calls += 1
        if call_index == 0:
            raise RuntimeError("injected producer executor failure")
        return real_executor(*args, **kwargs)

    monkeypatch.setattr(shared_host_stream_module, "ThreadPoolExecutor", fail_first_producer)
    returned = [False] * 4
    errors = [None] * 4

    def run_rank(world_rank):
        node_index = world_rank // 2
        node_rank = world_rank % 2
        node_communicator = _ThreadedCommunicator(node_groups[node_index], node_rank, 2)
        world_communicator = _ThreadedCommunicator(world_group, world_rank, 4)
        try:
            open_shared_host_weight_stream(
                preflight,
                node_communicator,
                world_communicator,
                group_manifest=groups,
                slot_bytes=4,
                buffer_budget_bytes=8,
                strict=False,
                strict_host_registration=True,
                window_factory=window_factories[node_index],
                reader=readers[world_rank],
                host_registrar=registrations[world_rank],
            )
            returned[world_rank] = True
        except BaseException as error:
            errors[world_rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert executor_calls == 2
    assert not any(returned)
    assert all(isinstance(error, SharedHostStreamError) for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert "injected producer executor failure" in str(errors[0])
    assert all(window.freed for factory in window_factories for window in factory.windows)
    assert all(registration.closed for registration in registrations)
    assert all(reader.closed for reader in readers)


def test_unlock_all_failure_skips_window_free_on_every_node_rank(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = b"abcdefgh"
    _u8_checkpoint(checkpoint, {"a": expected})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    window_factory = _ThreadedEpochFailureWindowFactory(failing_unlock_rank=1)
    readers = (_RecordingReader(), _ForbiddenReader())
    registrations = (_RegistrationFactory(), _RegistrationFactory())
    errors = [None, None]

    def run_rank(rank):
        try:
            node_communicator = _ThreadedCommunicator(node_group, rank, 2)
            world_communicator = _ThreadedCommunicator(world_group, rank, 2)
            stream = open_shared_host_weight_stream(
                preflight,
                node_communicator,
                world_communicator,
                group_manifest=groups,
                slot_bytes=8,
                buffer_budget_bytes=16,
                strict=True,
                strict_host_registration=True,
                window_factory=window_factory,
                reader=readers[rank],
                host_registrar=registrations[rank],
            )
            assert stream is not None
            stream.start()
            lease = stream.begin_next()
            assert lease is not None
            view = lease.view(lease.batch.segments[0])
            try:
                assert bytes(view) == expected
            finally:
                view.release()
            stream.complete(lease)
            assert stream.begin_next() is None
            stream.finalize()
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert all(isinstance(error, SharedHostStreamError) for error in errors)
    assert all(
        "node-local rank 1: RuntimeError: injected Win.Unlock_all failure" in str(error)
        for error in errors
    )
    assert all(window.unlock_calls == 1 for window in window_factory.windows.values())
    assert not window_factory.windows[0].locked
    assert window_factory.windows[1].locked
    assert all(not window.freed for window in window_factory.windows.values())
    assert all(registration.closed for registration in registrations)


def test_unregister_failure_skips_window_free_on_every_node_rank(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = b"abcdefgh"
    _u8_checkpoint(checkpoint, {"a": expected})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    window_factory = _ThreadedWindowFactory()
    readers = (_RecordingReader(), _ForbiddenReader())
    registrations = (_RegistrationFactory(), _FailingRegistrationFactory())
    errors = [None, None]

    def run_rank(rank):
        try:
            node_communicator = _ThreadedCommunicator(node_group, rank, 2)
            world_communicator = _ThreadedCommunicator(world_group, rank, 2)
            stream = open_shared_host_weight_stream(
                preflight,
                node_communicator,
                world_communicator,
                group_manifest=groups,
                slot_bytes=8,
                buffer_budget_bytes=16,
                strict=True,
                strict_host_registration=True,
                window_factory=window_factory,
                reader=readers[rank],
                host_registrar=registrations[rank],
            )
            assert stream is not None
            stream.start()
            lease = stream.begin_next()
            assert lease is not None
            view = lease.view(lease.batch.segments[0])
            try:
                assert bytes(view) == expected
            finally:
                view.release()
            stream.complete(lease)
            assert stream.begin_next() is None
            stream.finalize()
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert all(isinstance(error, SharedHostStreamError) for error in errors)
    assert all(
        "node-local rank 1: RuntimeError: injected cudaHostUnregister failure" in str(error)
        for error in errors
    )
    assert len(window_factory.windows) == 2
    assert all(not window.freed for window in window_factory.windows)
    assert all(window.locked for window in window_factory.windows)
    assert all(window.unlock_calls == 0 for window in window_factory.windows)
    assert registrations[0].closed
    assert registrations[1].close_calls == 1


def test_setup_arena_release_failure_skips_window_free_on_every_node_rank(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    window_factory = _ThreadedReleaseFailureWindowFactory(failing_rank=1)
    errors = [None, None]

    def unavailable_registrar(arena):
        del arena
        return HostMemoryRegistration(False, "injected registration unavailability")

    def run_rank(rank):
        node_communicator = _ThreadedCommunicator(node_group, rank, 2)
        world_communicator = _ThreadedCommunicator(world_group, rank, 2)
        try:
            open_shared_host_weight_stream(
                preflight,
                node_communicator,
                world_communicator,
                group_manifest=groups,
                slot_bytes=4,
                buffer_budget_bytes=8,
                strict=True,
                strict_host_registration=True,
                window_factory=window_factory,
                host_registrar=unavailable_registrar,
            )
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert all(isinstance(error, SharedHostStreamError) for error in errors)
    assert all(
        "node-local rank 1: BufferError: injected exported arena release failure" in str(error)
        for error in errors
    )
    assert len(window_factory.windows) == 2
    assert all(not window.freed for window in window_factory.windows)
    assert all(window.locked for window in window_factory.windows)
    assert all(window.unlock_calls == 0 for window in window_factory.windows)
    assert all(arena.release_calls == 1 for arena in window_factory.arenas.values())


def test_consumer_start_error_is_coordinated_before_node_publish(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])

    def remote_start_error(value):
        if isinstance(value, str):
            return [value, value]
        if isinstance(value, tuple):
            return [value, (1, *value[1:])]
        remote = replace(value, rank=1)
        if value.phase == "start":
            remote = replace(
                remote,
                error_rank=1,
                error_type="ValueError",
                error_message="mapper begin failed",
            )
        return [value, remote]

    world_communicator = _FakeCommunicator(size=2, allgather_hook=remote_start_error)
    groups = (WeightGroup("a", ("a",)),)
    stream, node_communicator, _, window_factory, registration_factory = _open_test_stream(
        preflight,
        groups,
        slot_bytes=4,
        world_communicator=world_communicator,
        start=False,
    )
    setup_call_count = len(node_communicator.call_threads)

    with pytest.raises(SharedHostStreamError, match="world rank 1.*mapper begin failed"):
        stream.start()
    # No node bcast/Barrier is entered before world start consensus succeeds.
    assert len(node_communicator.call_threads) == setup_call_count
    stream.finalize()
    assert window_factory.window.freed
    assert registration_factory.closed


def test_batch_payload_offsets_preserve_tensor_alignment(tmp_path):
    checkpoint = tmp_path / "aligned.safetensors"
    entries = {
        "a": {
            "dtype": "U8",
            "shape": [1],
            "data_offsets": [0, 1],
        },
        "b": {
            "dtype": "F64",
            "shape": [1],
            "data_offsets": [1, 9],
        },
    }
    _write_safetensors(checkpoint, entries, b"a12345678")
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("aligned", ("a", "b")),)
    stream, _, _, _, _ = _open_test_stream(preflight, groups, slot_bytes=16)

    lease = stream.begin_next()
    assert lease is not None
    segments = {segment.key: segment for segment in lease.batch.segments}
    assert segments["a"].payload_offset == 0
    assert segments["b"].payload_offset == 8
    assert lease.batch.payload_nbytes == 16
    stream.complete(lease)
    assert stream.begin_next() is None
    stream.finalize()


def test_production_mpi_shared_window_epoch_smoke():
    mpi = pytest.importorskip("mpi4py.MPI", reason="mpi4py is required for MPI smoke")
    world_communicator = mpi.COMM_WORLD
    if world_communicator.Get_size() < 2:
        pytest.skip("run with mpiexec -n 2 or more")

    node_communicator = world_communicator.Split_type(
        mpi.COMM_TYPE_SHARED,
        key=world_communicator.Get_rank(),
    )
    if node_communicator.Get_size() < 2:
        node_communicator.Free()
        pytest.skip("MPI smoke requires at least two ranks on one shared-memory node")

    window, arena = _allocate_mpi_shared_window(node_communicator, 64)
    window.Lock_all()
    node_communicator.Barrier()
    expected = b"trtllm-mpi-epoch"
    if node_communicator.Get_rank() == 0:
        arena[: len(expected)] = expected
    window.Sync()
    node_communicator.Barrier()
    window.Sync()
    observed = bytes(arena[: len(expected)])
    observations = node_communicator.allgather(observed)
    node_communicator.Barrier()
    arena.release()
    window.Unlock_all()
    node_communicator.allgather(None)
    window.Free()
    node_communicator.Free()

    assert observations == [expected] * len(observations)


@pytest.mark.parametrize(
    "entries,payload,match",
    [
        (
            {
                "bad": {
                    "dtype": "U8",
                    "shape": [4],
                    "data_offsets": [0, 3],
                }
            },
            b"abc",
            "requires 4 bytes",
        ),
        (
            {
                "a": {
                    "dtype": "U8",
                    "shape": [4],
                    "data_offsets": [0, 4],
                },
                "b": {
                    "dtype": "U8",
                    "shape": [4],
                    "data_offsets": [2, 6],
                },
            },
            b"abcdef",
            "overlap",
        ),
        (
            {
                "a": {
                    "dtype": "U8",
                    "shape": [2],
                    "data_offsets": [0, 2],
                },
                "b": {
                    "dtype": "U8",
                    "shape": [2],
                    "data_offsets": [3, 5],
                },
            },
            b"abcde",
            "unindexed gap",
        ),
    ],
)
def test_preflight_rejects_invalid_shape_size_and_overlapping_ranges(
    tmp_path, entries, payload, match
):
    checkpoint = tmp_path / "invalid.safetensors"
    _write_safetensors(checkpoint, entries, payload)

    with pytest.raises(SharedHostStreamUnavailableError, match=match):
        prepare_shared_host_weight_stream([checkpoint])


def test_missing_atomic_manifest_is_soft_or_strict_ineligible(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    node_communicator = _FakeCommunicator()
    world_communicator = _FakeCommunicator()
    window_factory = _FakeWindowFactory()

    stream = open_shared_host_weight_stream(
        preflight,
        node_communicator,
        world_communicator,
        slot_bytes=4,
        window_factory=window_factory,
    )
    assert stream is None
    assert window_factory.calls == 0

    with pytest.raises(SharedHostStreamUnavailableError, match="atomic weight-group manifest"):
        open_shared_host_weight_stream(
            preflight,
            node_communicator,
            world_communicator,
            slot_bytes=4,
            strict=True,
            window_factory=window_factory,
        )
    assert window_factory.calls == 0


def test_buffer_budget_must_hold_two_configured_slots(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    node_communicator = _FakeCommunicator()
    world_communicator = _FakeCommunicator()
    window_factory = _FakeWindowFactory()

    with pytest.raises(
        SharedHostStreamUnavailableError,
        match="buffer_budget_bytes must hold two configured slots",
    ):
        open_shared_host_weight_stream(
            preflight,
            node_communicator,
            world_communicator,
            group_manifest=(WeightGroup("a", ("a",)),),
            slot_bytes=8,
            buffer_budget_bytes=15,
            strict=True,
            window_factory=window_factory,
        )
    assert window_factory.calls == 0


def test_remote_materialization_error_cancels_and_is_deterministic(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])

    def remote_error(value):
        if isinstance(value, str):
            return [value, value]
        if isinstance(value, tuple):
            return [value, (1, *value[1:])]
        remote = replace(value, rank=1)
        if value.phase.startswith("complete:"):
            remote = replace(
                remote,
                error_rank=1,
                error_type="ValueError",
                error_message="remote materialization failure",
            )
        return [value, remote]

    world_communicator = _FakeCommunicator(size=2, allgather_hook=remote_error)
    groups = (WeightGroup("a", ("a",)),)
    stream, _, _, window_factory, registration_factory = _open_test_stream(
        preflight, groups, slot_bytes=4, world_communicator=world_communicator
    )
    lease = stream.begin_next()
    assert lease is not None

    with pytest.raises(SharedHostStreamError, match="world rank 1.*remote materialization failure"):
        stream.complete(lease)
    stream.finalize()

    assert window_factory.window.freed
    assert registration_factory.closed


def test_producer_read_error_propagates_and_cleanup_completes(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcdefgh"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    failing_offset = preflight.tensors[0].source_offset + 4
    reader = _RecordingReader(failing_offsets=(failing_offset,))
    window_factory = _ExportCheckingWindowFactory()
    groups = (WeightGroup("a", ("a",)),)
    stream, _, _, _, registration_factory = _open_test_stream(
        preflight,
        groups,
        slot_bytes=4,
        reader=reader,
        window_factory=window_factory,
    )

    lease = stream.begin_next()
    assert lease is not None
    stream.complete(lease)
    with pytest.raises(SharedHostStreamError, match="injected producer read failure"):
        stream.begin_next()
    # The failed read was awaited and removed from the pending-future set by
    # begin_next(). Its traceback must no longer export a view of the arena when
    # finalization frees the window.
    stream.finalize()

    assert window_factory.window.freed
    assert registration_factory.closed


def test_abort_drops_pending_failure_views_before_window_free(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcdefgh"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    failing_offset = preflight.tensors[0].source_offset + 4
    reader = _RecordingReader(watched_offsets=(failing_offset,), failing_offsets=(failing_offset,))
    window_factory = _ExportCheckingWindowFactory()
    groups = (WeightGroup("a", ("a",)),)
    stream, _, _, _, registration_factory = _open_test_stream(
        preflight,
        groups,
        slot_bytes=4,
        reader=reader,
        window_factory=window_factory,
    )

    lease = stream.begin_next()
    assert lease is not None
    # Batch N+1 has failed in the producer pool and its Future now owns an
    # exception traceback whose frame exported a view of the second slot.
    assert reader.watched.wait(timeout=2)
    with pytest.raises(SharedHostStreamError, match="stop after current batch"):
        stream.complete(lease, RuntimeError("stop after current batch"))

    stream.finalize()

    assert window_factory.window.freed
    assert registration_factory.closed


def test_prefetch_submit_failure_drops_all_destination_views(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcdefgh"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    failing_offset = preflight.tensors[0].source_offset + 4
    reader = _RecordingReader(watched_offsets=(failing_offset,), failing_offsets=(failing_offset,))
    real_executor = shared_host_stream_module.ThreadPoolExecutor

    class _RejectFourthSubmitExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._submit_calls = 0

        def submit(self, *args, **kwargs):
            self._submit_calls += 1
            if self._submit_calls == 4:
                # The first extent of the prefetched batch has already failed,
                # so its Future owns a traceback containing an arena view when
                # submitting the second extent fails independently.
                assert reader.watched.wait(timeout=2)
                raise RuntimeError("injected prefetch submit failure")
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(
        shared_host_stream_module, "ThreadPoolExecutor", _RejectFourthSubmitExecutor
    )
    window_factory = _ExportCheckingWindowFactory()
    groups = (WeightGroup("a", ("a",)),)
    stream, _, _, _, registration_factory = _open_test_stream(
        preflight,
        groups,
        slot_bytes=4,
        reader=reader,
        window_factory=window_factory,
    )

    with pytest.raises(SharedHostStreamError, match="injected prefetch submit failure"):
        stream.begin_next()
    stream.finalize()

    assert window_factory.window.freed
    assert registration_factory.closed


def test_asymmetric_window_factory_failure_does_not_deadlock_cleanup(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _u8_checkpoint(checkpoint, {"a": b"abcd"})
    preflight = prepare_shared_host_weight_stream([checkpoint])
    groups = (WeightGroup("a", ("a",)),)
    node_group = _ThreadedCollectiveGroup(2)
    world_group = _ThreadedCollectiveGroup(2)
    storage = bytearray(8)
    healthy_window = _FakeWindow()
    registration = _RegistrationFactory()
    errors = [None, None]

    def run_rank(rank):
        node_communicator = _ThreadedCommunicator(node_group, rank, 2)
        world_communicator = _ThreadedCommunicator(world_group, rank, 2)

        def asymmetric_factory(communicator, total_bytes):
            assert total_bytes == len(storage)
            if communicator.Get_rank() == 1:
                raise RuntimeError("injected rank-local allocation failure")
            return healthy_window, memoryview(storage)

        try:
            open_shared_host_weight_stream(
                preflight,
                node_communicator,
                world_communicator,
                group_manifest=groups,
                slot_bytes=4,
                buffer_budget_bytes=8,
                strict=True,
                strict_host_registration=True,
                window_factory=asymmetric_factory,
                host_registrar=registration,
            )
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert all(isinstance(error, SharedHostStreamError) for error in errors)
    assert all("ownership on only node-local ranks [0]" in str(error) for error in errors)
    # A partially owned MPI window cannot be freed collectively. The important
    # recovery guarantee is that every rank detects this and exits without
    # entering a collective that its peer skips.
    assert not healthy_window.freed
    assert healthy_window.locked
    assert healthy_window.unlock_calls == 0
    assert not registration.closed
