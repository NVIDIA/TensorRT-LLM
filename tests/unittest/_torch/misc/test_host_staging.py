# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned uploads remain correct while callers recycle pageable source storage."""

import gc
import time
import weakref

import pytest
import torch

from tensorrt_llm._torch import host_staging


@pytest.mark.cpu_only
def test_host_staging_cpu_destination():
    store = {}
    source = torch.arange(24).view(4, 6)[:, ::2]
    destination = torch.empty_like(source)
    host_staging.copy_host_to_device(store, "test", destination, source)
    torch.testing.assert_close(destination, source, rtol=0, atol=0)
    assert not store
    host_staging.copy_host_to_device(store, "empty", destination[:0], source[:0])
    assert not store


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("pinned_source", [False, True])
def test_host_staging_retains_inflight_sources(pinned_source):
    store, observed = {}, []
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        for iteration, count in enumerate((8, 13, 4, 31, 7, 2, 31)):
            expected = torch.arange(count, dtype=torch.int64) + iteration * 100
            destination = torch.empty(count, dtype=torch.int64, device="cuda")
            if pinned_source:
                source = expected.pin_memory()
            else:
                source = torch.empty(count * 2, dtype=torch.int64)[::2]
                source.copy_(expected)
            host_staging.copy_host_to_device(store, "test", destination, source)
            if not pinned_source:
                source.fill_(-1)
            del source
            observed.append((destination, expected))
    stream.synchronize()
    for actual, expected in observed:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    if pinned_source:
        assert not store
    else:
        assert len(store) == 1
        ring = next(iter(store.values()))
        assert len(ring["buffers"]) == 2
        assert all(buffer.is_pinned() for buffer in ring["buffers"])
        assert all(buffer.numel() <= 62 for buffer in ring["buffers"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_host_staging_respects_unpinned_policy(monkeypatch):
    monkeypatch.setattr(host_staging, "prefer_pinned", lambda: False)
    source = torch.arange(13, dtype=torch.int32)
    destination = torch.empty_like(source, device="cuda")
    store = {}
    host_staging.copy_host_to_device(store, "test", destination, source)
    torch.testing.assert_close(destination.cpu(), source, rtol=0, atol=0)
    assert not store


def _reference(pieces, shape, fill_value):
    """The tensor `upload_pieces` must produce, built on the host."""
    reference = torch.full(shape, 0 if fill_value is None else fill_value, dtype=pieces[0][0].dtype)
    for value, start in pieces:
        reference[tuple(slice(s, s + n) for s, n in zip(start, value.shape))] = value
    return reference


def _assert_same_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))


def _uploader(buffer_bytes, buffer_count=2):
    return host_staging.StagedUploader(
        0, host_staging.PinnedStagingPool(buffer_bytes=buffer_bytes, buffer_count=buffer_count)
    )


def _native_allocator():
    """The native caching allocator: 512-byte blocks, a pool per stream (not cudaMallocAsync)."""
    return torch.cuda.get_allocator_backend() == "native"


def _sleep_cycles(milliseconds):
    """`torch.cuda._sleep` cycles that keep a stream busy for about ``milliseconds``."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(50_000_000)
    end.record()
    end.synchronize()
    return int(50_000_000 * milliseconds / start.elapsed_time(end))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "case",
    [
        # (piece shapes, item dim, shape, fill value, buffer bytes, strided first piece, path)
        ("rows cut by bytes", [(5, 6), (3, 6), (9, 6)], 0, (17, 6), None, 64, False, "rows"),
        (
            "small pieces share a buffer",
            [(1, 3), (2, 3), (1, 3)],
            0,
            (4, 3),
            None,
            1 << 10,
            False,
            "rows",
        ),
        ("strided rows", [(5, 6), (3, 6)], 0, (8, 6), None, 64, True, "boxes"),
        ("ragged rows", [(2, 3, 4), (3, 5, 2), (1, 1, 1)], 0, (6, 5, 4), -1, 96, False, "boxes"),
        ("item axis 1", [(4, 2, 3), (2, 5, 3)], 1, (4, 7, 3), 0, 48, False, "boxes"),
        ("row above a buffer", [(2, 40), (1, 33)], 0, (3, 40), 7, 64, True, "boxes"),
        ("one piece", [(11, 4)], 0, (11, 4), None, 1 << 20, False, "rows"),
    ],
    ids=lambda case: case[0],
)
def test_upload_pieces_writes_each_piece_in_place(monkeypatch, case):
    """The device tensor equals the host layout, through many ranges of a small pool."""
    _, shapes, axis, shape, fill_value, buffer_bytes, strided, path = case
    used = []
    for name in ("rows", "boxes"):
        stage = getattr(host_staging, f"_stage_{name}")
        monkeypatch.setattr(
            host_staging,
            f"_stage_{name}",
            lambda *args, name=name, stage=stage: used.append(name) or stage(*args),
        )
    generator = torch.Generator().manual_seed(7)
    pieces, offset = [], 0
    for piece_shape in shapes:
        value = torch.randn(piece_shape, generator=generator).to(torch.bfloat16)
        start = [0] * len(shape)
        start[axis] = offset
        offset += piece_shape[axis]
        pieces.append((value, start))
    if strided:
        # A strided host view is read in place, without a contiguous copy first.
        first_value, first_start = pieces[0]
        pieces[0] = (torch.cat([first_value, first_value], dim=-1)[..., ::2], first_start)
    snapshot = [value.clone() for value, _ in pieces]
    uploader = _uploader(buffer_bytes)
    destination = uploader.upload_pieces(shape, torch.bfloat16, pieces, fill_value)
    torch.cuda.synchronize()
    assert used == [path]
    _assert_same_bits(destination.cpu(), _reference(pieces, shape, fill_value))
    for (value, _), before in zip(pieces, snapshot):
        assert value.device.type == "cpu"
        assert torch.equal(value, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_staging_buffers_are_reused_when_copies_completed():
    """Uploads on an idle device reuse the same pinned buffers; nothing else is pinned."""
    uploader = _uploader(1 << 10)
    pieces = [(torch.arange(4096, dtype=torch.int32).view(64, 64), [0, 0])]
    pointers = set()
    for _ in range(3):
        destination = uploader.upload_pieces((64, 64), torch.int32, pieces)
        torch.cuda.synchronize()
        assert torch.equal(destination.cpu(), pieces[0][0])
        pointers.update(buffer.data_ptr() for buffer in uploader.pool.buffers)
    assert len(pointers) == len(uploader.pool.buffers) <= 2
    assert all(buffer.is_pinned() for buffer in uploader.pool.buffers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_memory_stays_within_the_pool_while_the_consumer_is_busy():
    """With the consumer stream busy, an upload of many buffers' worth reuses the pool's buffers
    on the copy stream: no pinned memory is allocated beyond the pool."""
    uploader = _uploader(1 << 12)
    source = torch.arange(1 << 16, dtype=torch.float32).view(256, 256)  # 64 buffers' worth
    # Make every buffer of the pool first: an upload whose copies finish quickly may need only one.
    buffers = [uploader.pool.acquire() for _ in range(uploader.pool.buffer_count)]
    for buffer in buffers:
        uploader.pool.release(buffer, uploader.copy_stream)
    torch.cuda.synchronize()
    before = torch.cuda.host_memory_stats()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(_sleep_cycles(200))
        for _ in range(3):
            destination = uploader.upload_pieces((256, 256), torch.float32, [(source, [0, 0])])
            copy = destination.clone()
    after = torch.cuda.host_memory_stats()
    assert after["num_host_alloc"] == before["num_host_alloc"]
    assert after["allocated_bytes.current"] == before["allocated_bytes.current"]
    assert len(uploader.pool.buffers) == 2
    consumer.synchronize()
    assert torch.equal(copy.cpu(), source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_the_host_never_waits_for_a_busy_consumer():
    """The copies run on the copy stream while the consumer is busy for 200 ms: the host and the
    copy stream finish first, and the consumer reads the batch once it is free."""
    uploader = _uploader(1 << 16)
    source = torch.randn(1 << 20)  # 4 MiB, 64 buffers' worth through 2 buffers
    uploader.upload_pieces(source.shape, source.dtype, [(source, [0])])
    torch.cuda.synchronize()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        busy_done = torch.cuda.Event()
        torch.cuda._sleep(_sleep_cycles(200))
        busy_done.record()
        start = time.perf_counter()
        destination = uploader.upload_pieces(source.shape, source.dtype, [(source, [0])])
        host_ms = (time.perf_counter() - start) * 1e3
        seen = destination.clone()  # Runs after the sleep.
    uploader.copy_stream.synchronize()
    assert not busy_done.query(), "the copy stream waited for the consumer"
    assert host_ms < 200
    consumer.synchronize()
    assert torch.equal(seen.cpu(), source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a_batch_is_reused_only_after_its_consumer_finished():
    """A batch freed while its consumer is still busy keeps its block: the next upload, whose
    copies do not wait for the consumer, gets another one; after the consumer is done the native
    allocator's copy-stream pool hands the block out again."""
    uploader = _uploader(1 << 20)
    ones, twos = (
        torch.ones(1 << 16, dtype=torch.int32),
        torch.full((1 << 16,), 2, dtype=torch.int32),
    )
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        busy_done = torch.cuda.Event()
        torch.cuda._sleep(_sleep_cycles(200))
        busy_done.record()
        first = uploader.upload_pieces(ones.shape, ones.dtype, [(ones, [0])])
        seen = first.clone()  # Runs after the sleep.
        pointer = first.data_ptr()
        del first
        second = uploader.upload_pieces(twos.shape, twos.dtype, [(twos, [0])])
        assert second.data_ptr() != pointer
        uploader.copy_stream.synchronize()  # The second batch's bytes are written by now.
        assert not busy_done.query()
    consumer.synchronize()
    assert torch.equal(seen.cpu(), ones)
    assert torch.equal(second.cpu(), twos)
    reserved = uploader.reserved_bytes()
    del second
    third = uploader.upload_pieces(ones.shape, ones.dtype, [(ones, [0])])
    torch.cuda.synchronize()
    assert torch.equal(third.cpu(), ones)
    if _native_allocator():
        assert uploader.reserved_bytes() == reserved > 0
    else:  # cudaMallocAsync keeps no pool apart for the copy stream.
        assert uploader.reserved_bytes() == reserved == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_upload_pieces_makes_no_device_copy():
    """Ragged pieces cross host to device once each, with no device copy or temporary."""
    from torch.profiler import ProfilerActivity, profile

    uploader = _uploader(1 << 16, 4)
    pieces = [(torch.ones(3, 50, 7), [0, 0, 0]), (torch.ones(2, 20, 7), [3, 0, 0])]
    warm = uploader.upload_pieces((5, 50, 7), torch.float32, pieces, 0.0)  # First-use allocations.
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        destination = uploader.upload_pieces((5, 50, 7), torch.float32, pieces, 0.0)
        torch.cuda.synchronize()
    # The batch only: 7000 bytes, in 512-byte blocks with the native allocator.
    block_bytes = 7168 if _native_allocator() else 7000
    assert warm.nbytes == 7000 and torch.cuda.max_memory_allocated() - before == block_bytes
    kinds = [event.name for event in prof.events() if event.device_type.name == "CUDA"]
    assert kinds and all("HtoD" in kind for kind in kinds), kinds
    assert torch.equal(destination.cpu(), _reference(pieces, (5, 50, 7), 0.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_rows_are_copied_from_their_storage_and_kept_until_copied(monkeypatch):
    """Pinned pieces of the row layout skip the staging copy, and each is kept alive until its
    copy ran, even when the caller drops it while the copy stream is still busy. The executor's
    reclaim keeps them while the copy is pending and, with no further upload and no wait for the
    copy stream, drops them once the consumer's work is done (the sampler wait)."""
    uploader = _uploader(64, 4)  # Pageable rows go through 64-byte staging ranges.
    monkeypatch.setattr(host_staging, "_STAGED_UPLOADERS", {torch.cuda.current_device(): uploader})
    acquire = uploader.pool.acquire
    acquired = []
    uploader.pool.acquire = lambda: acquired.append(1) or acquire()
    generator = torch.Generator().manual_seed(3)
    pieces, offset = [], 0
    for rows, pinned in ((5, False), (3, True), (9, True), (2, False)):
        value = torch.randn(rows, 6, generator=generator)
        pieces.append((value.pin_memory() if pinned else value, [offset, 0]))
        offset += rows
    reference = _reference(pieces, (offset, 6), None)
    kept = [weakref.ref(value) for value, _ in pieces if value.is_pinned()]
    with torch.cuda.stream(uploader.copy_stream):
        torch.cuda._sleep(_sleep_cycles(200))
    destination = uploader.upload_pieces((offset, 6), torch.float32, pieces)
    del pieces, value
    host_staging.reclaim_staged_uploads()  # The copies still wait behind the sleep.
    gc.collect()
    assert all(ref() is not None for ref in kept)
    seen = destination.clone()
    consumed = torch.cuda.Event()
    consumed.record()
    consumed.synchronize()  # The consumer stream only, as the sampler wait.
    host_staging.reclaim_staged_uploads()
    gc.collect()
    assert all(ref() is None for ref in kept)
    _assert_same_bits(seen.cpu(), reference)
    # Staged: 120 pageable bytes in ranges of 64 and 56, then 48 after the pinned pieces.
    assert len(acquired) == 3 and uploader.pool.waits == 0


@pytest.mark.cpu_only
def test_staged_uploader_is_one_per_device(monkeypatch):
    """Each CUDA device gets its own uploader, copy stream and pool, whatever the device spec."""

    class FakeStream:
        def __init__(self, device):
            self.device = device

    monkeypatch.setattr(host_staging.torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(host_staging.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(host_staging, "_STAGED_UPLOADERS", {})
    first = host_staging.staged_uploader(0)
    assert host_staging.staged_uploader("cuda:0") is first
    assert host_staging.staged_uploader(torch.device("cuda", 0)) is first
    second = host_staging.staged_uploader("cuda")  # The current device, 1.
    assert host_staging.staged_uploader(1) is second is not first
    assert [u.copy_stream.device for u in (first, second)] == [
        torch.device("cuda", 0),
        torch.device("cuda", 1),
    ]
    assert first.pool is not second.pool
    with pytest.raises(ValueError, match="CUDA device"):
        host_staging.staged_uploader("cpu")


@pytest.mark.cpu_only
def test_reserved_bytes_counts_only_the_native_copy_stream_pool(monkeypatch):
    """The native allocator's segments of this device's copy stream are summed; under
    cudaMallocAsync, which has no block snapshot, the snapshot is not called and the size is 0."""

    class FakeStream:
        def __init__(self, device):
            self.device, self.cuda_stream = device, 0x1000

    monkeypatch.setattr(host_staging.torch.cuda, "Stream", FakeStream)
    uploader = host_staging.StagedUploader(0)
    segments = [
        dict(device=0, stream=0x1000, total_size=2 << 20),
        dict(device=0, stream=0x1000, total_size=20 << 20),
        dict(device=0, stream=0x2000, total_size=4 << 20),  # Another stream.
        dict(device=1, stream=0x1000, total_size=8 << 20),  # Another device.
    ]
    monkeypatch.setattr(host_staging.torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(host_staging.torch.cuda, "memory_snapshot", lambda: segments)
    assert uploader.reserved_bytes() == 22 << 20

    def snapshot():
        raise RuntimeError("Calling snapshot with backend:cudaMallocAsync is not meaningful.")

    monkeypatch.setattr(host_staging.torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
    monkeypatch.setattr(host_staging.torch.cuda, "memory_snapshot", snapshot)
    assert uploader.reserved_bytes() == 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_staged_uploader_uploads_to_its_own_device():
    source = torch.arange(1 << 12, dtype=torch.float32)
    with torch.cuda.device(0):
        destination = host_staging.staged_uploader(1).upload_pieces(
            source.shape, source.dtype, [(source, [0])]
        )
    assert destination.device == torch.device("cuda", 1)
    assert host_staging.staged_uploader(1).copy_stream.device == torch.device("cuda", 1)
    torch.cuda.synchronize(1)
    assert torch.equal(destination.cpu(), source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_encoder_decoder_features_go_through_the_shared_uploader(monkeypatch):
    """The encoder-decoder runner packs host features with the device's uploader: one batch made
    on its copy stream while the consumer is busy, the same bytes as concatenating them."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderStage

    uploader = _uploader(1 << 16)
    monkeypatch.setattr(host_staging, "_STAGED_UPLOADERS", {torch.cuda.current_device(): uploader})
    stage = object.__new__(EncoderStage)
    generator = torch.Generator().manual_seed(3)
    features = [torch.randn(rows, 80, 30, generator=generator).half() for rows in (1, 2, 1)]
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        busy_done = torch.cuda.Event()
        torch.cuda._sleep(_sleep_cycles(200))
        busy_done.record()
        packed = stage._pack_features(features)
        seen = packed.clone()  # Runs after the sleep.
    uploader.copy_stream.synchronize()
    assert not busy_done.query()
    assert 1 <= len(uploader.pool.buffers) <= 2
    assert (uploader.reserved_bytes() > 0) == _native_allocator()
    consumer.synchronize()
    assert packed.is_cuda and torch.equal(seen.cpu(), torch.cat(features))
