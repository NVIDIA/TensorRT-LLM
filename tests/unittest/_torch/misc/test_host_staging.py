# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned uploads remain correct while callers recycle pageable source storage."""

import gc
import traceback
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from utils.util import DeviceSleepCtl, device_sleep

import tensorrt_llm
from tensorrt_llm._torch import host_staging
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


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


_TOKENS_PER_BLOCK = 16
_MAX_SEQ_LEN = 256
_REQUESTS = 4


def _kv_cache_manager() -> KVCacheManager:
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=_REQUESTS * 2 * _MAX_SEQ_LEN),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=_MAX_SEQ_LEN,
        max_batch_size=_REQUESTS,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    manager.add_dummy_requests(list(range(_REQUESTS)), [_MAX_SEQ_LEN // 2] * _REQUESTS)
    return manager


def _metadata(
    manager: KVCacheManager, uploader: host_staging.HostUploader | None
) -> TrtllmAttentionMetadata:
    metadata = TrtllmAttentionMetadata(
        max_num_requests=_REQUESTS,
        max_num_tokens=_REQUESTS * 8,
        kv_cache_manager=manager,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
    )
    metadata.host_uploader = uploader
    return metadata


def _prepare(metadata: TrtllmAttentionMetadata, step: int, num_seqs: int) -> None:
    metadata.request_ids = list(range(num_seqs))
    metadata.prompt_lens = [17 + 3 * row + step for row in range(num_seqs)]
    metadata.num_contexts = 0
    metadata.seq_lens = torch.full((num_seqs,), 1 + step % 3, dtype=torch.int)
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=[40 + 5 * row + 2 * step for row in range(num_seqs)],
    )
    metadata.prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trtllm_metadata_uploads_match_plain_copies():
    """Metadata prepared through a HostUploader while the host runs ahead of a busy
    stream holds the same device lengths as metadata prepared with plain copies."""
    manager = _kv_cache_manager()
    staged = _metadata(manager, host_staging.HostUploader())
    steps = list(enumerate((2, 4, 3, 4, 1, 4, 2)))
    observed = []
    busy = torch.cuda.Stream()
    with torch.cuda.stream(busy):
        for step, num_seqs in steps:
            # Earlier work still occupies the stream while the host prepares.
            torch.cuda._sleep(1_000_000)
            _prepare(staged, step, num_seqs)
            observed.append(
                (
                    staged.kv_lens_cuda[:num_seqs].clone(),
                    staged.prompt_lens_cuda[:num_seqs].clone(),
                )
            )
    busy.synchronize()
    # The plain copies read persistent host buffers, so they are prepared one step at a time.
    plain = _metadata(manager, None)
    for (step, num_seqs), (kv_lens, prompt_lens) in zip(steps, observed):
        _prepare(plain, step, num_seqs)
        torch.cuda.synchronize()
        assert torch.equal(kv_lens, plain.kv_lens_cuda[:num_seqs])
        assert torch.equal(prompt_lens, plain.prompt_lens_cuda[:num_seqs])
    manager.shutdown()


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


@contextmanager
def _held(stream):
    """Hold ``stream`` until the block ends: work queued on it meanwhile waits. Yields an event
    that completes only after the hold, which a host wait on the stream would outlast (60 s)."""
    ctl, released = DeviceSleepCtl(), torch.cuda.Event()
    with torch.cuda.stream(stream):
        device_sleep(60, ctl=ctl)
        released.record()
    try:
        yield released
    finally:
        ctl.cancel()


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
    """With the consumer stream held, an upload of many buffers' worth reuses the pool's buffers
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
    with _held(consumer), torch.cuda.stream(consumer):
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
    """The copies run on the copy stream while the consumer is held: the upload returns and the
    copy stream finishes before the hold ends, and the consumer reads the batch once released."""
    uploader = _uploader(1 << 16)
    source = torch.randn(1 << 20)  # 4 MiB, 64 buffers' worth through 2 buffers
    uploader.upload_pieces(source.shape, source.dtype, [(source, [0])])
    torch.cuda.synchronize()
    consumer = torch.cuda.Stream()
    with _held(consumer) as released, torch.cuda.stream(consumer):
        destination = uploader.upload_pieces(source.shape, source.dtype, [(source, [0])])
        seen = destination.clone()  # Runs after the hold.
        uploader.copy_stream.synchronize()
        assert not released.query(), "the upload waited for the consumer"
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
    with _held(consumer) as released, torch.cuda.stream(consumer):
        first = uploader.upload_pieces(ones.shape, ones.dtype, [(ones, [0])])
        seen = first.clone()  # Runs after the hold.
        pointer = first.data_ptr()
        del first
        second = uploader.upload_pieces(twos.shape, twos.dtype, [(twos, [0])])
        assert second.data_ptr() != pointer
        uploader.copy_stream.synchronize()  # The second batch's bytes are written by now.
        assert not released.query()
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
    with _held(uploader.copy_stream):
        destination = uploader.upload_pieces((offset, 6), torch.float32, pieces)
        del pieces, value
        host_staging.reclaim_staged_uploads()  # The copies still wait behind the hold.
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


@pytest.fixture
def fake_encodes(monkeypatch):
    """The encoder entry point on fake streams whose work completes only when the host waits for
    an event recorded on them. `encode(stream, marker, fail)` runs one encode; `in_use` gets, for
    each batch the uploader makes, the earlier encodes whose encoder has not finished, and `waits`
    the unfinished events the host waited for."""
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
        EncoderGroup,
        MultimodalEncoderContractError,
        encode_multimodal_by_groups,
    )
    from tensorrt_llm.inputs import multimodal
    from tensorrt_llm.inputs.multimodal import (
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
        MultimodalItemMetadata,
        MultimodalItemPartition,
        MultimodalParams,
    )

    current, waits = [], []

    class FakeStream:
        """Its work completes only when the host waits for an event recorded on it."""

        def __init__(self, device=None):
            self.device, self.events = device, []

        def wait_event(self, event):
            pass

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            self.stream, self.done = None, False

        def record(self, stream=None):
            self.stream = stream if stream is not None else current[-1]
            self.stream.events.append(self)

        def query(self):
            return self.done

        def synchronize(self):
            if not self.done and self.stream in (main, aux):
                waits.append(self)
            events = self.stream.events
            for event in events[: len(events) - events[::-1].index(self)]:
                event.done = True

    @contextmanager
    def on(stream):
        current.append(stream)
        try:
            yield
        finally:
            current.pop()

    cuda = host_staging.torch.cuda
    monkeypatch.setattr(cuda, "Stream", FakeStream)
    monkeypatch.setattr(cuda, "Event", FakeEvent)
    monkeypatch.setattr(cuda, "stream", on)
    monkeypatch.setattr(cuda, "current_stream", lambda device=None: current[-1])
    monkeypatch.setattr(cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(multimodal, "prefer_pinned", lambda: True)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    in_use, empty = [], torch.empty

    def empty_on_host(*args, device=None, **kwargs):
        if device is not None and torch.device(device).type == "cuda":
            # A batch is made: count the earlier encodes whose encoder has not finished.
            in_use.append(sum(not used.done for used, _ in encodes))
            device = None
        return empty(*args, device=device, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_on_host)
    pool = host_staging.PinnedStagingPool(buffer_bytes=1 << 10, buffer_count=2)
    uploader = host_staging.StagedUploader(0, pool)
    monkeypatch.setattr(host_staging, "_STAGED_UPLOADERS", {0: uploader})
    main, aux = FakeStream(), FakeStream()
    encodes = []  # (event after the encoder's work, its stream)

    def encode(stream, marker, fail=None):
        """One encode on `stream`; with `fail`, the encoder raises ("encoder") or returns too few
        rows ("rows") after its work."""

        def encoder(pixels, grid):
            used = FakeEvent()
            used.record()
            encodes.append((used, current[-1]))
            if fail == "encoder":
                raise RuntimeError("encoder failed")
            return pixels[: 2 if fail == "rows" else None] * 2

        pixels = torch.full((4, 8), float(marker))
        metadata = MultimodalItemMetadata(
            [("image", 0)],
            output_embedding_lengths=[4],
            input_partitions={
                "image": {
                    "pixels": MultimodalItemPartition(item_lengths=(4,)),
                    "grid": MultimodalItemPartition(),
                }
            },
        )
        data = {
            "image": {"pixels": pixels, "grid": torch.tensor([[4]])},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
        }
        group = EncoderGroup(("image",), encoder)
        params = [MultimodalParams(multimodal_data=data)]
        with on(stream):
            if fail is None:
                out = encode_multimodal_by_groups((group,), params, ["image.pixels"])
                assert out[:, 0].tolist() == [2.0 * marker] * 4
                return
            error = RuntimeError if fail == "encoder" else MultimodalEncoderContractError
            with pytest.raises(error):
                encode_multimodal_by_groups((group,), params, ["image.pixels"])

    return SimpleNamespace(
        encode=encode,
        in_use=in_use,
        waits=waits,
        encodes=encodes,
        main=main,
        aux=aux,
        uploader=uploader,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("bound", [1, 2])
def test_back_to_back_encodes_stay_within_the_copy_stream_bound(fake_encodes, bound):
    """Encoder-only passes back to back (no LLM step, so no sampler wait) and a prefetch encode
    on a side stream, through the encoder entry point, while the streams that read the batches
    never finish on their own: when the uploader makes a batch, at most `bound - 1` earlier
    encodes may still read their batches, so the copy stream's pool holds at most `bound`
    encodes' batches (what the KV-cache estimate reserves). The host waits only for the encode
    `bound` back, and not at all once the consumers are done."""
    fake = fake_encodes
    fake.uploader.max_encodes_in_flight = bound
    for marker, stream in enumerate((fake.main,) * 4 + (fake.aux, fake.main), start=1):
        fake.encode(stream, marker)
    assert fake.in_use == [0] + [bound - 1] * 5
    # Each wait was for the encode `bound` back, on the stream that read its batch.
    assert len(fake.waits) == 6 - bound
    for wait, (used, stream) in zip(fake.waits, fake.encodes):
        assert wait.stream is stream and stream.events.index(wait) > stream.events.index(used)
    # Consumers done (the sampler wait): the next encode makes its batch without a wait.
    for stream in (fake.main, fake.aux):
        stream.events[-1].synchronize()
    fake.waits.clear()
    fake.encode(fake.main, 7)
    assert fake.in_use[-1] == 0 and not fake.waits


@pytest.mark.cpu_only
@pytest.mark.parametrize("fail", ["encoder", "rows"])
def test_a_failed_encode_counts_toward_the_copy_stream_bound(fake_encodes, fail):
    """Item scheduling and prefetch keep serving after an encode fails, so a failed encode ends
    like the others: after one failed encode and two that succeed, at most one earlier encode may
    still read its batch when the uploader makes one."""
    fake = fake_encodes
    for marker, failure in enumerate((fail, None, None), start=1):
        fake.encode(fake.main, marker, failure)
    assert fake.in_use == [0, 1, 1]
    # The second success waited for the failed encode's batch.
    assert len(fake.waits) == 1 and fake.encodes[0][0].done


@pytest.mark.cpu_only
@pytest.mark.parametrize("first_fails", [False, True])
def test_encoder_decoder_encodes_stay_within_the_copy_stream_bound(fake_encodes, first_fails):
    """Encoder-decoder passes back to back over host features (one request each, no CUDA graph),
    through `EncoderStage.forward`: each pass ends one encode, also when its encoder fails, so at
    most one earlier encode may still read its batch when the uploader makes one."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderStage
    from tensorrt_llm._torch.pyexecutor.engine.runners.interface import ScheduledInputs
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    fake = fake_encodes

    def execute(prepared):
        used = torch.cuda.Event()
        used.record()
        fake.encodes.append((used, fake.main))
        if first_fails and len(fake.encodes) == 1:
            raise RuntimeError("encoder failed")
        return prepared.kwargs["input_features"] * 2

    stage = object.__new__(EncoderStage)
    stage._config = SimpleNamespace(max_num_tokens=64)
    stage._prepare_encoder_feature_graph_inputs = lambda *args: None
    stage._build_attention_metadata = lambda sequence_lengths, request_ids: None
    stage._execute_prepared = execute
    for marker in (1, 2, 3):
        batch = ScheduledRequests()
        batch.encoder_requests = [
            SimpleNamespace(
                py_encoder_input_features=torch.full((4, 8), float(marker)),
                encoder_output_len=4,
                py_request_id=marker,
            )
        ]
        with torch.cuda.stream(fake.main):
            if first_fails and marker == 1:
                with pytest.raises(RuntimeError, match="encoder failed"):
                    stage.forward(ScheduledInputs(batch=batch), resource_manager=None)
                continue
            out = stage.forward(ScheduledInputs(batch=batch), resource_manager=None)
        assert out["encoder_hidden_states"][:, 0].tolist() == [2.0 * marker] * 4
    assert fake.in_use == [0, 1, 1]
    # The third pass waited for the first one's batch.
    assert len(fake.waits) == 1 and fake.encodes[0][0].done


@pytest.mark.cpu_only
@pytest.mark.parametrize("chain", ["direct", "cause", "context", "cycle"])
def test_a_kept_error_keeps_no_batch(monkeypatch, chain):
    """A failed encode ends after its batch is gone, also when the caller keeps the error and that
    error wraps the error of a helper that read the batch (`raise ... from`, raised while handling
    it, or causes in a loop). The errors and their messages stay."""

    class Batch:
        pass

    def read(batch):
        raise ValueError("kernel failed")

    def encoder(batch):
        if chain == "direct":
            read(batch)
        try:
            read(batch)
        except ValueError as inner:
            error = RuntimeError("encoder failed")
            if chain == "cycle":
                inner.__cause__ = error
            if chain == "context":
                raise error
            raise error from inner

    def encode():
        batch = Batch()
        batches.append(weakref.ref(batch))
        encoder(batch)

    batches, alive = [], []
    uploader = SimpleNamespace(end_encode=lambda: alive.append(batches[0]() is not None))
    monkeypatch.setattr(host_staging, "_STAGED_UPLOADERS", {0: uploader})
    monkeypatch.setattr(host_staging.torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(Exception) as failed:
        with host_staging.staged_encode():
            encode()
    assert alive == [False]
    error = failed.value
    if chain == "direct":
        assert str(error) == "kernel failed"
        return
    assert str(error) == "encoder failed" and str(error.__context__) == "kernel failed"
    assert (error.__cause__ is error.__context__) == (chain != "context")
    assert any("kernel failed" in line for line in traceback.format_exception(error))


# 16 MiB of float32: the native allocator gives each batch a segment of its own.
_GPU_BATCH = (1024, 4096)


def _read_batch(batch, fail):
    """The encoder's read of its batch on the GPU: `2 * batch` in 8 columns. With `fail` it raises
    after that read: "direct", or "wrapped" from the error of a helper that read the batch."""

    def read(batch):
        out = batch[:, :8] * 2
        if fail:
            raise RuntimeError("encoder failed")
        return out

    if fail != "wrapped":
        return read(batch)
    try:
        read(batch)
    except RuntimeError as error:
        raise RuntimeError("encoder failed") from error


def _gpu_encode(path):
    """`encode(marker, fail=None)`: one encode through the encoder entry point of `path` on the
    current stream, of a host batch full of `marker`. Its encoder reads the batch on the GPU
    (`_read_batch`) and returns `2 * marker` in 8 columns, or raises after that read with `fail`."""
    rows = _GPU_BATCH[0]
    if path == "multimodal":
        from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
            EncoderGroup,
            encode_multimodal_by_groups,
        )
        from tensorrt_llm.inputs.multimodal import (
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
            MultimodalItemMetadata,
            MultimodalItemPartition,
            MultimodalParams,
        )

        def encode(marker, fail=None):
            def encoder(pixels, grid):
                return _read_batch(pixels, fail)

            metadata = MultimodalItemMetadata(
                [("image", 0)],
                output_embedding_lengths=[rows],
                input_partitions={
                    "image": {
                        "pixels": MultimodalItemPartition(item_lengths=(rows,)),
                        "grid": MultimodalItemPartition(),
                    }
                },
            )
            pixels = torch.full(_GPU_BATCH, float(marker))
            data = {
                "image": {"pixels": pixels, "grid": torch.tensor([[rows]])},
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
            }
            group = EncoderGroup(("image",), encoder)
            params = [MultimodalParams(multimodal_data=data)]
            return encode_multimodal_by_groups((group,), params, ["image.pixels"])

        return encode

    from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderStage
    from tensorrt_llm._torch.pyexecutor.engine.runners.interface import ScheduledInputs
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    stage = object.__new__(EncoderStage)
    stage._config = SimpleNamespace(max_num_tokens=rows)
    stage._prepare_encoder_feature_graph_inputs = lambda *args: None
    stage._build_attention_metadata = lambda sequence_lengths, request_ids: None

    def encode(marker, fail=None):
        def execute(prepared):
            return _read_batch(prepared.kwargs["input_features"], fail)

        stage._execute_prepared = execute
        batch = ScheduledRequests()
        batch.encoder_requests = [
            SimpleNamespace(
                py_encoder_input_features=torch.full(_GPU_BATCH, float(marker)),
                encoder_output_len=rows,
                py_request_id=marker,
            )
        ]
        out = stage.forward(ScheduledInputs(batch=batch), resource_manager=None)
        return out["encoder_hidden_states"]

    return encode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("first_fails", [None, "direct", "wrapped"])
@pytest.mark.parametrize("path", ["multimodal", "encoder_decoder"])
def test_gpu_encodes_stay_within_the_copy_stream_bound(monkeypatch, path, first_fails):
    """Four encodes back to back through the encoder entry point, with the real allocator,
    streams and stream records, the second and third while their consumer stream is held. Each
    encode drops its batch before it ends, also when its encoder fails and the caller keeps the
    error (as item scheduling keeps it as the cause of the error it raises), also when that error
    wraps the error of a helper that read the batch. So once the host waited for the first encode,
    its block can be reused, and the copy stream's pool never holds more than the two batches of
    the bound (native allocator)."""
    encode = _gpu_encode(path)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()  # So the copy stream's pool holds only this test's batches.
    uploader = _uploader(4 << 20)
    monkeypatch.setattr(host_staging, "_STAGED_UPLOADERS", {torch.cuda.current_device(): uploader})
    batch_bytes = _GPU_BATCH[0] * _GPU_BATCH[1] * 4
    base = uploader.reserved_bytes()
    consumer, ctl = torch.cuda.Stream(), DeviceSleepCtl()
    errors, outputs, pool = [], [], []
    try:
        with torch.cuda.stream(consumer):
            for marker in (1, 2, 3, 4):
                if marker == 2:
                    device_sleep(60, ctl=ctl)  # Holds the work of the second and third encodes.
                elif marker == 4:
                    ctl.cancel()  # The fourth encode waits for the second one.
                if first_fails and marker == 1:
                    with pytest.raises(RuntimeError, match="encoder failed") as failed:
                        encode(marker, fail=first_fails)
                    errors.append(failed.value)  # Kept until the end.
                else:
                    outputs.append((marker, encode(marker)))
                pool.append((uploader.reserved_bytes() - base) / batch_bytes)
    finally:
        ctl.cancel()
    torch.cuda.synchronize()
    for marker, out in outputs:
        assert torch.equal(out.cpu(), torch.full((_GPU_BATCH[0], 8), 2.0 * marker))
    if _native_allocator():
        assert max(pool) <= 2, pool
    else:  # cudaMallocAsync keeps no pool apart for the copy stream.
        assert pool == [0] * 4
    assert len(errors) == bool(first_fails)


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
    on its copy stream while the consumer is held, the same bytes as concatenating them."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderStage

    uploader = _uploader(1 << 16)
    monkeypatch.setattr(host_staging, "_STAGED_UPLOADERS", {torch.cuda.current_device(): uploader})
    stage = object.__new__(EncoderStage)
    generator = torch.Generator().manual_seed(3)
    features = [torch.randn(rows, 80, 30, generator=generator).half() for rows in (1, 2, 1)]
    consumer = torch.cuda.Stream()
    with _held(consumer) as released, torch.cuda.stream(consumer):
        packed = stage._pack_features(features)
        seen = packed.clone()  # Runs after the hold.
        uploader.copy_stream.synchronize()
        assert not released.query()
    assert 1 <= len(uploader.pool.buffers) <= 2
    assert (uploader.reserved_bytes() > 0) == _native_allocator()
    consumer.synchronize()
    assert packed.is_cuda and torch.equal(seen.cpu(), torch.cat(features))
