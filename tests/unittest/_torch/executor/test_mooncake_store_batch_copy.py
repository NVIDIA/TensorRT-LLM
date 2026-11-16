# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched staging copies (TRTLLM_MOONCAKE_STORE_BATCH_COPY) put the same bytes in the same places as the loop.

The CPU tests run every copy path against host memory: the CUDA calls are replaced by `ctypes.memmove`
emulators with the runtime's semantics (1D, pitched 2D, batch), and "device" pages live in a host buffer. What
is under test is the planning (which byte goes where), which is where a batched copy can go wrong. The GPU test
at the end runs the real runtime calls when a device is present.
"""

import ctypes
import random

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import staging


class FakeStore:
    def register_buffer(self, base, size):
        return 0

    def unregister_buffer(self, base):
        return 0


class Recorder:
    """Host-memory stand-ins for the three runtime calls, counting how often each is used."""

    def __init__(self):
        self.calls = {"1d": 0, "2d": 0, "batch": 0}
        self.streams = []

    def memcpy(self, dst, src, size, kind, stream):
        self.calls["1d"] += 1
        self.streams.append(int(stream))
        ctypes.memmove(int(dst), int(src), int(size))

    def memcpy2d(self, dst, dpitch, src, spitch, width, height, kind, stream):
        self.calls["2d"] += 1
        self.streams.append(int(stream))
        assert dpitch >= width and spitch >= width and height >= 2
        for row in range(height):
            ctypes.memmove(int(dst) + row * dpitch, int(src) + row * spitch, int(width))

    def batch(self, dsts, srcs, sizes, stream):
        self.calls["batch"] += 1
        self.streams.append(int(stream))
        for dst, src, size in zip(dsts, srcs, sizes):
            ctypes.memmove(int(dst), int(src), int(size))


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(staging, "_memcpy_async", rec.memcpy)
    monkeypatch.setattr(staging, "_memcpy2d_async", rec.memcpy2d)
    monkeypatch.setattr(staging, "_memcpy_batch_async", rec.batch)
    monkeypatch.setattr(staging, "_batch_api_available", lambda: True)
    monkeypatch.setattr(staging, "_BATCH_API_BROKEN", False)
    return rec


class Layout:
    """`regions` per page, each region of a page `stride` apart per page index, as PageAddressing lays them out.

    `cut_every` splits a region at an allocation boundary for some pages, so pages differ in region count.
    """

    def __init__(self, num_pages, region_sizes, gap=32, cut_pages=()):
        self.num_pages = num_pages
        self.region_sizes = region_sizes
        self.strides = [size + gap for size in region_sizes]
        total = sum(stride * num_pages for stride in self.strides)
        self.buffer = torch.zeros(total, dtype=torch.uint8)
        self.base = self.buffer.data_ptr()
        self.region_bases = []
        offset = 0
        for stride in self.strides:
            self.region_bases.append(self.base + offset)
            offset += stride * num_pages
        self.cut_pages = set(cut_pages)

    def buffers(self, page):
        addresses, sizes = [], []
        for base, stride, size in zip(self.region_bases, self.strides, self.region_sizes):
            start = base + stride * page
            if page in self.cut_pages and size > 8:
                addresses += [start, start + 8]
                sizes += [8, size - 8]
            else:
                addresses.append(start)
                sizes.append(size)
        return addresses, sizes

    def fill_random(self, seed):
        gen = torch.Generator().manual_seed(seed)
        self.buffer.copy_(torch.randint(0, 256, self.buffer.shape, dtype=torch.uint8, generator=gen))


def _pool(layout, num_slots):
    return staging.HostStagingPool(
        slot_bytes=sum(layout.region_sizes), num_slots=num_slots, store=FakeStore(), label="t"
    )


def _random_page_map(rng, num_pages, count, consecutive):
    if consecutive:
        start = rng.randrange(0, num_pages - count + 1)
        pages = list(range(start, start + count))
        # Mix in a jump, as a reused prefix rarely occupies one run of page indices.
        if count > 4:
            pages[count // 2 :] = [min(num_pages - 1, p + 3) for p in pages[count // 2 :]]
            pages = list(dict.fromkeys(pages))
        return pages
    return rng.sample(range(num_pages), count)


def _gather(mode, layout, pages, monkeypatch):
    monkeypatch.setattr(staging, "_BATCH_COPY", mode)
    pool = _pool(layout, len(pages))
    addresses, sizes = zip(*(layout.buffers(p) for p in pages))
    staged, totals = staging.stage_batch_for_put(pool, list(addresses), list(sizes), stream=7)
    out = pool._buffer.clone()
    pool.close()
    return out, staged, totals


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("mode", ["2d", "batch", "auto"])
def test_gather_matches_the_loop_on_random_page_maps(recorder, monkeypatch, seed, mode):
    rng = random.Random(seed)
    layout = Layout(64, [rng.choice([96, 128, 200]) for _ in range(rng.randint(1, 3))],
                    cut_pages={rng.randrange(64) for _ in range(5)} if seed % 2 else ())
    layout.fill_random(seed)
    pages = _random_page_map(rng, 64, rng.randint(1, 24), consecutive=seed % 3 != 0)

    expect, expect_staged, expect_totals = _gather(None, layout, pages, monkeypatch)
    got, staged, totals = _gather(mode, layout, pages, monkeypatch)

    assert torch.equal(got, expect)
    # The store is handed the same slot addresses and lengths either way (pool buffers differ, so compare offsets).
    assert [t for t in totals] == [t for t in expect_totals]
    assert len(staged) == len(expect_staged)


def _scatter(mode, layout, pages, payload, only, monkeypatch):
    monkeypatch.setattr(staging, "_BATCH_COPY", mode)
    layout.buffer.zero_()
    pool = _pool(layout, len(pages))
    pool._buffer.copy_(payload)
    addresses, sizes = zip(*(layout.buffers(p) for p in pages))
    staging.unstage_batch_after_get(pool, list(addresses), list(sizes), stream=7, only=only)
    pool.close()
    return layout.buffer.clone()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("mode", ["2d", "batch"])
def test_scatter_matches_the_loop_on_random_page_maps(recorder, monkeypatch, seed, mode):
    rng = random.Random(100 + seed)
    layout = Layout(48, [rng.choice([64, 136, 256]) for _ in range(rng.randint(1, 3))],
                    cut_pages={rng.randrange(48) for _ in range(4)} if seed % 2 else ())
    pages = _random_page_map(rng, 48, rng.randint(1, 20), consecutive=seed % 3 != 0)
    gen = torch.Generator().manual_seed(seed)
    payload = torch.randint(0, 256, (len(pages) * sum(layout.region_sizes),), dtype=torch.uint8, generator=gen)
    only = None if seed % 2 == 0 else sorted(rng.sample(range(len(pages)), max(1, len(pages) // 2)))

    expect = _scatter(None, layout, pages, payload, only, monkeypatch)
    got = _scatter(mode, layout, pages, payload, only, monkeypatch)
    assert torch.equal(got, expect)
    assert got.any()


def test_round_trip_is_byte_exact(recorder, monkeypatch):
    layout = Layout(32, [128, 64])
    layout.fill_random(3)
    original = layout.buffer.clone()
    pages = [5, 6, 7, 8, 20, 21, 2]
    slots_payload, _, _ = _gather("2d", layout, pages, monkeypatch)
    restored = _scatter("batch", layout, pages, slots_payload, None, monkeypatch)
    for page in pages:
        for address, size in zip(*layout.buffers(page)):
            offset = address - layout.base
            assert torch.equal(restored[offset : offset + size], original[offset : offset + size])


def test_2d_runs_collapse_consecutive_pages(recorder, monkeypatch):
    layout = Layout(64, [128, 128])
    pages = list(range(10, 42))
    _gather("2d", layout, pages, monkeypatch)
    # One pitched copy per region instead of one call per page per region.
    assert recorder.calls == {"1d": 0, "2d": 2, "batch": 0}


def test_batch_mode_is_one_call_per_batch(recorder, monkeypatch):
    layout = Layout(64, [128, 96, 64])
    _gather("batch", layout, [3, 9, 1, 40, 41], monkeypatch)
    assert recorder.calls == {"1d": 0, "2d": 0, "batch": 1}


def test_batch_mode_never_uses_the_null_stream(recorder, monkeypatch):
    monkeypatch.setattr(staging, "_BATCH_COPY", "batch")
    layout = Layout(16, [64])
    pool = _pool(layout, 4)
    addresses, sizes = zip(*(layout.buffers(p) for p in [1, 2, 3]))
    staging.stage_batch_for_put(pool, list(addresses), list(sizes), stream=0)
    assert recorder.calls["batch"] == 0 and recorder.calls["2d"] == 1
    pool.close()


def test_a_failing_batch_call_falls_back_to_2d(recorder, monkeypatch):
    def broken(*_args):
        raise RuntimeError("cudaErrorNotSupported")

    monkeypatch.setattr(staging, "_memcpy_batch_async", broken)
    layout = Layout(32, [128])
    layout.fill_random(9)
    expect, _, _ = _gather(None, layout, [4, 5, 6, 11], monkeypatch)
    got, _, _ = _gather("auto", layout, [4, 5, 6, 11], monkeypatch)
    assert torch.equal(got, expect)
    assert staging._BATCH_API_BROKEN


def test_overlapping_destinations_keep_the_loop(recorder, monkeypatch):
    # The same device page twice in one load: only the loop's order decides which slot wins.
    layout = Layout(16, [64])
    pages = [3, 3]
    payload = torch.arange(128, dtype=torch.uint8)
    expect = _scatter(None, layout, pages, payload, None, monkeypatch)
    before = dict(recorder.calls)
    got = _scatter("batch", layout, pages, payload, None, monkeypatch)
    assert torch.equal(got, expect)
    assert recorder.calls["batch"] == before["batch"] and recorder.calls["2d"] == before["2d"]


def test_snapshot_ring_gather_many_matches_gather(recorder, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.snapshot import SnapshotRing

    layout = Layout(32, [96, 32])
    layout.fill_random(5)
    pages = [7, 2, 30]
    slots = [5, 0, 3]  # ring slots come back in any order
    buffers = [layout.buffers(p) for p in pages]

    monkeypatch.setattr(staging, "_BATCH_COPY", None)
    ring = SnapshotRing(_pool(layout, 8))
    expect_placed = [ring.gather(s, a, z, 7) for s, (a, z) in zip(slots, buffers)]
    expect = ring._pool._buffer.clone()
    expect_base = ring._pool.slot_address(0)
    ring.close()

    monkeypatch.setattr(staging, "_BATCH_COPY", "2d")
    ring = SnapshotRing(_pool(layout, 8))
    placed = ring.gather_many(slots, [b[0] for b in buffers], [b[1] for b in buffers], 7)
    got = ring._pool._buffer.clone()
    base = ring._pool.slot_address(0)
    ring.close()
    assert torch.equal(got, expect)
    assert [(a - base, t) for a, t in placed] == [(a - expect_base, t) for a, t in expect_placed]


def test_off_by_default_and_parsing():
    assert staging._parse_batch_copy(None) is None
    assert staging._parse_batch_copy("0") is None
    assert staging._parse_batch_copy("1") == "auto"
    assert staging._parse_batch_copy("2d") == "2d"
    assert staging._parse_batch_copy("batch") == "batch"
    with pytest.raises(ValueError):
        staging._parse_batch_copy("yes please")


def test_a_page_too_large_for_its_slot_is_refused(recorder, monkeypatch):
    monkeypatch.setattr(staging, "_BATCH_COPY", "2d")
    layout = Layout(8, [64])
    pool = staging.HostStagingPool(slot_bytes=32, num_slots=2, store=FakeStore(), label="t")
    with pytest.raises(ValueError, match="does not fit"):
        staging.stage_batch_for_put(pool, [layout.buffers(1)[0]], [layout.buffers(1)[1]], stream=7)
    pool.close()


# ---- real runtime calls ----


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("mode", ["2d", "batch", "auto"])
def test_device_round_trip_matches_the_loop(monkeypatch, mode):
    if mode == "batch" and not staging._batch_api_available():
        pytest.skip("cuda-python has no cudaMemcpyBatchAsync")
    monkeypatch.setattr(staging, "_BATCH_API_BROKEN", False)
    device = torch.device("cuda")
    regions, region_bytes, num_pages = [512, 448], 512, 64
    strides = [r + 64 for r in regions]
    dev = torch.randint(0, 256, (sum(s * num_pages for s in strides),), dtype=torch.uint8, device=device)
    bases = [dev.data_ptr(), dev.data_ptr() + strides[0] * num_pages]

    def buffers(page):
        return [b + s * page for b, s in zip(bases, strides)], list(regions)

    pages = [3, 4, 5, 6, 30, 31, 9, 60]
    addresses, sizes = zip(*(buffers(p) for p in pages))
    stream = torch.cuda.Stream()
    results = {}
    for m in (None, mode):
        monkeypatch.setattr(staging, "_BATCH_COPY", m)
        pool = staging.HostStagingPool(slot_bytes=sum(regions), num_slots=len(pages), store=FakeStore(), label="t")
        staging.stage_batch_for_put(pool, list(addresses), list(sizes), stream.cuda_stream)
        stream.synchronize()
        slots = pool._buffer.clone()
        target = torch.zeros_like(dev)
        shift = target.data_ptr() - dev.data_ptr()
        staging.unstage_batch_after_get(
            pool, [[a + shift for a in page] for page in addresses], list(sizes), stream.cuda_stream, only=[0, 2, 3, 7]
        )
        stream.synchronize()
        results[m] = (slots, target.cpu())
        pool.close()
    assert torch.equal(results[None][0], results[mode][0])
    assert torch.equal(results[None][1], results[mode][1])
