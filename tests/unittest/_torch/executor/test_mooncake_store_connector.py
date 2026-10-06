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
"""Unit tests for the Mooncake store KV cache connector.

Covers the three places a defect is a wrong answer rather than a slow one: the
arithmetic that maps a store key onto KV cache memory, the host staging that
has to leave a staged transfer byte-identical to a zero-copy one, and the
scheduler state that decides which pages are published under which key.

Runs without a Mooncake installation and without a GPU: the store handle is
replaced by an in-process fake, and the KV cache layout is synthesized from
plain integers, which is all the addressing arithmetic needs.
"""

import contextlib
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    RequestData,
    SchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import (
    KvCacheBufferRef,
    KvCacheLayerGroupLayout,
    KvCacheLayout,
    KvCacheRegion,
    _gpu_pool_mapping_bytes,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import addressing as addressing_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import gpudirect as gpudirect_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import staging as staging_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.addressing import (
    PageAddressing,
    mapping_origin,
    merge_intervals,
    split_at_boundaries,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.gpudirect import (
    RangeFacts,
    describe_range,
    format_diagnosis,
    reservation_start,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import BlockHashChain
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    PageTransfer,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler import (
    MooncakeStoreConnectorScheduler,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.worker import (
    MooncakeStoreConnectorWorker,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

TOKENS_PER_BLOCK = 4


# ---- fixtures and fakes ----


class FakeStore:
    """Records calls and remembers which keys exist, nothing more."""

    def __init__(self):
        self.objects = set()
        self.registered = []
        self.put_calls = []
        self.get_calls = []
        self.exist_calls = []
        self.closed = False
        self.fail_gets_for = set()
        #: Workers built against this store. `make_worker` shuts each one down;
        #: the fixture repeats it as a backstop for early failures.
        self.workers = []

    def register_buffer(self, address, size):
        self.registered.append((address, size))
        return 0

    def batch_is_exist(self, keys):
        self.exist_calls.append(list(keys))
        return [1 if key in self.objects else 0 for key in keys]

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, *_args, **_kwargs):
        self.put_calls.append((list(keys), [list(a) for a in addresses], [list(s) for s in sizes]))
        self.objects.update(keys)
        return [sum(size) for size in sizes]

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        self.get_calls.append((list(keys), [list(a) for a in addresses], [list(s) for s in sizes]))
        return [
            -1 if (key in self.fail_gets_for or key not in self.objects) else sum(size)
            for key, size in zip(keys, sizes)
        ]

    def close(self):
        self.closed = True


def make_layout(*, num_groups=1, regions_per_group=1, num_slots=8):
    """A layout whose regions are laid out back to back in a fake address space."""
    groups = []
    base = 0x1000
    for group_id in range(num_groups):
        regions = []
        for region_id in range(regions_per_group):
            size = 64 * (region_id + 1)
            stride = size
            regions.append(
                KvCacheRegion(
                    base=base,
                    size=size,
                    stride=stride,
                    num_slots=num_slots,
                    buffers=(KvCacheBufferRef(layer_id=group_id, role="key"),),
                )
            )
            base += stride * num_slots
        groups.append(
            KvCacheLayerGroupLayout(
                layer_group_id=group_id,
                layer_ids=(group_id,),
                window_size=None,
                regions=tuple(regions),
            )
        )
    return KvCacheLayout(tokens_per_block=TOKENS_PER_BLOCK, groups=tuple(groups))


#: Small enough that a single page crosses it, so the splitting is visible in a
#: layout built from plain integers. The real value is 2 to 32 MiB.
MAPPING_BYTES = 256
#: Mapping-aligned, so a page starting here ends 44 bytes into the next mapping.
REGION_BASE = 0x1000
REGION_SIZE = 300
REGION_SLOTS = 4
#: A reservation base that is not a multiple of the mapping size, which is what
#: `cuMemAddressReserve` may return and what the multiples would get wrong.
UNALIGNED_ORIGIN = REGION_BASE - MAPPING_BYTES // 4


def make_mapped_layout(*, mapping_bytes=MAPPING_BYTES):
    """One region wide enough to straddle `mapping_bytes`, packed slot to slot."""
    return KvCacheLayout(
        tokens_per_block=TOKENS_PER_BLOCK,
        groups=(
            KvCacheLayerGroupLayout(
                layer_group_id=0,
                layer_ids=(0,),
                window_size=None,
                regions=(
                    KvCacheRegion(
                        base=REGION_BASE,
                        size=REGION_SIZE,
                        stride=REGION_SIZE,
                        num_slots=REGION_SLOTS,
                        buffers=(KvCacheBufferRef(layer_id=0, role="key"),),
                    ),
                ),
            ),
        ),
        gpu_pool_mapping_bytes=mapping_bytes,
    )


@pytest.fixture(autouse=True)
def unknown_reservation_base(monkeypatch):
    """None of these fake addresses is in a reservation, so skip the driver.

    Keeps the cuts the same whether or not the host running the tests has a
    usable CUDA driver. Tests that care where a pool's mappings begin install a
    base of their own with `reservation_base`.
    """
    monkeypatch.setattr(addressing_module, "reservation_start", lambda _address: None)


@pytest.fixture
def reservation_base(monkeypatch):
    """Report a chosen reservation base for every address, as the driver would."""

    def install(base):
        monkeypatch.setattr(addressing_module, "reservation_start", lambda _address: base)
        return base

    return install


@pytest.fixture
def store_config(tmp_path, monkeypatch):
    path = tmp_path / "mooncake.json"
    path.write_text(
        json.dumps(
            {
                "metadata_server": "http://127.0.0.1:8080/metadata",
                "master_server_address": "127.0.0.1:50051",
                "protocol": "tcp",
                "device_name": "",
                "global_segment_size": "1GiB",
                "local_buffer_size": "256MiB",
                "model_key": "test-model",
            }
        )
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    return path


def set_pool_setting(store_config, **settings):
    """Rewrite the rendered client config, as a differently configured server would."""
    raw = json.loads(store_config.read_text())
    raw.update(settings)
    store_config.write_text(json.dumps(raw))


def make_llm_args(
    *, enable_attention_dp: bool = False, run_dir: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        model="/models/test-model",
        kv_cache_config=SimpleNamespace(tokens_per_block=TOKENS_PER_BLOCK),
        tensor_parallel_size=1,
        enable_attention_dp=enable_attention_dp,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        sparse_attention_config=None,
        # How a rank reaches settings the process that rendered the client
        # config could not hand it; see config.pool_config.
        kv_connector_config=SimpleNamespace(
            connector="mooncake-store",
            mooncake_store=SimpleNamespace(run_dir=run_dir),
        ),
    )


@pytest.fixture
def fake_store(monkeypatch):
    """Replace the store handle, and tear down any worker a test builds."""
    store = FakeStore()
    monkeypatch.setattr(worker_module, "_open_store", lambda _config: (store, "10.0.0.1"))
    # The node budget is a property of the machine the test happens to run on,
    # which is not what any of these tests are about.
    monkeypatch.setattr(worker_module, "validate_node_budget", lambda _config: None)
    yield store
    for worker in store.workers:
        worker.shutdown()
    worker_module._LOCAL_WORKER = None
    worker_module._LOCAL_WORKER_READY.clear()


@contextlib.contextmanager
def make_worker(
    fake_store: FakeStore,
    *,
    layout: KvCacheLayout | None = None,
    run_dir: str | None = None,
    enable_attention_dp: bool = False,
) -> Iterator[MooncakeStoreConnectorWorker]:
    """Build a worker and shut it down before the test call phase ends.

    Registering a layout starts the background save thread, and
    pytest-threadleak snapshots threads around the call phase only, so
    fixture teardown would run too late to keep it quiet.
    """
    worker = MooncakeStoreConnectorWorker(
        make_llm_args(run_dir=run_dir, enable_attention_dp=enable_attention_dp)
    )
    fake_store.workers.append(worker)
    if layout is not None:
        worker.register_kv_cache_layout(layout)
    try:
        yield worker
    finally:
        worker.shutdown()


def make_request(
    request_id,
    tokens,
    cache_salt=None,
    lora_task_id=None,
    multimodal_hashes=None,
    multimodal_positions=None,
    multimodal_lengths=None,
):
    """A stand-in for the fields the leader reads off an `LlmRequest`.

    The multimodal three travel together on a real request, so a caller that
    names any of them gets plausible values for the others.
    """
    if multimodal_positions is None and (
        multimodal_hashes is not None or multimodal_lengths is not None
    ):
        multimodal_positions = [0]
    if multimodal_lengths is None and multimodal_positions is not None:
        multimodal_lengths = [1] * len(multimodal_positions)
    return SimpleNamespace(
        request_id=request_id,
        cache_salt=cache_salt,
        lora_task_id=lora_task_id,
        multimodal_hashes=multimodal_hashes,
        multimodal_positions=multimodal_positions,
        multimodal_lengths=multimodal_lengths,
        get_tokens=lambda _beam=0, _tokens=tuple(tokens): list(_tokens),
    )


#: One item's content hash in the form the bindings report: 8 int32 chunks.
IMAGE_A_HASH = [1, 2, 3, 4, 5, 6, 7, 8]
IMAGE_B_HASH = [1, 2, 3, 4, 5, 6, 7, 9]


# ---- addressing ----


@pytest.mark.parametrize(
    "intervals,expected",
    [
        ([], []),
        ([(0, 10)], [(0, 10)]),
        ([(0, 10), (10, 20)], [(0, 20)]),
        ([(0, 10), (5, 20)], [(0, 20)]),
        ([(0, 10), (20, 30)], [(0, 10), (20, 30)]),
        ([(20, 30), (0, 10)], [(0, 10), (20, 30)]),
        ([(0, 100), (10, 20)], [(0, 100)]),
        ([(0, 0), (5, 10)], [(5, 10)]),
    ],
)
def test_merge_intervals(intervals, expected):
    assert merge_intervals(intervals) == expected


def test_page_addressing_resolves_every_region_of_a_page():
    layout = make_layout(regions_per_group=3, num_slots=4)
    addressing = PageAddressing(layout)
    regions = layout.groups[0].regions

    addresses, sizes = addressing.buffers(0, 2)
    assert sizes == [region.size for region in regions]
    assert addresses == [region.base + region.stride * 2 for region in regions]
    assert addressing.bytes_per_page(0) == sum(region.size for region in regions)


def test_page_addressing_rejects_out_of_range_page():
    addressing = PageAddressing(make_layout(num_slots=4))
    with pytest.raises(IndexError):
        addressing.buffers(0, 4)
    with pytest.raises(IndexError):
        addressing.buffers(0, -1)


def test_page_addressing_registration_covers_every_slot_once():
    layout = make_layout(num_groups=2, regions_per_group=2, num_slots=4)
    ranges = PageAddressing(layout).registration_ranges()

    # Regions were laid out back to back, so the whole span merges into one.
    all_regions = [region for group in layout.groups for region in group.regions]
    lowest = min(region.base for region in all_regions)
    highest = max(
        region.base + region.stride * (region.num_slots - 1) + region.size for region in all_regions
    )
    assert ranges == [(lowest, highest)]


@pytest.mark.parametrize(
    "start,size",
    [
        (0, MAPPING_BYTES),  # exactly one mapping
        (0, 2 * MAPPING_BYTES),  # two whole mappings
        (MAPPING_BYTES // 2, MAPPING_BYTES),  # straddles one boundary
        (0, 3 * MAPPING_BYTES + 7),  # ragged tail
        (MAPPING_BYTES + 8, 16),  # well inside one mapping
        (REGION_BASE, REGION_SIZE),
    ],
    ids=["one_mapping", "two_mappings", "straddling", "ragged_tail", "interior", "region"],
)
def test_split_at_boundaries_preserves_the_bytes_and_their_order(start, size):
    """A page's payload is the same concatenation however the range is cut."""
    pieces = split_at_boundaries(start, size, MAPPING_BYTES)

    assert sum(piece_size for _, piece_size in pieces) == size
    address = start
    for piece_address, piece_size in pieces:
        assert piece_address == address
        address += piece_size
        # The point of splitting: no piece may cover two mappings.
        assert piece_address % MAPPING_BYTES + piece_size <= MAPPING_BYTES


@pytest.mark.parametrize("boundary", [None, 0], ids=["unknown", "zero"])
def test_split_at_boundaries_leaves_the_range_whole_without_a_boundary(boundary):
    assert split_at_boundaries(REGION_BASE, 4096, boundary) == [(REGION_BASE, 4096)]


def test_split_at_boundaries_passes_an_empty_range_through():
    assert split_at_boundaries(REGION_BASE, 0, MAPPING_BYTES) == [(REGION_BASE, 0)]


@pytest.mark.parametrize(
    "start,size",
    [
        (UNALIGNED_ORIGIN, MAPPING_BYTES),  # the first whole mapping
        (UNALIGNED_ORIGIN, 3 * MAPPING_BYTES + 7),  # ragged tail
        (REGION_BASE, REGION_SIZE),  # starts mid-mapping
        (REGION_BASE, 8),  # well inside one mapping
    ],
    ids=["one_mapping", "ragged_tail", "region", "interior"],
)
def test_split_at_boundaries_counts_from_the_origin(start, size):
    """An unaligned reservation is what the multiples of the size would miss."""
    pieces = split_at_boundaries(start, size, MAPPING_BYTES, UNALIGNED_ORIGIN)

    assert sum(piece_size for _, piece_size in pieces) == size
    address = start
    for piece_address, piece_size in pieces:
        assert piece_address == address
        address += piece_size
        offset = piece_address - UNALIGNED_ORIGIN
        assert offset % MAPPING_BYTES + piece_size <= MAPPING_BYTES


def test_split_at_boundaries_on_an_unaligned_origin_cuts_elsewhere():
    """Otherwise there would be nothing to tell the two apart."""
    pieces = split_at_boundaries(REGION_BASE, REGION_SIZE, MAPPING_BYTES, UNALIGNED_ORIGIN)

    assert pieces != split_at_boundaries(REGION_BASE, REGION_SIZE, MAPPING_BYTES)
    first = MAPPING_BYTES - (REGION_BASE - UNALIGNED_ORIGIN)
    assert pieces == [(REGION_BASE, first), (REGION_BASE + first, REGION_SIZE - first)]


def test_page_addressing_splits_a_page_that_crosses_a_mapping_boundary():
    addressing = PageAddressing(make_mapped_layout())
    assert addressing.mapping_bytes == MAPPING_BYTES

    addresses, sizes = addressing.buffers(0, 0)
    assert list(zip(addresses, sizes)) == [
        (REGION_BASE, MAPPING_BYTES),
        (REGION_BASE + MAPPING_BYTES, REGION_SIZE - MAPPING_BYTES),
    ]
    # One region still contributes its whole payload, just in several buffers.
    assert sum(sizes) == addressing.bytes_per_page(0)


def test_page_addressing_registration_stays_inside_one_mapping():
    ranges = PageAddressing(make_mapped_layout()).registration_ranges()
    span_end = REGION_BASE + REGION_SIZE * REGION_SLOTS

    # The same bytes as the unsplit span, handed over one mapping at a time.
    assert len(ranges) > 1
    assert sum(end - start for start, end in ranges) == span_end - REGION_BASE
    previous_end = REGION_BASE
    for start, end in ranges:
        assert start == previous_end
        previous_end = end
        assert start % MAPPING_BYTES + (end - start) <= MAPPING_BYTES
    assert previous_end == span_end


def test_page_addressing_cuts_where_the_pool_reservation_puts_the_boundaries(reservation_base):
    """A pool's mappings tile from its reservation base, wherever that lands."""
    origin = reservation_base(UNALIGNED_ORIGIN)
    addressing = PageAddressing(make_mapped_layout())
    assert addressing.mapping_origins == (origin,)

    first = MAPPING_BYTES - (REGION_BASE - origin) % MAPPING_BYTES
    addresses, sizes = addressing.buffers(0, 0)
    assert list(zip(addresses, sizes)) == [
        (REGION_BASE, first),
        (REGION_BASE + first, REGION_SIZE - first),
    ]
    assert sum(sizes) == addressing.bytes_per_page(0)


def test_page_addressing_registers_inside_the_mappings_of_an_unaligned_pool(reservation_base):
    origin = reservation_base(UNALIGNED_ORIGIN)
    ranges = PageAddressing(make_mapped_layout()).registration_ranges()
    span_end = REGION_BASE + REGION_SIZE * REGION_SLOTS

    assert sum(end - start for start, end in ranges) == span_end - REGION_BASE
    previous_end = REGION_BASE
    for start, end in ranges:
        assert start == previous_end
        previous_end = end
        assert (start - origin) % MAPPING_BYTES + (end - start) <= MAPPING_BYTES
    assert previous_end == span_end


def test_page_addressing_assumes_the_multiples_when_the_reservation_is_unknown():
    """The behavior before the pools were split at all, and the worker warns."""
    addressing = PageAddressing(make_mapped_layout())
    assert addressing.mapping_origins == (0,)
    assert addressing.buffers(0, 0) == (
        [REGION_BASE, REGION_BASE + MAPPING_BYTES],
        [MAPPING_BYTES, REGION_SIZE - MAPPING_BYTES],
    )


def test_mapping_origin_leaves_the_driver_alone_without_a_boundary(monkeypatch):
    """Nothing to count from is needed when nothing is being cut."""
    monkeypatch.setattr(
        addressing_module, "reservation_start", lambda _address: pytest.fail("driver consulted")
    )
    assert mapping_origin(REGION_BASE, None) == 0


def test_page_addressing_leaves_everything_whole_when_the_mapping_size_is_unknown():
    """Without a granularity the pools are addressed and registered unsplit."""
    addressing = PageAddressing(make_mapped_layout(mapping_bytes=None))
    assert addressing.mapping_bytes is None
    assert addressing.buffers(0, 1) == ([REGION_BASE + REGION_SIZE], [REGION_SIZE])
    assert addressing.registration_ranges() == [
        (REGION_BASE, REGION_BASE + REGION_SIZE * REGION_SLOTS)
    ]


def test_page_addressing_rejects_mixed_slot_counts():
    region_a = KvCacheRegion(base=0, size=8, stride=8, num_slots=4, buffers=())
    region_b = KvCacheRegion(base=64, size=8, stride=8, num_slots=8, buffers=())
    layout = KvCacheLayout(
        tokens_per_block=TOKENS_PER_BLOCK,
        groups=(
            KvCacheLayerGroupLayout(
                layer_group_id=0,
                layer_ids=(0,),
                window_size=None,
                regions=(region_a, region_b),
            ),
        ),
    )
    with pytest.raises(ValueError, match="slot counts"):
        PageAddressing(layout)


# ---- GPU pool mapping size ----


def make_v2_impl(granularity):
    """Just enough of a `KVCacheManagerV2` impl to read the GPU granularity from."""
    return SimpleNamespace(
        _storage=SimpleNamespace(
            _levels=[SimpleNamespace(storage=SimpleNamespace(pool_size_granularity=granularity))]
        )
    )


def test_gpu_pool_mapping_bytes_reads_the_granularity_of_the_gpu_level():
    assert _gpu_pool_mapping_bytes(make_v2_impl(32 << 20)) == 32 << 20


@pytest.mark.parametrize(
    "impl",
    [
        SimpleNamespace(),
        SimpleNamespace(_storage=SimpleNamespace(_levels=[])),
        make_v2_impl(0),
        make_v2_impl(-1),
        make_v2_impl(None),
        make_v2_impl("32MiB"),
    ],
    ids=["no_storage", "no_gpu_level", "zero", "negative", "none", "not_a_number"],
)
def test_gpu_pool_mapping_bytes_returns_none_rather_than_guessing(impl):
    assert _gpu_pool_mapping_bytes(impl) is None


# ---- GPUDirect diagnosis ----

_GRANULARITY = 32 << 20
_RESERVATION = 0x7F0000000000


@pytest.fixture
def stub_driver_facts(monkeypatch):
    """Report chosen facts instead of asking a driver that may not be present."""

    def install(*facts):
        by_address = {fact.address: fact for fact in facts}
        monkeypatch.setattr(
            gpudirect_module, "describe_range", lambda address, _length: by_address[address]
        )
        return [(fact.address, fact.address + fact.length) for fact in facts]

    return install


def test_range_facts_detects_a_range_that_crosses_a_mapping_boundary():
    inside = RangeFacts(
        address=_RESERVATION + _GRANULARITY, length=_GRANULARITY, range_start=_RESERVATION
    )
    assert inside.fits_one_mapping(_GRANULARITY) is True

    crossing = RangeFacts(
        address=_RESERVATION + _GRANULARITY // 2, length=_GRANULARITY, range_start=_RESERVATION
    )
    assert crossing.fits_one_mapping(_GRANULARITY) is False


@pytest.mark.parametrize(
    "mapping_bytes,range_start",
    [(None, _RESERVATION), (0, _RESERVATION), (_GRANULARITY, None)],
    ids=["no_granularity", "zero_granularity", "no_reservation"],
)
def test_range_facts_will_not_judge_without_the_granularity_and_the_base(
    mapping_bytes, range_start
):
    facts = RangeFacts(address=_RESERVATION, length=64, range_start=range_start)
    assert facts.fits_one_mapping(mapping_bytes) is None


def test_reservation_start_answers_nothing_for_an_address_the_driver_disowns():
    """Callers get the documented fallback rather than an exception."""
    assert reservation_start(_RESERVATION) is None


def test_describe_range_reports_a_failure_instead_of_raising():
    facts = describe_range(_RESERVATION, 1 << 20)
    assert isinstance(facts, RangeFacts)
    assert facts.address == _RESERVATION
    assert facts.describe()


def test_format_diagnosis_blames_the_missing_peermem_module(monkeypatch, stub_driver_facts):
    monkeypatch.setattr(gpudirect_module, "peermem_loaded", lambda: False)
    monkeypatch.delenv("WITH_NVIDIA_PEERMEM", raising=False)
    ranges = stub_driver_facts(
        RangeFacts(address=_RESERVATION, length=64, range_start=_RESERVATION)
    )

    report = format_diagnosis(ranges, rank=0, mapping_bytes=_GRANULARITY)
    assert "ibv_reg_mr" in report
    assert "nvidia_peermem is not loaded" in report


def test_format_diagnosis_blames_the_boundary_on_the_dmabuf_path(monkeypatch, stub_driver_facts):
    monkeypatch.setattr(gpudirect_module, "peermem_loaded", lambda: True)
    monkeypatch.setenv("WITH_NVIDIA_PEERMEM", "0")
    ranges = stub_driver_facts(
        RangeFacts(
            address=_RESERVATION + _GRANULARITY // 2,
            length=_GRANULARITY,
            range_start=_RESERVATION,
        )
    )

    report = format_diagnosis(ranges, rank=3, mapping_bytes=_GRANULARITY)
    assert "rank 3" in report
    assert "1 range(s) cross a 32 MiB pool mapping boundary" in report
    assert "EINVAL" in report


def test_format_diagnosis_reports_memory_the_driver_withholds_from_rdma(
    monkeypatch, stub_driver_facts
):
    monkeypatch.setattr(gpudirect_module, "peermem_loaded", lambda: True)
    monkeypatch.setenv("WITH_NVIDIA_PEERMEM", "0")
    ranges = stub_driver_facts(
        RangeFacts(address=_RESERVATION, length=64, range_start=_RESERVATION, gdr_capable=False)
    )

    report = format_diagnosis(ranges, rank=0, mapping_bytes=_GRANULARITY)
    assert "not GPUDirect-RDMA capable" in report


def test_format_diagnosis_admits_when_it_explains_nothing(monkeypatch, stub_driver_facts):
    monkeypatch.setattr(gpudirect_module, "peermem_loaded", lambda: True)
    monkeypatch.setenv("WITH_NVIDIA_PEERMEM", "1")
    ranges = stub_driver_facts(
        RangeFacts(address=_RESERVATION, length=64, range_start=_RESERVATION, gdr_capable=True)
    )

    report = format_diagnosis(ranges, rank=0, mapping_bytes=_GRANULARITY)
    assert "Nothing here explains a registration failure" in report


# ---- worker ----


def test_worker_registers_each_pool_mapping_separately(store_config, fake_store):
    with make_worker(fake_store, layout=make_mapped_layout()) as worker:
        assert worker.is_registered
        assert len(fake_store.registered) > 1
        for address, size in fake_store.registered:
            assert address % MAPPING_BYTES + size <= MAPPING_BYTES
        assert sum(size for _, size in fake_store.registered) == REGION_SIZE * REGION_SLOTS


def test_worker_registers_each_mapping_of_an_unaligned_pool(
    store_config, fake_store, reservation_base
):
    origin = reservation_base(UNALIGNED_ORIGIN)
    with make_worker(fake_store, layout=make_mapped_layout()) as worker:
        assert worker.is_registered
        for address, size in fake_store.registered:
            assert (address - origin) % MAPPING_BYTES + size <= MAPPING_BYTES
        assert sum(size for _, size in fake_store.registered) == REGION_SIZE * REGION_SLOTS


def test_worker_says_when_it_is_assuming_where_the_mappings_begin(
    store_config, fake_store, monkeypatch
):
    """Silence would leave a misplaced cut looking like a Mooncake failure."""
    messages = []
    monkeypatch.setattr(worker_module.logger, "warning", messages.append)

    with make_worker(fake_store, layout=make_mapped_layout()):
        pass
    assert any("could not read the base" in message for message in messages)


def test_worker_reports_the_driver_facts_only_on_request(store_config, fake_store, monkeypatch):
    messages = []
    # The logger does not propagate to the root logger, so caplog cannot see it.
    monkeypatch.setattr(worker_module.logger, "info", messages.append)

    monkeypatch.delenv(worker_module.REGISTRATION_DEBUG_ENV, raising=False)
    with make_worker(fake_store, layout=make_mapped_layout()):
        pass
    assert not any("GPU registration diagnosis" in message for message in messages)

    messages.clear()
    monkeypatch.setenv(worker_module.REGISTRATION_DEBUG_ENV, "1")
    with make_worker(fake_store, layout=make_mapped_layout()):
        pass
    assert any("GPU registration diagnosis" in message for message in messages)


def test_worker_prefix_hit_needs_every_layer_group(store_config, fake_store):
    layout = make_layout(num_groups=2)
    with make_worker(fake_store, layout=layout) as worker:
        hashes = [bytes([index]) * 16 for index in range(3)]

        assert worker.count_prefix_hit(hashes) == 0

        # Populate blocks 0 and 1 completely, and block 2 only partially.
        for block in range(2):
            for group_id in range(2):
                fake_store.objects.add(worker._namespaces[group_id].key(hashes[block]))
        fake_store.objects.add(worker._namespaces[0].key(hashes[2]))

        assert worker.count_prefix_hit(hashes) == 2


def test_worker_prefix_hit_stops_at_the_first_gap(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        hashes = [bytes([index]) * 16 for index in range(3)]
        # With block 1 missing, block 2 is unusable even though it is present,
        # because a prefix is replayed contiguously.
        fake_store.objects.add(worker._namespaces[0].key(hashes[0]))
        fake_store.objects.add(worker._namespaces[0].key(hashes[2]))
        assert worker.count_prefix_hit(hashes) == 1


def test_worker_load_raises_when_a_page_is_missing(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        transfers = RequestTransfers(7, [PageTransfer(b"\x00" * 16, 0, 1)])
        worker.bind_connector_meta(SimpleNamespace(loads=[transfers], saves=[]))
        with pytest.raises(RuntimeError, match="already"):
            worker.start_load_kv(None)


def test_worker_load_addresses_the_requested_page(store_config, fake_store):
    layout = make_layout(regions_per_group=2)
    with make_worker(fake_store, layout=layout) as worker:
        block_hash = b"\x00" * 16
        key = worker._namespaces[0].key(block_hash)
        fake_store.objects.add(key)

        transfers = RequestTransfers(7, [PageTransfer(block_hash, 0, 3)])
        worker.bind_connector_meta(SimpleNamespace(loads=[transfers], saves=[]))
        worker.start_load_kv(None)

        (keys, addresses, sizes) = fake_store.get_calls[0]
        expected_addresses, expected_sizes = PageAddressing(layout).buffers(0, 3)
        assert keys == [key]
        assert addresses == [expected_addresses]
        assert sizes == [expected_sizes]


def test_worker_reports_a_request_finished_once_its_saves_drain(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        # One submission outstanding: the request is closed but must not be released.
        worker._outstanding_saves[42] = 1
        assert worker.get_finished([42], []) == ([], [])

        worker._outstanding_saves.pop(42)
        assert worker.get_finished([], []) == ([42], [])
        # Reported once only.
        assert worker.get_finished([], []) == ([], [])


# ---- host staging ----


@contextlib.contextmanager
def make_staged_worker(fake_store, store_config, *, layout, budget=None):
    """A worker configured to pass pages through pinned host slots.

    `budget` patches the pinned-memory ceiling rather than setting a field:
    the allocation follows from the layout and the transfer batch now, so the
    ceiling is the only thing left that can bind.
    """
    set_pool_setting(store_config, stage_through_host=True)
    with contextlib.ExitStack() as stack:
        if budget is not None:
            patch = stack.enter_context(pytest.MonkeyPatch.context())
            patch.setattr(worker_module, "MAX_STAGING_BUFFER_BYTES", budget)
        worker = stack.enter_context(make_worker(fake_store, layout=layout))
        yield worker


@pytest.fixture
def staged_copies(monkeypatch):
    """Record the copies staging would issue, instead of running them."""
    copies = []
    monkeypatch.setattr(
        staging_module,
        "_memcpy_async",
        lambda dst, src, size, kind, stream: copies.append((int(dst), int(src), int(size))),
    )
    # Imported into the worker by name, so replace the worker's binding.
    monkeypatch.setattr(worker_module, "_sync_stream", lambda _stream: None)
    return copies


@pytest.fixture
def fake_cuda(monkeypatch):
    """Present a CUDA device on a host that has none, recording set_device calls.

    Only safe for paths that do not allocate or launch, and exists to exercise
    the device bookkeeping around the save thread.
    """
    recorded = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: recorded.append(index))
    return recorded


def test_staging_put_hands_the_store_one_host_buffer_per_page(
    store_config, fake_store, staged_copies
):
    layout = make_layout(regions_per_group=3)
    with make_staged_worker(fake_store, store_config, layout=layout) as worker:
        block_hash = b"\x07" * 16
        worker._put([RequestTransfers(1, [PageTransfer(block_hash, 0, 2)])])

        device_addresses, device_sizes = PageAddressing(layout).buffers(0, 2)
        slot = worker._save_staging.slot_address(0)
        (keys, addresses, sizes) = fake_store.put_calls[0]

        assert keys == [worker._namespaces[0].key(block_hash)]
        # The store sees one contiguous host buffer whose length is the sum of
        # the device regions. That equality is what keeps a staged write
        # byte-identical to a zero-copy one.
        assert addresses == [[slot]]
        assert sizes == [[sum(device_sizes)]]

        # Every region was gathered, in order, into its place in the slot.
        expected = []
        offset = 0
        for address, size in zip(device_addresses, device_sizes):
            expected.append((slot + offset, address, size))
            offset += size
        assert staged_copies == expected


def test_staging_get_scatters_back_to_the_device_regions(store_config, fake_store, staged_copies):
    layout = make_layout(regions_per_group=3)
    with make_staged_worker(fake_store, store_config, layout=layout) as worker:
        block_hash = b"\x03" * 16
        fake_store.objects.add(worker._namespaces[0].key(block_hash))

        worker.bind_connector_meta(
            SimpleNamespace(loads=[RequestTransfers(7, [PageTransfer(block_hash, 0, 5)])], saves=[])
        )
        worker.start_load_kv(None)

        device_addresses, device_sizes = PageAddressing(layout).buffers(0, 5)
        slot = worker._load_staging.slot_address(0)
        (_keys, addresses, sizes) = fake_store.get_calls[0]
        assert addresses == [[slot]]
        assert sizes == [[sum(device_sizes)]]

        # The scatter is the mirror of the gather: same split, opposite direction.
        expected = []
        offset = 0
        for address, size in zip(device_addresses, device_sizes):
            expected.append((address, slot + offset, size))
            offset += size
        assert staged_copies == expected


def test_staging_does_not_scatter_a_failed_load(store_config, fake_store, staged_copies):
    layout = make_layout()
    with make_staged_worker(fake_store, store_config, layout=layout) as worker:
        block_hash = b"\x09" * 16
        key = worker._namespaces[0].key(block_hash)
        fake_store.objects.add(key)
        fake_store.fail_gets_for.add(key)

        worker.bind_connector_meta(
            SimpleNamespace(loads=[RequestTransfers(7, [PageTransfer(block_hash, 0, 1)])], saves=[])
        )
        with pytest.raises(RuntimeError, match="failed to load"):
            worker.start_load_kv(None)

        # A failed read leaves the slot holding whatever it held before, and
        # copying that onto the page would put unrelated bytes where the
        # runtime already promised computed KV.
        assert staged_copies == []


def test_the_ranks_device_is_captured_and_adopted_by_the_save_thread(
    store_config, fake_store, fake_cuda
):
    """The save thread must not run on torch's default device.

    It issues staging copies against pointers owned by the rank's device. A
    stream created on device 0 instead fails every copy with
    cudaErrorInvalidValue, and only on ranks other than 0.
    """
    with make_worker(fake_store, layout=make_layout()) as worker:
        assert worker._device_index == 3

        deadline = time.monotonic() + 5.0
        while 3 not in fake_cuda and time.monotonic() < deadline:
            time.sleep(0.01)
        assert 3 in fake_cuda, f"save thread set devices {fake_cuda}, expected the rank's 3"
        # Never the thread-local default.
        assert 0 not in fake_cuda


def test_a_save_thread_that_cannot_start_fails_registration(
    store_config, fake_store, fake_cuda, monkeypatch
):
    """A worker whose save thread died must not go on to report itself ready.

    The thread's setup binds this rank's device. If that raises and the thread
    exits, every later `wait_for_save` counts a save that nothing will ever
    consume, and the requests holding those pages stay pinned for good.
    """

    def refuse(_index):
        raise RuntimeError("cudaSetDevice failed")

    monkeypatch.setattr(torch.cuda, "set_device", refuse)

    with pytest.raises(RuntimeError, match="save thread"):
        with make_worker(fake_store, layout=make_layout()):
            pytest.fail("registration should not have completed")


def test_shutdown_leaves_the_store_open_under_a_save_still_reading(
    store_config, fake_store, monkeypatch
):
    """A timed join is not evidence that the thread stopped.

    The save thread reads the KV pools through the store handle and the staging
    slots, so closing the handle or dropping those buffers mid-transfer takes
    the memory out from under it.
    """
    monkeypatch.setattr(worker_module, "SAVE_DRAIN_TIMEOUT", 0.1)
    reading = threading.Event()
    release = threading.Event()
    store_put = fake_store.batch_put_from_multi_buffers

    def blocking_put(*args, **kwargs):
        reading.set()
        assert release.wait(30.0), "the test never released the save thread"
        return store_put(*args, **kwargs)

    fake_store.batch_put_from_multi_buffers = blocking_put

    with make_worker(fake_store, layout=make_layout()) as worker:
        # Queued directly: what matters is a thread inside the store call, not
        # how the pass that produced the pages reached it.
        worker._save_queue.put(
            (
                SimpleNamespace(synchronize=lambda: None),
                [RequestTransfers(1, [PageTransfer(b"\x05" * 16, 0, 0)])],
            )
        )
        assert reading.wait(30.0), "the save thread never reached the store"

        worker.shutdown()

        assert not fake_store.closed
        assert worker._store is fake_store
        # Kept rather than dropped, so a later call retries the join.
        assert worker._save_thread is not None

        release.set()
        worker.shutdown()

        assert fake_store.closed
        assert worker._store is None
        assert worker._save_thread is None


def test_staging_narrows_the_batch_to_the_budget(store_config, fake_store, staged_copies):
    layout = make_layout(regions_per_group=2, num_slots=8)
    page_bytes = PageAddressing(layout).bytes_per_page(0)
    with make_staged_worker(
        fake_store, store_config, layout=layout, budget=2 * page_bytes
    ) as worker:
        assert worker._save_staging.num_slots == 2
        assert worker._batch_size == 2

        hashes = [bytes([index]) * 16 for index in range(5)]
        worker._put(
            [
                RequestTransfers(
                    1,
                    [PageTransfer(block_hash, 0, index) for index, block_hash in enumerate(hashes)],
                )
            ]
        )
        # Five pages through two slots, and no call wider than the slot count.
        assert [len(keys) for keys, _a, _s in fake_store.put_calls] == [2, 2, 1]


# ---- scheduler ----


class FakeWorker:
    """Stands in for the process-local worker's lookup service."""

    def __init__(self, hit_blocks=0):
        self.hit_blocks = hit_blocks
        self.queries = []

    def count_prefix_hit(self, block_hashes):
        self.queries.append(list(block_hashes))
        return min(self.hit_blocks, len(block_hashes))


def make_scheduler(store_config, hit_blocks=0):
    scheduler = MooncakeStoreConnectorScheduler(make_llm_args())
    scheduler._worker = FakeWorker(hit_blocks)
    return scheduler


def request_data(request_id, new_tokens, page_indices):
    return RequestData(
        request_id=request_id,
        new_tokens=list(new_tokens),
        new_block_ids=list(page_indices),
        computed_position=0,
        num_scheduled_tokens=len(new_tokens),
        # Keyed by layer group id.
        new_block_ids_by_layer_group={0: list(page_indices)},
    )


def test_scheduler_never_offers_the_whole_prompt(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=99)
    # Exactly three full blocks: the last one is withheld so the runtime still
    # has a token to run a forward pass on.
    request = make_request(1, list(range(3 * TOKENS_PER_BLOCK)))
    matched, _ = scheduler.get_num_new_matched_tokens(request, 0)
    assert matched == 2 * TOKENS_PER_BLOCK


def test_scheduler_declines_partial_local_matches(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=2)
    request = make_request(1, list(range(5 * TOKENS_PER_BLOCK)))
    assert scheduler.get_num_new_matched_tokens(request, TOKENS_PER_BLOCK + 1) == (0, False)


def test_scheduler_skips_local_prefix_when_looking_up(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=1)
    tokens = list(range(6 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 2 * TOKENS_PER_BLOCK)
    # Blocks 0 and 1 are on device already; candidates start at block 2 and stop
    # short of the final block.
    full_chain = list(BlockHashChain(TOKENS_PER_BLOCK).extend(tokens))
    assert scheduler._worker.queries[0] == full_chain[2:5]


def test_scheduler_builds_loads_for_the_offered_blocks(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=2)
    tokens = list(range(5 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)

    output = SchedulerOutput(new_requests=[request_data(1, tokens, [10, 11, 12, 13, 14])])
    metadata = scheduler.build_connector_meta(output)

    assert [page.page_index for page in metadata.loads[0].pages] == [10, 11]
    # Blocks 0 and 1 came from the store, so only blocks 2..4 are written back.
    assert [page.page_index for page in metadata.saves[0].pages] == [12, 13, 14]


def test_scheduler_does_not_resave_blocks_across_iterations(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=0)
    tokens = list(range(2 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)

    first = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, [4, 5])])
    )
    assert [page.page_index for page in first.saves[0].pages] == [4, 5]

    # A generation step completes one more block; only that block is saved.
    more_tokens = list(range(2 * TOKENS_PER_BLOCK, 3 * TOKENS_PER_BLOCK))
    second = scheduler.build_connector_meta(
        SchedulerOutput(cached_requests=[request_data(1, more_tokens, [6])])
    )
    assert [page.page_index for page in second.saves[0].pages] == [6]


def test_scheduler_waits_for_a_block_to_fill_before_saving(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=0)
    tokens = list(range(TOKENS_PER_BLOCK + 1))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)

    metadata = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, [4, 5])])
    )
    # Page 5 holds a single token, so only the full block is offered up.
    assert [page.page_index for page in metadata.saves[0].pages] == [4]


def test_scheduler_skips_blocks_without_a_page_in_every_group(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=0)
    tokens = list(range(2 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)

    data = request_data(1, tokens, [4, 5])
    data.new_block_ids_by_layer_group[1] = [7, BAD_PAGE_INDEX]
    metadata = scheduler.build_connector_meta(SchedulerOutput(new_requests=[data]))

    # Block 1 has no page in group 1, so neither of its halves is stored; block 0
    # contributes one page per group.
    assert [(page.layer_group_id, page.page_index) for page in metadata.saves[0].pages] == [
        (0, 4),
        (1, 7),
    ]


def test_scheduler_request_finished_pins_pages_only_when_saving(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=0)
    tokens = list(range(2 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.build_connector_meta(SchedulerOutput(new_requests=[request_data(1, tokens, [4, 5])]))
    assert scheduler.request_finished(request, [4, 5]) is True
    # State is dropped with the request, so a second call reports nothing pending.
    assert scheduler.request_finished(request, [4, 5]) is False


def test_scheduler_request_finished_is_false_without_saves(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=0)
    request = make_request(1, list(range(TOKENS_PER_BLOCK - 1)))
    scheduler.get_num_new_matched_tokens(request, 0)
    assert scheduler.request_finished(request, []) is False


def test_scheduler_saves_a_replayed_request_from_its_new_pages(store_config):
    """Rollback frees the pages the first attempt recorded, and another request
    may own them by the time the replay saves.
    """
    scheduler = make_scheduler(store_config, hit_blocks=0)
    tokens = list(range(2 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)

    scheduler.get_num_new_matched_tokens(request, 0)
    first = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, [4, 5])])
    )
    assert [page.page_index for page in first.saves[0].pages] == [4, 5]

    scheduler.request_reset(request)

    # The replay is admitted to different pages and must save from those.
    scheduler.get_num_new_matched_tokens(request, 0)
    replay = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, [9, 10])])
    )
    assert [page.page_index for page in replay.saves[0].pages] == [9, 10]


def test_scheduler_isolates_requests_by_cache_salt(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=1)
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    scheduler.get_num_new_matched_tokens(make_request(1, tokens, cache_salt="a"), 0)
    scheduler.get_num_new_matched_tokens(make_request(2, tokens, cache_salt="b"), 0)
    assert scheduler._worker.queries[0] != scheduler._worker.queries[1]


def test_scheduler_isolates_requests_by_lora_adapter(store_config):
    """Same prompt, different adapter, different keys.

    Sharing them would serve one adapter's KV to the other, which is a wrong
    answer rather than a slow one.
    """
    scheduler = make_scheduler(store_config, hit_blocks=1)
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    scheduler.get_num_new_matched_tokens(make_request(1, tokens), 0)
    scheduler.get_num_new_matched_tokens(make_request(2, tokens, lora_task_id=0), 0)
    scheduler.get_num_new_matched_tokens(make_request(3, tokens, lora_task_id=1), 0)

    queries = scheduler._worker.queries
    assert len({tuple(query) for query in queries}) == 3


def test_scheduler_isolates_requests_by_multimodal_content(store_config):
    """Two images behind the same placeholder tokens are two prefixes."""
    scheduler = make_scheduler(store_config, hit_blocks=1)
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    scheduler.get_num_new_matched_tokens(
        make_request(1, tokens, multimodal_hashes=[IMAGE_A_HASH]), 0
    )
    scheduler.get_num_new_matched_tokens(
        make_request(2, tokens, multimodal_hashes=[IMAGE_B_HASH]), 0
    )
    scheduler.get_num_new_matched_tokens(make_request(3, tokens), 0)

    queries = scheduler._worker.queries
    assert len({tuple(query) for query in queries}) == 3

    # The same image again is the same prefix, which is the point of keying on
    # the content rather than refusing the request.
    scheduler.get_num_new_matched_tokens(
        make_request(4, tokens, multimodal_hashes=[IMAGE_A_HASH]), 0
    )
    assert queries[3] == queries[0]


def test_scheduler_bypasses_multimodal_requests_it_cannot_identify(store_config):
    """Media without hashes gets neither a lookup nor a save.

    The prompt's placeholder tokens say nothing about the content behind them,
    so there is no key that names this request's pages and no one else's.
    """
    scheduler = make_scheduler(store_config, hit_blocks=2)
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens, multimodal_positions=[4], multimodal_lengths=[4])

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert scheduler._worker.queries == []

    metadata = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, [7, 8, 9])])
    )
    assert metadata.loads == []
    assert metadata.saves == []
    # Nothing was handed to a save thread, so nothing pins its pages either.
    assert scheduler.request_finished(request, [7, 8, 9]) is False


def test_scheduler_does_no_per_request_work_for_a_capacity_only_role(store_config):
    """It transfers nothing, so a request costs it nothing.

    No hash chain is built, nothing is looked up, and no state is kept for a
    save to be planned from.
    """
    set_pool_setting(store_config, role="capacity")
    scheduler = make_scheduler(store_config, hit_blocks=2)
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert scheduler._worker.queries == []
    assert scheduler._requests == {}

    metadata = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, [7, 8, 9])])
    )
    assert metadata.loads == []
    assert metadata.saves == []
    assert scheduler.request_finished(request, [7, 8, 9]) is False


def test_mooncake_declares_adp_support() -> None:
    assert MooncakeStoreConnectorScheduler.supports_attention_dp
    assert MooncakeStoreConnectorWorker.supports_attention_dp


@pytest.mark.parametrize("producer_rank,consumer_rank", [(0, 3), (3, 1), (1, 0)])
def test_adp_reuses_another_owners_pages(
    store_config: Path,
    fake_store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
    producer_rank: int,
    consumer_rank: int,
) -> None:
    """A full prefix needs one copy per layer group, independent of its owner."""
    monkeypatch.setattr(worker_module, "mpi_world_size", lambda: 4)
    monkeypatch.setattr(worker_module, "mpi_rank", lambda: producer_rank)
    layout = make_layout(num_groups=3, regions_per_group=2)
    hashes = BlockHashChain(TOKENS_PER_BLOCK).extend(list(range(8)))
    with make_worker(fake_store, layout=layout, enable_attention_dp=True) as producer:
        producer._put(
            [RequestTransfers(7, [PageTransfer(hashes[0], group, 1) for group in range(3)])]
        )
        # An incompletely published next block must not extend the offer.
        producer._put([RequestTransfers(7, [PageTransfer(hashes[1], 0, 2)])])
        monkeypatch.setattr(worker_module, "mpi_rank", lambda: consumer_rank)
        with make_worker(fake_store, layout=layout, enable_attention_dp=True) as consumer:
            assert consumer._namespaces == producer._namespaces
            assert consumer.count_prefix_hit(hashes) == 1
            assert len(fake_store.exist_calls[-1]) == 2 * 3
            transfers = RequestTransfers(
                19, [PageTransfer(hashes[0], group, 5) for group in range(3)]
            )
            consumer.bind_connector_meta(SimpleNamespace(loads=[transfers], saves=[]))
            consumer.start_load_kv(None)
            keys, addresses, sizes = fake_store.get_calls[-1]
            assert keys == [producer._namespaces[group].key(hashes[0]) for group in range(3)]
            addressing = PageAddressing(layout)
            assert addresses == [addressing.buffers(group, 5)[0] for group in range(3)]
            assert sizes == [addressing.buffers(group, 5)[1] for group in range(3)]
            # Reuse is shared, while request completion belongs to each worker.
            producer._outstanding_saves[7] = 1
            assert producer.get_finished([7], []) == ([], [])
            assert consumer.get_finished([7], []) == ([7], [])
            producer._outstanding_saves.clear()
            assert producer.get_finished([], []) == ([7], [])


def test_tp_lookup_still_requires_all_attention_shards(
    store_config: Path, fake_store: FakeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_module, "mpi_world_size", lambda: 4)
    monkeypatch.setattr(worker_module, "mpi_rank", lambda: 0)
    block_hash = BlockHashChain(TOKENS_PER_BLOCK).extend(list(range(4)))[0]
    with make_worker(fake_store, layout=make_layout(num_groups=2)) as worker:
        worker._put(
            [RequestTransfers(7, [PageTransfer(block_hash, group, 1) for group in range(2)])]
        )
        assert worker.count_prefix_hit([block_hash]) == 0
        for namespaces in worker._peer_namespaces.values():
            for namespace in namespaces:
                fake_store.objects.add(namespace.key(block_hash))
        assert worker.count_prefix_hit([block_hash]) == 1
        assert len(fake_store.exist_calls[-1]) == 4 * 2
        with make_worker(
            fake_store, layout=make_layout(num_groups=2), enable_attention_dp=True
        ) as adp:
            assert adp.count_prefix_hit([block_hash]) == 0


def test_scheduler_cancel_load_truncates_the_offer(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=3)
    tokens = list(range(6 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)

    # The runtime will not consume anything from block 1 onwards.
    scheduler.cancel_load(request, TOKENS_PER_BLOCK, 6 * TOKENS_PER_BLOCK)
    metadata = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, list(range(10, 16)))])
    )
    assert [page.page_index for page in metadata.loads[0].pages] == [10]


@pytest.mark.parametrize(
    "cancel_ranges,expected_pages",
    [
        ([(4, 8)], [12, 13, 14]),
        ([(4, 12)], [13, 14]),
        ([(4, 8), (16, 20)], [12, 13]),
        ([(16, 20), (4, 8)], [12, 13]),
        ([(4, 20)], []),
        ([(0, 24)], []),
        ([(0, 4), (20, 24)], [11, 12, 13, 14]),
        ([(8, 8)], [11, 12, 13, 14]),
        ([(4, 5)], [12, 13, 14]),
        ([(19, 24)], [11, 12, 13]),
    ],
)
def test_scheduler_cancel_load_preserves_uncanceled_range(
    store_config: Path,
    cancel_ranges: list[tuple[int, int]],
    expected_pages: list[int],
) -> None:
    """Cancel from either end of a nonzero-offset offer without losing its remainder."""
    scheduler = make_scheduler(store_config, hit_blocks=4)
    tokens = list(range(6 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    assert scheduler.get_num_new_matched_tokens(request, TOKENS_PER_BLOCK) == (16, False)
    for start, end in cancel_ranges:
        scheduler.cancel_load(request, start, end)
    metadata = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, list(range(10, 16)))])
    )
    assert [page.page_index for load in metadata.loads for page in load.pages] == expected_pages
    # A consumed offer must not be issued again on the next iteration.
    next_metadata = scheduler.build_connector_meta(
        SchedulerOutput(cached_requests=[request_data(1, [], [])])
    )
    assert next_metadata.loads == []


def test_scheduler_rejects_cancellation_that_splits_an_offer(store_config: Path) -> None:
    scheduler = make_scheduler(store_config, hit_blocks=4)
    tokens = list(range(6 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)
    with pytest.raises(ValueError, match="beginning or end"):
        scheduler.cancel_load(request, TOKENS_PER_BLOCK, 2 * TOKENS_PER_BLOCK)
