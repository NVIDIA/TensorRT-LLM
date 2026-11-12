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
import os
import threading
import time
from collections.abc import Iterator
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
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import gpudirect as gpudirect_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import staging as staging_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.addressing import (
    PageAddressing,
    allocation_spans,
    split_at_cuts,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.gpudirect import (
    RangeFacts,
    allocation_range,
    describe_range,
    format_diagnosis,
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

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 4


# ---- fixtures and fakes ----


class FakeStore:
    """Records calls and remembers which keys exist, nothing more."""

    def __init__(self):
        self.objects = set()
        self.registered = []
        self.unregistered = []
        #: Registrations, unregistrations and the close, in call order.
        self.events = []
        #: Index of the `register_buffer` call to fail, counting from 0.
        self.fail_register_at = None
        self.put_calls = []
        self.get_calls = []
        self.exist_calls = []
        self.closed = False
        self.fail_gets_for = set()
        #: Workers built against this store. `make_worker` shuts each one down;
        #: the fixture repeats it as a backstop for early failures.
        self.workers = []

    def register_buffer(self, address, size):
        if self.fail_register_at == len(self.registered):
            self.events.append(("register_failed", address))
            return -600
        self.registered.append((address, size))
        self.events.append(("register", address))
        return 0

    def unregister_buffer(self, address):
        self.unregistered.append(address)
        self.events.append(("unregister", address))
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
        self.events.append(("close",))


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
#: layout built from plain integers. A real V2 pool maps 2 to 32 MiB at a time.
CHUNK_BYTES = 256
#: Not a multiple of the chunk size: `cuMemAddressReserve` with the default
#: alignment may return any base, and the chunks tile from there.
POOL_BASE = 0x1000 + CHUNK_BYTES // 4
REGION_SIZE = 300
REGION_SLOTS = 4
POOL_BYTES = REGION_SIZE * REGION_SLOTS
#: The whole reservation, which `VirtMem` sizes to a multiple of the chunk.
RESERVATION_BYTES = -(-POOL_BYTES // CHUNK_BYTES) * CHUNK_BYTES


def make_pool_layout(*, base=POOL_BASE):
    """One region wide enough to straddle a chunk, packed slot to slot."""
    return KvCacheLayout(
        tokens_per_block=TOKENS_PER_BLOCK,
        groups=(
            KvCacheLayerGroupLayout(
                layer_group_id=0,
                layer_ids=(0,),
                window_size=None,
                regions=(
                    KvCacheRegion(
                        base=base,
                        size=REGION_SIZE,
                        stride=REGION_SIZE,
                        num_slots=REGION_SLOTS,
                        buffers=(KvCacheBufferRef(layer_id=0, role="key"),),
                    ),
                ),
            ),
        ),
    )


def chunked_lookup(*reservations, chunk_bytes=CHUNK_BYTES):
    """Answer like `cuMemGetAddressRange` for reservations mapped chunk by chunk.

    Args:
        reservations: `(base, size)` of each reservation, sized in chunks.
        chunk_bytes: Size of every `cuMemCreate` allocation mapped into them.
    """

    def lookup(address):
        for base, size in reservations:
            if base <= address < base + size:
                index = (address - base) // chunk_bytes
                return base + index * chunk_bytes, chunk_bytes
        return None

    return lookup


def chunk_pieces(start, end, *, chunk_base=POOL_BASE, chunk_bytes=CHUNK_BYTES):
    """`[start, end)` cut wherever a chunk counted from `chunk_base` begins."""
    pieces = []
    cursor = start
    while cursor < end:
        chunk_end = chunk_base + ((cursor - chunk_base) // chunk_bytes + 1) * chunk_bytes
        stop = min(chunk_end, end)
        pieces.append((cursor, stop))
        cursor = stop
    return pieces


@pytest.fixture(autouse=True)
def no_driver_allocations(monkeypatch):
    """None of these fake addresses is a CUDA allocation, so skip the driver.

    Keeps the cuts the same whether or not the host running the tests has a
    usable CUDA driver. Tests that need a pool backed by allocations install
    one with `mapped_pool`.
    """
    monkeypatch.setattr(worker_module, "allocation_range", lambda _address: None)


@pytest.fixture(autouse=True)
def no_mooncake_client_started(monkeypatch):
    """Start each test as a process whose first Mooncake client is still to come."""
    monkeypatch.setattr(gpudirect_module, "_peermem_at_first_client", gpudirect_module._NOT_STARTED)


@pytest.fixture
def mapped_pool(monkeypatch):
    """Back the fake pool with chunked allocations, as the driver would report them."""

    def install(*reservations):
        lookup = chunked_lookup(*(reservations or ((POOL_BASE, RESERVATION_BYTES),)))
        monkeypatch.setattr(worker_module, "allocation_range", lookup)
        return lookup

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


def make_llm_args(*, run_dir: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        model="/models/test-model",
        kv_cache_config=SimpleNamespace(tokens_per_block=TOKENS_PER_BLOCK),
        tensor_parallel_size=1,
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
) -> Iterator[MooncakeStoreConnectorWorker]:
    """Build a worker and shut it down before the test call phase ends.

    Registering a layout starts the background save thread, and
    pytest-threadleak snapshots threads around the call phase only, so
    fixture teardown would run too late to keep it quiet.
    """
    worker = MooncakeStoreConnectorWorker(make_llm_args(run_dir=run_dir))
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


def test_allocation_spans_cuts_at_every_allocation_boundary():
    lookup = chunked_lookup((POOL_BASE, RESERVATION_BYTES))
    spans = allocation_spans(POOL_BASE, POOL_BASE + POOL_BYTES, lookup)

    assert spans == chunk_pieces(POOL_BASE, POOL_BASE + POOL_BYTES)
    assert len(spans) == RESERVATION_BYTES // CHUNK_BYTES
    for start, end in spans:
        base, size = lookup(start)
        assert end <= base + size, "a piece runs into the next allocation"


def test_allocation_spans_starts_and_ends_mid_allocation():
    lookup = chunked_lookup((POOL_BASE, RESERVATION_BYTES))
    start, end = POOL_BASE + 10, POOL_BASE + CHUNK_BYTES + 20
    assert allocation_spans(start, end, lookup) == [
        (start, POOL_BASE + CHUNK_BYTES),
        (POOL_BASE + CHUNK_BYTES, end),
    ]


def test_allocation_spans_gives_up_on_an_address_it_cannot_resolve():
    """An unsplittable range must not be mistaken for one that needs no cut."""
    lookup = chunked_lookup((POOL_BASE, CHUNK_BYTES))
    assert allocation_spans(POOL_BASE, POOL_BASE + 2 * CHUNK_BYTES, lookup) is None


def test_allocation_spans_rejects_an_answer_that_misses_the_address():
    """An answer for some other address is rejected rather than trusted."""
    assert allocation_spans(POOL_BASE, POOL_BASE + 64, lambda _address: (0, 1)) is None


@pytest.mark.parametrize(
    "start,size,cuts,expected",
    [
        (100, 50, [], [(100, 50)]),
        (100, 50, [100, 150], [(100, 50)]),  # cuts at the ends cut nothing
        (100, 50, [120], [(100, 20), (120, 30)]),
        (100, 50, [50, 110, 130, 200], [(100, 10), (110, 20), (130, 20)]),
        (100, 0, [100], [(100, 0)]),
    ],
    ids=["no_cuts", "cuts_at_the_ends", "one_cut", "several_cuts", "empty"],
)
def test_split_at_cuts_preserves_the_bytes_and_their_order(start, size, cuts, expected):
    pieces = split_at_cuts(start, size, cuts)
    assert pieces == expected
    assert sum(piece_size for _, piece_size in pieces) == size


def test_page_addressing_splits_a_page_that_crosses_an_allocation():
    lookup = chunked_lookup((POOL_BASE, RESERVATION_BYTES))
    addressing = PageAddressing(make_pool_layout(), allocation_lookup=lookup)

    addresses, sizes = addressing.buffers(0, 0)
    assert list(zip(addresses, sizes)) == [
        (POOL_BASE, CHUNK_BYTES),
        (POOL_BASE + CHUNK_BYTES, REGION_SIZE - CHUNK_BYTES),
    ]
    # One region still contributes its whole payload, just in several buffers.
    assert sum(sizes) == addressing.bytes_per_page(0)


def test_page_addressing_registers_one_range_per_allocation():
    lookup = chunked_lookup((POOL_BASE, RESERVATION_BYTES))
    ranges = PageAddressing(make_pool_layout(), allocation_lookup=lookup).registration_ranges()

    # The same bytes as the unsplit span, handed over one allocation at a time.
    assert ranges == chunk_pieces(POOL_BASE, POOL_BASE + POOL_BYTES)


def test_page_addressing_separates_pools_whose_reservations_touch():
    """Merging adjacent pools into one span must not merge their registrations."""
    second_base = POOL_BASE + POOL_BYTES
    first = make_pool_layout().groups[0]
    second = make_pool_layout(base=second_base).groups[0]
    layout = KvCacheLayout(
        tokens_per_block=TOKENS_PER_BLOCK,
        groups=(
            first,
            KvCacheLayerGroupLayout(
                layer_group_id=1,
                layer_ids=(1,),
                window_size=None,
                regions=second.regions,
            ),
        ),
    )
    # Two reservations, each filled exactly by its pool, so the second starts
    # where the first ends and the spans merge.
    chunk_bytes = POOL_BYTES // 3
    lookup = chunked_lookup(
        (POOL_BASE, POOL_BYTES), (second_base, POOL_BYTES), chunk_bytes=chunk_bytes
    )

    ranges = PageAddressing(layout, allocation_lookup=lookup).registration_ranges()
    assert ranges == [
        (POOL_BASE + index * chunk_bytes, POOL_BASE + (index + 1) * chunk_bytes)
        for index in range(6)
    ]
    assert second_base in [start for start, _ in ranges]


def test_page_addressing_leaves_everything_whole_without_a_lookup():
    """Host staging never registers the pools, so it never asks the driver."""
    addressing = PageAddressing(make_pool_layout())
    assert addressing.buffers(0, 0) == ([POOL_BASE], [REGION_SIZE])
    assert addressing.registration_ranges() == [(POOL_BASE, POOL_BASE + POOL_BYTES)]
    assert not addressing.unresolved_ranges


def test_page_addressing_reports_the_ranges_it_could_not_resolve():
    addressing = PageAddressing(make_pool_layout(), allocation_lookup=lambda _address: None)
    span = (POOL_BASE, POOL_BASE + POOL_BYTES)
    assert addressing.registration_ranges() == [span]
    assert addressing.unresolved_ranges == (span,)


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


# ---- GPUDirect diagnosis ----

_ALLOCATION = 0x7F0000000000
_ALLOCATION_BYTES = 32 << 20


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


def _facts(*, offset=0, length=64, gdr_capable=True):
    return RangeFacts(
        address=_ALLOCATION + offset,
        length=length,
        allocation_start=_ALLOCATION,
        allocation_size=_ALLOCATION_BYTES,
        gdr_capable=gdr_capable,
    )


def test_range_facts_detects_a_range_that_runs_past_its_allocation():
    assert _facts(length=_ALLOCATION_BYTES).fits_one_allocation is True
    assert (
        _facts(offset=_ALLOCATION_BYTES // 2, length=_ALLOCATION_BYTES).fits_one_allocation is False
    )


def test_range_facts_will_not_judge_without_the_allocation():
    facts = RangeFacts(address=_ALLOCATION, length=64)
    assert facts.fits_one_allocation is None


def test_allocation_range_answers_nothing_for_an_address_the_driver_disowns():
    """Callers get the documented fallback rather than an exception."""
    assert allocation_range(_ALLOCATION) is None


def test_describe_range_reports_a_failure_instead_of_raising():
    facts = describe_range(_ALLOCATION, 1 << 20)
    assert isinstance(facts, RangeFacts)
    assert facts.address == _ALLOCATION
    assert facts.describe()


def test_format_diagnosis_blames_the_missing_peermem_module(monkeypatch, stub_driver_facts):
    monkeypatch.setattr(gpudirect_module, "peermem_loaded", lambda: False)
    monkeypatch.delenv("WITH_NVIDIA_PEERMEM", raising=False)
    ranges = stub_driver_facts(_facts())

    report = format_diagnosis(ranges, rank=0)
    assert "ibv_reg_mr" in report
    assert "nvidia_peermem is not loaded" in report
    assert "WITH_NVIDIA_PEERMEM=0" in report


@pytest.mark.parametrize("peermem_env", [None, "0"], ids=["peermem_path", "dmabuf_path"])
def test_format_diagnosis_blames_a_range_that_crosses_allocations(
    monkeypatch, stub_driver_facts, peermem_env
):
    """Both paths refuse it, and under peermem it looks like a missing module."""
    monkeypatch.setattr(gpudirect_module, "peermem_loaded", lambda: True)
    if peermem_env is None:
        monkeypatch.delenv("WITH_NVIDIA_PEERMEM", raising=False)
    else:
        monkeypatch.setenv("WITH_NVIDIA_PEERMEM", peermem_env)
    ranges = stub_driver_facts(_facts(offset=_ALLOCATION_BYTES // 2, length=_ALLOCATION_BYTES))

    report = format_diagnosis(ranges, rank=3)
    assert "rank 3" in report
    assert "1 range(s) run past the cuMemCreate allocation" in report
    assert "EFAULT" in report and "EINVAL" in report


def test_format_diagnosis_sends_memory_without_gdr_to_dmabuf(monkeypatch, stub_driver_facts):
    """peermem cannot register it at any size, while dma-buf can."""
    monkeypatch.setattr(gpudirect_module, "peermem_loaded", lambda: True)
    monkeypatch.delenv("WITH_NVIDIA_PEERMEM", raising=False)
    ranges = stub_driver_facts(_facts(gdr_capable=False))

    report = format_diagnosis(ranges, rank=0)
    assert "without gpuDirectRDMACapable" in report
    assert "WITH_NVIDIA_PEERMEM=0 selects dma-buf" in report


def test_format_diagnosis_does_not_blame_gdr_on_the_dmabuf_path(monkeypatch, stub_driver_facts):
    monkeypatch.setattr(gpudirect_module, "peermem_loaded", lambda: False)
    monkeypatch.setenv("WITH_NVIDIA_PEERMEM", "0")
    ranges = stub_driver_facts(_facts(gdr_capable=False))

    report = format_diagnosis(ranges, rank=0)
    assert "gpuDirectRDMACapable" not in report
    assert "Nothing here explains a registration failure" in report


def test_format_diagnosis_admits_when_it_explains_nothing(monkeypatch, stub_driver_facts):
    monkeypatch.setattr(gpudirect_module, "peermem_loaded", lambda: True)
    monkeypatch.setenv("WITH_NVIDIA_PEERMEM", "1")
    ranges = stub_driver_facts(_facts())

    report = format_diagnosis(ranges, rank=0)
    assert "Nothing here explains a registration failure" in report


@pytest.mark.parametrize(
    "module_line, expected",
    [
        ("nvidia_peermem 16384 0 - Live 0x0000000000000000", True),
        ("nv_peer_mem 16384 0 - Live 0x0000000000000000", True),
        ("nvidia 1234 1 - Live 0x0000000000000000", False),
    ],
    ids=["nvidia_peermem", "nv_peer_mem", "neither"],
)
def test_peermem_loaded_recognizes_both_modules(tmp_path, monkeypatch, module_line, expected):
    modules = tmp_path / "modules"
    modules.write_text(module_line + "\n")
    real_open = open
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *args, **kwargs: real_open(
            modules if path == "/proc/modules" else path, *args, **kwargs
        ),
    )
    assert gpudirect_module.peermem_loaded() is expected


@pytest.fixture
def registration_host(monkeypatch):
    """A host whose peer-memory module and dma-buf support the test chooses."""
    monkeypatch.delenv("WITH_NVIDIA_PEERMEM", raising=False)

    def install(*, peermem, dmabuf):
        monkeypatch.setattr(gpudirect_module, "peermem_loaded", lambda: peermem)
        monkeypatch.setattr(gpudirect_module, "dmabuf_supported_everywhere", lambda: dmabuf)

    return install


def _select(protocol="rdma", *, registers_device_memory=True):
    return gpudirect_module.select_registration_path(
        protocol, registers_device_memory=registers_device_memory
    )


@pytest.mark.parametrize("registers_device_memory", [True, False], ids=["kv_pools", "capacity"])
def test_select_registration_path_picks_dmabuf_without_peermem(
    registration_host, registers_device_memory
):
    """Any role selects, since the first client fixes the path for the process."""
    registration_host(peermem=False, dmabuf=True)
    assert _select(registers_device_memory=registers_device_memory) is True
    assert os.environ["WITH_NVIDIA_PEERMEM"] == "0"
    assert gpudirect_module.dmabuf_selected()


def test_select_registration_path_leaves_tcp_alone(registration_host):
    registration_host(peermem=False, dmabuf=True)
    assert _select("tcp") is False
    assert "WITH_NVIDIA_PEERMEM" not in os.environ


@pytest.mark.parametrize(
    "peermem, dmabuf",
    [(True, True), (None, True), (False, False), (False, None)],
    ids=["peermem_loaded", "modules_unreadable", "no_dmabuf", "dmabuf_unknown"],
)
def test_select_registration_path_leaves_an_uncertain_host_alone(
    registration_host, peermem, dmabuf
):
    registration_host(peermem=peermem, dmabuf=dmabuf)
    assert _select() is False
    assert "WITH_NVIDIA_PEERMEM" not in os.environ


@pytest.mark.parametrize("value", ["1", "0"])
def test_select_registration_path_never_overrides_the_deployment(
    registration_host, monkeypatch, value
):
    registration_host(peermem=False, dmabuf=True)
    monkeypatch.setenv("WITH_NVIDIA_PEERMEM", value)
    assert _select() is False
    assert os.environ["WITH_NVIDIA_PEERMEM"] == value


def test_select_registration_path_never_writes_after_the_first_client(registration_host):
    """Mooncake would ignore the write, so the warning would claim a path it never takes."""
    registration_host(peermem=False, dmabuf=True)
    assert _select("tcp") is False
    assert _select(registers_device_memory=False) is False
    assert "WITH_NVIDIA_PEERMEM" not in os.environ
    assert not gpudirect_module.dmabuf_selected()


def test_select_registration_path_refuses_a_late_client_that_needs_dmabuf(registration_host):
    registration_host(peermem=False, dmabuf=True)
    _select("tcp")
    with pytest.raises(RuntimeError, match="export WITH_NVIDIA_PEERMEM=0 before the first"):
        _select()
    assert "WITH_NVIDIA_PEERMEM" not in os.environ


def test_select_registration_path_refuses_a_value_changed_after_the_first_client(
    registration_host, monkeypatch
):
    registration_host(peermem=True, dmabuf=True)
    _select()
    monkeypatch.setenv("WITH_NVIDIA_PEERMEM", "0")
    with pytest.raises(RuntimeError, match="reads it only then"):
        _select()
    # Clients that register no GPU memory are unaffected by the path.
    assert _select(registers_device_memory=False) is False
    assert _select("tcp") is False


def test_select_registration_path_accepts_a_late_client_on_the_same_path(registration_host):
    registration_host(peermem=False, dmabuf=True)
    assert _select(registers_device_memory=False) is True
    assert _select() is False
    assert os.environ["WITH_NVIDIA_PEERMEM"] == "0"


def test_format_diagnosis_reports_the_value_mooncake_read(
    registration_host, monkeypatch, stub_driver_facts
):
    registration_host(peermem=False, dmabuf=True)
    _select("tcp")
    monkeypatch.setenv("WITH_NVIDIA_PEERMEM", "0")

    report = format_diagnosis(stub_driver_facts(_facts()), rank=0)
    assert "WITH_NVIDIA_PEERMEM=None as Mooncake read it" in report
    assert "is '0' in the environment now, which Mooncake ignores" in report
    assert "nvidia_peermem is not loaded, so every range fails" in report


# ---- worker registration ----


def test_worker_registers_one_buffer_per_allocation(store_config, fake_store, mapped_pool):
    lookup = mapped_pool()
    with make_worker(fake_store, layout=make_pool_layout()) as worker:
        assert worker.is_registered
        expected = chunk_pieces(POOL_BASE, POOL_BASE + POOL_BYTES)
        assert fake_store.registered == [(start, end - start) for start, end in expected]
        for address, size in fake_store.registered:
            base, length = lookup(address)
            assert address + size <= base + length, "a registration crosses two allocations"


def test_worker_page_buffers_stay_inside_one_allocation(store_config, fake_store, mapped_pool):
    lookup = mapped_pool()
    with make_worker(fake_store, layout=make_pool_layout()) as worker:
        worker._put([RequestTransfers(1, [PageTransfer(b"\x01" * 16, 0, 0)])])
    _, addresses, sizes = fake_store.put_calls[0]
    assert sum(sizes[0]) == REGION_SIZE
    for address, size in zip(addresses[0], sizes[0]):
        base, length = lookup(address)
        assert address + size <= base + length


def test_worker_takes_back_earlier_pieces_when_one_fails(store_config, fake_store, mapped_pool):
    mapped_pool()
    fake_store.fail_register_at = 2

    worker = MooncakeStoreConnectorWorker(make_llm_args())
    fake_store.workers.append(worker)
    with pytest.raises(RuntimeError, match="register_buffer failed") as excinfo:
        worker.register_kv_cache_layout(make_pool_layout())

    registered = [address for address, _ in fake_store.registered]
    assert len(registered) == 2
    # Latest first, and nothing left for shutdown to hand back.
    assert fake_store.unregistered == list(reversed(registered))
    worker.shutdown()
    assert fake_store.unregistered == list(reversed(registered))

    message = str(excinfo.value)
    assert "WITH_NVIDIA_PEERMEM" in message
    assert "gpuDirectRDMACapable" in message
    assert "GPU registration diagnosis" in message


def test_worker_unregisters_every_piece_at_shutdown_before_closing(
    store_config, fake_store, mapped_pool
):
    mapped_pool()
    with make_worker(fake_store, layout=make_pool_layout()):
        assert not fake_store.unregistered
    registered = [address for address, _ in fake_store.registered]
    assert sorted(fake_store.unregistered) == sorted(registered)
    assert fake_store.events[-1] == ("close",)
    teardown = fake_store.events[len(fake_store.registered) :]
    assert all(event[0] == "unregister" for event in teardown[:-1])


def test_worker_says_when_it_cannot_resolve_the_allocations(store_config, fake_store, monkeypatch):
    """Silence would leave a crossing range looking like a Mooncake failure."""
    messages = []
    monkeypatch.setattr(worker_module.logger, "warning", messages.append)

    with make_worker(fake_store, layout=make_pool_layout()):
        pass
    assert any("could not ask the driver" in message for message in messages)
    assert fake_store.registered == [(POOL_BASE, POOL_BYTES)]


@pytest.mark.parametrize("debug", [False, True], ids=["default", "debug"])
def test_worker_reports_the_driver_facts_only_at_debug_level(
    store_config, fake_store, mapped_pool, monkeypatch, debug
):
    """The report queries the driver per range, so it is not even built otherwise."""
    mapped_pool()
    messages = []
    built = []
    real_format_diagnosis = worker_module.format_diagnosis

    def format_diagnosis(*args, **kwargs):
        built.append(args)
        return real_format_diagnosis(*args, **kwargs)

    # The logger does not propagate to the root logger, so caplog cannot see it.
    monkeypatch.setattr(worker_module.logger, "is_debug_enabled", lambda: debug)
    monkeypatch.setattr(worker_module.logger, "debug", messages.append)
    monkeypatch.setattr(worker_module, "format_diagnosis", format_diagnosis)

    with make_worker(fake_store, layout=make_pool_layout()):
        pass
    assert bool(built) is debug
    assert any("GPU registration diagnosis" in message for message in messages) is debug


def test_staging_never_asks_the_driver_about_the_pools(
    store_config, fake_store, staged_copies, monkeypatch
):
    """Only the pinned slots are registered, so the pools are never cut."""
    monkeypatch.setattr(
        worker_module, "allocation_range", lambda _address: pytest.fail("driver consulted")
    )
    with make_staged_worker(fake_store, store_config, layout=make_pool_layout()) as worker:
        assert worker._addressing.buffers(0, 0) == ([POOL_BASE], [REGION_SIZE])
    assert POOL_BASE not in [address for address, _ in fake_store.registered]


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
        worker.bind_allocation_fence(AllocationFence(fake_store))
        worker.bind_connector_meta(SimpleNamespace(loads=[transfers], saves=[]))
        with pytest.raises(RuntimeError, match="already"):
            worker.start_load_kv(None)


def test_worker_zero_copy_load_refuses_to_run_unordered(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        bind_one_load(worker, fake_store, b"\x06" * 16)
        with pytest.raises(RuntimeError, match="no allocation fence"):
            worker.start_load_kv(None)
    assert fake_store.get_calls == []


class AllocationFence:
    """Stands in for the KV cache manager's fence, noting each wait."""

    def __init__(self, store):
        self._store = store
        self.reads_before_wait = []
        self.waited_for = []

    def __call__(self, request_ids):
        self.reads_before_wait.append(len(self._store.get_calls))
        self.waited_for.append(set(request_ids))


def bind_one_load(worker, fake_store, block_hash):
    fake_store.objects.add(worker._namespaces[0].key(block_hash))
    worker.bind_connector_meta(
        SimpleNamespace(loads=[RequestTransfers(7, [PageTransfer(block_hash, 0, 1)])], saves=[])
    )


def test_worker_zero_copy_load_waits_for_the_allocation_first(store_config, fake_store):
    """The NIC writes into pages the previous pass may still be writing."""
    with make_worker(fake_store, layout=make_layout()) as worker:
        fence = AllocationFence(fake_store)
        worker.bind_allocation_fence(fence)
        bind_one_load(worker, fake_store, b"\x04" * 16)
        worker.start_load_kv(None)
    assert fence.reads_before_wait == [0]
    # Only the loading request's pages, so other requests' pending work is not waited for.
    assert fence.waited_for == [{7}]
    assert len(fake_store.get_calls) == 1


def test_worker_reports_a_request_finished_once_its_saves_drain(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        # One submission outstanding: the request is closed but must not be released.
        worker._outstanding_saves[42] = 1
        assert worker.get_finished([42], []) == ([], [])

        worker._outstanding_saves.pop(42)
        assert worker.get_finished([], []) == ([42], [])
        # Reported once only.
        assert worker.get_finished([], []) == ([], [])


def test_a_save_that_raises_is_dropped_and_its_request_still_retires(
    store_config, fake_store, monkeypatch
):
    """A write that did not land costs a cache miss, not the server.

    Failing the batch instead would also strand the request whose pages it was
    pinned on, which is what a thread that stops draining its queue leaves
    behind.
    """

    def refuse(_keys):
        raise RuntimeError("store RPC failed")

    with make_worker(fake_store, layout=make_layout()) as worker:
        monkeypatch.setattr(fake_store, "batch_is_exist", refuse)

        # What `wait_for_save` hands the thread, minus the CUDA event this host
        # has no device for.
        with worker._save_lock:
            worker._outstanding_saves[42] += 1
        worker._save_queue.put(
            (
                SimpleNamespace(synchronize=lambda: None),
                [RequestTransfers(42, [PageTransfer(b"\x01" * 16, 0, 1)])],
            )
        )

        deadline = time.monotonic() + 5.0
        while worker._outstanding_saves.get(42) and time.monotonic() < deadline:
            time.sleep(0.01)

        assert worker.get_finished([42], []) == ([42], [])


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


class StubStagingPool:
    """Stands in for `HostStagingPool` where no device backs its slots.

    The real pool page-locks its buffer whenever a device is reported, which
    is the one claim `fake_cuda` cannot make true.
    """

    def __init__(self, *, slot_bytes, num_slots, store, label):
        self.slot_bytes = slot_bytes
        self.num_slots = num_slots
        self.label = label

    def close(self) -> None:
        pass


@pytest.fixture
def fake_cuda(monkeypatch):
    """Present a CUDA device on a host that has none, recording set_device calls.

    The device bookkeeping around the save thread is control flow, so testing
    it should not need a GPU. The two things that path does need one for, the
    staging slots registration pins and the stream the thread opens over them,
    are stood in for, so claiming a device cannot reach a real allocation.
    """
    recorded = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: recorded.append(index))
    monkeypatch.setattr(torch.cuda, "Stream", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(worker_module, "HostStagingPool", StubStagingPool)
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
        (keys, addresses, sizes) = fake_store.get_calls[0]
        assert keys == [worker._namespaces[0].key(block_hash)]
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


def test_staging_load_leaves_the_ordering_to_the_stream(store_config, fake_store, staged_copies):
    """The scatter is queued on the stream, behind the previous pass."""
    with make_staged_worker(fake_store, store_config, layout=make_layout()) as worker:
        fence = AllocationFence(fake_store)
        worker.bind_allocation_fence(fence)
        bind_one_load(worker, fake_store, b"\x05" * 16)
        worker.start_load_kv(None)
    assert fence.reads_before_wait == []
    assert len(staged_copies) == 1


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
    store_config, fake_store, staged_copies, monkeypatch
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
        # One entry per layer group, indexed by layer group id.
        new_block_ids_by_layer_group=[list(page_indices)],
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
    data.new_block_ids_by_layer_group.append([7, BAD_PAGE_INDEX])
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


def test_scheduler_bypasses_requests_carrying_a_lora_adapter(store_config):
    """An adapter id gets neither a lookup nor a save.

    The id is assigned by whoever submitted the request, so two servers sharing
    a pool can name different weights by the same id. A page keyed by it would
    be served against the wrong weights, which is a wrong answer rather than a
    slow one.
    """
    scheduler = make_scheduler(store_config, hit_blocks=2)
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    # Adapter 0 is a real adapter, so the bypass turns on presence rather than
    # on the id being truthy.
    request = make_request(1, tokens, lora_task_id=0)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert scheduler._worker.queries == []

    metadata = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, [7, 8, 9])])
    )
    assert metadata.loads == []
    assert metadata.saves == []
    assert scheduler.request_finished(request, [7, 8, 9]) is False

    # A request without one is served as before, so the bypass is scoped to the
    # requests it cannot key rather than to the deployment.
    scheduler.get_num_new_matched_tokens(make_request(2, tokens), 0)
    assert len(scheduler._worker.queries) == 1


def test_scheduler_does_no_per_request_work_for_a_capacity_only_role(store_config):
    """It transfers nothing, so a request costs it nothing.

    No hash chain is built, nothing is looked up, and the request is left
    without the state a save would be planned from.
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
