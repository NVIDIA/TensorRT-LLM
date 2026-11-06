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
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    PrefixLoad,
    RequestData,
    SchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import (
    KvCacheBufferRef,
    KvCacheLayerGroupLayout,
    KvCacheLayout,
    KvCacheRegion,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import scheduler as scheduler_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import staging as staging_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.addressing import PageAddressing
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import BlockHashChain
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    MooncakeStoreMetadata,
    PageTransfer,
    PrefixLoadTransfers,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler import (
    MooncakeStoreConnectorScheduler,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.validation import (
    NODE_BUDGET_RESERVE_BYTES,
    validate_layout,
    validate_llm_args,
    validate_node_budget,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.worker import (
    MooncakeStoreConnectorWorker,
    PrefixHit,
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
    monkeypatch.setattr(worker_module, "validate_node_budget", lambda _config, **_kwargs: None)
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


# ---- worker ----


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
                time.monotonic(),
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
    monkeypatch.setattr(
        torch.cuda, "Stream", lambda *_args, **_kwargs: SimpleNamespace(cuda_stream=0)
    )
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


def test_a_load_thread_that_cannot_start_fails_registration(
    store_config, fake_store, fake_cuda, monkeypatch
):
    """A parked request waits on the load thread, so one that died must stop bringup."""
    calls = []

    def refuse_after_the_save_thread(index):
        calls.append(index)
        # The save thread binds the device first and registration waits for it.
        if len(calls) > 1:
            raise RuntimeError("cudaSetDevice failed")

    monkeypatch.setattr(torch.cuda, "set_device", refuse_after_the_save_thread)

    worker = MooncakeStoreConnectorWorker(make_llm_args())
    fake_store.workers.append(worker)
    try:
        with pytest.raises(RuntimeError, match="load thread"):
            worker.register_kv_cache_layout(make_layout())
    finally:
        # The save thread did start, so it has to be stopped.
        worker.shutdown()


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
                time.monotonic(),
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


# ---- sliding windows ----

#: The window of the sliding group in these tests: two blocks.
WINDOW = 2 * TOKENS_PER_BLOCK


def make_window_layout(*, num_slots=16):
    """Group 0 is full attention, group 1 a sliding window of `WINDOW` tokens."""
    full = make_layout(num_slots=num_slots).groups[0]
    windowed = KvCacheLayerGroupLayout(
        layer_group_id=1,
        layer_ids=(1,),
        window_size=WINDOW,
        regions=(
            KvCacheRegion(
                base=0x100000,
                size=128,
                stride=128,
                num_slots=num_slots,
                buffers=(KvCacheBufferRef(layer_id=1, role="key"),),
            ),
        ),
    )
    return KvCacheLayout(tokens_per_block=TOKENS_PER_BLOCK, groups=(full, windowed))


def store_window(fake_store, worker, hashes, end_block):
    """Store the window pages a prefix of `end_block` blocks ends with."""
    _, stale_end = worker.stale_blocks(1, end_block * TOKENS_PER_BLOCK)
    for block in range(stale_end, end_block):
        fake_store.objects.add(worker._namespaces[1].key(hashes[block]))
    return stale_end


def test_validate_layout_accepts_a_sliding_window():
    validate_layout(make_window_layout())


def reference_chain(tokens, tokens_per_block):
    """The chain under the per-token `int.to_bytes` encoding, which the packed fast path must match byte for byte."""
    import hashlib

    seed = BlockHashChain(tokens_per_block)._seed
    hashes = []
    for ordinal in range(len(tokens) // tokens_per_block):
        block = tokens[ordinal * tokens_per_block : (ordinal + 1) * tokens_per_block]
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(hashes[-1] if hashes else seed)
        hasher.update(b"".join(int(t).to_bytes(8, "little", signed=True) for t in block))
        hashes.append(hasher.digest())
    return hashes


def test_hash_chain_matches_the_per_token_encoding_across_extends():
    import random

    rng = random.Random(7)
    tokens = [rng.randrange(-(2**40), 2**40) for _ in range(40 * TOKENS_PER_BLOCK + 5)]
    chain = BlockHashChain(TOKENS_PER_BLOCK)
    for end in (3, TOKENS_PER_BLOCK, 7 * TOKENS_PER_BLOCK + 1, len(tokens)):
        chain.extend(tokens[:end])
        assert list(chain.hashes) == reference_chain(tokens[:end], TOKENS_PER_BLOCK)


def test_interleaved_keys_match_namespace_keys(store_config, fake_store):
    with make_worker(fake_store, layout=make_window_layout()) as worker:
        namespaces = (
            list(worker._namespaces.values()) * 3
        )  # several namespaces, as a multi-rank peer list
        hashes = [bytes([index]) * 16 for index in range(9)]
        expected = [namespace.key(block_hash) for block_hash in hashes for namespace in namespaces]
        assert worker_module._interleaved_keys(namespaces, hashes) == expected
        assert worker_module._interleaved_keys(namespaces[:1], hashes) == [
            namespaces[0].key(h) for h in hashes
        ]
        assert worker_module._interleaved_keys(namespaces, []) == []


def test_worker_window_hit_needs_the_window_at_the_prefix_end(store_config, fake_store):
    with make_worker(fake_store, layout=make_window_layout()) as worker:
        hashes = [bytes([index]) * 16 for index in range(6)]
        for block_hash in hashes:
            fake_store.objects.add(worker._namespaces[0].key(block_hash))

        # Every full-attention page but no window anywhere: no end is servable.
        assert worker.find_prefix_hit(hashes, 0) == PrefixHit(0)
        fake_store.exist_calls.clear()

        window_from = store_window(fake_store, worker, hashes, 6)
        assert 0 < window_from < 6
        assert worker.find_prefix_hit(hashes, 0) == PrefixHit(6, {1: window_from})
        # The window group's pages before the window are never asked for.
        probed = {key for call in fake_store.exist_calls for key in call}
        assert worker._namespaces[1].key(hashes[0]) not in probed


def test_worker_window_hit_walks_back_over_the_rewind_only(store_config, fake_store):
    with make_worker(fake_store, layout=make_window_layout()) as worker:
        hashes = [bytes([index]) * 16 for index in range(6)]
        for block_hash in hashes:
            fake_store.objects.add(worker._namespaces[0].key(block_hash))
        # An earlier prompt ended at block 5, one block short of the full hit.
        store_window(fake_store, worker, hashes, 5)

        worker._window_fallback = False
        assert worker.find_prefix_hit(hashes, 0, rewind_tokens=0).blocks == 0
        assert worker.find_prefix_hit(hashes, 0, rewind_tokens=TOKENS_PER_BLOCK).blocks == 5


def test_worker_window_fallback_resumes_at_an_earlier_prompt_end(store_config, fake_store):
    """Full-attention pages that outrun every stored window cost one prompt's delta, not the prefix."""
    with make_worker(fake_store, layout=make_window_layout(num_slots=32)) as worker:
        hashes = [bytes([index]) * 16 for index in range(12)]
        for block_hash in hashes:
            fake_store.objects.add(worker._namespaces[0].key(block_hash))
        # Earlier prompts ended at blocks 4 and 7; the latest one, at 12, has no window yet.
        store_window(fake_store, worker, hashes, 4)
        window_from = store_window(fake_store, worker, hashes, 7)

        assert worker.find_prefix_hit(hashes, 0, rewind_tokens=TOKENS_PER_BLOCK) == PrefixHit(
            7, {1: window_from}
        )
        assert worker._lookup_stats.counts["recovered"] == 1

        worker._window_fallback = False
        assert worker.find_prefix_hit(hashes, 0, rewind_tokens=TOKENS_PER_BLOCK).blocks == 0
        assert worker._lookup_stats.counts["window_miss"] == 1


def test_worker_window_fallback_needs_a_complete_window(store_config, fake_store):
    with make_worker(fake_store, layout=make_window_layout(num_slots=32)) as worker:
        hashes = [bytes([index]) * 16 for index in range(12)]
        for block_hash in hashes:
            fake_store.objects.add(worker._namespaces[0].key(block_hash))
        window_from = store_window(fake_store, worker, hashes, 4)
        # The window at block 8 lost its last page, so only the end at 4 is complete.
        store_window(fake_store, worker, hashes, 8)
        fake_store.objects.discard(worker._namespaces[1].key(hashes[7]))

        assert worker.find_prefix_hit(hashes, 0) == PrefixHit(4, {1: window_from})


def test_worker_window_fallback_respects_what_is_on_device(store_config, fake_store):
    with make_worker(fake_store, layout=make_window_layout(num_slots=32)) as worker:
        hashes = [bytes([index]) * 16 for index in range(12)]
        for block_hash in hashes[3:]:
            fake_store.objects.add(worker._namespaces[0].key(block_hash))
        store_window(fake_store, worker, hashes, 8)
        # Resuming at 8 from a device prefix of 3 blocks: the store serves 5.
        assert worker.find_prefix_hit(hashes[3:], 3).blocks == 5
        # No stored end above the device prefix: nothing to serve.
        assert worker.find_prefix_hit(hashes[9:], 9).blocks == 0


def test_worker_window_hit_skips_what_is_on_device(store_config, fake_store):
    with make_worker(fake_store, layout=make_window_layout()) as worker:
        hashes = [bytes([index]) * 16 for index in range(6)]
        for block_hash in hashes[4:]:
            fake_store.objects.add(worker._namespaces[0].key(block_hash))
        # Blocks 0..3 are on device, window included, so the store only needs
        # the part of the end's window from block 4 on.
        _, stale_end = worker.stale_blocks(1, 6 * TOKENS_PER_BLOCK)
        for block in range(max(4, stale_end), 6):
            fake_store.objects.add(worker._namespaces[1].key(hashes[block]))
        assert worker.find_prefix_hit(hashes[4:], 4).blocks == 2


class FakeWindowWorker:
    """A lookup service over group 0 (full) and group 1 (a `WINDOW` window).

    The window reads the blocks overlapping its last `WINDOW` tokens; the real
    life cycle may keep a little more, which these tests do not depend on.
    """

    windowed_group_ids = (1,)

    def __init__(self, hit: PrefixHit):
        self.hit = hit
        self.queries = []

    def find_prefix_hit(self, block_hashes, first_block, rewind_tokens=0):
        self.queries.append((list(block_hashes), first_block, rewind_tokens))
        return PrefixHit(min(self.hit.blocks, len(block_hashes)), self.hit.window_present_from)

    def sink_blocks(self, layer_group_id):
        return 0

    def stale_blocks(self, layer_group_id, history_length):
        return 0, max(0, (history_length - WINDOW) // TOKENS_PER_BLOCK)


def make_window_scheduler(store_config, hit=PrefixHit(0), rewind_tokens=0):
    llm_args = make_llm_args()
    llm_args.kv_cache_config.block_reuse_config = SimpleNamespace(
        swa_endpoint_rewind_tokens=rewind_tokens
    )
    scheduler = MooncakeStoreConnectorScheduler(llm_args)
    scheduler._worker = FakeWindowWorker(hit)
    return scheduler


def make_context_request(request_id, tokens, *, position=0, last_chunk=True):
    request = make_request(request_id, tokens)
    request.context_current_position = position
    request.is_context_init_state = True
    request.is_last_context_chunk = last_chunk
    return request


def window_data(request_id, tokens, full_pages, window_pages, computed_position=0):
    data = request_data(request_id, tokens, full_pages)
    data.computed_position = computed_position
    data.new_block_ids_by_layer_group = [list(full_pages), list(window_pages)]
    return data


def pages_of(transfers, layer_group_id):
    return [page.page_index for page in transfers.pages if page.layer_group_id == layer_group_id]


def test_scheduler_saves_the_window_once_at_the_prompts_end(store_config):
    scheduler = make_window_scheduler(store_config)
    tokens = list(range(6 * TOKENS_PER_BLOCK))
    request = make_context_request(1, tokens, last_chunk=False)
    scheduler.get_num_new_matched_tokens(request, 0)

    # The first report carries the whole prompt and the first chunk's pages.
    first = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[window_data(1, tokens, [10, 11, 12], [20, 21, 22])])
    )
    # A chunk short of the prompt's end saves full attention only.
    assert pages_of(first.saves[0], 0) == [10, 11, 12]
    assert pages_of(first.saves[0], 1) == []

    request.context_current_position = 12
    request.is_last_context_chunk = True
    last = scheduler.build_connector_meta(
        SchedulerOutput(cached_requests=[window_data(1, [], [13, 14, 15], [23, 24, 25], 12)])
    )
    # The window the prompt ends with: the blocks overlapping its last WINDOW
    # tokens, blocks 4 and 5.
    assert pages_of(last.saves[0], 0) == [13, 14, 15]
    assert pages_of(last.saves[0], 1) == [24, 25]
    assert scheduler.request_finished(request, []) is True


def test_scheduler_saves_the_window_ahead_of_the_full_attention_blocks(store_config, monkeypatch):
    for window_first, first_group in (("1", 1), ("0", 0)):
        monkeypatch.setenv(scheduler_module.WINDOW_FIRST_ENV, window_first)
        scheduler = make_window_scheduler(store_config)
        tokens = list(range(6 * TOKENS_PER_BLOCK))
        request = make_context_request(1, tokens)
        scheduler.get_num_new_matched_tokens(request, 0)
        metadata = scheduler.build_connector_meta(
            SchedulerOutput(
                new_requests=[
                    window_data(1, tokens, [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25])
                ]
            )
        )
        (saves,) = metadata.saves
        assert saves.pages[0].layer_group_id == first_group
        assert pages_of(saves, 0) == [10, 11, 12, 13, 14, 15]
        assert pages_of(saves, 1) == [24, 25]


def test_scheduler_window_save_stops_at_what_the_request_still_holds(store_config):
    """The last chunk's commit releases the rewind margin before the save thread runs."""
    scheduler = make_window_scheduler(store_config, rewind_tokens=TOKENS_PER_BLOCK)
    tokens = list(range(6 * TOKENS_PER_BLOCK))
    request = make_context_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)

    metadata = scheduler.build_connector_meta(
        SchedulerOutput(
            new_requests=[
                window_data(1, tokens, [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25])
            ]
        )
    )
    # Only the window the prompt ends with, despite one block of rewind.
    assert pages_of(metadata.saves[0], 1) == [24, 25]


def test_scheduler_window_save_skips_blocks_the_window_has_passed(store_config):
    scheduler = make_window_scheduler(store_config, rewind_tokens=4 * TOKENS_PER_BLOCK)
    tokens = list(range(6 * TOKENS_PER_BLOCK))
    request = make_context_request(1, tokens, position=4 * TOKENS_PER_BLOCK)
    scheduler.get_num_new_matched_tokens(request, 4 * TOKENS_PER_BLOCK)

    # The last chunk starts at block 4: blocks 0 and 1 have left the window and
    # may hold another request's KV by now, even with a rewind that reaches them.
    metadata = scheduler.build_connector_meta(
        SchedulerOutput(
            new_requests=[
                window_data(1, tokens, [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25], 16)
            ]
        )
    )
    assert pages_of(metadata.saves[0], 1) == [24, 25]


def chunked_prompt(scheduler, tokens):
    """A cold prompt of 8 blocks in two chunks of 4; returns the two iterations' metadata."""
    request = make_context_request(1, tokens, last_chunk=False)
    scheduler.get_num_new_matched_tokens(request, 0)
    first = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[window_data(1, tokens, [10, 11, 12, 13], [20, 21, 22, 23])])
    )
    request.context_current_position = 4 * TOKENS_PER_BLOCK
    request.is_last_context_chunk = True
    last = scheduler.build_connector_meta(
        SchedulerOutput(
            cached_requests=[
                window_data(1, [], [14, 15, 16, 17], [24, 25, 26, 27], 4 * TOKENS_PER_BLOCK)
            ]
        )
    )
    return first, last


def test_scheduler_resaves_a_long_prompt_at_its_last_chunk(store_config, monkeypatch):
    monkeypatch.setenv(scheduler_module.RESAVE_MIN_TOKENS_ENV, str(6 * TOKENS_PER_BLOCK))
    first, last = chunked_prompt(
        make_window_scheduler(store_config), list(range(8 * TOKENS_PER_BLOCK))
    )
    assert pages_of(first.saves[0], 0) == [10, 11, 12, 13]
    # The whole prompt again, so the save path refreshes the head the first chunk wrote.
    assert pages_of(last.saves[0], 0) == [10, 11, 12, 13, 14, 15, 16, 17]
    assert pages_of(last.saves[0], 1) == [26, 27]


def test_scheduler_does_not_resave_short_prompts_or_by_default(store_config, monkeypatch):
    tokens = list(range(8 * TOKENS_PER_BLOCK))
    _, last = chunked_prompt(make_window_scheduler(store_config), tokens)
    assert pages_of(last.saves[0], 0) == [14, 15, 16, 17]
    monkeypatch.setenv(scheduler_module.RESAVE_MIN_TOKENS_ENV, str(16 * TOKENS_PER_BLOCK))
    _, last = chunked_prompt(make_window_scheduler(store_config), tokens)
    assert pages_of(last.saves[0], 0) == [14, 15, 16, 17]


def test_scheduler_loads_window_pages_only_where_the_runtime_resumes(store_config):
    scheduler = make_window_scheduler(store_config, hit=PrefixHit(4, {1: 2}))
    tokens = list(range(6 * TOKENS_PER_BLOCK + 1))
    request = make_context_request(1, tokens)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (4 * TOKENS_PER_BLOCK, False)

    # Honoured in full: the cache gives the window pages for blocks 2 and 3 only.
    request.context_current_position = 4 * TOKENS_PER_BLOCK
    metadata = scheduler.build_connector_meta(
        SchedulerOutput(
            new_requests=[
                window_data(
                    1,
                    tokens,
                    [10, 11, 12, 13, 14, 15, 16],
                    [BAD_PAGE_INDEX, BAD_PAGE_INDEX, 22, 23, 24, 25, 26],
                )
            ]
        )
    )
    assert pages_of(metadata.loads[0], 0) == [10, 11, 12, 13]
    assert pages_of(metadata.loads[0], 1) == [22, 23]


def test_scheduler_refuses_a_short_honour_whose_window_is_not_stored(store_config):
    scheduler = make_window_scheduler(store_config, hit=PrefixHit(4, {1: 2}))
    tokens = list(range(6 * TOKENS_PER_BLOCK + 1))
    request = make_context_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)

    # The runtime resumes at block 2, whose window reads blocks 0 and 1: pages
    # the store was never found to hold, since prompts do not end there.
    request.context_current_position = 2 * TOKENS_PER_BLOCK
    data = window_data(1, tokens, [10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26])
    with pytest.raises(RuntimeError, match="not found to hold"):
        scheduler.build_connector_meta(SchedulerOutput(new_requests=[data]))


def test_scheduler_passes_the_rewind_and_local_prefix_to_the_lookup(store_config):
    scheduler = make_window_scheduler(store_config, rewind_tokens=8)
    tokens = list(range(6 * TOKENS_PER_BLOCK))
    scheduler.get_num_new_matched_tokens(make_context_request(1, tokens), 2 * TOKENS_PER_BLOCK)
    _, first_block, rewind_tokens = scheduler._worker.queries[0]
    assert (first_block, rewind_tokens) == (2, 8)


def test_scheduler_saves_only_prompt_blocks_under_speculative_decoding(store_config):
    """Rejected drafts rewind a request's page list, so a generated block's page may change hands."""
    llm_args = make_llm_args()
    llm_args.speculative_config = SimpleNamespace(max_draft_len=3)
    scheduler = MooncakeStoreConnectorScheduler(llm_args)
    scheduler._worker = FakeWorker(0)
    tokens = list(range(2 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)

    first = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, [4, 5])])
    )
    assert [page.page_index for page in first.saves[0].pages] == [4, 5]

    more_tokens = list(range(2 * TOKENS_PER_BLOCK, 3 * TOKENS_PER_BLOCK))
    second = scheduler.build_connector_meta(
        SchedulerOutput(cached_requests=[request_data(1, more_tokens, [6])])
    )
    assert second.saves == []


def test_scheduler_implements_the_per_layer_group_callbacks(store_config):
    """A sliding-window cache has several layer groups, and the flat callbacks are never called for it."""
    scheduler = make_window_scheduler(store_config)
    tokens = list(range(2 * TOKENS_PER_BLOCK))
    request = make_context_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc_by_layer_group(request, [[4, 5], [6, 7]])
    scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[window_data(1, tokens, [4, 5], [6, 7])])
    )
    assert scheduler.request_finished_by_layer_group(request, [[4, 5], [6, 7]]) is True
    assert scheduler.request_finished_by_layer_group(request, [[4, 5], [6, 7]]) is False


# ---- prefix reservations ----


def wait_for(probe, timeout=5.0):
    """Poll `probe` until it returns something truthy or `timeout` passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = probe()
        if value:
            return value
        time.sleep(0.01)
    return probe()


def confirmed_load(reservation_id, request_id, start, end, pages_by_group, tokens, is_async=True):
    return PrefixLoad(
        reservation_id,
        request_id,
        start,
        end,
        is_async,
        [list(p) for p in pages_by_group],
        list(tokens),
        None,
    )


def test_scheduler_reserve_prefix_promises_an_asynchronous_load(store_config):
    scheduler = make_window_scheduler(store_config, hit=PrefixHit(4, {1: 2}))
    request = make_context_request(1, list(range(6 * TOKENS_PER_BLOCK + 1)))
    assert scheduler.reserve_prefix(request, 0, 11) == (4 * TOKENS_PER_BLOCK, True)


def test_scheduler_reserve_prefix_loads_synchronously_on_request(store_config, monkeypatch):
    monkeypatch.setenv(scheduler_module.SYNC_LOAD_ENV, "1")
    scheduler = make_window_scheduler(store_config, hit=PrefixHit(4, {1: 2}))
    request = make_context_request(1, list(range(6 * TOKENS_PER_BLOCK + 1)))
    assert scheduler.reserve_prefix(request, 0, 11) == (4 * TOKENS_PER_BLOCK, False)


def test_scheduler_reserve_prefix_offers_nothing_without_a_hit(store_config):
    scheduler = make_window_scheduler(store_config, hit=PrefixHit(0))
    request = make_context_request(1, list(range(6 * TOKENS_PER_BLOCK + 1)))
    assert scheduler.reserve_prefix(request, 0, 11) == (0, False)


def test_scheduler_turns_a_confirmed_reservation_into_page_transfers(store_config):
    scheduler = make_window_scheduler(store_config, hit=PrefixHit(4, {1: 2}))
    tokens = list(range(6 * TOKENS_PER_BLOCK + 1))
    request = make_context_request(1, tokens)
    scheduler.reserve_prefix(request, 0, 11)

    full = [10, 11, 12, 13, 14, 15, 16]
    window = [BAD_PAGE_INDEX, BAD_PAGE_INDEX, 22, 23, 24, 25, 26]
    load = confirmed_load(11, 1, 0, 4 * TOKENS_PER_BLOCK, [full, window], tokens)
    metadata = scheduler.build_connector_meta(SchedulerOutput(prefix_loads=[load]))
    (prefix_load,) = metadata.prefix_loads
    assert (prefix_load.reservation_id, prefix_load.is_async) == (11, True)
    assert pages_of(prefix_load.transfers, 0) == [10, 11, 12, 13]
    # The window reads blocks 2 and 3 at the load's end; 0 and 1 have no page.
    assert pages_of(prefix_load.transfers, 1) == [22, 23]

    # Once the request runs, nothing is loaded again and the store's blocks are
    # not written back.
    request.context_current_position = 4 * TOKENS_PER_BLOCK
    later = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[window_data(1, tokens, full, window, 4 * TOKENS_PER_BLOCK)])
    )
    assert later.loads == [] and later.prefix_loads == []
    assert pages_of(later.saves[0], 0) == [14, 15]
    assert pages_of(later.saves[0], 1) == [24, 25]


def test_scheduler_refuses_a_confirmed_load_whose_window_was_not_found(store_config):
    scheduler = make_window_scheduler(store_config, hit=PrefixHit(4, {1: 2}))
    tokens = list(range(6 * TOKENS_PER_BLOCK + 1))
    request = make_context_request(1, tokens)
    scheduler.reserve_prefix(request, 0, 11)

    # Confirmed only up to block 2, whose window reads blocks 0 and 1: pages the
    # lookup never found, since no prompt ended there.
    load = confirmed_load(
        11, 1, 0, 2 * TOKENS_PER_BLOCK, [[10, 11, 12, 13], [20, 21, 22, 23]], tokens
    )
    with pytest.raises(RuntimeError, match="not found to hold"):
        scheduler.build_connector_meta(SchedulerOutput(prefix_loads=[load]))


def test_scheduler_release_trims_then_forgets_a_reservation(store_config):
    scheduler = make_window_scheduler(store_config, hit=PrefixHit(4, {1: 2}))
    request = make_context_request(1, list(range(6 * TOKENS_PER_BLOCK + 1)))
    scheduler.reserve_prefix(request, 0, 11)
    reservations = scheduler._requests[1].reservations

    scheduler.release_prefix_reservation(request, 11, 2 * TOKENS_PER_BLOCK, 4 * TOKENS_PER_BLOCK)
    assert (reservations[11].start, reservations[11].end) == (0, 2 * TOKENS_PER_BLOCK)
    scheduler.release_prefix_reservation(request, 11, 0, 2 * TOKENS_PER_BLOCK)
    assert 11 not in reservations


def test_worker_loads_a_parked_requests_pages_off_the_executor_thread(
    store_config, fake_store, staged_copies
):
    layout = make_layout()
    with make_staged_worker(fake_store, store_config, layout=layout) as worker:
        block_hash = b"\x05" * 16
        key = worker._namespaces[0].key(block_hash)
        fake_store.objects.add(key)
        transfers = RequestTransfers(7, [PageTransfer(block_hash, 0, 3)])
        worker.bind_connector_meta(
            MooncakeStoreMetadata(prefix_loads=[PrefixLoadTransfers(21, True, transfers)])
        )
        worker.start_load_kv(None)

        assert wait_for(worker.get_finished_prefix_loads) == [21]
        (keys, addresses, _sizes) = fake_store.get_calls[0]
        assert keys == [key]
        # Through a loader's own slots, never the executor's.
        loader_slots = {pool.slot_address(0) for pool in worker._async_load_stagings}
        assert addresses[0][0] in loader_slots
        assert addresses[0][0] != worker._load_staging.slot_address(0)
        assert len(staged_copies) == 1


def test_worker_starts_one_loader_per_configured_thread(
    store_config, fake_store, staged_copies, monkeypatch
):
    monkeypatch.setenv(worker_module.LOAD_THREADS_ENV, "3")
    with make_staged_worker(fake_store, store_config, layout=make_layout()) as worker:
        assert len(worker._load_threads) == 3
        # Each with slots of its own.
        assert len({id(pool) for pool in worker._async_load_stagings}) == 3
        assert all(thread.is_alive() for thread in worker._load_threads)
    assert worker._load_threads == []


def test_worker_refuses_a_loader_count_below_one(store_config, fake_store, monkeypatch):
    monkeypatch.setenv(worker_module.LOAD_THREADS_ENV, "0")
    with pytest.raises(ValueError, match="at least 1"):
        MooncakeStoreConnectorWorker(make_llm_args())


def test_node_budget_counts_a_staging_pool_per_loader_thread():
    gib = 1 << 30
    config = SimpleNamespace(global_segment_size=0, role=SimpleNamespace(loads=True, saves=True))
    # The executor's pool, two loaders' and the save pool: 4 GiB of pinned staging.
    validate_node_budget(
        config, ranks_on_node=1, available_bytes=NODE_BUDGET_RESERVE_BYTES + 4 * gib, load_threads=2
    )
    with pytest.raises(ValueError, match="pin 4.0 GiB of staging"):
        validate_node_budget(
            config,
            ranks_on_node=1,
            available_bytes=NODE_BUDGET_RESERVE_BYTES + 4 * gib - 1,
            load_threads=2,
        )


def test_worker_reports_a_synchronous_prefix_load_at_once(store_config, fake_store, staged_copies):
    layout = make_layout()
    with make_staged_worker(fake_store, store_config, layout=layout) as worker:
        block_hash = b"\x06" * 16
        fake_store.objects.add(worker._namespaces[0].key(block_hash))
        transfers = RequestTransfers(7, [PageTransfer(block_hash, 0, 3)])
        worker.bind_connector_meta(
            MooncakeStoreMetadata(prefix_loads=[PrefixLoadTransfers(22, False, transfers)])
        )
        worker.start_load_kv(None)

        assert worker.get_finished_prefix_loads() == [22]
        assert worker.get_finished_prefix_loads() == []
        (_keys, addresses, _sizes) = fake_store.get_calls[0]
        assert addresses == [[worker._load_staging.slot_address(0)]]


def test_worker_raises_a_failed_asynchronous_load_on_the_executor_thread(
    store_config, fake_store, staged_copies
):
    layout = make_layout()
    with make_staged_worker(fake_store, store_config, layout=layout) as worker:
        block_hash = b"\x08" * 16
        key = worker._namespaces[0].key(block_hash)
        fake_store.objects.add(key)
        fake_store.fail_gets_for.add(key)
        transfers = RequestTransfers(7, [PageTransfer(block_hash, 0, 3)])
        worker.bind_connector_meta(
            MooncakeStoreMetadata(prefix_loads=[PrefixLoadTransfers(23, True, transfers)])
        )
        worker.start_load_kv(None)

        def raised():
            try:
                worker.get_finished_prefix_loads()
            except RuntimeError:
                return True
            return False

        # The parked request can neither run nor be reported done, so the
        # failure surfaces on the executor thread instead of hanging it.
        assert wait_for(raised)
        # A failed read is never scattered over the destination page.
        assert staged_copies == []


def test_worker_splits_a_long_probe_into_concurrent_chunks(store_config, fake_store, monkeypatch):
    monkeypatch.setenv(worker_module.PROBE_CHUNK_ENV, "3")
    with make_worker(fake_store, layout=make_layout()) as worker:
        keys = [f"k{index}" for index in range(8)]
        fake_store.objects.update(keys[:5])
        fake_store.exist_calls.clear()

        assert worker._probe(keys) == [1, 1, 1, 1, 1, 0, 0, 0]
        # Three calls of at most three keys, answered in key order.
        assert sorted(len(call) for call in fake_store.exist_calls) == [2, 3, 3]
        assert sorted(key for call in fake_store.exist_calls for key in call) == sorted(keys)


def test_worker_probe_chunking_can_be_turned_off(store_config, fake_store, monkeypatch):
    monkeypatch.setenv(worker_module.PROBE_CHUNK_ENV, "0")
    with make_worker(fake_store, layout=make_layout()) as worker:
        keys = [f"k{index}" for index in range(8)]
        fake_store.exist_calls.clear()
        assert worker._probe(keys) == [0] * 8
        assert [len(call) for call in fake_store.exist_calls] == [8]


def test_worker_reports_a_failed_chunk_as_a_miss(store_config, fake_store, monkeypatch):
    monkeypatch.setenv(worker_module.PROBE_CHUNK_ENV, "3")
    with make_worker(fake_store, layout=make_layout()) as worker:
        real = fake_store.batch_is_exist

        def flaky(keys):
            if "k4" in keys:
                raise RuntimeError("master unreachable")
            return real(keys)

        monkeypatch.setattr(fake_store, "batch_is_exist", flaky)
        assert worker._probe([f"k{index}" for index in range(8)]) is None


def test_scheduler_saves_only_the_blocks_a_chunk_completes(store_config):
    """A drafter's extra KV tokens give the next chunk's first block a page before it is computed."""
    scheduler = make_window_scheduler(store_config)
    tokens = list(range(6 * TOKENS_PER_BLOCK))
    request = make_context_request(1, tokens, last_chunk=False)
    request.context_chunk_size = 2 * TOKENS_PER_BLOCK
    scheduler.get_num_new_matched_tokens(request, 0)

    first = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[window_data(1, tokens, [10, 11, 12], [20, 21, 22])])
    )
    # Blocks 0 and 1 are computed; block 2 only has its page.
    assert pages_of(first.saves[0], 0) == [10, 11]

    request.context_current_position = 2 * TOKENS_PER_BLOCK
    request.context_chunk_size = 4 * TOKENS_PER_BLOCK
    request.is_last_context_chunk = True
    last = scheduler.build_connector_meta(
        SchedulerOutput(cached_requests=[window_data(1, [], [13, 14, 15], [23, 24, 25], 8)])
    )
    assert pages_of(last.saves[0], 0) == [12, 13, 14, 15]


def test_validate_refuses_a_spec_recompute_tail():
    llm_args = make_llm_args()
    llm_args.speculative_config = SimpleNamespace(max_draft_len=7, context_recompute_tail=2048)
    with pytest.raises(NotImplementedError, match="context_recompute_tail"):
        validate_llm_args(llm_args)
    llm_args.speculative_config.context_recompute_tail = 0
    validate_llm_args(llm_args)


def test_worker_starts_no_loader_threads_for_synchronous_loads(
    store_config, fake_store, staged_copies, monkeypatch
):
    monkeypatch.setenv(worker_module.SYNC_LOAD_ENV, "1")
    with make_staged_worker(fake_store, store_config, layout=make_layout()) as worker:
        assert worker._load_threads == []
        assert worker._async_load_stagings == []


def test_worker_refuses_a_window_layout_without_prefix_reservations(store_config, fake_store):
    llm_args = make_llm_args()
    llm_args.scheduler_config = SimpleNamespace(enable_prefix_aware_scheduling=False)
    worker = MooncakeStoreConnectorWorker(llm_args)
    fake_store.workers.append(worker)
    with pytest.raises(ValueError, match="enable_prefix_aware_scheduling"):
        worker.register_kv_cache_layout(make_window_layout())
