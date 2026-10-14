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
from collections.abc import Iterator, Sequence
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
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import staging as staging_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.addressing import PageAddressing
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import MooncakeStoreConnectorConfig
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
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
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


def make_llm_args(*, run_dir: str | None = None, enable_attention_dp: bool = False,
                  speculative_config: MTPDecodingConfig | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        model="/models/test-model",
        enable_attention_dp=enable_attention_dp,
        speculative_config=speculative_config,
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
    enable_attention_dp: bool = False,
) -> Iterator[MooncakeStoreConnectorWorker]:
    """Build a worker and shut it down before the test call phase ends.

    Registering a layout starts the background save thread, and
    pytest-threadleak snapshots threads around the call phase only, so
    fixture teardown would run too late to keep it quiet.
    """
    worker = MooncakeStoreConnectorWorker(make_llm_args(run_dir=run_dir, enable_attention_dp=enable_attention_dp))
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


def test_worker_load_reports_the_request_when_a_page_is_missing(store_config, fake_store):
    """A single-rank worker hands a request whose page is gone back to the executor."""
    with make_worker(fake_store, layout=make_layout()) as worker:
        transfers = RequestTransfers(7, [PageTransfer(b"\x00" * 16, 0, 1)])
        worker.bind_connector_meta(SimpleNamespace(loads=[transfers], saves=[]))
        worker.start_load_kv(None)
        assert worker.take_failed_load_requests() == {7}
        # Cleared on return: the next iteration starts clean.
        assert worker.take_failed_load_requests() == set()


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
def make_staged_worker(
    fake_store, store_config, *, layout, budget=None, enable_attention_dp=False
):
    """A worker using the configured bounded host staging pools."""
    options = {"stage_through_host": True}
    if budget is not None:
        options["staging_buffer_bytes"] = budget
    set_pool_setting(store_config, **options)
    with make_worker(
        fake_store, layout=layout, enable_attention_dp=enable_attention_dp
    ) as worker:
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
        worker.start_load_kv(None)
        assert worker.take_failed_load_requests() == {7}

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


def make_scheduler(
    store_config: Path,
    hit_blocks: int = 0,
    speculative_config: MTPDecodingConfig | None = None,
) -> MooncakeStoreConnectorScheduler:
    scheduler = MooncakeStoreConnectorScheduler(
        make_llm_args(speculative_config=speculative_config)
    )
    scheduler._worker = FakeWorker(hit_blocks)
    return scheduler


def request_data(
    request_id: int,
    new_tokens: Sequence[int],
    page_indices: Sequence[int],
    layer_group_id: int = 0,
    *,
    computed_position: int = 0,
    num_scheduled_tokens: int | None = None,
) -> RequestData:
    by_layer_group = [[] for _ in range(layer_group_id + 1)]
    by_layer_group[layer_group_id] = list(page_indices)
    return RequestData(
        request_id=request_id,
        new_tokens=list(new_tokens),
        new_block_ids=list(page_indices),
        computed_position=computed_position,
        num_scheduled_tokens=(
            len(new_tokens) if num_scheduled_tokens is None else num_scheduled_tokens
        ),
        new_block_ids_by_layer_group=by_layer_group,
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
        SchedulerOutput(cached_requests=[request_data(1, more_tokens, [6], computed_position=2 * TOKENS_PER_BLOCK)])
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


@pytest.mark.parametrize("lookahead", [1, TOKENS_PER_BLOCK + 1])
def test_hash_chain_requires_lookahead_and_matches_incremental_extension(lookahead: int) -> None:
    tokens = list(range(4 * TOKENS_PER_BLOCK + lookahead))
    chain = BlockHashChain(TOKENS_PER_BLOCK, prompt_lookahead=lookahead)
    assert chain.extend(tokens[: TOKENS_PER_BLOCK + lookahead - 1]) == []
    assert len(chain.extend(tokens[: TOKENS_PER_BLOCK + lookahead])) == 1
    for end in range(TOKENS_PER_BLOCK + lookahead + 1, len(tokens) + 1):
        chain.extend(tokens[:end])
    assert len(chain.hashes) == 4
    assert chain.hashes == BlockHashChain(TOKENS_PER_BLOCK, prompt_lookahead=lookahead).extend(
        tokens
    )

    divergent = list(tokens)
    divergent[TOKENS_PER_BLOCK + lookahead - 1] += 1000
    assert (
        chain.hashes[0]
        != BlockHashChain(TOKENS_PER_BLOCK, prompt_lookahead=lookahead).extend(divergent)[0]
    )


def test_hash_chain_rejects_negative_lookahead() -> None:
    with pytest.raises(ValueError, match="prompt_lookahead"):
        BlockHashChain(TOKENS_PER_BLOCK, prompt_lookahead=-1)


def test_scheduler_publishes_only_computed_full_blocks(store_config: Path) -> None:
    scheduler = make_scheduler(store_config)
    tokens = list(range(3 * TOKENS_PER_BLOCK + 1))
    scheduler.get_num_new_matched_tokens(make_request(1, tokens), 0)
    data = request_data(1, tokens, [4, 5, 6, 7], num_scheduled_tokens=TOKENS_PER_BLOCK - 1)
    first = scheduler.build_connector_meta(SchedulerOutput(new_requests=[data]))
    assert first.saves == []

    # Full prompt hashes and every capacity page were reported up front, but
    # no page may be published until its final token has actually been computed.
    for ordinal in range(3):
        computed = TOKENS_PER_BLOCK - 1 if ordinal == 0 else ordinal * TOKENS_PER_BLOCK
        scheduled = 1 if ordinal == 0 else TOKENS_PER_BLOCK
        data = request_data(1, [], [], computed_position=computed, num_scheduled_tokens=scheduled)
        metadata = scheduler.build_connector_meta(SchedulerOutput(cached_requests=[data]))
        assert [page.page_index for page in metadata.saves[0].pages] == [4 + ordinal]


@pytest.mark.parametrize("local_blocks", [0, 1])
@pytest.mark.parametrize("use_mtp", [False, True])
def test_scheduler_saves_computed_tail_after_restore(
    store_config: Path, local_blocks: int, use_mtp: bool
) -> None:
    config = MTPDecodingConfig(max_draft_len=1) if use_mtp else None
    scheduler = make_scheduler(store_config, hit_blocks=2, speculative_config=config)
    tokens = list(range(6 * TOKENS_PER_BLOCK + 1))
    scheduler.get_num_new_matched_tokens(make_request(1, tokens), local_blocks * TOKENS_PER_BLOCK)
    # The scheduler subtracts connector restores from computed_position.
    # Full prompt hashes and capacity pages are available ahead of computation.
    data = request_data(
        1,
        tokens,
        list(range(10, 17)),
        computed_position=local_blocks * TOKENS_PER_BLOCK,
        num_scheduled_tokens=TOKENS_PER_BLOCK + 1,
    )
    metadata = scheduler.build_connector_meta(SchedulerOutput(new_requests=[data]))
    tail_block = local_blocks + 2
    assert [page.page_index for page in metadata.loads[0].pages] == [
        10 + local_blocks,
        11 + local_blocks,
    ]
    assert [page.page_index for page in metadata.saves[0].pages] == [10 + tail_block]

    # A later chunk includes the restored prefix in its position. Count the
    # restore only once, and wait for this partial page to become complete.
    data = request_data(
        1,
        [],
        [],
        computed_position=(tail_block + 1) * TOKENS_PER_BLOCK + 1,
        num_scheduled_tokens=TOKENS_PER_BLOCK - 1,
    )
    metadata = scheduler.build_connector_meta(SchedulerOutput(cached_requests=[data]))
    assert metadata.loads == []
    assert [page.page_index for page in metadata.saves[0].pages] == [11 + tail_block]


def test_scheduler_mtp_keys_require_matching_lookahead(store_config: Path) -> None:
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    other_tokens = list(tokens)
    other_tokens[TOKENS_PER_BLOCK] += 1000
    config = MTPDecodingConfig(max_draft_len=1)
    scheduler = make_scheduler(store_config, hit_blocks=99, speculative_config=config)
    scheduler.get_num_new_matched_tokens(make_request(1, tokens), 0)
    scheduler.get_num_new_matched_tokens(make_request(2, other_tokens), 0)
    first, other = scheduler._worker.queries
    assert first == list(BlockHashChain(TOKENS_PER_BLOCK, prompt_lookahead=1).extend(tokens))
    assert first[0] != other[0]
    assert first[0] != BlockHashChain(TOKENS_PER_BLOCK).extend(tokens)[0]

    metadata = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(2, other_tokens, [4, 5, 6])])
    )
    assert len(metadata.loads[0].pages) == 2
    assert metadata.saves == []  # The final block lacks the draft's following token.


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


def test_mooncake_declares_adp_support() -> None:
    assert MooncakeStoreConnectorScheduler.supports_attention_dp
    assert MooncakeStoreConnectorWorker.supports_attention_dp


def test_worker_load_raises_when_a_page_is_missing_under_tensor_parallelism(
    store_config, fake_store, monkeypatch
):
    """TP ranks hold shards of one page and cannot restart a request alone."""
    monkeypatch.setattr(worker_module, "mpi_world_size", lambda: 2)
    with make_worker(fake_store, layout=make_layout()) as worker:
        transfers = RequestTransfers(7, [PageTransfer(b"\x00" * 16, 0, 1)])
        worker.bind_connector_meta(SimpleNamespace(loads=[transfers], saves=[]))
        with pytest.raises(RuntimeError, match="already"):
            worker.start_load_kv(None)


def test_worker_load_skips_the_rest_of_a_failed_request_and_keeps_the_others(
    store_config, fake_store
):
    """Only the request that lost a page restarts; its later pages are not fetched."""
    with make_worker(fake_store, layout=make_layout(), enable_attention_dp=True) as worker:
        present, missing, later = (bytes([1]) * 16, bytes([2]) * 16, bytes([3]) * 16)
        for block_hash in (present, later):
            fake_store.objects.add(worker._namespaces[0].key(block_hash))
        worker._batch_size = 2
        loads = [
            RequestTransfers(7, [PageTransfer(present, 0, 1)]),
            RequestTransfers(8, [PageTransfer(missing, 0, 2), PageTransfer(later, 0, 3)]),
        ]
        worker.bind_connector_meta(SimpleNamespace(loads=loads, saves=[]))
        worker.start_load_kv(None)
        assert worker.take_failed_load_requests() == {8}
        fetched = [key for keys, _addresses, _sizes in fake_store.get_calls for key in keys]
        assert worker._namespaces[0].key(later) not in fetched
        assert worker._namespaces[0].key(present) in fetched


def test_staging_scatters_only_the_pages_that_landed(store_config, fake_store, staged_copies):
    """In a mixed batch the filled slots reach their pages, the failed one does not."""
    layout = make_layout()
    with make_staged_worker(fake_store, store_config, layout=layout) as worker:
        good, bad = bytes([0x0A]) * 16, bytes([0x0B]) * 16
        fake_store.objects.add(worker._namespaces[0].key(good))
        fake_store.objects.add(worker._namespaces[0].key(bad))
        fake_store.fail_gets_for.add(worker._namespaces[0].key(bad))
        loads = [
            RequestTransfers(7, [PageTransfer(good, 0, 1)]),
            RequestTransfers(8, [PageTransfer(bad, 0, 2)]),
        ]
        worker.bind_connector_meta(SimpleNamespace(loads=loads, saves=[]))
        worker.start_load_kv(None)
        assert worker.take_failed_load_requests() == {8}
        good_addresses, _sizes = worker._addressing.buffers(0, 1)
        assert [dst for dst, _src, _size in staged_copies] == list(good_addresses)


def test_worker_failed_load_drops_that_requests_saves_from_the_pass(store_config, fake_store):
    """A mixed batch: the request that lost a page must not publish its scheduled tail.

    `build_connector_meta` bound the saves of every scheduled request before the
    load ran. The failed request leaves the batch without computing anything, so
    only the surviving request's saves may reach `wait_for_save`.
    """
    with make_worker(fake_store, layout=make_layout(), enable_attention_dp=True) as worker:
        present, missing = bytes([1]) * 16, bytes([2]) * 16
        tail_ok, tail_failed = bytes([3]) * 16, bytes([4]) * 16
        fake_store.objects.add(worker._namespaces[0].key(present))
        loads = [
            RequestTransfers(7, [PageTransfer(present, 0, 1)]),
            RequestTransfers(8, [PageTransfer(missing, 0, 2)]),
        ]
        saves = [
            RequestTransfers(7, [PageTransfer(tail_ok, 0, 3)]),
            RequestTransfers(8, [PageTransfer(tail_failed, 0, 4)]),
        ]
        metadata = SimpleNamespace(loads=loads, saves=saves)
        worker.bind_connector_meta(metadata)
        worker.start_load_kv(None)
        assert worker.take_failed_load_requests() == {8}
        # What `wait_for_save` will hand to the save thread after the pass:
        assert [transfers.request_id for transfers in worker.get_connector_meta().saves] == [7]
        assert metadata.saves is worker.get_connector_meta().saves  # edited in place
        assert 8 not in worker._outstanding_saves
        # Dropping a request that has nothing bound leaves the survivor alone.
        assert worker.drop_bound_saves([9]) == 0
        assert [transfers.request_id for transfers in metadata.saves] == [7]


def test_worker_drops_bound_saves_of_a_restarted_request_even_when_it_was_the_whole_batch(
    store_config, fake_store
):
    """An attention-DP owner whose only request failed runs a dummy forward: no saves at all."""
    with make_worker(fake_store, layout=make_layout(), enable_attention_dp=True) as worker:
        missing, tail = bytes([2]) * 16, bytes([4]) * 16
        metadata = SimpleNamespace(
            loads=[RequestTransfers(8, [PageTransfer(missing, 0, 2)])],
            saves=[RequestTransfers(8, [PageTransfer(tail, 0, 4)])],
        )
        worker.bind_connector_meta(metadata)
        worker.start_load_kv(None)
        assert worker.take_failed_load_requests() == {8}
        assert metadata.saves == []
        # With nothing bound to save, the post-pass handoff is a no-op.
        worker.wait_for_save(None)
        assert worker._save_queue.empty()
        assert not worker._outstanding_saves
        # A second drop is harmless, as is dropping an unknown request.
        assert worker.drop_bound_saves([8, 9]) == 0


def set_store_options(store_config: Path, **options) -> None:
    """Update keys of the rendered client config."""
    raw = json.loads(store_config.read_text())
    raw.update(options)
    store_config.write_text(json.dumps(raw))


def enable_async_load(store_config: Path, workers: int | None = None) -> None:
    """Turn on `async_load` in the rendered client config."""
    options = {"async_load": True}
    if workers is not None:
        options["async_load_workers"] = workers
    set_store_options(store_config, **options)


def wait_async_idle(worker: MooncakeStoreConnectorWorker, timeout: float = 5.0) -> None:
    """Block until the load thread has drained every queued request."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with worker._load_lock:
            if not worker._async_pending:
                return
        time.sleep(0.005)
    raise AssertionError("the asynchronous load did not finish in time")


def test_config_async_load_defaults_off_and_reads_the_json(store_config):
    assert MooncakeStoreConnectorConfig.from_file(str(store_config)).async_load is False
    enable_async_load(store_config)
    assert MooncakeStoreConnectorConfig.from_file(str(store_config)).async_load is True


def test_worker_has_no_load_thread_by_default(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        assert worker._load_threads == []
        assert worker._async_stagings == []
        with pytest.raises(RuntimeError, match="not enabled"):
            worker.start_async_load(1, RequestTransfers(1, []))
        # Without asynchronous offers a parked id is echoed straight back.
        assert worker.get_finished([], [5]) == ([], [5])


def test_worker_rejects_async_load_under_tensor_parallelism(store_config, fake_store, monkeypatch):
    enable_async_load(store_config)
    monkeypatch.setattr(worker_module, "mpi_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="attention DP"):
        MooncakeStoreConnectorWorker(make_llm_args())


def test_worker_async_load_is_reported_once_the_runtime_announces_it(store_config, fake_store):
    """A landed load waits for the runtime to say it parked the request."""
    enable_async_load(store_config)
    layout = make_layout()
    with make_worker(fake_store, layout=layout, enable_attention_dp=True) as worker:
        assert len(worker._load_threads) == 1
        block_hash = b"\x05" * 16
        fake_store.objects.add(worker._namespaces[0].key(block_hash))

        worker.start_async_load(7, RequestTransfers(7, [PageTransfer(block_hash, 0, 2)]))
        wait_async_idle(worker)

        # Landed, but not announced: nothing to report yet.
        assert worker.get_finished([], []) == ([], [])
        assert worker.get_finished([], [7]) == ([], [7])
        assert worker.take_failed_async_load_requests() == set()
        # Reported once only.
        assert worker.get_finished([], []) == ([], [])

        (keys, addresses, sizes) = fake_store.get_calls[0]
        expected_addresses, expected_sizes = PageAddressing(layout).buffers(0, 2)
        assert keys == [worker._namespaces[0].key(block_hash)]
        assert addresses == [expected_addresses]
        assert sizes == [expected_sizes]


def test_worker_async_load_announced_first_waits_for_the_pages(store_config, fake_store):
    """The runtime is never told a load finished while its bytes are in flight."""
    enable_async_load(store_config)
    with make_worker(fake_store, layout=make_layout(), enable_attention_dp=True) as worker:
        block_hash = b"\x06" * 16
        fake_store.objects.add(worker._namespaces[0].key(block_hash))
        gate = threading.Event()
        original = fake_store.batch_get_into_multi_buffers

        def gated(*args, **kwargs):
            assert gate.wait(5.0)
            return original(*args, **kwargs)

        fake_store.batch_get_into_multi_buffers = gated
        worker.start_async_load(7, RequestTransfers(7, [PageTransfer(block_hash, 0, 1)]))
        assert worker.get_finished([], [7]) == ([], [])
        gate.set()
        wait_async_idle(worker)
        assert worker.get_finished([], []) == ([], [7])
        assert worker.take_failed_async_load_requests() == set()


def test_worker_async_load_failure_is_handed_to_the_executor(store_config, fake_store):
    """A missing page still releases the parked request, marked for a restart."""
    enable_async_load(store_config)
    with make_worker(fake_store, layout=make_layout(), enable_attention_dp=True) as worker:
        present, missing, later = (bytes([1]) * 16, bytes([2]) * 16, bytes([3]) * 16)
        for block_hash in (present, later):
            fake_store.objects.add(worker._namespaces[0].key(block_hash))
        # One page per store call, so the page after the missing one is a
        # separate batch the loader can skip.
        worker._async_batch_size = 1
        transfers = RequestTransfers(
            8,
            [PageTransfer(present, 0, 1), PageTransfer(missing, 0, 2), PageTransfer(later, 0, 3)],
        )
        worker.start_async_load(8, transfers)
        wait_async_idle(worker)

        assert worker.get_finished([], [8]) == ([], [8])
        assert worker.take_failed_async_load_requests() == {8}
        # Cleared on return.
        assert worker.take_failed_async_load_requests() == set()
        # The pages after the missing one were not fetched: the request restarts.
        fetched = [key for keys, _addresses, _sizes in fake_store.get_calls for key in keys]
        assert worker._namespaces[0].key(later) not in fetched


def test_worker_async_load_with_nothing_to_load_completes_at_once(store_config, fake_store):
    enable_async_load(store_config)
    with make_worker(fake_store, layout=make_layout(), enable_attention_dp=True) as worker:
        worker.start_async_load(9, RequestTransfers(9, []))
        assert worker.get_finished([], [9]) == ([], [9])
        assert worker.take_failed_async_load_requests() == set()
        assert fake_store.get_calls == []


def test_worker_restarts_a_parked_request_it_was_never_handed(store_config, fake_store):
    """An announced request with no load behind it must not stay parked."""
    enable_async_load(store_config)
    with make_worker(fake_store, layout=make_layout(), enable_attention_dp=True) as worker:
        assert worker.get_finished([], [11]) == ([], [11])
        assert worker.take_failed_async_load_requests() == {11}


class FakeAsyncWorker(FakeWorker):
    """A lookup service that also records the loads handed to the load thread."""

    def __init__(self, hit_blocks=0):
        super().__init__(hit_blocks)
        self.async_loads = []
        self.failed_async_loads = []

    def start_async_load(self, request_id, transfers):
        self.async_loads.append((request_id, transfers))

    def fail_async_load(self, request_id):
        self.failed_async_loads.append(request_id)


def make_async_scheduler(store_config: Path, hit_blocks: int) -> MooncakeStoreConnectorScheduler:
    enable_async_load(store_config)
    scheduler = MooncakeStoreConnectorScheduler(make_llm_args(enable_attention_dp=True))
    scheduler._worker = FakeAsyncWorker(hit_blocks)
    return scheduler


def test_scheduler_offers_asynchronously_when_enabled(store_config):
    scheduler = make_async_scheduler(store_config, hit_blocks=2)
    request = make_request(1, list(range(5 * TOKENS_PER_BLOCK)))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (2 * TOKENS_PER_BLOCK, True)
    # A miss is never asynchronous: there would be nothing to wait for.
    scheduler._worker.hit_blocks = 0
    request = make_request(2, list(range(5 * TOKENS_PER_BLOCK)))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)


def test_scheduler_rejects_async_load_under_tensor_parallelism(store_config):
    enable_async_load(store_config)
    llm_args = make_llm_args()
    llm_args.tensor_parallel_size = 2
    with pytest.raises(NotImplementedError, match="attention DP"):
        MooncakeStoreConnectorScheduler(llm_args)


def test_scheduler_starts_the_async_load_when_the_pages_are_reported(store_config):
    """The offer goes to the load thread with the pages the runtime reserved."""
    scheduler = make_async_scheduler(store_config, hit_blocks=2)
    tokens = list(range(5 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (2 * TOKENS_PER_BLOCK, True)

    scheduler.update_state_after_alloc_by_layer_group(request, [[10, 11, 12, 13, 14]])

    ((request_id, transfers),) = scheduler._worker.async_loads
    state = scheduler._requests[1]
    assert request_id == 1
    assert [(page.layer_group_id, page.page_index) for page in transfers.pages] == [
        (0, 10),
        (0, 11),
    ]
    assert [page.block_hash for page in transfers.pages] == list(state.chain.hashes[:2])

    # The offer is consumed: the request comes back as a new scheduler-output
    # entry once its load landed, issues no synchronous load, and does not
    # write the stored blocks back.
    output = SchedulerOutput(
        new_requests=[
            request_data(
                1,
                tokens,
                [10, 11, 12, 13, 14],
                computed_position=2 * TOKENS_PER_BLOCK,
                num_scheduled_tokens=3 * TOKENS_PER_BLOCK,
            )
        ]
    )
    metadata = scheduler.build_connector_meta(output)
    assert metadata.loads == []
    assert [page.page_index for page in metadata.saves[0].pages] == [12, 13, 14]
    # Handed over once only.
    assert len(scheduler._worker.async_loads) == 1


def test_scheduler_async_load_stops_at_the_pages_the_runtime_reserved(store_config):
    scheduler = make_async_scheduler(store_config, hit_blocks=3)
    request = make_request(1, list(range(6 * TOKENS_PER_BLOCK)))
    scheduler.get_num_new_matched_tokens(request, 0)
    # An accepted three-block offer cannot complete with only two destinations.
    scheduler.update_state_after_alloc_by_layer_group(request, [[20, 21]])
    assert scheduler._worker.async_loads == []
    assert scheduler._worker.failed_async_loads == [request.request_id]


def test_scheduler_async_flat_alloc_is_a_single_group(store_config):
    scheduler = make_async_scheduler(store_config, hit_blocks=1)
    request = make_request(1, list(range(4 * TOKENS_PER_BLOCK)))
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.update_state_after_alloc(request, [30, 31, 32, 33])
    ((_request_id, transfers),) = scheduler._worker.async_loads
    assert [(page.layer_group_id, page.page_index) for page in transfers.pages] == [(0, 30)]


def test_scheduler_alloc_hooks_stay_no_ops_for_synchronous_loads(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=2)
    tokens = list(range(5 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)
    # The synchronous worker has no `start_async_load`; the hooks must not reach for it.
    scheduler.update_state_after_alloc_by_layer_group(request, [[10, 11, 12, 13, 14]])
    scheduler.update_state_after_alloc(request, [10, 11, 12, 13, 14])
    output = SchedulerOutput(new_requests=[request_data(1, tokens, [10, 11, 12, 13, 14])])
    metadata = scheduler.build_connector_meta(output)
    assert [page.page_index for page in metadata.loads[0].pages] == [10, 11]


def test_config_async_load_workers_defaults_to_one_and_rejects_zero(store_config):
    assert MooncakeStoreConnectorConfig.from_file(str(store_config)).async_load_workers == 1
    set_store_options(store_config, async_load_workers=3)
    assert MooncakeStoreConnectorConfig.from_file(str(store_config)).async_load_workers == 3
    set_store_options(store_config, async_load_workers=0)
    with pytest.raises(ValueError, match="async_load_workers"):
        MooncakeStoreConnectorConfig.from_file(str(store_config))


def test_worker_pool_default_is_one_worker_with_the_full_staging(
    store_config, fake_store, staged_copies
):
    """N=1 keeps the previous behaviour: one thread, one pool, the sync slot count."""
    enable_async_load(store_config)
    layout = make_layout(regions_per_group=2)
    page_bytes = PageAddressing(layout).bytes_per_page(0)
    with make_staged_worker(
        fake_store, store_config, layout=layout, budget=8 * page_bytes, enable_attention_dp=True
    ) as worker:
        assert len(worker._load_threads) == 1
        assert len(worker._async_stagings) == 1
        assert worker._async_stagings[0].num_slots == worker._load_staging.num_slots
        assert worker._async_batch_size == worker._batch_size


def test_worker_pool_splits_the_staging_budget_without_exceeding_it(
    store_config, fake_store, staged_copies
):
    layout = make_layout(regions_per_group=2)
    page_bytes = PageAddressing(layout).bytes_per_page(0)
    # 512 pages of budget across 4 workers -> 128 slots each, batch narrowed to 128.
    enable_async_load(store_config, workers=4)
    set_store_options(store_config, transfer_batch_size=256)
    with make_staged_worker(
        fake_store, store_config, layout=layout, budget=512 * page_bytes, enable_attention_dp=True
    ) as worker:
        assert [pool.num_slots for pool in worker._async_stagings] == [128] * 4
        assert worker._async_batch_size == 128
        # The synchronous path keeps the whole budget.
        assert worker._batch_size == 256
    fake_store.registered.clear()
    # 8 pages of budget across 2 workers stays at 4 slots each.
    enable_async_load(store_config, workers=2)
    set_store_options(store_config, transfer_batch_size=16)
    with make_staged_worker(
        fake_store, store_config, layout=layout, budget=8 * page_bytes, enable_attention_dp=True
    ) as worker:
        assert [pool.num_slots for pool in worker._async_stagings] == [4, 4]
        assert worker._async_batch_size == 4


def test_worker_pool_loads_two_requests_concurrently(store_config, fake_store):
    """Two workers transfer at the same time; each request is handled once."""
    enable_async_load(store_config, workers=2)
    with make_worker(fake_store, layout=make_layout(), enable_attention_dp=True) as worker:
        assert len(worker._load_threads) == 2
        hashes = [bytes([0x21]) * 16, bytes([0x22]) * 16]
        for block_hash in hashes:
            fake_store.objects.add(worker._namespaces[0].key(block_hash))
        # Both workers must be inside the store call at once, or this times out.
        rendezvous = threading.Barrier(2, timeout=5.0)
        original = fake_store.batch_get_into_multi_buffers

        def meet_then_get(*args, **kwargs):
            rendezvous.wait()
            return original(*args, **kwargs)

        fake_store.batch_get_into_multi_buffers = meet_then_get
        worker.start_async_load(7, RequestTransfers(7, [PageTransfer(hashes[0], 0, 1)]))
        worker.start_async_load(8, RequestTransfers(8, [PageTransfer(hashes[1], 0, 2)]))
        wait_async_idle(worker)

        _saving, finished = worker.get_finished([], [7, 8])
        assert sorted(finished) == [7, 8]
        assert worker.take_failed_async_load_requests() == set()
        fetched = [key for keys, _addresses, _sizes in fake_store.get_calls for key in keys]
        assert sorted(fetched) == sorted(worker._namespaces[0].key(h) for h in hashes)


def test_worker_pool_keeps_failure_and_success_apart(store_config, fake_store):
    enable_async_load(store_config, workers=2)
    with make_worker(fake_store, layout=make_layout(), enable_attention_dp=True) as worker:
        good, bad = bytes([0x31]) * 16, bytes([0x32]) * 16
        fake_store.objects.add(worker._namespaces[0].key(good))
        for request_id, block_hash, page in ((7, good, 1), (8, bad, 2), (9, good, 3)):
            worker.start_async_load(
                request_id, RequestTransfers(request_id, [PageTransfer(block_hash, 0, page)])
            )
        wait_async_idle(worker)
        _saving, finished = worker.get_finished([], [7, 8, 9])
        assert sorted(finished) == [7, 8, 9]
        assert worker.take_failed_async_load_requests() == {8}


def test_worker_pool_stats_line_summarises_the_interval(store_config, fake_store, monkeypatch):
    enable_async_load(store_config, workers=2)
    with make_worker(fake_store, layout=make_layout(), enable_attention_dp=True) as worker:
        # Idle and nothing completed: no line, even when forced.
        assert worker._maybe_log_async_stats(force=True) is None

        good, bad = bytes([0x41]) * 16, bytes([0x42]) * 16
        fake_store.objects.add(worker._namespaces[0].key(good))
        worker.start_async_load(7, RequestTransfers(7, [PageTransfer(good, 0, 1)]))
        worker.start_async_load(8, RequestTransfers(8, [PageTransfer(bad, 0, 2)]))
        wait_async_idle(worker)

        lines = []
        monkeypatch.setattr(worker_module.logger, "info", lambda message: lines.append(message))
        # Not due yet: the interval gates the line.
        assert worker._maybe_log_async_stats() is None
        line = worker._maybe_log_async_stats(force=True)
        assert line is not None and lines == [line]
        assert line.startswith("mooncake-store async-load stats: rank 0 workers 2 ")
        assert "completed=2" in line and "failed_pages=1" in line and "queue=0 active=0" in line
        # The counters reset after a line.
        assert "completed=0" in worker._format_async_stats()
