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

Runs without a Mooncake installation and without a GPU: the store handle is
replaced by an in-process fake, and the KV cache layout is synthesized from
plain integers, which is all the addressing arithmetic needs.
"""

import contextlib
import json
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
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import staging as staging_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import validation as validation_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.addressing import (
    PageAddressing,
    merge_intervals,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    MooncakeStoreConnectorConfig,
    StoreRole,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import BlockHashChain
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.ledger import read_segments
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    PageTransfer,
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
)
from tensorrt_llm._torch.pyexecutor.connectors.registry import uses_connector
from tensorrt_llm._torch.pyexecutor.py_executor_creator import _disable_native_kv_offload
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, KvCacheConnectorConfig
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


def make_layout(*, num_groups=1, regions_per_group=1, num_slots=8, window_size=None):
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
                window_size=window_size,
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


def resolve_config(run_dir: str | None = None) -> MooncakeStoreConnectorConfig:
    """The client config a rank resolves, given what its worker config said."""
    return MooncakeStoreConnectorConfig.resolve(make_llm_args(run_dir=run_dir))


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


# ---- validation ----


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("context_parallel_size", 2, "context parallelism"),
        ("pipeline_parallel_size", 2, "pipeline parallelism"),
    ],
)
def test_validate_llm_args_rejects_unsupported_parallelism(field, value, match):
    args = make_llm_args()
    setattr(args, field, value)
    with pytest.raises(NotImplementedError, match=match):
        validate_llm_args(args)


def test_validate_llm_args_rejects_m3_index_value_cache():
    args = make_llm_args()
    args.sparse_attention_config = SimpleNamespace(sparse_disable_index_value=False)
    with pytest.raises(NotImplementedError, match="sparse_disable_index_value"):
        validate_llm_args(args)

    args.sparse_attention_config = SimpleNamespace(sparse_disable_index_value=True)
    validate_llm_args(args)


def test_validate_layout_rejects_sliding_window():
    with pytest.raises(NotImplementedError, match="sliding-window"):
        validate_layout(make_layout(window_size=1024))
    validate_layout(make_layout())


# ---- the per-node budget ----
#
# Contribution is per rank and host DRAM is per node, so the ranks sharing one
# each claim the segment independently. Unchecked, the answer is the OOM killer
# minutes later during weight loading, naming no cause.


def node_budget_config(segment_size, **overrides):
    return MooncakeStoreConnectorConfig(
        master_server_address="10.0.0.1:50051",
        global_segment_size=segment_size,
        **overrides,
    )


def test_a_segment_the_node_can_afford_is_accepted():
    validate_node_budget(
        node_budget_config(16 * 1024**3),
        ranks_on_node=4,
        available_bytes=256 * 1024**3,
    )


def test_a_segment_the_node_cannot_afford_is_refused():
    """Eight attention-DP owners at 160GiB against a ~956GiB node."""
    with pytest.raises(ValueError, match="cannot afford"):
        validate_node_budget(
            node_budget_config(160 * 1024**3),
            ranks_on_node=8,
            available_bytes=956 * 1024**3,
        )


def test_the_same_segment_is_fine_with_fewer_ranks_on_the_node():
    """Four ranks of the same size fit, which is why the multiplier matters."""
    validate_node_budget(
        node_budget_config(160 * 1024**3),
        ranks_on_node=4,
        available_bytes=956 * 1024**3,
    )


def test_refusing_says_what_would_fit():
    """A rejection a user cannot act on is only marginally better than an OOM."""
    with pytest.raises(ValueError, match="at most"):
        validate_node_budget(
            node_budget_config(512 * 1024**3),
            ranks_on_node=4,
            available_bytes=256 * 1024**3,
        )


def test_the_budget_leaves_room_for_the_weights():
    """Permitting every last byte would pass configs that die under load."""
    available = 256 * 1024**3
    # Exactly the available memory, which is precisely what must not be taken.
    with pytest.raises(ValueError, match="cannot afford"):
        validate_node_budget(
            node_budget_config(available), ranks_on_node=1, available_bytes=available
        )
    validate_node_budget(
        node_budget_config(available - NODE_BUDGET_RESERVE_BYTES),
        ranks_on_node=1,
        available_bytes=available,
    )


def test_staging_counts_against_the_same_budget():
    """It is pinned for the process's lifetime and comes out of the same DRAM."""
    segment = 100 * 1024**3
    available = 128 * 1024**3 + NODE_BUDGET_RESERVE_BYTES
    validate_node_budget(node_budget_config(segment), ranks_on_node=1, available_bytes=available)
    # `both` stages in each direction, so it asks for two pools.
    with pytest.raises(ValueError, match="cannot afford"):
        validate_node_budget(
            node_budget_config(available - NODE_BUDGET_RESERVE_BYTES, stage_through_host=True),
            ranks_on_node=1,
            available_bytes=available,
        )


def test_a_capacity_only_rank_is_charged_no_staging():
    """It opens no staging pool in either direction, because it transfers nothing."""
    available = 128 * 1024**3 + NODE_BUDGET_RESERVE_BYTES
    validate_node_budget(
        node_budget_config(128 * 1024**3, role=StoreRole.CAPACITY, stage_through_host=True),
        ranks_on_node=1,
        available_bytes=available,
    )


def test_a_zero_segment_is_always_affordable():
    """A rank that contributes nothing cannot exhaust anything."""
    validate_node_budget(node_budget_config(0), ranks_on_node=64, available_bytes=1024)


def test_an_unreadable_budget_warns_rather_than_blocks(monkeypatch):
    """Refusing to start because a figure could not be read helps nobody."""
    monkeypatch.setattr(validation_module, "_available_host_memory", lambda: None)
    validate_node_budget(node_budget_config(160 * 1024**3), ranks_on_node=8)


# ---- connector identification ----
#
# py_executor_creator turns partial reuse off for this connector and finds it
# through uses_connector. Failing to recognize the config would silently cost
# the reuse the store exists to provide.


@pytest.mark.parametrize(
    "config, expected",
    [
        (KvCacheConnectorConfig(connector="mooncake-store"), True),
        (
            KvCacheConnectorConfig(
                connector_module="tensorrt_llm._torch.pyexecutor.connectors.mooncake_store",
                connector_scheduler_class="MooncakeStoreConnectorScheduler",
                connector_worker_class="MooncakeStoreConnectorWorker",
            ),
            True,
        ),
        (KvCacheConnectorConfig(connector="kvbm"), False),
        (None, False),
    ],
    ids=["preset", "hand_written_module", "another_connector", "no_connector"],
)
def test_uses_connector_recognizes_the_connector_however_it_is_spelled(config, expected):
    assert uses_connector(config, "mooncake-store") is expected


def test_uses_connector_rejects_an_unknown_preset():
    config = KvCacheConnectorConfig(connector="mooncake-store")
    with pytest.raises(ValueError, match="Unknown connector preset"):
        uses_connector(config, "mooncake-stroe")


# ---- native offload tiers ----
#
# The pool is the deployment's offload tier, so a native one would claim a
# second share of the same node's DRAM.


def test_default_cache_sizes_are_pinned_to_zero():
    """Unset is not the same as no tier: None asks V2 to size one itself."""
    config = KvCacheConfig()
    assert config.host_cache_size is None
    assert config.disk_cache_size is None

    _disable_native_kv_offload(config)

    assert config.host_cache_size == 0
    assert config.disk_cache_size == 0


def test_an_explicitly_sized_host_tier_is_overridden():
    config = KvCacheConfig(host_cache_size=64 * 1024**3)
    _disable_native_kv_offload(config)
    assert config.host_cache_size == 0


def test_an_explicitly_sized_disk_tier_is_overridden():
    config = KvCacheConfig(disk_cache_size=64 * 1024**3, disk_cache_path="/tmp/kv")
    _disable_native_kv_offload(config)
    assert config.disk_cache_size == 0


def test_overriding_an_explicit_size_says_so(caplog):
    """Silently dropping a memory budget someone sized on purpose is a trap."""
    config = KvCacheConfig(host_cache_size=64 * 1024**3)
    with caplog.at_level("WARNING"):
        _disable_native_kv_offload(config)
    assert "host_cache_size" in caplog.text
    assert "segment_size" in caplog.text


def test_leaving_the_defaults_alone_warns_about_nothing(caplog):
    """There is no budget to report having ignored."""
    with caplog.at_level("WARNING"):
        _disable_native_kv_offload(KvCacheConfig())
    assert caplog.text == ""


# ---- worker ----


def test_worker_registers_every_pool_range(store_config, fake_store):
    layout = make_layout(num_groups=2, regions_per_group=2)
    with make_worker(fake_store, layout=layout) as worker:
        assert fake_store.registered == [
            (start, end - start) for start, end in PageAddressing(layout).registration_ranges()
        ]
        assert worker.is_registered


def test_worker_rejects_v1_pool_registration(store_config, fake_store):
    with make_worker(fake_store) as worker:
        with pytest.raises(NotImplementedError, match="KVCacheManagerV2"):
            worker.register_kv_caches(None)


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


def test_worker_save_skips_pages_already_in_the_store(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        hashes = [bytes([index]) * 16 for index in range(2)]
        fake_store.objects.add(worker._namespaces[0].key(hashes[0]))

        worker._put(
            [
                RequestTransfers(
                    1,
                    [PageTransfer(hashes[0], 0, 0), PageTransfer(hashes[1], 0, 1)],
                )
            ]
        )
        assert len(fake_store.put_calls) == 1
        assert fake_store.put_calls[0][0] == [worker._namespaces[0].key(hashes[1])]


def test_worker_reports_a_request_finished_once_its_saves_drain(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        # One submission outstanding: the request is closed but must not be released.
        worker._outstanding_saves[42] = 1
        assert worker.get_finished([42], []) == ([], [])

        worker._outstanding_saves.pop(42)
        assert worker.get_finished([], []) == ([42], [])
        # Reported once only.
        assert worker.get_finished([], []) == ([], [])


def test_worker_reports_a_request_with_no_saves_immediately(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        assert worker.get_finished([9], [5]) == ([9], [5])


def test_worker_shutdown_closes_the_store(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker.shutdown()
        assert fake_store.closed
        # Idempotent: a second call must not raise or reopen anything.
        worker.shutdown()


def test_worker_records_the_segment_it_mounted(store_config, fake_store, tmp_path):
    """Capacity is a sum no participant can see, so each writes down its term."""
    run_dir = tmp_path / "run"

    with make_worker(fake_store, run_dir=str(run_dir)):
        pass

    (record,) = read_segments(str(run_dir))
    assert record.host == "10.0.0.1"
    assert record.segment_size == 1024**3
    assert record.role == "both"
    assert record.model_key == "test-model"


# ---- the capacity-only role ----
#
# Its segment is mounted by the same connector every other rank uses, and every
# path that would then touch the pool is short-circuited. The tests below are
# one per path.


@contextlib.contextmanager
def make_capacity_worker(fake_store, store_config, *, layout=None, run_dir=None):
    """A worker whose role contributes memory and drives no traffic."""
    set_pool_setting(store_config, role="capacity")
    with make_worker(fake_store, layout=layout, run_dir=run_dir) as worker:
        yield worker


def test_capacity_only_says_so(store_config, fake_store):
    with make_capacity_worker(fake_store, store_config) as worker:
        assert worker.capacity_only


def test_a_transferring_worker_is_not_capacity_only(store_config, fake_store):
    with make_worker(fake_store) as worker:
        assert not worker.capacity_only


def test_capacity_only_registers_no_kv_cache(store_config, fake_store):
    """What removes the GPUDirect RDMA dependency from this side entirely."""
    with make_capacity_worker(fake_store, store_config, layout=make_layout()) as worker:
        assert fake_store.registered == []
        assert not worker.is_registered


def test_capacity_only_starts_no_save_thread(store_config, fake_store):
    with make_capacity_worker(fake_store, store_config, layout=make_layout()) as worker:
        assert worker._save_thread is None


def test_capacity_only_opens_no_staging_pool(store_config, fake_store):
    """Staging exists to reach GPU pages; there are none to reach here."""
    set_pool_setting(store_config, stage_through_host=True)

    with make_capacity_worker(fake_store, store_config, layout=make_layout()) as worker:
        assert worker._load_staging is None
        assert worker._save_staging is None
        assert fake_store.registered == []


def test_capacity_only_accepts_the_v1_registration(store_config, fake_store):
    """Addressing nothing means the choice of KV cache manager stops mattering."""
    with make_capacity_worker(fake_store, store_config) as worker:
        worker.register_kv_caches(torch.empty(0))


def test_a_transferring_worker_still_refuses_the_v1_registration(store_config, fake_store):
    with make_worker(fake_store) as worker:
        with pytest.raises(NotImplementedError, match="KVCacheManagerV2"):
            worker.register_kv_caches(torch.empty(0))


def test_capacity_only_looks_nothing_up(store_config, fake_store):
    fake_store.objects.update({"anything"})
    with make_capacity_worker(fake_store, store_config, layout=make_layout()) as worker:
        assert worker.count_prefix_hit([b"a", b"b"]) == 0
        assert fake_store.exist_calls == []


def test_capacity_only_loads_nothing(store_config, fake_store):
    with make_capacity_worker(fake_store, store_config, layout=make_layout()) as worker:
        worker.start_load_kv(None)
        assert fake_store.get_calls == []


def test_capacity_only_saves_nothing(store_config, fake_store):
    with make_capacity_worker(fake_store, store_config, layout=make_layout()) as worker:
        worker.wait_for_save(None)
        assert fake_store.put_calls == []


def test_capacity_only_reports_no_transfers_outstanding(store_config, fake_store):
    """Nothing is in flight, so a request is never held waiting on this rank."""
    with make_capacity_worker(fake_store, store_config, layout=make_layout()) as worker:
        assert worker.get_finished([7], []) == ([7], [])


def test_capacity_only_still_mounts_its_segment(store_config, fake_store, tmp_path):
    """The one thing it does do, and the only reason it exists."""
    run_dir = tmp_path / "run"

    with make_capacity_worker(fake_store, store_config, layout=make_layout(), run_dir=str(run_dir)):
        pass

    (record,) = read_segments(str(run_dir))
    assert record.role == "capacity"
    assert record.segment_size == 1024**3


def test_capacity_only_scheduler_offers_and_saves_nothing(store_config):
    """The leader half of the same short-circuit."""
    set_pool_setting(store_config, role="capacity")

    scheduler = MooncakeStoreConnectorScheduler(make_llm_args())
    scheduler._worker = FakeWorker(hit_blocks=4)
    request = make_request(1, list(range(4 * TOKENS_PER_BLOCK)))

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    output = SimpleNamespace(
        new_requests=[
            SimpleNamespace(
                request_id=1,
                new_tokens=list(range(4 * TOKENS_PER_BLOCK)),
                new_block_ids_by_layer_group={0: [0, 1, 2, 3]},
            )
        ],
        cached_requests=[],
    )
    metadata = scheduler.build_connector_meta(output)
    assert metadata.loads == []
    assert metadata.saves == []


# ---- host staging ----


@contextlib.contextmanager
def make_staged_worker(fake_store, store_config, *, layout, budget=None, batch=None):
    """A worker configured to pass pages through pinned host slots.

    `budget` patches the pinned-memory ceiling rather than setting a field:
    the allocation follows from the layout and the transfer batch now, so the
    ceiling is the only thing left that can bind.
    """
    set_pool_setting(store_config, stage_through_host=True)
    if batch is not None:
        set_pool_setting(store_config, transfer_batch_size=batch)
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


def test_staging_registers_host_buffers_and_never_the_pools(
    store_config, fake_store, staged_copies
):
    layout = make_layout(regions_per_group=2)
    with make_staged_worker(fake_store, store_config, layout=layout) as worker:
        # Registering the pools is the step that needs GPUDirect, so staging
        # must not do it at all.
        pool_ranges = PageAddressing(layout).registration_ranges()
        registered_starts = {address for address, _size in fake_store.registered}
        assert registered_starts.isdisjoint({start for start, _end in pool_ranges})

        # One pinned buffer per direction, since the default role is `both`.
        assert len(fake_store.registered) == 2
        assert registered_starts == {
            worker._load_staging.slot_address(0),
            worker._save_staging.slot_address(0),
        }


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


def test_staging_is_sized_from_the_layout_and_the_batch(store_config, fake_store):
    """The allocation follows from the geometry rather than from a field.

    A slot holds the largest page any layer group produces, and no more slots
    are useful than one store call has pages, so both numbers follow from the
    registered layout. A configured size could only be too small, which costs
    throughput quietly by narrowing the batch.
    """
    layout = make_layout(regions_per_group=2)
    page_bytes = PageAddressing(layout).bytes_per_page(0)

    with make_staged_worker(fake_store, store_config, layout=layout, batch=3) as worker:
        assert worker._save_staging.slot_bytes == page_bytes
        assert worker._save_staging.num_slots == 3
        assert worker._batch_size == 3


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


def test_scheduler_offers_the_stored_prefix(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=2)
    request = make_request(1, list(range(5 * TOKENS_PER_BLOCK)))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (2 * TOKENS_PER_BLOCK, False)


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


def test_scheduler_offers_nothing_as_a_producer(store_config):
    set_pool_setting(store_config, role="producer")
    scheduler = make_scheduler(store_config, hit_blocks=2)
    request = make_request(1, list(range(5 * TOKENS_PER_BLOCK)))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert scheduler._worker.queries == []


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


def test_scheduler_saves_nothing_as_a_consumer(store_config):
    set_pool_setting(store_config, role="consumer")
    scheduler = make_scheduler(store_config, hit_blocks=0)
    tokens = list(range(2 * TOKENS_PER_BLOCK))
    request = make_request(1, tokens)
    scheduler.get_num_new_matched_tokens(request, 0)
    metadata = scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, [4, 5])])
    )
    assert metadata.saves == []


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
