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
from types import SimpleNamespace

import pytest

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
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.addressing import (
    PageAddressing,
    merge_intervals,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    MooncakeStoreConnectorConfig,
    StoreRole,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import (
    BlockHashChain,
    KeyNamespace,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    PageTransfer,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler import (
    MooncakeStoreConnectorScheduler,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.validation import (
    validate_layout,
    validate_llm_args,
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
        #: Workers built against this store. ``make_worker`` shuts each one
        #: down; the fixture repeats it as a backstop for early failures.
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
    monkeypatch.delenv("TRTLLM_MOONCAKE_STORE_ROLE", raising=False)
    monkeypatch.delenv("TRTLLM_MOONCAKE_STORE_PREFIX", raising=False)
    monkeypatch.delenv("TRTLLM_MOONCAKE_STORE_MODEL_KEY", raising=False)
    return path


def make_llm_args():
    return SimpleNamespace(
        model="/models/test-model",
        kv_cache_config=SimpleNamespace(tokens_per_block=TOKENS_PER_BLOCK),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        sparse_attention_config=None,
    )


@pytest.fixture
def fake_store(monkeypatch):
    """Replace the store handle, and tear down any worker a test builds."""
    store = FakeStore()
    monkeypatch.setattr(worker_module, "_open_store", lambda _config: store)
    yield store
    for worker in store.workers:
        worker.shutdown()
    worker_module._LOCAL_WORKER = None
    worker_module._LOCAL_WORKER_READY.clear()


@contextlib.contextmanager
def make_worker(fake_store, *, layout=None):
    """Build a worker and shut it down before the test call phase ends.

    Registering a layout starts the background save thread, and
    pytest-threadleak snapshots threads around the call phase only, so
    fixture teardown would run too late to keep it quiet.
    """
    worker = MooncakeStoreConnectorWorker(make_llm_args())
    fake_store.workers.append(worker)
    if layout is not None:
        worker.register_kv_cache_layout(layout)
    try:
        yield worker
    finally:
        worker.shutdown()


def make_request(request_id, tokens, cache_salt=None):
    return SimpleNamespace(
        request_id=request_id,
        cache_salt=cache_salt,
        get_tokens=lambda _beam=0, _tokens=tuple(tokens): list(_tokens),
    )


# ---- keys ----


def test_hash_chain_is_deterministic_and_prefix_sensitive():
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    first = list(BlockHashChain(TOKENS_PER_BLOCK).extend(tokens))
    second = list(BlockHashChain(TOKENS_PER_BLOCK).extend(tokens))
    assert first == second

    # Changing a token in block 0 must change every hash after it, which is what
    # makes a key safe to share: a hit implies the whole prefix matched.
    altered = list(tokens)
    altered[0] += 1
    changed = list(BlockHashChain(TOKENS_PER_BLOCK).extend(altered))
    assert all(a != b for a, b in zip(first, changed))


def test_hash_chain_ignores_partial_trailing_block():
    full = list(range(2 * TOKENS_PER_BLOCK))
    chain = BlockHashChain(TOKENS_PER_BLOCK)
    assert len(chain.extend(full)) == 2
    assert len(chain.extend(full + [99])) == 2


def test_hash_chain_extends_incrementally():
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    incremental = BlockHashChain(TOKENS_PER_BLOCK)
    for end in range(0, len(tokens) + 1, TOKENS_PER_BLOCK):
        incremental.extend(tokens[:end])
    assert list(incremental.hashes) == list(BlockHashChain(TOKENS_PER_BLOCK).extend(tokens))


def test_hash_chain_separates_cache_salts():
    tokens = list(range(TOKENS_PER_BLOCK))
    unsalted = BlockHashChain(TOKENS_PER_BLOCK).extend(tokens)
    salted = BlockHashChain(TOKENS_PER_BLOCK, cache_salt="tenant-a").extend(tokens)
    other = BlockHashChain(TOKENS_PER_BLOCK, cache_salt="tenant-b").extend(tokens)
    assert unsalted[0] != salted[0] != other[0]
    assert salted[0] != other[0]


def test_hash_chain_rejects_shrinking_token_list():
    chain = BlockHashChain(TOKENS_PER_BLOCK)
    chain.extend(list(range(2 * TOKENS_PER_BLOCK)))
    with pytest.raises(ValueError, match="shrank"):
        chain.extend(list(range(TOKENS_PER_BLOCK)))


def test_key_namespace_separates_every_dimension():
    base = dict(
        cache_prefix="trtllm",
        model_key="m",
        rank=0,
        world_size=2,
        layer_group_id=0,
        tokens_per_block=32,
        bytes_per_page=1024,
    )
    block_hash = b"\x01" * 16
    reference = KeyNamespace(**base).key(block_hash)
    for field, value in [
        ("cache_prefix", "other"),
        ("model_key", "n"),
        ("rank", 1),
        ("world_size", 4),
        ("layer_group_id", 1),
        ("tokens_per_block", 64),
        ("bytes_per_page", 2048),
    ]:
        assert KeyNamespace(**{**base, field: value}).key(block_hash) != reference


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


# ---- config ----


def test_config_reads_sizes_with_units(store_config):
    config = MooncakeStoreConnectorConfig.from_env()
    assert config.global_segment_size == 1024**3
    assert config.local_buffer_size == 256 * 1024**2
    assert config.role is StoreRole.BOTH
    assert config.resolve_model_key("/models/ignored") == "test-model"


def test_config_role_comes_from_environment(store_config, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_ROLE", "producer")
    config = MooncakeStoreConnectorConfig.from_env()
    assert config.role is StoreRole.PRODUCER
    assert config.role.saves and not config.role.loads

    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_ROLE", "consumer")
    config = MooncakeStoreConnectorConfig.from_env()
    assert config.role.loads and not config.role.saves

    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_ROLE", "nonsense")
    with pytest.raises(ValueError, match="TRTLLM_MOONCAKE_STORE_ROLE"):
        MooncakeStoreConnectorConfig.from_env()


def test_config_requires_the_env_var(monkeypatch):
    monkeypatch.delenv("MOONCAKE_CONFIG_PATH", raising=False)
    with pytest.raises(ValueError, match="MOONCAKE_CONFIG_PATH"):
        MooncakeStoreConnectorConfig.from_env()


def test_config_model_key_defaults_to_basename(store_config, tmp_path, monkeypatch):
    path = tmp_path / "no_model_key.json"
    path.write_text(json.dumps({"master_server_address": "127.0.0.1:50051"}))
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    config = MooncakeStoreConnectorConfig.from_env()
    assert config.resolve_model_key("/models/MiniMax-M3/") == "MiniMax-M3"


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
        # Block 1 missing: block 2 is unusable even though it is present, because a
        # prefix is replayed contiguously.
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


def request_data(request_id, new_tokens, page_indices, layer_group_id=0):
    return RequestData(
        request_id=request_id,
        new_tokens=list(new_tokens),
        new_block_ids=list(page_indices),
        computed_position=0,
        num_scheduled_tokens=len(new_tokens),
        new_block_ids_by_layer_group={layer_group_id: list(page_indices)},
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


def test_scheduler_offers_nothing_as_a_producer(store_config, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_ROLE", "producer")
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


def test_scheduler_saves_nothing_as_a_consumer(store_config, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_ROLE", "consumer")
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
    data.new_block_ids_by_layer_group[1] = [7, BAD_PAGE_INDEX]
    metadata = scheduler.build_connector_meta(SchedulerOutput(new_requests=[data]))

    # Block 1 has no page in group 1, so neither of its halves is stored; block 0
    # contributes one page per group.
    assert [(page.layer_group_id, page.page_index) for page in metadata.saves[0].pages] == [
        (0, 4),
        (1, 7),
    ]


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


def test_scheduler_isolates_requests_by_cache_salt(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=1)
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    scheduler.get_num_new_matched_tokens(make_request(1, tokens, cache_salt="a"), 0)
    scheduler.get_num_new_matched_tokens(make_request(2, tokens, cache_salt="b"), 0)
    assert scheduler._worker.queries[0] != scheduler._worker.queries[1]
