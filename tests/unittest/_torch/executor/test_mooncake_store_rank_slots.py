# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel ranks may put the same block on different page slots (pgfix, lanes/mc/STORE_PAGE_RACE.md).

KV cache manager V2 prefers a recycled slot whose last GPU use has finished over a never-used one, so which slot a
block gets depends on each rank's GPU progress. The connector metadata is built on rank 0 and broadcast; a worker
must map every block ordinal back to its own slot before it loads or saves.

CPU only: the store is the in-process fake from test_mooncake_store_connector.
"""

from types import SimpleNamespace

import pytest
from test_mooncake_store_connector import (  # noqa: F401 (fixtures)
    TOKENS_PER_BLOCK,
    fake_store,
    make_layout,
    make_request,
    make_scheduler,
    make_staged_worker,
    make_worker,
    request_data,
    staged_copies,
    store_config,
)

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import SchedulerOutput
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.addressing import PageAddressing
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    MooncakeStoreMetadata,
    PageTransfer,
    PrefixLoadTransfers,
    RequestTransfers,
)

pytestmark = pytest.mark.cpu_only

HASH = b"\x05" * 16


def test_scheduler_tags_every_page_with_its_block_ordinal(store_config):
    scheduler = make_scheduler(store_config, hit_blocks=2)
    tokens = list(range(5 * TOKENS_PER_BLOCK))
    scheduler.get_num_new_matched_tokens(make_request(1, tokens), 0)
    output = SchedulerOutput(new_requests=[request_data(1, tokens, [10, 11, 12, 13, 14])])
    metadata = scheduler.build_connector_meta(output)
    assert [(p.page_index, p.block_ordinal) for p in metadata.loads[0].pages] == [(10, 0), (11, 1)]
    assert [(p.page_index, p.block_ordinal) for p in metadata.saves[0].pages] == [(12, 2), (13, 3), (14, 4)]


def test_narrowing_does_not_replay_deltas(store_config):
    """release_unstarted_prefix_loads asks again for the trimmed output: ordinals must not shift."""
    scheduler = make_scheduler(store_config, hit_blocks=0)
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    for request_id in (1, 2):
        scheduler.get_num_new_matched_tokens(make_request(request_id, tokens), 0)
    output = SchedulerOutput(
        new_requests=[request_data(1, tokens, [10, 11, 12]), request_data(2, tokens, [20, 21, 22])]
    )
    first = scheduler.build_connector_meta(output)
    pages_before = {rid: list(scheduler._requests[rid].pages[0]) for rid in (1, 2)}
    tokens_before = {rid: len(scheduler._requests[rid].tokens) for rid in (1, 2)}
    output.new_requests = [item for item in output.new_requests if item.request_id != 2]
    narrowed = scheduler.narrow_connector_meta(output)
    assert {entry.request_id for entry in first.saves} == {1, 2}
    assert [entry.request_id for entry in narrowed.saves] == [1]
    assert narrowed.saves[0].pages == first.saves[0].pages
    assert {rid: list(scheduler._requests[rid].pages[0]) for rid in (1, 2)} == pages_before
    assert {rid: len(scheduler._requests[rid].tokens) for rid in (1, 2)} == tokens_before
    # A different output object is built normally.
    other = SchedulerOutput(new_requests=[])
    assert scheduler.narrow_connector_meta(other).saves == []


def test_localize_uses_this_ranks_slots(store_config, fake_store):
    # Rank 0 put request 7's blocks 0..2 on slots 1, 2, 3; this rank on 1, 6, 3.
    tables = {7: [[1, 6, 3]]}
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker.bind_local_page_indices(lambda request_id: tables[request_id])
        metadata = MooncakeStoreMetadata(
            loads=[RequestTransfers(7, [PageTransfer(HASH, 0, 1, 0)])],
            saves=[RequestTransfers(7, [PageTransfer(HASH, 0, 2, 1), PageTransfer(HASH, 0, 3, 2)])],
            prefix_loads=[PrefixLoadTransfers(4, True, RequestTransfers(7, [PageTransfer(HASH, 0, 2, 1)]))],
        )
        worker._localize(metadata)
        assert [p.page_index for p in metadata.loads[0].pages] == [1]
        assert [p.page_index for p in metadata.saves[0].pages] == [6, 3]
        assert [p.page_index for p in metadata.prefix_loads[0].transfers.pages] == [6]
        assert metadata.prefix_loads[0].reservation_id == 4 and metadata.prefix_loads[0].is_async
        assert worker._remapped_pages == 2
        # Once per metadata object: a second call (start_load_kv, then wait_for_save) changes nothing.
        tables[7] = [[5, 5, 5]]
        worker._localize(metadata)
        assert [p.page_index for p in metadata.saves[0].pages] == [6, 3]


def test_a_transfer_without_an_ordinal_keeps_its_slot(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker.bind_local_page_indices(lambda request_id: [[4, 4, 4]])
        metadata = SimpleNamespace(loads=[RequestTransfers(7, [PageTransfer(HASH, 0, 2)])], saves=[])
        worker._localize(metadata)
        assert [p.page_index for p in metadata.loads[0].pages] == [2]


def test_without_a_resolver_nothing_changes(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        assert worker.get_local_page_indices() is None
        metadata = MooncakeStoreMetadata(saves=[RequestTransfers(7, [PageTransfer(HASH, 0, 2, 1)])])
        worker._localize(metadata)
        assert [p.page_index for p in metadata.saves[0].pages] == [2]


def test_a_save_page_with_no_slot_here_is_dropped(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        # This rank's cache for request 7 is gone (or the block has no page): never save someone else's page.
        worker.bind_local_page_indices(lambda request_id: [[1, -1]])
        metadata = MooncakeStoreMetadata(
            saves=[RequestTransfers(7, [PageTransfer(HASH, 0, 1, 0), PageTransfer(HASH, 0, 2, 1)])]
        )
        worker._localize(metadata)
        assert [p.page_index for p in metadata.saves[0].pages] == [1]
        assert worker._unresolved_save_pages == 1


def test_a_load_with_no_slot_here_raises(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker.bind_local_page_indices(lambda request_id: [[]])
        metadata = MooncakeStoreMetadata(loads=[RequestTransfers(7, [PageTransfer(HASH, 0, 1, 0)])])
        with pytest.raises(RuntimeError, match="no page for block 0"):
            worker._localize(metadata)


def test_a_load_lands_in_this_ranks_slot(store_config, fake_store, staged_copies):
    layout = make_layout(regions_per_group=2)
    with make_staged_worker(fake_store, store_config, layout=layout) as worker:
        fake_store.objects.add(worker._namespaces[0].key(HASH))
        # Rank 0's slot for block 3 is 5; this rank's is 2.
        worker.bind_local_page_indices(lambda request_id: [[0, 0, 0, 2]])
        worker.bind_connector_meta(
            SimpleNamespace(loads=[RequestTransfers(7, [PageTransfer(HASH, 0, 5, 3)])], saves=[])
        )
        worker.start_load_kv(None)
        device_addresses, _ = PageAddressing(layout).buffers(0, 2)
        assert [dst for dst, _, _ in staged_copies] == device_addresses
        rank0_addresses, _ = PageAddressing(layout).buffers(0, 5)
        assert not set(rank0_addresses) & {dst for dst, _, _ in staged_copies}


def test_a_save_reads_this_ranks_slot(store_config, fake_store, staged_copies):
    layout = make_layout(regions_per_group=2)
    with make_staged_worker(fake_store, store_config, layout=layout) as worker:
        # Rank 0's slot for block 1 is 4; this rank's is 6.
        worker.bind_local_page_indices(lambda request_id: [[0, 6]])
        metadata = MooncakeStoreMetadata(saves=[RequestTransfers(9, [PageTransfer(HASH, 0, 4, 1)])])
        worker._localize(metadata)
        worker._put(metadata.saves)
        assert len(fake_store.put_calls) == 1
        # The gather reads this rank's page, never rank 0's slot.
        local, _ = PageAddressing(layout).buffers(0, 6)
        rank0, _ = PageAddressing(layout).buffers(0, 4)
        assert [src for _, src, _ in staged_copies] == local
        assert not set(rank0) & {src for _, src, _ in staged_copies}


class _Stream:
    def __init__(self, handle):
        self.cuda_stream = handle


def test_hand_off_also_orders_behind_the_execution_stream(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout()) as worker:
        current, execution = _Stream(11), _Stream(22)
        assert worker._handoff_streams(current) == [current]
        worker.bind_execution_stream(execution)
        assert worker._handoff_streams(current) == [current, execution]
        # The same stream is waited on once.
        assert len(worker._handoff_streams(_Stream(22))) == 1


def test_narrowing_keeps_the_window_retirements_of_the_kept_requests(store_config):
    """Window retirement (TRTLLM_MOONCAKE_STORE_WINDOW_KEEP) lists prompt ends in the metadata; narrowing keeps them."""
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import WindowRetire

    scheduler = make_scheduler(store_config, hit_blocks=0)
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    for request_id in (1, 2):
        scheduler.get_num_new_matched_tokens(make_request(request_id, tokens), 0)
    output = SchedulerOutput(
        new_requests=[request_data(1, tokens, [10, 11, 12]), request_data(2, tokens, [20, 21, 22])]
    )
    first = scheduler.build_connector_meta(output)
    first.retires = [WindowRetire(1, 2, HASH, 1, [HASH]), WindowRetire(2, 2, HASH, 1, [HASH])]
    output.new_requests = [item for item in output.new_requests if item.request_id != 2]
    narrowed = scheduler.narrow_connector_meta(output)
    assert [item.request_id for item in narrowed.retires] == [1]
    assert narrowed.retires[0] is first.retires[0]


def test_handle_metadata_binds_this_ranks_resolver():
    """The manager hands the worker the KV cache manager's own slots (unmasked) with every metadata."""
    from tensorrt_llm._torch.pyexecutor.connectors import kv_cache_connector as kvc

    calls = []
    bound = {}
    resolver = lambda request_id: calls.append(request_id) or [[9]]  # noqa: E731
    manager = SimpleNamespace(
        _scheduler_output=SimpleNamespace(prefix_loads=[]),
        _bound_prefix_load_ids=set(),
        _bound_scheduler_output=None,
        _allocation_fence=None,
        _execution_stream="exec",
        _local_page_indices=resolver,
        _run_on_leader=lambda f: f(),
        scheduler=SimpleNamespace(build_connector_meta=lambda output: "meta"),
        worker=SimpleNamespace(
            bind_allocation_fence=lambda fence: None,
            bind_execution_stream=lambda stream: bound.__setitem__("stream", stream),
            bind_local_page_indices=lambda r: bound.__setitem__("resolver", r),
            bind_connector_meta=lambda meta: bound.__setitem__("meta", meta),
        ),
    )
    kvc.KvCacheConnectorManager.handle_metadata(manager)
    assert bound == {"stream": "exec", "resolver": resolver, "meta": "meta"}
    assert bound["resolver"](3) == [[9]] and calls == [3]
