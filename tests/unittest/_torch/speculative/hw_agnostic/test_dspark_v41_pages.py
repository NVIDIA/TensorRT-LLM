# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V4.1 retained draft pages and bounded context seed lifecycle."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import device_copy
from tensorrt_llm._torch.speculative.dspark_v41 import DSv41DSparkWorker
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, ReusePolicy

pytestmark = pytest.mark.cpu_only


def test_context_worker_rebinds_managed_pages_without_draft_decode():
    spec = SimpleNamespace(
        num_kv_heads=1,
        kv_factor=1,
        head_dim=512,
        sliding_window_size=129,
        reuse_policy=ReusePolicy.PRIVATE,
    )
    manager = SimpleNamespace(
        draft_layer_ids=(2, 3, 4),
        get_layer_cache_spec=lambda _: spec,
        get_buffers=Mock(return_value=object()),
        max_blocks_per_seq=8,
        draft_page_table=lambda: torch.empty((2, 8), dtype=torch.int32),
    )
    manager.get_draft_kv_cache_view = lambda: manager
    worker = SimpleNamespace(
        _lazy_init=Mock(),
        _draft_kv_manager=object(),
        max_draft_len=5,
        _win=128,
        _batch_to_slot=torch.empty(2),
        _ctx_len=torch.empty(3),
        _draft_page_versions={0: object()},
        _draft_prepared_batch=object(),
        _draft_page_indices={0: (0, (1,))},
        _managed_residency={42: object()},
        _uploaded_slots=(0,),
        _uploaded_capacities=(128,),
        _table_slots=(0,),
        _prepare_page_tables=Mock(),
        get_draft_kv_cache_manager=lambda _: manager,
    )
    metadata = SimpleNamespace(request_ids=[42])
    draft = SimpleNamespace(
        num_stages=3,
        _attn_params={"head_dim": 512},
        dspark_model=SimpleNamespace(context_kv_only=True),
    )
    DSv41DSparkWorker.prepare_draft_context(
        worker,
        draft,
        metadata,
        SimpleNamespace(num_contexts=1),
        SimpleNamespace(get_resource_manager=lambda _: manager),
    )
    assert worker._draft_kv_manager is manager
    assert not worker._draft_page_versions
    assert worker._draft_prepared_batch is None
    assert not worker._draft_page_indices
    assert not worker._managed_residency
    assert manager.get_buffers.call_count == 3
    assert worker._draft_slot_block_tables.shape == (3, 8)
    assert worker._uploaded_slots is worker._uploaded_capacities is worker._table_slots is None
    worker._prepare_page_tables.assert_called_once_with([42], 1)
    for return_tokens in (False, True):
        proposals = DSv41DSparkWorker._draft_gen_block_batched(
            worker,
            draft,
            metadata,
            None,
            torch.empty((1, 6), dtype=torch.int32),
            None,
            num_contexts=0,
            batch_size=1,
            total_target_tokens=6,
            position_ids=None,
            return_tokens=return_tokens,
        )
        if return_tokens:
            assert proposals.dtype == torch.int32 and proposals.tolist() == [[0] * 5]
        else:
            assert proposals is None


def test_retained_pages_preserve_context_holes_and_reject_missing_generation_kv():
    manager = SimpleNamespace(
        tokens_per_block=128,
        max_blocks_per_seq=16,
        layer_offsets={8: 11, 9: 12},
        layer_to_pool_mapping_dict={11: 3, 12: 3},
        get_layer_page_index_scale=lambda _: 2,
        _get_cache_view_page_stride=lambda scale, factor: 1,
    )
    worker = SimpleNamespace(_draft_kv_manager=manager, _draft_layer_ids=(8, 9), _win=128)
    indices = [BAD_PAGE_INDEX] * 4 + [7, 2]
    cache = SimpleNamespace(
        history_length=640, num_blocks=6, get_base_page_indices=lambda pool: indices
    )
    assert DSv41DSparkWorker._page_indices(worker, cache, False) == (4, (14, 4))
    indices[4] = BAD_PAGE_INDEX
    assert DSv41DSparkWorker._page_indices(worker, cache, True) == (4, (BAD_PAGE_INDEX, 4))
    with pytest.raises(ValueError, match="missing retained draft KV"):
        DSv41DSparkWorker._page_indices(worker, cache, False)


def test_unchanged_private_pages_skip_staging_events_and_scratch_resets(monkeypatch):
    class StablePages(SimpleNamespace):
        def __getattribute__(self, name):
            if name in ("history_length", "num_blocks"):
                pytest.fail("Unchanged native page versions must not re-read page geometry")
            return super().__getattribute__(name)

    cache = StablePages(history_length=140, page_storage_version=3, num_blocks=2, is_active=True)
    manager = SimpleNamespace(
        kv_cache_map={42: cache}, tokens_per_block=128, draft_page_table_prepared=lambda _: False
    )
    worker = SimpleNamespace(
        _draft_kv_manager=manager,
        _draft_reuse_policy=ReusePolicy.PRIVATE,
        _graph_dummy_id_floor=1 << 60,
        _req_to_slot={42: 0},
        _managed_residency={42: cache},
        _scratch_slot=1,
        _win=128,
        _draft_capacities=torch.tensor([256]),
        _draft_page_versions={0: (cache, 3, 0, 2, False)},
        _draft_prepared_batch=None,
        _draft_page_indices={0: (0, (7, 8))},
        _uploaded_slots=(0,),
        _uploaded_capacities=(256,),
        _table_slots=(0,),
        _draft_block_tables=None,
        _draft_managed_block_tables=None,
        _draft_fallback_block_tables=None,
        _ctx_len=torch.tensor([140, 17]),
        _valid_len=torch.tensor([128, 9]),
    )
    worker._assign_slot = DSv41DSparkWorker._assign_slot.__get__(worker)

    def unexpected_work(*args, **kwargs):
        pytest.fail("Unchanged PRIVATE metadata performed staging or scratch-reset work")

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "Event", unexpected_work)
        patch.setattr(torch, "empty", unexpected_work)
        patch.setattr(torch, "tensor", unexpected_work)
        patch.setattr(torch.Tensor, "zero_", unexpected_work)
        for history in (140, 141, 142):
            cache.history_length = history
            DSv41DSparkWorker._prepare_page_tables(worker, [42], 0)
    assert worker._ctx_len.tolist() == [140, 17]
    assert worker._valid_len.tolist() == [128, 9]


def test_private_fallback_pages_preserve_execution_order_and_managed_pages_skip_copies(monkeypatch):
    version_reads = []

    class Cache(SimpleNamespace):
        def __getattribute__(self, name):
            if name == "page_storage_version":
                version_reads.append(name)
            return super().__getattribute__(name)

    caches = {
        42: Cache(history_length=140, page_storage_version=1, num_blocks=2, is_active=True),
        43: Cache(history_length=268, page_storage_version=1, num_blocks=3, is_active=True),
    }
    tables = {id(caches[42]): (0, (7, 8)), id(caches[43]): (1, (9, 10))}
    storage = torch.full((6, 8), BAD_PAGE_INDEX, dtype=torch.int32)
    worker = SimpleNamespace(
        _draft_kv_manager=SimpleNamespace(
            kv_cache_map=caches, tokens_per_block=128, draft_page_table_prepared=lambda _: False
        ),
        _draft_reuse_policy=ReusePolicy.PRIVATE,
        _graph_dummy_id_floor=1 << 60,
        _req_to_slot={42: 0, 43: 1},
        _managed_residency=dict(caches),
        _free_slots=deque(),
        _scratch_slot=2,
        _win=128,
        _ctx_len=torch.tensor([140, 268, 0]),
        _valid_len=torch.tensor([128, 128, 0]),
        _position_initialized=torch.ones(3, dtype=torch.bool),
        _kv_windows=None,
        _confidence_logits=None,
        _draft_capacities=torch.zeros(3, dtype=torch.long),
        _batch_to_slot=torch.full((3,), 2, dtype=torch.long),
        _draft_table_storage=storage,
        _draft_slot_block_tables=storage[:3],
        _draft_block_tables=storage[3:],
        _draft_managed_block_tables=storage[3:],
        _draft_fallback_block_tables=storage[3:],
        _draft_page_versions={},
        _draft_prepared_batch=None,
        _draft_page_indices={},
        _draft_metadata_host_buffers=[],
        _draft_updates_device=None,
        _draft_indices_device=None,
        _uploaded_slots=None,
        _uploaded_capacities=None,
        _table_slots=None,
        _page_indices=Mock(side_effect=lambda cache, _: tables[id(cache)]),
    )
    worker._assign_slot = DSv41DSparkWorker._assign_slot.__get__(worker)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(query=lambda: True, record=lambda: None)
    )
    monkeypatch.setattr("tensorrt_llm._torch.speculative.dspark_v41.prefer_pinned", lambda: False)
    worker._clear_slot = DSv41DSparkWorker._clear_slot.__get__(worker)
    gather = Mock(wraps=torch.index_select)
    monkeypatch.setattr(torch, "index_select", gather)
    upload = Mock(wraps=device_copy.copy)
    monkeypatch.setattr(device_copy, "copy", upload)
    DSv41DSparkWorker._prepare_page_tables(worker, [42, 43], 0)
    assert worker._draft_block_tables[0, :2].tolist() == [7, 8]
    assert worker._draft_block_tables[1, 1:3].tolist() == [9, 10]
    DSv41DSparkWorker._prepare_page_tables(worker, [43, 42], 0)
    assert worker._draft_block_tables[0, 1:3].tolist() == [9, 10]
    assert worker._draft_block_tables[1, :2].tolist() == [7, 8]
    assert worker._batch_to_slot.tolist() == [1, 0, 2]
    assert worker._draft_capacities.tolist() == [384, 256, 0]
    assert worker._page_indices.call_count == gather.call_count == 2

    caches[42].page_storage_version += 1
    tables[id(caches[42])] = (0, (7, 11))
    DSv41DSparkWorker._prepare_page_tables(worker, [43, 42], 0)
    assert worker._draft_slot_block_tables[0, :2].tolist() == [7, 11]
    assert worker._draft_block_tables[1, :2].tolist() == [7, 11]
    assert worker._page_indices.call_count == 3
    assert gather.call_count == 3

    DSv41DSparkWorker._prepare_page_tables(worker, [43], 0)
    assert worker._batch_to_slot.tolist() == [1, 2, 2]
    assert worker._draft_capacities.tolist() == [384, 0, 0]
    assert gather.call_count == 4

    # A narrower window can advance inside one native history block.
    worker._draft_kv_manager.tokens_per_block = 256
    caches[43].history_length = 383
    caches[43].num_blocks = 2
    tables[id(caches[43])] = (0, (9, 10))
    DSv41DSparkWorker._prepare_page_tables(worker, [43], 0)
    calls = worker._page_indices.call_count
    caches[43].history_length = 384
    tables[id(caches[43])] = (1, (10,))
    DSv41DSparkWorker._prepare_page_tables(worker, [43], 0)
    assert worker._page_indices.call_count == calls + 1
    assert worker._draft_page_indices[1] == (1, (10,))

    # Once target preparation published this execution order, draft generation
    # uses its addressable extent without uploading per-request block growth.
    worker._draft_kv_manager.draft_page_table_prepared = lambda _: True
    snapshot = [None]
    worker._draft_kv_manager.draft_page_table_state = lambda: snapshot[0]
    calls, gathers = worker._page_indices.call_count, gather.call_count
    native_uploads = None
    for blocks in (2, 3, 3):
        caches[43].num_blocks = blocks
        caches[43].page_storage_version += 1
        snapshot[0] = object()
        version_reads.clear()
        DSv41DSparkWorker._prepare_page_tables(worker, [43], 0)
        assert not version_reads
        assert worker._draft_capacities.tolist() == [8 * 256, 0, 0]
        if native_uploads is None:
            native_uploads = upload.call_count
        else:
            assert upload.call_count == native_uploads
        # The target's unchanged snapshot also covers the slot and capacity
        # metadata. Reading native properties again would duplicate its work.
        with monkeypatch.context() as patch:
            patch.setattr(worker._draft_kv_manager, "kv_cache_map", {})
            DSv41DSparkWorker._prepare_page_tables(worker, [43], 0)
        assert worker._batch_to_slot.tolist() == [1, 2, 2]
        assert worker._draft_capacities.tolist() == [8 * 256, 0, 0]
    assert worker._page_indices.call_count == calls
    assert gather.call_count == gathers
    # A different request still needs its own slot even if it aliases the
    # same cache and the target's physical page conversion can be reused.
    caches[44] = caches[43]
    DSv41DSparkWorker._prepare_page_tables(worker, [44], 0)
    assert 43 not in worker._req_to_slot
    assert 44 in worker._req_to_slot


@pytest.mark.parametrize(
    "chunks,captures",
    [([180], [180]), ([100, 80], [100, 80]), ([40, 140], [0, 140])],
    ids=["single", "chunked", "held-tail"],
)
def test_private_context_seed_rebuilds_final_window(chunks, captures):
    worker = SimpleNamespace(
        _draft_reuse_policy=ReusePolicy.PRIVATE,
        _draft_kv_manager=SimpleNamespace(enable_block_reuse=True),
        _win=128,
        _req_to_slot={42: 0},
        _scratch_slot=1,
        _ctx_len=torch.zeros(2, dtype=torch.long),
        _valid_len=torch.zeros(2, dtype=torch.long),
        _batch_to_slot=torch.tensor([0]),
        _draft_kv_buffers=(),
        _draft_block_tables=torch.zeros((1, 2), dtype=torch.int32),
        _draft_capacities=torch.tensor([256]),
    )
    written = {}

    def write(hidden, frames, mask, *args):
        for frame, value in zip(frames.flatten().tolist(), hidden.flatten().tolist()):
            written[frame - 1] = value

    draft = SimpleNamespace(write_context_pages=Mock(side_effect=write))
    offset = 0
    for length, capture_len in zip(chunks, captures):
        positions = torch.arange(offset, offset + length)
        metadata = SimpleNamespace(
            context_capture_lens=(capture_len,),
            request_ids=[42],
            get_hidden_states=lambda count: positions[:, None].float(),
        )
        DSv41DSparkWorker._seed_context_windows(
            worker,
            draft,
            metadata,
            SimpleNamespace(num_contexts=1, _seq_lens=[length]),
            positions,
            length,
        )
        offset += length
        if capture_len == 0:
            assert not written
            assert worker._ctx_len[0] == offset
            assert worker._valid_len[0] == 0
    assert worker._ctx_len[0] == 180
    assert worker._valid_len[0] == 128
    assert [written[token] for token in range(52, 180)] == list(range(52, 180))
