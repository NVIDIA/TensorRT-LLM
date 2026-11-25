# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V4.1 retained draft pages and bounded context seed lifecycle."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import device_copy
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.speculative.dspark_v41 import DSparkCopySite, DSv41DSparkWorker
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
        _prewarm_draft_pages=Mock(),
        _draft_kv_manager=object(),
        max_draft_len=5,
        _win=128,
        _batch_to_slot=torch.empty(2),
        _ctx_len=torch.empty(3),
        _draft_page_versions={0: object()},
        _draft_prepared_batch=object(),
        _draft_page_indices={0: (0, (1,))},
        _managed_residency={42: object()},
        _chunking_ids={42},
        _table_slots=(0,),
        _prepare_page_tables=Mock(),
        get_draft_kv_cache_manager=lambda _: manager,
        return_confidence=True,
    )
    metadata = SimpleNamespace(request_ids=[42], context_chunking_ids=frozenset())
    draft = SimpleNamespace(
        num_stages=3,
        _attn_params={"head_dim": 512},
        dspark_model=SimpleNamespace(context_kv_only=True),
    )

    def initialize(*args):
        assert not worker.return_confidence

    worker._lazy_init.side_effect = initialize
    DSv41DSparkWorker.prepare_draft_context(
        worker,
        draft,
        metadata,
        SimpleNamespace(num_contexts=1),
        SimpleNamespace(get_resource_manager=lambda _: manager),
    )
    assert worker._draft_kv_manager is manager
    assert worker._draft_prepared_batch is None
    assert not worker._managed_residency
    assert torch.all(worker._draft_block_tables == BAD_PAGE_INDEX)
    worker._prepare_page_tables.assert_called_once_with(
        [42], 1, paused_in_place_ids=frozenset(), context_chunking_ids=frozenset()
    )
    for return_tokens in (False, True):
        proposals = DSv41DSparkWorker._draft_gen_block_batched(
            worker,
            draft,
            SimpleNamespace(request_ids=[CUDA_GRAPH_DUMMY_REQUEST_ID]),
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
            assert proposals.dtype == torch.int32 and proposals.shape == (1, 5)
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


def test_private_page_tables_follow_request_lifetime_and_batch_order(monkeypatch):
    caches = {
        42: SimpleNamespace(
            history_length=140, page_storage_version=1, num_blocks=2, is_active=True
        ),
        43: SimpleNamespace(
            history_length=268, page_storage_version=1, num_blocks=3, is_active=True
        ),
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
        _chunking_ids=set(),
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
        _draft_managed_block_tables=torch.full((3, 8), BAD_PAGE_INDEX, dtype=torch.int32),
        _draft_fallback_block_tables=storage[3:],
        _draft_page_versions={},
        _draft_prepared_batch=None,
        _draft_page_indices={},
        _draft_metadata_host_buffers=[],
        _draft_updates_device=None,
        _draft_indices_device=None,
        _host_slots=torch.empty(3, dtype=torch.long),
        _host_capacities=torch.empty(3, dtype=torch.long),
        _table_slots=None,
        _page_indices=lambda cache, _: tables[id(cache)],
    )
    worker._slot_upload = device_copy.DeviceCopyBuffer(
        worker._batch_to_slot, site=DSparkCopySite.REQUEST_SLOTS
    )
    worker._capacity_upload = device_copy.DeviceCopyBuffer(
        worker._draft_capacities, site=DSparkCopySite.PAGE_CAPACITIES
    )
    worker._assign_slot = DSv41DSparkWorker._assign_slot.__get__(worker)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(query=lambda: True, record=lambda: None)
    )
    monkeypatch.setattr("tensorrt_llm._torch.speculative.dspark_v41.prefer_pinned", lambda: False)
    worker._clear_slot = DSv41DSparkWorker._clear_slot.__get__(worker)
    DSv41DSparkWorker._prepare_page_tables(worker, [42, 43], 0)
    assert worker._draft_block_tables[0, :2].tolist() == [7, 8]
    assert worker._draft_block_tables[1, 1:3].tolist() == [9, 10]
    DSv41DSparkWorker._prepare_page_tables(worker, [43, 42], 0)
    assert worker._draft_block_tables[0, 1:3].tolist() == [9, 10]
    assert worker._draft_block_tables[1, :2].tolist() == [7, 8]
    assert worker._batch_to_slot.tolist() == [1, 0, 2]
    assert worker._draft_capacities.tolist() == [384, 256, 0]

    caches[42].page_storage_version += 1
    tables[id(caches[42])] = (0, (7, 11))
    DSv41DSparkWorker._prepare_page_tables(worker, [43, 42], 0)
    assert worker._draft_block_tables[1, :2].tolist() == [7, 11]

    DSv41DSparkWorker._prepare_page_tables(worker, [43], 0, paused_in_place_ids=frozenset({42}))
    assert worker._req_to_slot[42] == 0
    assert worker._batch_to_slot.tolist() == [1, 2, 2]
    assert worker._draft_capacities.tolist() == [384, 0, 0]

    # A narrower window can advance inside one native history block.
    worker._draft_kv_manager.tokens_per_block = 256
    caches[43].history_length = 383
    caches[43].num_blocks = 2
    tables[id(caches[43])] = (0, (9, 10))
    DSv41DSparkWorker._prepare_page_tables(worker, [43], 0)
    caches[43].history_length = 384
    tables[id(caches[43])] = (1, (10,))
    DSv41DSparkWorker._prepare_page_tables(worker, [43], 0)
    assert worker._draft_block_tables[0, :3].tolist() == [BAD_PAGE_INDEX, 10, BAD_PAGE_INDEX]

    # Once the target publishes a native table, generation reads its addressable extent.
    worker._draft_kv_manager.draft_page_table_prepared = lambda _: True
    native = torch.full((3, 8), BAD_PAGE_INDEX, dtype=torch.int32)
    native[0, 1:3] = torch.tensor([21, 22])
    worker._draft_managed_block_tables.copy_(native)
    worker._draft_kv_manager.draft_page_table_state = lambda: native
    DSv41DSparkWorker._prepare_page_tables(worker, [43], 0)
    torch.testing.assert_close(worker._draft_block_tables, native)
    assert worker._draft_capacities.tolist() == [8 * 256, 0, 0]
    # A discarded metadata copy must be retried even if the target's batch is unchanged.
    with monkeypatch.context() as patch:
        patch.setattr(device_copy, "CHECK_MODE", False)
        worker._slot_upload.invalidate()
        worker._batch_to_slot.fill_(2)
        DSv41DSparkWorker._prepare_page_tables(worker, [43], 0)
        assert worker._batch_to_slot.tolist() == [1, 2, 2]
    # A different request still needs its own slot even if it aliases the
    # same cache and the target's physical page conversion can be reused.
    caches[44] = caches[43]
    DSv41DSparkWorker._prepare_page_tables(worker, [44], 0)
    assert 43 not in worker._req_to_slot
    assert 44 in worker._req_to_slot

    DSv41DSparkWorker._prepare_page_tables(worker, [42], 1, context_chunking_ids=frozenset({42}))
    slot = worker._req_to_slot[42]
    worker._valid_len[slot].fill_(48)
    DSv41DSparkWorker._prepare_page_tables(worker, [44], 1)
    DSv41DSparkWorker._prepare_page_tables(worker, [42], 1)
    assert worker._req_to_slot[42] == slot
    assert worker._valid_len[slot] == 48
    DSv41DSparkWorker._prepare_page_tables(worker, [44], 1)
    assert 42 not in worker._req_to_slot


@pytest.mark.parametrize(
    "chunks,captures",
    [([100, 80], [100, 80]), ([40, 140], [0, 140])],
    ids=["chunked", "held-tail"],
)
def test_private_context_seed_rebuilds_final_window(chunks, captures):
    if captures[0] == 0:
        from tensorrt_llm._torch.pyexecutor.scheduler.ctx_decoder_batch import (
            ContextPassPlan,
            CtxDecoderBatcher,
        )
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

        scheduler = object.__new__(KVCacheV2Scheduler)
        scheduler.max_num_tokens, scheduler.chunk_unit_size = 32, 1
        scheduler._ctx_decoder_held = False
        scheduler._num_ctx_decoder_hold_passes = scheduler._num_ctx_decoder_held_requests = 0
        scheduler.plan_context_pass = Mock(return_value=ContextPassPlan(140, 140, 180))
        executor = SimpleNamespace(
            model_engine=SimpleNamespace(replays_final_windows_only=lambda: True),
            dist=SimpleNamespace(tp_allgather_int64=lambda values: torch.tensor([values, values])),
        )
        batcher = CtxDecoderBatcher(executor, scheduler, 140)
        request = SimpleNamespace(
            is_context_only_request=True,
            is_dummy=False,
            py_ctx_decoder_held=False,
            py_csa2_remote_tail_mode=None,
            context_current_position=0,
            prompt_len=180,
        )
        batcher.before_schedule([request], set())
        assert scheduler._ctx_decoder_hold_window == 140
        first_chunk = scheduler._ctx_decoder_hold_chunk(request, 180)
        assert request.py_ctx_decoder_held and scheduler._ctx_decoder_held
        chunks = [first_chunk, 180 - first_chunk]
        assert chunks == [40, 140]
        batcher.after_schedule()
        assert scheduler._ctx_decoder_hold_window == 0
        assert scheduler._num_ctx_decoder_hold_passes == 1
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
        for frame, value in zip(frames[mask].tolist(), hidden[mask].flatten().tolist()):
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
