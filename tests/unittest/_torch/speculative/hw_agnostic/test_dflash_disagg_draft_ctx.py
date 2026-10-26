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
"""DFlash worker state for the disagg drafter context-window transfer (F0a).

Context side: ``_forward_impl`` records one event right after ``_store_prefill_context`` when the
batch has context requests and the stream is not capturing, and each context request keeps it
with its slot's host ``(ctx_len, ctx_base)``. Generation side: received lengths are seeded by
``DFlashSpecMetadata.prepare`` when the request first holds a real slot, never on the dummy slot.
"""

from collections import deque
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.speculative import dflash as dflash_module
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker

pytestmark = pytest.mark.cpu_only

FLOOR = 1 << 40  # request ids from here on are CUDA-graph padding dummies


class _StopAfterStore(Exception):
    """Ends the forward right after the context store; drafting is not under test."""


class _Event:
    recorded = []

    def __init__(self):
        self.record_calls = 0

    def record(self, stream=None):
        self.record_calls += 1
        _Event.recorded.append(self)


@pytest.fixture
def events(monkeypatch):
    _Event.recorded = []
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    return _Event.recorded


def _ctx_worker(engaged=True, pool_bound=True):
    worker = DFlashWorker.__new__(DFlashWorker)
    worker._ctx_buf_inited = False  # skips the batch-to-slot rebuild after the store
    worker._ctx_block_tables = torch.zeros(2, 4, dtype=torch.int32) if pool_bound else None
    worker._ctx_len = torch.zeros(3, dtype=torch.long)
    worker._ctx_base_pages = torch.zeros(3, dtype=torch.long)
    worker._ctx_len_host = [0, 0, 0]
    worker._ctx_base_host = [0, 0, 0]
    worker._req_ctx_pos = {}
    worker._req_to_slot = {}
    worker._accept_stats = None
    worker.get_draft_kv_cache_manager = Mock(return_value=None)
    worker._lazy_init_ctx_buffers = Mock()
    worker._refresh_ctx_block_tables = Mock(return_value=True)
    worker._execute_guided_decoder_if_present = Mock()
    worker.sample_and_accept_draft_tokens = Mock(return_value=(None, None))
    worker._prepare_attn_metadata_for_dflash = Mock()
    worker._prepare_kv_for_draft_forward = Mock()
    worker.prepare_1st_drafter_inputs = Mock(side_effect=_StopAfterStore)

    def store(draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens):
        # What the real store leaves on the host: slots and their stored lengths.
        for slot, rid in enumerate(spec_metadata.request_ids[: attn_metadata.num_contexts]):
            worker._req_to_slot[rid] = slot
            worker._ctx_len_host[slot] = 100 + slot
            worker._ctx_base_host[slot] = slot

    worker._store_prefill_context = Mock(side_effect=store)
    if engaged:
        worker.engage_disagg_draft_kv()
    return worker


def _forward(worker, request_ids, num_contexts):
    num_tokens = 8
    attn_metadata = SimpleNamespace(
        num_seqs=len(request_ids), num_contexts=num_contexts, kv_cache_manager=None
    )
    spec_metadata = SimpleNamespace(
        runtime_draft_len=3, is_cuda_graph=False, request_ids=list(request_ids)
    )
    with pytest.raises(_StopAfterStore):
        worker._forward_impl(
            torch.zeros(num_tokens, dtype=torch.int32),
            torch.arange(num_tokens),
            None,
            torch.zeros(num_tokens, 4),
            attn_metadata,
            spec_metadata,
            SimpleNamespace(),
        )


def test_event_is_recorded_after_the_context_store(monkeypatch, events):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    worker = _ctx_worker()
    order = []
    store = worker._store_prefill_context.side_effect

    def ordered_store(*args):
        order.append("store")
        store(*args)

    def ordered_record(event, stream=None):
        order.append("record")
        events.append(event)

    worker._store_prefill_context.side_effect = ordered_store
    monkeypatch.setattr(_Event, "record", ordered_record)

    _forward(worker, [11, 12, 13], num_contexts=2)

    assert order == ["store", "record"]
    assert len(events) == 1
    ready = worker.take_disagg_draft_ctx_ready(11)
    assert ready == (events[0], 100, 0)
    assert worker.take_disagg_draft_ctx_ready(12) == (events[0], 101, 1)
    # The generation request in the batch gets nothing; a taken entry is gone.
    assert worker.take_disagg_draft_ctx_ready(13) is None
    assert worker.take_disagg_draft_ctx_ready(11) is None


def test_no_event_without_context_requests(monkeypatch, events):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    worker = _ctx_worker()
    _forward(worker, [21, 22], num_contexts=0)
    worker._store_prefill_context.assert_not_called()
    assert events == []
    assert worker._disagg_draft_ctx_ready == {}


def test_no_event_while_capturing(monkeypatch, events):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    worker = _ctx_worker()
    _forward(worker, [31, 32], num_contexts=1)
    worker._store_prefill_context.assert_called_once()
    assert events == []
    assert worker._disagg_draft_ctx_ready == {}


def test_no_event_unless_engaged(monkeypatch, events):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    worker = _ctx_worker(engaged=False)
    _forward(worker, [41], num_contexts=1)
    worker._store_prefill_context.assert_called_once()
    assert events == []
    assert worker.take_disagg_draft_ctx_ready(41) is None


def test_private_arena_is_refused(monkeypatch, events):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    worker = _ctx_worker(pool_bound=False)
    with pytest.raises(RuntimeError, match="managed draft KV pool"):
        worker._record_disagg_draft_ctx_ready(SimpleNamespace(request_ids=[51]), 1)
    assert events == []


def test_latest_chunk_store_wins(monkeypatch, events):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    worker = _ctx_worker()
    _forward(worker, [61], num_contexts=1)
    worker._ctx_len_host[0] = 4096  # the next chunk stored more of the prompt
    worker._record_disagg_draft_ctx_ready(SimpleNamespace(request_ids=[61]), 1)
    assert len(events) == 2
    assert worker.take_disagg_draft_ctx_ready(61) == (events[1], 4096, 0)


def test_event_map_is_bounded(monkeypatch, events):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(dflash_module, "_DISAGG_DRAFT_CTX_MAP_CAP", 2)
    worker = _ctx_worker()
    for rid in (71, 72, 73):
        worker._record_disagg_draft_ctx_ready(SimpleNamespace(request_ids=[rid]), 1)
    assert list(worker._disagg_draft_ctx_ready) == [72, 73]


# ---------------------------------------------------------------------------
# Generation side: seeding at first sight in prepare()
# ---------------------------------------------------------------------------


def _gen_worker(num_slots=2, engaged=True):
    worker = SimpleNamespace(
        _ctx_buf_inited=True,
        _req_to_slot={},
        _req_ctx_pos={},
        _free_slots=deque(range(num_slots)),
        _graph_dummy_id_floor=FLOOR,
        _dummy_slot=num_slots,
        _ctx_block_tables=torch.zeros(1),
        _batch_to_slot=torch.zeros(8, dtype=torch.long),
        _disagg_draft_ctx_seeds={} if engaged else None,
        writes=[],
    )
    worker._write_ctx_len = lambda updates, bases=None: worker.writes.append(
        (dict(updates), dict(bases or {}))
    )
    worker._assign_slot = partial(DFlashWorker._assign_slot, worker)
    worker._check_disagg_draft_ctx_pool_bound = partial(
        DFlashWorker._check_disagg_draft_ctx_pool_bound, worker
    )
    worker._bound_disagg_draft_ctx_map = DFlashWorker._bound_disagg_draft_ctx_map
    worker.seed = partial(DFlashWorker.seed_disagg_draft_ctx, worker)
    return worker


def _seed_writes(worker):
    """The writes that carry base pages: only a seed does (slot clears and evictions do not)."""
    return [write for write in worker.writes if write[1]]


def _prepare(monkeypatch, worker, request_ids, num_generations):
    monkeypatch.setattr(dflash_module, "prefer_pinned", lambda: False)
    meta = SimpleNamespace(
        request_ids=request_ids,
        num_generations=num_generations,
        batch_indices_cuda=torch.zeros(8, dtype=torch.int),
        _dflash_worker=worker,
    )
    DFlashSpecMetadata.prepare(meta)


def test_received_lengths_are_seeded_when_the_request_first_holds_a_slot(monkeypatch):
    worker = _gen_worker()
    worker.seed(7, 9000, 3, 9000)
    worker.seed(8, 33, 0, 33)

    _prepare(monkeypatch, worker, [7, FLOOR + 1], num_generations=2)

    slot7 = worker._req_to_slot[7]
    # _assign_slot clears the slot, then the seed lands in one batched write.
    assert _seed_writes(worker) == [({slot7: 9000}, {slot7: 3})]
    assert worker.writes[-1] == ({slot7: 9000}, {slot7: 3})
    assert worker._req_ctx_pos[7] == 9000
    # A request not in the batch keeps its seed for its own first step.
    assert worker._disagg_draft_ctx_seeds == {8: (33, 0, 33)}
    # The padding dummy never holds a slot, so nothing seeds the dummy slot.
    assert all(worker._dummy_slot not in lens for lens, _ in worker.writes)

    _prepare(monkeypatch, worker, [7, 8], num_generations=2)
    slot8 = worker._req_to_slot[8]
    assert _seed_writes(worker)[1:] == [({slot8: 33}, {slot8: 0})]
    assert worker._disagg_draft_ctx_seeds == {}
    # Seeds apply once: a later step leaves the slots to the drafter.
    _prepare(monkeypatch, worker, [7, 8], num_generations=2)
    assert len(_seed_writes(worker)) == 2


def test_seed_waits_for_a_free_slot(monkeypatch):
    worker = _gen_worker(num_slots=1)
    worker.seed(5, 64, 0, 64)
    worker.seed(6, 96, 0, 96)
    _prepare(monkeypatch, worker, [5, 6], num_generations=2)
    assert 6 not in worker._req_to_slot  # shares the dummy slot this step, unseeded
    assert worker._disagg_draft_ctx_seeds == {6: (96, 0, 96)}
    _prepare(monkeypatch, worker, [6], num_generations=1)  # request 5 left, its slot freed
    assert worker._req_to_slot[6] == 0
    assert worker.writes[-1] == ({0: 96}, {0: 0})


def test_without_engagement_prepare_writes_no_seed(monkeypatch):
    worker = _gen_worker(engaged=False)
    _prepare(monkeypatch, worker, [5], num_generations=1)
    assert worker._req_to_slot == {5: 0}
    assert worker._req_ctx_pos == {}
    assert _seed_writes(worker) == []


def test_seeding_a_received_window_requires_the_pool_bound_context(monkeypatch):
    worker = _gen_worker()
    worker._disagg_draft_kv_engaged = True  # the window transfer is engaged
    worker._ctx_block_tables = None
    worker.seed(5, 64, 0, 64)
    with pytest.raises(RuntimeError, match="managed draft KV pool"):
        _prepare(monkeypatch, worker, [5], num_generations=1)


def test_private_arena_without_the_transfer_skips_the_seed(monkeypatch):
    """Gen-init window sizing alone: a private-arena drafter numbers from 0, nothing to rebase."""
    worker = _gen_worker()
    worker._ctx_block_tables = None
    worker.seed(5, 64, 2, 64)
    _prepare(monkeypatch, worker, [5], num_generations=1)
    assert _seed_writes(worker) == []


def test_dump_hook_runs_once_right_after_the_seed_lands(monkeypatch):
    worker = _gen_worker()
    worker._disagg_draft_ctx_seed_hooks = None
    seen = []
    worker.seed(7, 4097, 0, 4097, on_seed=lambda: seen.append(list(worker.writes)))
    worker.seed(8, 64, 0, 64)  # no knob: no hook

    _prepare(monkeypatch, worker, [7, 8], num_generations=2)

    assert len(seen) == 1
    # The batch's seed write (both requests in one) already happened when the hook ran.
    slot7, slot8 = worker._req_to_slot[7], worker._req_to_slot[8]
    assert seen[0][-1] == ({slot7: 4097, slot8: 64}, {slot7: 0, slot8: 0})
    assert worker._disagg_draft_ctx_seed_hooks == {}
    _prepare(monkeypatch, worker, [7, 8], num_generations=2)
    assert len(seen) == 1
