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
"""The Kimi K3 DSpark drafter's slot bookkeeping on the unified draft cache must not wait for the GPU.

``DFlashWorker._prepare_managed_history`` runs before every forward of a disaggregated
generation server, while the previous decode step is still in flight: it releases the slots of
requests that left the batch and assigns and restores the slots of requests that joined. Every
write there has to stay stream-ordered (pinned uploads, ``zero_()`` on element views), or the
host waits for the step in flight.

``DFlashWorker._store_prefill_context`` runs right after the target forward of a context server
has been enqueued. On the unified draft cache it bounds each chunk by the row's page count,
which the host built this iteration; reading the counts back from the device would make the
host wait for that whole forward before the KV of the step can be handed to the transceiver.
"""

import collections
import contextlib
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.spec.dflash import (
    DFlashWorker,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

FLOOR = 1 << 40  # request ids from here on are CUDA-graph padding dummies
NUM_SLOTS = 8


@contextlib.contextmanager
def _no_host_sync():
    """Fail on any host-device synchronization inside the block."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _worker(histories):
    """A worker holding only the state ``_prepare_managed_history`` touches.

    ``histories`` maps a request id to its ``(valid_length, position)`` in the manager, or None.
    """
    lengths = list(range(10, 10 + NUM_SLOTS + 1))
    manager = SimpleNamespace(
        draft_layer_ids=(61,),
        _draft_dummy_request_ids=set(),
        kv_cache_map={request_id: object() for request_id in histories},
        get_draft_history=lambda request_id: (
            None
            if histories[request_id] is None
            else SimpleNamespace(
                valid_length=histories[request_id][0], position=histories[request_id][1]
            )
        ),
    )
    worker = SimpleNamespace(
        _ctx_kv_manager=manager,
        _req_to_slot={},
        _req_ctx_pos={},
        _managed_cache_bindings={},
        _managed_request_ids=(),
        _free_slots=collections.deque(range(NUM_SLOTS)),
        _graph_dummy_id_floor=FLOOR,
        _dummy_slot=NUM_SLOTS,
        _ctx_len=torch.tensor(lengths, dtype=torch.long, device="cuda"),
        _ctx_len_host=list(lengths),
        _ctx_position_offset=torch.full((NUM_SLOTS + 1,), 3, dtype=torch.long, device="cuda"),
        _batch_to_slot_buffer=DeviceCopyBuffer(
            torch.zeros(NUM_SLOTS, dtype=torch.long, device="cuda"), pin_memory=True
        ),
        _managed_snapshot_slots=DeviceCopyBuffer(
            torch.zeros(NUM_SLOTS, dtype=torch.long, device="cuda"), pin_memory=True
        ),
    )
    for name in (
        "_has_unified_draft_cache",
        "_assign_slot",
        "_write_ctx_len",
        "_write_slot_values",
        "_rebuild_batch_to_slot",
        "_prepare_managed_history",
    ):
        setattr(worker, name, partial(getattr(DFlashWorker, name), worker))
    return worker


def test_join_and_leave_on_the_unified_draft_cache_are_stream_ordered():
    worker = _worker({100: (40, 50), 101: (7, 7), 102: None})
    with _no_host_sync():
        first_rows = worker._prepare_managed_history([100, 101])
        # Request 100 leaves and request 102 joins; 101 keeps its slot and its binding.
        second_rows = worker._prepare_managed_history([101, 102])
    torch.cuda.synchronize()

    assert (first_rows, second_rows) == ([0, 1], [1])
    assert worker._req_to_slot == {101: 1, 102: 2}
    assert list(worker._free_slots) == [3, 4, 5, 6, 7, 0]
    expected_lengths = list(range(10, 10 + NUM_SLOTS + 1))
    expected_lengths[0], expected_lengths[1], expected_lengths[2] = 0, 7, 0
    expected_lengths[NUM_SLOTS] = 0
    assert worker._ctx_len.tolist() == expected_lengths
    assert worker._ctx_len_host == expected_lengths
    expected_offsets = [3] * (NUM_SLOTS + 1)
    expected_offsets[0] = expected_offsets[1] = expected_offsets[2] = 0
    expected_offsets[NUM_SLOTS] = 0
    assert worker._ctx_position_offset.tolist() == expected_offsets
    assert worker._batch_to_slot_buffer.tensor[:2].tolist() == [1, 2]
    assert worker._managed_snapshot_slots.tensor[:2].tolist() == [1, 2]
    assert worker._managed_request_ids == (101, 102)


def test_slot_reset_on_the_unified_draft_cache_is_stream_ordered():
    worker = _worker({7: (5, 9)})
    worker._req_to_slot = {7: 4}
    worker._free_slots.remove(4)
    worker._ctx_position_offset[4] = 11
    with _no_host_sync():
        slot = worker._assign_slot(7, reset=True)
    torch.cuda.synchronize()
    assert slot == 0
    assert worker._ctx_len[4].item() == 0 and worker._ctx_len[0].item() == 0
    # A freed slot keeps its offset until it is assigned; an assigned one starts at 0.
    assert worker._ctx_position_offset[4].item() == 11
    assert worker._ctx_position_offset[0].item() == 0


PAGE = 32
HIDDEN = 16


def _store_worker(host_counts):
    """A worker that also holds what ``_store_prefill_context`` touches, on the unified cache."""
    worker = _worker({100: None, 101: None})
    stored = []
    worker.__dict__.update(
        _helix_mapping=None,
        _ctx_page_size=PAGE,
        _ctx_block_tables=torch.zeros(NUM_SLOTS, 4, dtype=torch.int32, device="cuda"),
        _ctx_block_counts=torch.tensor([1, 2] + [0] * (NUM_SLOTS - 2), device="cuda"),
        _ctx_block_counts_host=host_counts,
        _max_ctx=8192,
        _ctx_reuse_addressable=False,
        _ctx_paged=True,
        _ctx_kv_buf=[torch.empty(1, dtype=torch.bfloat16, device="cuda")],
        _ctx_k_buf=None,
        _store_context_kv_paged=lambda k, v, rows, positions: stored.append((k, rows, positions)),
    )
    for name in ("_helix_shard", "_store_prefill_context"):
        setattr(worker, name, partial(getattr(DFlashWorker, name), worker))
    return worker, stored


def _store(worker, seq_lens):
    num_tokens = sum(seq_lens)
    hidden = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    draft_model = SimpleNamespace(
        fc=None,
        hidden_norm=None,
        project_target_hidden=lambda captured: captured,
        precompute_context_kv=lambda proj, positions: (proj, None),
    )
    spec_metadata = SimpleNamespace(
        get_hidden_states=lambda total: hidden,
        host_position_ids=list(range(num_tokens)),
        request_ids=[100, 101],
    )
    attn_metadata = SimpleNamespace(
        num_ctx_tokens=num_tokens,
        num_contexts=len(seq_lens),
        _seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0] * len(seq_lens)),
    )
    position_ids = torch.arange(num_tokens, device="cuda")
    worker._store_prefill_context(draft_model, spec_metadata, attn_metadata, position_ids, num_tokens)


def test_prefill_store_on_the_unified_draft_cache_reads_page_counts_on_the_host():
    worker, stored = _store_worker(host_counts=[1, 2])
    worker._prepare_managed_history([100, 101])
    with _no_host_sync():
        # Row 1 holds 2 pages, so 40 tokens fit; row 0 holds 1 page, so 33 overflow.
        _store(worker, [33, 40])
    torch.cuda.synchronize()

    slot0, slot1 = worker._req_to_slot[100], worker._req_to_slot[101]
    assert worker._ctx_len_host[slot0] == 0 and worker._ctx_len_host[slot1] == 40
    assert worker._ctx_len[slot1].item() == 40
    # The overflowing row restarts empty and keeps its absolute position in the offset.
    assert worker._ctx_position_offset[slot0].item() == 33
    assert worker._ctx_position_offset[slot1].item() == 0
    assert len(stored) == 1 and stored[0][1].tolist() == [1] * 40


def test_prefill_store_without_host_page_counts_waits_for_the_device():
    """Control: the device read-back that the host counts replace does synchronize."""
    worker, _ = _store_worker(host_counts=None)
    worker._prepare_managed_history([100, 101])
    with pytest.raises(RuntimeError, match="synchroniz"), _no_host_sync():
        _store(worker, [33, 40])
