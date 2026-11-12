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

``DFlashSpecMetadata.prepare`` (main's) runs before every forward of a disaggregated generation
server, while the previous decode step is still in flight: it releases the slots of requests that
left the batch and assigns the slots of requests that joined, seeding a joining request's context
length at its first position (its prompt's drafter context arrived in the unified cache's draft
layers). Every write there has to stay stream-ordered (pinned uploads), or the host waits for the
step in flight.

``K3DFlashWorker._store_prefill_context`` runs right after the target forward of a context server
has been enqueued. On the unified draft cache it bounds each chunk by the row's page count, which
``_refresh_ctx_block_tables`` read from the manager's caches on the host; reading the counts back
from the device would make the host wait for that whole forward before the KV of the step can be
handed to the transceiver.
"""

import collections
import contextlib
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.drafter import (
    K3DSparkPath,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.worker import (
    K3DFlashWorker,
)
from tensorrt_llm._torch.speculative import DFlashSpecMetadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

FLOOR = 1 << 40  # request ids from here on are CUDA-graph padding dummies
NUM_SLOTS = 8
TOKENS_PER_GEN = 3  # K + 1 target tokens of a generation request


@contextlib.contextmanager
def _no_host_sync():
    """Fail on any host-device synchronization inside the block."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _bind(worker, *names):
    for name in names:
        setattr(worker, name, partial(getattr(K3DFlashWorker, name), worker))


def _worker():
    """A worker holding only the slot state main's ``DFlashSpecMetadata.prepare`` touches."""
    lengths = list(range(10, 10 + NUM_SLOTS + 1))
    worker = SimpleNamespace(
        _ctx_buf_inited=True,
        _ctx_unified_cache=True,
        _max_ctx=8192,
        _req_to_slot={},
        _req_ctx_pos={},
        _free_slots=collections.deque(range(NUM_SLOTS)),
        _graph_dummy_id_floor=FLOOR,
        _dummy_slot=NUM_SLOTS,
        _ctx_len=torch.tensor(lengths, dtype=torch.long, device="cuda"),
        _ctx_len_host=list(lengths),
        _batch_to_slot_buffer=DeviceCopyBuffer(
            torch.zeros(NUM_SLOTS, dtype=torch.long, device="cuda"), pin_memory=True
        ),
    )
    _bind(
        worker, "_assign_slot", "_seed_generation_slot", "_write_ctx_len", "_rebuild_batch_to_slot"
    )
    return worker


def _prepare(worker, request_ids, first_positions):
    """Main's ``DFlashSpecMetadata.prepare`` for a generation-only batch."""
    metadata = SimpleNamespace(
        request_ids=request_ids,
        num_generations=len(request_ids),
        seq_lens=[TOKENS_PER_GEN] * len(request_ids),
        host_position_ids=[p + j for p in first_positions for j in range(TOKENS_PER_GEN)],
        _batch_indices_buffer=DeviceCopyBuffer(
            torch.zeros(NUM_SLOTS, dtype=torch.int, device="cuda"), pin_memory=True
        ),
        _dflash_worker=worker,
    )
    DFlashSpecMetadata.prepare(metadata)


def test_join_and_leave_on_the_unified_draft_cache_are_stream_ordered():
    worker = _worker()
    with _no_host_sync():
        _prepare(worker, [100, 101], [40, 7])
        # Request 100 leaves and request 102 joins; 101 keeps its slot and its length.
        _prepare(worker, [101, 102], [8, 20])
    torch.cuda.synchronize()

    assert worker._req_to_slot == {101: 1, 102: 2}
    assert list(worker._free_slots) == [3, 4, 5, 6, 7, 0]
    expected_lengths = list(range(10, 10 + NUM_SLOTS + 1))
    expected_lengths[0], expected_lengths[1], expected_lengths[2] = 0, 7, 20
    assert worker._ctx_len.tolist() == expected_lengths
    assert worker._ctx_len_host == expected_lengths
    assert worker._batch_to_slot_buffer.tensor[:2].tolist() == [1, 2]


def test_slot_reset_on_the_unified_draft_cache_is_stream_ordered():
    worker = _worker()
    worker._req_to_slot = {7: 4}
    worker._free_slots.remove(4)
    with _no_host_sync():
        slot = worker._assign_slot(7, reset=True)
    torch.cuda.synchronize()
    assert slot == 0
    assert worker._ctx_len[4].item() == 0 and worker._ctx_len[0].item() == 0


PAGE = 32
HIDDEN = 16


def _store_worker(host_counts):
    """A worker that also holds what ``_store_prefill_context`` touches, on the unified cache."""
    worker = _worker()
    stored = []
    worker.__dict__.update(
        _helix_mapping=None,
        _ctx_page_size=PAGE,
        _ctx_block_tables=torch.zeros(NUM_SLOTS, 4, dtype=torch.int32, device="cuda"),
        _ctx_block_counts=torch.tensor([1, 2] + [0] * (NUM_SLOTS - 2), device="cuda"),
        _ctx_block_counts_host=host_counts,
        _ctx_reuse_addressable=True,
        _ctx_paged=True,
        _ctx_kv_buf=[torch.empty(1, dtype=torch.bfloat16, device="cuda")],
        _ctx_k_buf=None,
        _store_context_kv_paged=lambda k, v, rows, positions: stored.append((k, rows, positions)),
    )
    _bind(worker, "_helix_shard", "_store_prefill_context")
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
        # Each prompt starts at position 0: no reused prefix.
        host_position_ids=[p for n in seq_lens for p in range(n)],
        request_ids=[100, 101],
    )
    attn_metadata = SimpleNamespace(
        num_ctx_tokens=num_tokens,
        num_contexts=len(seq_lens),
        _seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
    )
    position_ids = torch.arange(num_tokens, device="cuda")
    worker._store_prefill_context(
        draft_model, spec_metadata, attn_metadata, position_ids, num_tokens
    )


def test_prefill_store_on_the_unified_draft_cache_reads_page_counts_on_the_host():
    worker, stored = _store_worker(host_counts=[1, 2])
    with _no_host_sync():
        # Row 1 holds 2 pages, so 40 tokens fit; row 0 holds 1 page, so 33 overflow.
        _store(worker, [33, 40])
    torch.cuda.synchronize()

    # The overflowing request drafts nothing: its slot goes back, emptied.
    assert 100 not in worker._req_to_slot
    slot1 = worker._req_to_slot[101]
    assert worker._ctx_len_host[0] == 0 and worker._ctx_len_host[slot1] == 40
    assert worker._ctx_len[slot1].item() == 40
    assert len(stored) == 1 and stored[0][1].tolist() == [1] * 40
    assert stored[0][2].tolist() == list(range(40))


def test_prefill_store_without_host_page_counts_waits_for_the_device():
    """Control: the device read-back that the host counts replace does synchronize."""
    worker, _ = _store_worker(host_counts=None)
    with pytest.raises(RuntimeError, match="synchroniz"), _no_host_sync():
        _store(worker, [33, 40])


OFFSET = 400  # the draft layers' K-buffer offset in a per-layer page table entry
WIDTH = 4


@pytest.fixture(
    params=[K3DSparkPath(), K3DSparkPath(spec_tail_fusion=True, tail_v2=True)],
    ids=["plain", "tail_v2"],
)
def path(request):
    """The deployment path: ``tail_v2`` decodes the block table over more programs."""
    return request.param


def _refresh_worker(num_blocks, checks, path):
    """A worker with what ``_refresh_ctx_block_tables`` touches, on the unified cache."""
    worker = SimpleNamespace(
        _path=path,
        _ctx_block_tables=torch.zeros(NUM_SLOTS + 1, WIDTH, dtype=torch.int32, device="cuda"),
        _ctx_block_counts=torch.zeros(NUM_SLOTS + 1, dtype=torch.long, device="cuda"),
        _ctx_block_partial=torch.zeros(NUM_SLOTS + 1, dtype=torch.int32, device="cuda"),
        _ctx_block_counts_host=None,
        _ctx_unified_cache=True,
        _ctx_unified_host_counts=True,
        _draft_table_checks_left=checks,
        _ctx_pool_idx=0,
        _ctx_block_divisor=1,
        _ctx_block_offset=OFFSET,
        _ctx_kv_manager=SimpleNamespace(
            kv_cache_map={
                request_id: SimpleNamespace(num_blocks=count)
                for request_id, count in num_blocks.items()
            }
        ),
    )
    _bind(worker, "_refresh_ctx_block_tables", "_check_host_block_counts")
    return worker


def _offsets(rows):
    """``kv_cache_block_offsets`` ``[pools, rows, K/V, blocks]``; 0 where a row holds no page,
    as the native backend writes an unallocated entry."""
    src = torch.zeros(1, NUM_SLOTS + 1, 2, WIDTH, dtype=torch.int32)
    for i, blocks in enumerate(rows):
        for j, block in enumerate(blocks):
            src[0, i, :, j] = OFFSET + block
    return src.cuda()


def test_block_tables_take_the_page_counts_from_the_manager_on_the_host(path):
    worker = _refresh_worker({100: 1, 101: 2}, checks=1, path=path)
    attn_metadata = SimpleNamespace(num_contexts=2, kv_cache_block_offsets=_offsets([[5], [7, 9]]))
    # The first iteration checks the host counts against the decoded table (it reads it back).
    assert worker._refresh_ctx_block_tables(attn_metadata, 2, [100, 101])
    assert worker._ctx_block_counts_host == [1, 2]
    with _no_host_sync():
        assert worker._refresh_ctx_block_tables(attn_metadata, 2, [100, 101])
    torch.cuda.synchronize()
    assert worker._ctx_block_counts_host == [1, 2]
    assert worker._ctx_block_tables[:2, :2].tolist() == [[5, 0], [7, 9]]
    assert worker._ctx_block_counts[:2].tolist() == [1, 2]


def test_a_saturated_decoded_count_bounds_nothing_and_is_accepted(path):
    worker = _refresh_worker({100: 1, 101: 2}, checks=1, path=path)
    # A zero layer offset: unallocated entries decode as page 0 and count as allocated.
    worker._ctx_block_offset = 0
    src = torch.zeros(1, NUM_SLOTS + 1, 2, WIDTH, dtype=torch.int32, device="cuda")
    attn_metadata = SimpleNamespace(num_contexts=2, kv_cache_block_offsets=src)
    assert worker._refresh_ctx_block_tables(attn_metadata, 2, [100, 101])
    assert worker._ctx_block_counts[:2].tolist() == [WIDTH, WIDTH]
    assert worker._ctx_block_counts_host == [1, 2]


def test_host_counts_that_disagree_with_the_table_are_dropped(path):
    worker = _refresh_worker({100: 3, 101: 2}, checks=1, path=path)
    attn_metadata = SimpleNamespace(num_contexts=2, kv_cache_block_offsets=_offsets([[5], [7, 9]]))
    assert worker._refresh_ctx_block_tables(attn_metadata, 2, [100, 101])
    assert worker._ctx_block_counts_host is None
    assert worker._ctx_unified_host_counts is False
    assert worker._refresh_ctx_block_tables(attn_metadata, 2, [100, 101])
    assert worker._ctx_block_counts_host is None
