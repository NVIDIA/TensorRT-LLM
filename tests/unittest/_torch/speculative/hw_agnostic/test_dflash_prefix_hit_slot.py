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
"""A prefix-cache hit on DFlash's dense context arena resumes from a slot holding the reused prefix.

With block reuse the target skips the cached prompt prefix, so its hidden states never reach the drafter. The paged
pools that join block reuse still address the prefix's drafter K / V; the dense private arena (VANILLA) does not, and
used to start such a request at column 0 with only the uncached tail. DFlashWorker now keeps a finished request's slot
and lets a later request whose prompt starts with the same tokens take it over at column ``first_pos``, or copies the
prefix columns of a running request that holds them.

The fake drafter stores each token's hidden value (its token id) as K and the column's RoPE position as V, so a test
can read back which token sits in which column and at which position.
"""

from collections import OrderedDict, deque
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import (
    DFlashPromptTracker,
    DFlashSpecMetadata,
    DFlashWorker,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the drafter context arena lives on the GPU"
)

MAX_BATCH = 3
CAPACITY = 64
LAYERS = 2
FLOOR = 1 << 40


class _Buffer:
    def __init__(self):
        self.value = None

    def update(self, value):
        self.value = value.tolist()


class _Drafter:
    fc = hidden_norm = object()

    @staticmethod
    def project_target_hidden(hidden):
        return hidden

    @staticmethod
    def precompute_context_kv(proj, positions):
        n = proj.shape[0]
        k = proj[:, :1].reshape(n, 1, 1, 1).expand(n, LAYERS, 1, 1)
        v = positions.to(proj.dtype).reshape(n, 1, 1, 1).expand(n, LAYERS, 1, 1)
        return k.contiguous(), v.contiguous()


def _worker():
    w = object.__new__(DFlashWorker)
    num_slots = MAX_BATCH + 1
    w._ctx_buf_inited = True
    w._ctx_paged = False
    w._ctx_reuse_addressable = False
    w._ctx_block_tables = None
    w._ctx_kv_buf = None
    w._max_ctx = CAPACITY
    w._dummy_slot = MAX_BATCH
    w._graph_dummy_id_floor = FLOOR
    w._ctx_k_buf = torch.full((num_slots, LAYERS, CAPACITY, 1, 1), -1.0, device="cuda")
    w._ctx_v_buf = torch.full((num_slots, LAYERS, CAPACITY, 1, 1), -1.0, device="cuda")
    w._ctx_len = torch.zeros(num_slots, dtype=torch.long, device="cuda")
    w._ctx_len_host = [0] * num_slots
    w._ctx_base_pages = torch.zeros(num_slots, dtype=torch.long, device="cuda")
    w._ctx_base_host = [0] * num_slots
    w._free_slots = deque(range(MAX_BATCH))
    w._req_to_slot = {}
    w._req_ctx_pos = {}
    w._slot_owner = {}
    w._kept_slots = OrderedDict()
    w._prefix_hits = 0
    w._prefix_misses = 0
    w._warned_prefix_miss = False
    w._batch_to_slot_buffer = _Buffer()
    return w


def _request(rid, tokens, hashes=None, positions=None, mm_data=None):
    req = SimpleNamespace(
        py_request_id=rid,
        is_dummy=False,
        tokens=list(tokens),
        multimodal_hashes=hashes,
        multimodal_positions=positions,
        py_multimodal_data=mm_data,
    )
    req.get_tokens = lambda beam: list(req.tokens)
    return req


def _schedule(tracker, *requests):
    tracker.prepare_resources(SimpleNamespace(context_requests=list(requests)))


def _prefill(worker, tracker, chunks):
    """One context forward. ``chunks``: (request, first_pos, num_tokens) per context request."""
    hidden, positions, host_pos, seq_lens = [], [], [], []
    for req, first_pos, n in chunks:
        toks = req.tokens[first_pos : first_pos + n]
        hidden += toks
        positions += list(range(first_pos, first_pos + n))
        host_pos += list(range(first_pos, first_pos + n))
        seq_lens.append(n)
    hs = torch.tensor(hidden, dtype=torch.float32, device="cuda").reshape(-1, 1)
    spec_metadata = SimpleNamespace(
        get_hidden_states=lambda total: hs,
        request_ids=[req.py_request_id for req, _, _ in chunks],
        host_position_ids=host_pos,
        spec_resource_manager=tracker,
    )
    attn_metadata = SimpleNamespace(
        num_ctx_tokens=len(hidden), num_contexts=len(chunks), _seq_lens=seq_lens
    )
    position_ids = torch.tensor(positions, dtype=torch.long, device="cuda")
    DFlashWorker._store_prefill_context(
        worker, _Drafter(), spec_metadata, attn_metadata, position_ids, len(hidden)
    )
    torch.cuda.synchronize()


def _generate(worker, req, num_tokens):
    """Generation appends accepted tokens' K / V and advances _ctx_len on the device only."""
    slot = worker._req_to_slot[req.py_request_id]
    start = int(worker._ctx_len[slot])
    for col in range(start, start + num_tokens):
        worker._ctx_k_buf[slot, :, col] = float(req.tokens[col])
        worker._ctx_v_buf[slot, :, col] = float(col)
    worker._ctx_len[slot] += num_tokens


def _prepare(worker, tracker, request_ids, num_generations):
    meta = SimpleNamespace(
        request_ids=list(request_ids),
        num_generations=num_generations,
        spec_resource_manager=tracker,
        _dflash_worker=worker,
        _batch_indices_buffer=_Buffer(),
        host_position_ids=None,
        seq_lens=None,
    )
    DFlashSpecMetadata.prepare(meta)
    torch.cuda.synchronize()


def _finish(worker, tracker, req, still_running=()):
    tracker.free_resources(req)
    _prepare(worker, tracker, still_running, num_generations=len(still_running))


def _columns(worker, slot, n):
    """(token, position) held by the first ``n`` columns of ``slot``, layer 0."""
    k = worker._ctx_k_buf[slot, 0, :n, 0, 0].tolist()
    v = worker._ctx_v_buf[slot, 0, :n, 0, 0].tolist()
    return [(int(a), int(b)) for a, b in zip(k, v)]


def _start(worker, tracker, req):
    _schedule(tracker, req)
    _prefill(worker, tracker, [(req, 0, len(req.tokens))])
    return worker._req_to_slot[req.py_request_id]


def test_follow_up_turn_takes_over_the_kept_slot():
    worker, tracker = _worker(), DFlashPromptTracker(MAX_BATCH)
    first = _request(1, range(100, 108))
    slot = _start(worker, tracker, first)
    first.tokens += [200, 201, 202, 203]  # generated; the last one has no K / V yet
    _generate(worker, first, 3)
    _finish(worker, tracker, first)
    assert slot in worker._kept_slots and slot in worker._free_slots

    # Turn two reuses 10 cached tokens (prompt + 2 generated) and prefills 3 new ones.
    second = _request(2, first.tokens[:10] + [300, 301, 302])
    _schedule(tracker, second)
    _prefill(worker, tracker, [(second, 10, 3)])

    assert worker._req_to_slot[2] == slot
    assert slot not in worker._free_slots and slot not in worker._kept_slots
    assert worker._ctx_len_host[slot] == 13 and int(worker._ctx_len[slot]) == 13
    assert worker._ctx_base_host[slot] == 0 and int(worker._ctx_base_pages[slot]) == 0
    assert _columns(worker, slot, 13) == [(t, p) for p, t in enumerate(second.tokens)]
    assert worker._prefix_hits == 1 and worker._prefix_misses == 0


def test_adopted_slot_keeps_a_chunked_tail_contiguous():
    worker, tracker = _worker(), DFlashPromptTracker(MAX_BATCH)
    first = _request(1, range(100, 108))
    slot = _start(worker, tracker, first)
    first.tokens += [200, 201, 202, 203]
    _generate(worker, first, 3)
    _finish(worker, tracker, first)

    # The uncached tail arrives in two prefill chunks; the second continues the adopted slot.
    second = _request(2, first.tokens[:10] + [300, 301, 302, 303])
    _schedule(tracker, second)
    _prefill(worker, tracker, [(second, 10, 2)])
    _prefill(worker, tracker, [(second, 12, 2)])

    assert worker._req_to_slot[2] == slot
    assert worker._ctx_len_host[slot] == 14 and int(worker._ctx_len[slot]) == 14
    assert _columns(worker, slot, 14) == [(t, p) for p, t in enumerate(second.tokens)]
    assert worker._prefix_hits == 1 and worker._prefix_misses == 0


def test_spec_resource_manager_is_the_prompt_tracker_for_dflash():
    """Without sa_config, DFlash gets the tracker that the prefix-hit resume reads."""
    from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
    from tensorrt_llm._torch.speculative.utils import get_spec_resource_manager

    # DFlashDecodingConfig has no sa_config attribute, so the stub doesn't either.
    engine = SimpleNamespace(
        spec_config=SimpleNamespace(spec_dec_mode=SpeculativeDecodingMode.DFLASH),
        model=SimpleNamespace(config=SimpleNamespace()),
        batch_size=MAX_BATCH,
        max_seq_len=CAPACITY,
        max_num_tokens=CAPACITY,
    )

    manager = get_spec_resource_manager(engine)
    assert isinstance(manager, DFlashPromptTracker)
    assert manager.max_num_requests == MAX_BATCH and manager.active


def test_prefix_mismatch_starts_from_the_tail_as_before():
    worker, tracker = _worker(), DFlashPromptTracker(MAX_BATCH)
    first = _request(1, range(100, 108))
    slot = _start(worker, tracker, first)
    _finish(worker, tracker, first)

    other = list(first.tokens)
    other[3] = 999  # the target matched different blocks; the kept slot is not this prefix
    second = _request(2, other + [300, 301])
    _schedule(tracker, second)
    _prefill(worker, tracker, [(second, 8, 2)])

    got = worker._req_to_slot[2]
    assert got != slot and slot in worker._kept_slots
    assert worker._ctx_len_host[got] == 2
    assert _columns(worker, got, 2) == [(300, 0), (301, 1)]
    assert worker._prefix_misses == 1 and worker._warned_prefix_miss


def test_kept_slot_shorter_than_the_reused_prefix_is_not_taken():
    worker, tracker = _worker(), DFlashPromptTracker(MAX_BATCH)
    first = _request(1, range(100, 108))
    slot = _start(worker, tracker, first)
    first.tokens += [200, 201, 202]
    _generate(worker, first, 1)  # only 9 columns hold K / V
    _finish(worker, tracker, first)

    second = _request(2, first.tokens[:10] + [300])
    _schedule(tracker, second)
    _prefill(worker, tracker, [(second, 10, 1)])
    assert worker._req_to_slot[2] != slot and worker._prefix_misses == 1


def test_running_request_with_the_same_prefix_is_copied_not_moved():
    worker, tracker = _worker(), DFlashPromptTracker(MAX_BATCH)
    system = list(range(100, 108))
    first = _request(1, system + [110, 111])
    src = _start(worker, tracker, first)

    second = _request(2, system + [120, 121, 122])
    _schedule(tracker, second)
    _prefill(worker, tracker, [(second, 8, 3)])

    dst = worker._req_to_slot[2]
    assert dst != src and worker._req_to_slot[1] == src
    assert _columns(worker, dst, 11) == [(t, p) for p, t in enumerate(second.tokens)]
    assert _columns(worker, src, 10) == [(t, p) for p, t in enumerate(first.tokens)]
    assert worker._ctx_len_host[dst] == 11 and worker._ctx_len_host[src] == 10


def test_same_batch_prefix_owner_and_follower():
    # Chunks of one forward are stored in order, so a follower later in the batch can copy.
    worker, tracker = _worker(), DFlashPromptTracker(MAX_BATCH)
    system = list(range(100, 108))
    first = _request(1, system + [110])
    second = _request(2, system + [120])
    _schedule(tracker, first, second)
    _prefill(worker, tracker, [(first, 0, 9), (second, 8, 1)])
    dst = worker._req_to_slot[2]
    assert _columns(worker, dst, 9) == [(t, p) for p, t in enumerate(second.tokens)]


def test_empty_slots_are_handed_out_before_kept_ones():
    worker, tracker = _worker(), DFlashPromptTracker(MAX_BATCH)
    a = _request(1, range(100, 104))
    kept = _start(worker, tracker, a)
    _finish(worker, tracker, a)

    b = _request(2, range(200, 204))
    c = _request(3, range(300, 304))
    assert _start(worker, tracker, b) != kept
    assert _start(worker, tracker, c) != kept
    assert kept in worker._kept_slots

    # No empty slot left: the least recently kept one is reused and forgotten.
    d = _request(4, range(400, 404))
    assert _start(worker, tracker, d) == kept
    assert kept not in worker._kept_slots and worker._slot_owner[kept] == 4


def test_slot_that_started_on_a_prefix_hit_without_a_match_is_not_kept():
    # Its columns hold the tail at positions from 0, which no later prefix can use.
    worker, tracker = _worker(), DFlashPromptTracker(MAX_BATCH)
    req = _request(1, range(100, 110))
    _schedule(tracker, req)
    _prefill(worker, tracker, [(req, 6, 4)])
    slot = worker._req_to_slot[1]
    assert slot not in worker._slot_owner
    _finish(worker, tracker, req)
    assert slot not in worker._kept_slots and slot in worker._free_slots


@pytest.mark.parametrize(
    "second_hashes, second_mm, adopted",
    [
        ([[7, 7]], {"image": 1}, True),  # same image inside the prefix
        ([[8, 8]], {"image": 1}, False),  # different image, identical placeholder ids
        (None, {"image": 1}, False),  # multimodal data without hashes: never matched
    ],
)
def test_images_inside_the_prefix_must_match(second_hashes, second_mm, adopted):
    worker, tracker = _worker(), DFlashPromptTracker(MAX_BATCH)
    tokens = [100, 5, 5, 5, 101, 102, 103, 104]
    first = _request(1, tokens, hashes=[[7, 7]], positions=[1], mm_data={"image": 1})
    slot = _start(worker, tracker, first)
    _finish(worker, tracker, first)

    positions = [1] if second_hashes is not None else None
    second = _request(
        2, tokens + [300], hashes=second_hashes, positions=positions, mm_data=second_mm
    )
    _schedule(tracker, second)
    _prefill(worker, tracker, [(second, 8, 1)])
    assert (worker._req_to_slot[2] == slot) is adopted


def test_paged_context_turns_the_tracker_off():
    worker, tracker = _worker(), DFlashPromptTracker(MAX_BATCH)
    worker._ctx_reuse_addressable = True  # a paged pool joined to block reuse
    req = _request(1, range(100, 104))
    _schedule(tracker, req)
    _prefill(worker, tracker, [(req, 0, 4)])
    assert not tracker.active and tracker.lookup(1) is None
    _schedule(tracker, _request(2, range(4)))
    assert tracker.lookup(2) is None


def test_prepare_without_a_tracker_frees_slots_as_before():
    worker = _worker()
    req = _request(1, range(100, 104))
    _prefill(worker, None, [(req, 0, 4)])
    slot = worker._req_to_slot[1]
    _prepare(worker, None, [], num_generations=0)
    assert slot in worker._free_slots and not worker._kept_slots
    assert worker._ctx_len_host[slot] == 0
