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
"""A disagg generation worker sizes a gen-init request's windowed draft mirror for its window.

The standalone DSpark drafter's separate pool is windowed and funded per request for its window
plus the step's tokens (here 16 requests x 4,276 tokens). A disagg generation worker receives the
prompt instead of prefilling it, so the mirror takes the whole prompt as history and holds only the
window behind the prompt end. With the model's ``disagg_draft_kv`` hook the drafter also starts
from an empty window there, never reading a page the window released. No GPU needed.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as mgr_mod
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.speculative.dflash import DFlashWorker, dflash_rebased_ctx_view

pytestmark = pytest.mark.cpu_only

TPB = 32
WINDOW = 4096
FUNDED_TOKENS = 16 * 4276
PROMPT = 100_013  # not a whole number of pages: the base rounds up
DRAFT = 7
EXTRA = 1


def _blocks(tokens):
    return -(-tokens // TPB)


class _WindowedPool:
    """A windowed draft pool's page budget: a cache holds its pages from the window start on."""

    def __init__(self, funded_tokens=FUNDED_TOKENS):
        self.budget_pages = funded_tokens // TPB
        self.caches = []
        self.peak_pages = 0

    def live_pages(self):
        return sum(cache.live_pages() for cache in self.caches)


class _Cache:
    def __init__(self, pool):
        self.pool = pool
        self.capacity = 0
        self.history_length = 0
        pool.caches.append(self)

    def live_pages(self, capacity=None, history=None):
        capacity = self.capacity if capacity is None else capacity
        history = self.history_length if history is None else history
        released = max(0, (history + 1 - WINDOW) // TPB)
        return max(0, _blocks(capacity) - released)

    def first_held_ordinal(self):
        return max(0, (self.history_length + 1 - WINDOW) // TPB)

    def resize(self, capacity, history_length=None):
        history = self.history_length if history_length is None else history_length
        others = self.pool.live_pages() - self.live_pages()
        if others + self.live_pages(capacity, history) > self.pool.budget_pages:
            return False
        self.capacity, self.history_length = capacity, history
        self.pool.peak_pages = max(self.pool.peak_pages, self.pool.live_pages())
        return True


def _manager(cache):
    m = object.__new__(KVCacheManagerV2)
    m.is_draft = True
    m.enable_joint_kv_cache_reuse = False
    # A one-engine external drafter (DSpark): its mirror follows the target's progress.
    m._mirror_uses_draft_view = False
    m.num_extra_kv_tokens = EXTRA
    m.tokens_per_block = TPB
    m.kv_cache_map = {1: cache}
    m._allocated_draft_lens = {}
    m._mirror_draft_kv_cache = lambda req: cache
    m._resume_and_restore = lambda request_id, kv_cache: True
    m._after_shrink = lambda request_id, kv_cache: None
    m._draft_pool_diagnostic = lambda: ""
    return m


def _gen_init_request():
    # What a disagg generation worker hands the draft manager for a received prompt.
    return SimpleNamespace(
        py_request_id=1,
        prompt_len=PROMPT,
        context_current_position=0,
        context_chunk_size=PROMPT,
        is_last_context_chunk=True,
        is_disagg_generation_init_state=True,
        py_draft_tokens=[0] * DRAFT,
    )


def _batch(req):
    return SimpleNamespace(
        context_requests=[req], generation_requests=[], all_requests=lambda: [req]
    )


def _worker():
    worker = DFlashWorker.__new__(DFlashWorker)
    worker.engage_disagg_gen_init_window()
    return worker


def test_stock_mirror_path_holds_the_window_without_a_seed(monkeypatch):
    """Without the hook the mirror already takes the received prompt as history, so it holds the
    window and stays inside the pool's funding; only the hook seeds the drafter's empty window."""
    monkeypatch.setattr(mgr_mod, "get_draft_token_length", lambda req: DRAFT)
    pool = _WindowedPool()
    cache = _Cache(pool)
    m = _manager(cache)
    assert m._disagg_gen_init_window is None

    m._prepare_draft_resources(_batch(_gen_init_request()))  # no resize failure

    assert (cache.capacity, cache.history_length) == (PROMPT + DRAFT + EXTRA, PROMPT)
    assert pool.peak_pages <= _blocks(WINDOW + DRAFT + EXTRA) + 1


def test_gen_init_mirror_holds_only_the_window_and_seeds_an_empty_one(monkeypatch):
    monkeypatch.setattr(mgr_mod, "get_draft_token_length", lambda req: DRAFT)
    pool = _WindowedPool()
    cache = _Cache(pool)
    m = _manager(cache)
    worker = _worker()
    m.engage_disagg_gen_init_window(worker.seed_disagg_empty_window)

    m._prepare_draft_resources(_batch(_gen_init_request()))  # no resize failure

    assert (cache.capacity, cache.history_length) == (PROMPT + DRAFT + EXTRA, PROMPT)
    # Peak live pages: the window behind the prompt end plus the draft / extra tail.
    assert pool.peak_pages <= _blocks(WINDOW + DRAFT + EXTRA) + 1
    assert pool.peak_pages <= pool.budget_pages
    base = _blocks(PROMPT)
    assert worker._disagg_draft_ctx_seeds == {1: (PROMPT, base, PROMPT)}

    # The drafter never reads a page the window released: its reads start at base * page, past
    # the prompt end and so past the window start L - W + 1.
    rows = torch.arange(base + 4, dtype=torch.long).unsqueeze(0)
    for accepted in (0, 1, 9, 40):
        num_ctx = torch.tensor([PROMPT + accepted])
        shifted, readable = dflash_rebased_ctx_view(rows, torch.tensor([base]), num_ctx, TPB)
        first_read_position = int(num_ctx - readable)  # rows below it are not read
        assert first_read_position >= PROMPT > PROMPT - WINDOW + 1
        first_page = int(shifted[0, 0])
        assert first_page >= cache.first_held_ordinal()


def test_seed_applies_once_at_first_sight_in_prepare(monkeypatch):
    from tensorrt_llm._torch.speculative import dflash as dflash_module
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

    monkeypatch.setattr(dflash_module, "prefer_pinned", lambda: False)
    worker = _worker()
    worker.seed_disagg_empty_window(7, PROMPT, TPB)
    writes = []
    worker._ctx_buf_inited = True
    worker._req_to_slot = {}
    worker._req_ctx_pos = {}
    worker._free_slots = __import__("collections").deque([0, 1])
    worker._graph_dummy_id_floor = 1 << 40
    worker._dummy_slot = 2
    worker._ctx_block_tables = torch.zeros(1)  # the drafter context is pool-bound
    worker._batch_to_slot = torch.zeros(4, dtype=torch.long)
    worker._write_ctx_len = lambda updates, bases=None: writes.append((dict(updates), bases))
    meta = SimpleNamespace(
        request_ids=[7],
        num_generations=1,
        batch_indices_cuda=torch.zeros(4, dtype=torch.int),
        _dflash_worker=worker,
    )
    DFlashSpecMetadata.prepare(meta)
    slot = worker._req_to_slot[7]
    assert writes[-1] == ({slot: PROMPT}, {slot: _blocks(PROMPT)})
    DFlashSpecMetadata.prepare(meta)
    assert sum(1 for _, bases in writes if bases) == 1


def test_without_the_hook_nothing_is_engaged():
    """Inert unless the model sets RuntimeHooks.disagg_draft_kv."""
    from tensorrt_llm._torch.pyexecutor._util import engage_disagg_gen_init_window
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

    m = object.__new__(KVCacheManagerV2)
    m.is_draft = True
    worker = DFlashWorker.__new__(DFlashWorker)
    assert not engage_disagg_gen_init_window(SimpleNamespace(spec_worker=worker), m)
    assert m._disagg_gen_init_window is None and worker._disagg_draft_ctx_seeds is None
    hooked = SimpleNamespace(spec_worker=worker, runtime_hooks=RuntimeHooks(disagg_draft_kv=True))
    assert not engage_disagg_gen_init_window(hooked, None)
    assert engage_disagg_gen_init_window(hooked, m)
    assert m._disagg_gen_init_window == worker.seed_disagg_empty_window
    assert worker._disagg_draft_ctx_seeds == {}


def test_kill_switch_moves_no_window_but_keeps_the_gen_init_sizing(monkeypatch):
    """TRTLLM_M31_DISABLE_DISAGG_DRAFT_KV means no draft transfer only: a hooked model still sizes
    the generation side's draft mirror for its window and seeds the empty window."""
    from tensorrt_llm._torch.disaggregation.draft_kv import (
        DISABLE_DISAGG_DRAFT_KV_ENV,
        select_disagg_draft_kv_cache_manager,
    )
    from tensorrt_llm._torch.pyexecutor._util import engage_disagg_gen_init_window
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

    monkeypatch.setenv(DISABLE_DISAGG_DRAFT_KV_ENV, "1")
    monkeypatch.setattr(mgr_mod, "get_draft_token_length", lambda req: DRAFT)
    pool = _WindowedPool()
    m = _manager(_Cache(pool))
    worker = DFlashWorker.__new__(DFlashWorker)
    model = SimpleNamespace(spec_worker=worker, runtime_hooks=RuntimeHooks(disagg_draft_kv=True))

    assert select_disagg_draft_kv_cache_manager(model, m) is None  # no draft transfer
    assert engage_disagg_gen_init_window(model, m)  # the window sizing stays engaged
    assert m._disagg_draft_kv_receive is False and worker._disagg_draft_kv_engaged is False

    m._prepare_draft_resources(_batch(_gen_init_request()))  # 100K prompt: no resize failure
    assert pool.peak_pages <= _blocks(WINDOW + DRAFT + EXTRA) + 1
    assert worker._disagg_draft_ctx_seeds == {1: (PROMPT, _blocks(PROMPT), PROMPT)}
    assert not getattr(m, "_disagg_receive_ready", {})  # no receive fence without a transfer


def _run_context_only_flow(monkeypatch, engaged):
    """A disagg context-only worker: chunked context requests through the draft mirror and the
    drafter's metadata prepare, with or without the gen-init window sizing engaged."""
    from collections import deque

    from tensorrt_llm._torch.speculative import dflash as dflash_module
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

    monkeypatch.setattr(mgr_mod, "get_draft_token_length", lambda req: DRAFT)
    monkeypatch.setattr(dflash_module, "prefer_pinned", lambda: False)
    pool = _WindowedPool()
    caches = {rid: _Cache(pool) for rid in (1, 2)}
    resizes = []
    for rid, cache in caches.items():
        plain_resize = cache.resize

        def resize(capacity, history_length=None, rid=rid, plain_resize=plain_resize):
            resizes.append((rid, capacity, history_length))
            return plain_resize(capacity, history_length)

        cache.resize = resize
    m = _manager(None)
    m._mirror_draft_kv_cache = lambda req: caches[req.py_request_id]
    worker = DFlashWorker.__new__(DFlashWorker)
    worker._ctx_buf_inited = True
    worker._req_to_slot = {}
    worker._req_ctx_pos = {}
    worker._free_slots = deque([0, 1, 2])
    worker._graph_dummy_id_floor = 1 << 40
    worker._dummy_slot = 3
    worker._ctx_block_tables = torch.zeros(1)  # pool-bound: a seed would be written
    worker._batch_to_slot = torch.zeros(4, dtype=torch.long)
    writes = []
    worker._write_ctx_len = lambda updates, bases=None: writes.append((dict(updates), bases))
    gen_init_calls = []
    if engaged:
        from tensorrt_llm._torch.pyexecutor._util import engage_disagg_gen_init_window
        from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

        model = SimpleNamespace(
            spec_worker=worker, runtime_hooks=RuntimeHooks(disagg_draft_kv=True)
        )
        assert engage_disagg_gen_init_window(model, m)
        seed = m._disagg_gen_init_window
        m._disagg_gen_init_window = lambda *args: (gen_init_calls.append(args), seed(*args))

    chunk = 8192
    prompts = {1: 3 * chunk, 2: 1000}
    positions = {1: 0, 2: 0}
    while any(positions[rid] < prompts[rid] for rid in prompts):
        reqs = []
        for rid, prompt in prompts.items():
            pos = positions[rid]
            if pos >= prompt:
                continue
            size = min(chunk, prompt - pos)
            # A context-only request: never in DISAGG_GENERATION_INIT.
            reqs.append(
                SimpleNamespace(
                    py_request_id=rid,
                    prompt_len=prompt,
                    context_current_position=pos,
                    context_chunk_size=size,
                    is_last_context_chunk=pos + size == prompt,
                    is_disagg_generation_init_state=False,
                    py_draft_tokens=[0] * DRAFT,
                )
            )
            positions[rid] = pos + size
        m._prepare_draft_resources(SimpleNamespace(context_requests=reqs, generation_requests=[]))
        meta = SimpleNamespace(
            request_ids=[req.py_request_id for req in reqs],
            num_generations=0,
            batch_indices_cuda=torch.zeros(4, dtype=torch.int),
            _dflash_worker=worker,
        )
        DFlashSpecMetadata.prepare(meta)
        # _store_prefill_context assigns the context requests' slots after the forward.
        for req in reqs:
            worker._assign_slot(req.py_request_id)
    # The context-only request has finished: it leaves the batch.
    DFlashSpecMetadata.prepare(
        SimpleNamespace(
            request_ids=[],
            num_generations=0,
            batch_indices_cuda=torch.zeros(4, dtype=torch.int),
            _dflash_worker=worker,
        )
    )
    state = {
        "resizes": resizes,
        "caches": {rid: (c.capacity, c.history_length) for rid, c in caches.items()},
        "peak_pages": pool.peak_pages,
        "writes": writes,
        "req_to_slot": dict(worker._req_to_slot),
        "req_ctx_pos": dict(worker._req_ctx_pos),
        "free_slots": list(worker._free_slots),
    }
    return state, worker, m, gen_init_calls


def test_gen_init_sizing_is_inert_on_a_context_only_worker(monkeypatch):
    """A CTX worker of a hooked model engages the gen-init window sizing too (is_disagg), but it
    never holds a DISAGG_GENERATION_INIT request: the draft mirror's sizing, the drafter's slots
    and context lengths are exactly the stock ones, and no seed is ever recorded."""
    stock, stock_worker, stock_m, _ = _run_context_only_flow(monkeypatch, engaged=False)
    engaged, worker, m, gen_init_calls = _run_context_only_flow(monkeypatch, engaged=True)

    assert m._disagg_gen_init_window is not None and stock_m._disagg_gen_init_window is None
    assert engaged == stock
    assert gen_init_calls == []
    assert worker._disagg_draft_ctx_seeds == {}  # engaged, never filled
    assert stock_worker._disagg_draft_ctx_seeds is None
    assert not getattr(m, "_disagg_receive_ready", {})
    # The stock chunked-prefill sizing: chunk start as committed history, draft tail on the last.
    assert engaged["caches"][1] == (3 * 8192 + DRAFT + EXTRA, 2 * 8192)
    assert engaged["caches"][2] == (1000 + DRAFT + EXTRA, 0)
