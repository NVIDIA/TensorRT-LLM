# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV connector prefix phases against a *real* ``KVCacheManagerV2``.

``test_kv_connector_v2_prefix.py`` drives the same three phases against a stub
cache. That is what makes it fast and exhaustive, and it is the right place for
the ordering and state-machine rules -- but a stub cannot show that a real
``_KVCache`` survives the sequence: that ``resize`` finds real pages for the
offered prefix, that ``history_length`` really moves, that suspend/resume across
a deferral leaves the request askable exactly once, and that the page slots
handed to the connector at delivery are distinct and real.

The engine-level suite cannot show it either, for a different reason. Whether a
request is deferred there depends on whether it reaches the scheduler in the
same pass as the request that outbids it, and that is a race: the same test was
observed asking both requests in the first pass when run alone, and asking the
second one an iteration later when run after the rest of the suite. Deferral is
therefore driven here directly -- ``prepare_context`` without the
``prepare_resources`` that would have followed it in an iteration where the
request actually ran -- which is exactly what the scheduler does to a request
that loses the token budget after being prepared (scheduler_v2.py:554-558).

These tests allocate device memory pools.
"""

import gc
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

# These build a real manager, which allocates device pools. The directory is
# listed in the GPU-less l0_cpu stage, so the requirement is declared rather
# than left to fail at `torch.cuda.init()`.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocates real KV cache pools"
)

TOKENS_PER_BLOCK = 32
PROMPT_LEN = 96
OFFER_TOKENS = 32


class FakeConnectorManager:
    """Records what the phases tell the connector, in order.

    Same shape as the one in ``test_kv_connector_v2_prefix.py``, plus the
    ``build_scheduler_output`` that ``prepare_resources`` calls after delivery.
    """

    def __init__(self, num_matched=OFFER_TOKENS, load_async=False):
        self.num_matched = num_matched
        self.load_async = load_async
        self.queries = []
        self.commits = []
        self.cancels = []
        self.allocs = []

    def query_num_new_matched_tokens(self, request, num_computed_tokens):
        self.queries.append((request.py_request_id, num_computed_tokens))
        return self.num_matched, self.load_async

    def commit_new_matched_tokens(self, request, num_tokens, load_kv_async):
        self.commits.append((request.py_request_id, num_tokens, load_kv_async))
        request.py_num_connector_matched_tokens = num_tokens

    def cancel_load(self, request, start, end):
        self.cancels.append((request.py_request_id, start, end))

    def update_state_after_alloc(self, request, block_ids):
        self.allocs.append((request.py_request_id, list(block_ids)))

    def build_scheduler_output(self, scheduled_batch, kv_cache_manager):
        pass


def make_manager(connector, **overrides):
    kwargs = dict(
        kv_cache_config=KvCacheConfig(max_tokens=2048, enable_block_reuse=True),
        kv_cache_type=CacheType.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
        kv_connector_manager=connector,
    )
    kwargs.update(overrides)
    return KVCacheManagerV2(**kwargs)


def make_request(request_id=1, prompt_len=PROMPT_LEN):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def deliver(manager, request):
    """One `prepare_resources`, i.e. the request reached the final batch."""
    manager.prepare_resources(SimpleNamespace(context_requests=[request], generation_requests=[]))


@pytest.fixture
def connector():
    return FakeConnectorManager()


@pytest.fixture
def manager(connector):
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    mgr = make_manager(connector)
    yield mgr
    mgr.shutdown()
    del mgr
    gc.collect()
    torch.cuda.empty_cache()


def test_offer_is_backed_by_real_pages(manager, connector):
    """Phase 2 against a real `_KVCache`: capacity, history, and slots.

    `history_length` is what the stub suite can only observe as an argument to
    a fake `resize`. Here it is read back off the cache that the forward pass
    would use, which is the difference between "we called resize correctly" and
    "the offered prefix is resident".
    """
    request = make_request()

    assert manager.prepare_context(request)

    kv_cache = manager.kv_cache_map[request.py_request_id]
    assert request.context_current_position == OFFER_TOKENS
    assert kv_cache.capacity >= OFFER_TOKENS
    assert kv_cache.history_length == OFFER_TOKENS
    assert kv_cache.is_active

    deliver(manager, request)

    assert connector.commits == [(request.py_request_id, OFFER_TOKENS, False)]
    assert len(connector.allocs) == 1

    _, page_indices = connector.allocs[0]
    assert len(page_indices) >= OFFER_TOKENS // TOKENS_PER_BLOCK
    assert all(index != BAD_PAGE_INDEX for index in page_indices)
    assert len(set(page_indices)) == len(page_indices)


def test_deferred_request_is_asked_once_and_delivered_when_it_runs(manager, connector):
    """Prepared, dropped before the batch, prepared again, then run.

    The cache is suspended in between, which is what `_revert_context_resize`
    does to a request that loses its slot after being prepared. Both attempts
    must reach the same offer end, and the connector must see exactly one query
    and exactly one commit -- it took ownership of remote blocks inside the
    first query, and a second would double-pin them.
    """
    request = make_request()

    # Iteration 1: prepared, then outbid before it reached the batch.
    assert manager.prepare_context(request)
    kv_cache = manager.kv_cache_map[request.py_request_id]
    kv_cache.suspend()

    assert connector.queries == [(request.py_request_id, 0)]
    assert connector.commits == [], "nothing may be recorded until it runs"

    # Iteration 2: prepared again, and this time it runs.
    assert manager.prepare_context(request)
    assert kv_cache.is_active, "phase 2 must have found an active cache"
    assert request.context_current_position == OFFER_TOKENS
    assert kv_cache.history_length == OFFER_TOKENS

    deliver(manager, request)

    assert connector.queries == [(request.py_request_id, 0)]
    assert connector.commits == [(request.py_request_id, OFFER_TOKENS, False)]
    assert len(connector.allocs) == 1


def test_delivering_twice_reports_one_allocation(manager, connector):
    """An asynchronously loaded request re-enters on its first context chunk.

    `py_connector_delivered` is what stops that second pass re-recording the
    load and firing a second `update_state_after_alloc` for one allocation.
    """
    request = make_request()

    assert manager.prepare_context(request)
    deliver(manager, request)
    deliver(manager, request)

    assert len(connector.commits) == 1
    assert len(connector.allocs) == 1


def test_undelivered_offer_is_released_when_the_request_is_freed(manager, connector):
    """Phase 1 is speculative, so an offer can outlive the request.

    Without this the connector holds the remote blocks it took ownership of for
    the rest of the process's life.
    """
    request = make_request()

    assert manager.prepare_context(request)
    manager.free_resources(request)

    assert connector.cancels == [(request.py_request_id, 0, OFFER_TOKENS)]


def test_delivered_offer_is_not_released_when_the_request_is_freed(manager, connector):
    request = make_request()

    assert manager.prepare_context(request)
    deliver(manager, request)
    manager.free_resources(request)

    assert connector.cancels == []
