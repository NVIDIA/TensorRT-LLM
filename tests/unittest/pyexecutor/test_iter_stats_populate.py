# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the per-iteration request-aggregate fields on InflightBatchingStats.

``PyExecutor._update_iter_stats`` populates the following members on
``stats.inflight_batching_stats`` in addition to the pre-existing
``num_context_requests`` / ``num_gen_requests`` / ``num_paused_requests`` /
``num_scheduled_requests`` / ``micro_batch_id``:

  * ``num_ctx_kv_tokens`` — tokens read from prior state
      (prefix-cache hits + previously-chunked tokens) summed across
      scheduled context requests; dummy-filtered.
  * ``num_gen_kv_tokens`` — total KV context length summed across
      scheduled generation requests; dummy-filtered.
  * ``num_queued_context_requests`` — normal context-type requests
      (``REQUEST_TYPE_CONTEXT_AND_GENERATION`` / ``REQUEST_TYPE_CONTEXT_ONLY``)
      sitting in the executor_request_queue; excludes shutdown/cancel
      control items and requests without a payload.
  * ``num_queued_ctx_tokens`` — prompt-token sum across the above.
  * ``num_queued_gen_requests`` — generation-only queued requests
      (``REQUEST_TYPE_GENERATION_ONLY``), typically disagg-decode items
      awaiting KV transfer before decoding.
  * ``num_queued_gen_kv_tokens`` — prompt-token sum across the above;
      acts as the KV budget each request will occupy post-transfer.
  * ``num_paused_kv_tokens`` — total KV context length summed across
      paused (preempted-decode) requests; dummy-filtered.

These tests invoke the real ``PyExecutor._update_iter_stats`` as an
unbound call against a minimal fake ``self``. Exercising the production
function directly (rather than a duplicated shim) catches drift from
refactors, renamed attributes, or silent unit changes in the populate
block.

Attention-DP-specific cases verify that dummy padding is excluded from
planner/FPM metrics and that rank-local iter-stats payloads are emitted
only when all ranks line up on the same iteration.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

from tensorrt_llm._torch.pyexecutor.adp_iter_stats import (
    _ITERATION_STATS_OPTIONAL_FIELDS,
    _ITERATION_STATS_SCALAR_FIELDS,
    ADPIterStatsBuffer,
)
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import RankIterStatsPayload, RankState
from tensorrt_llm.bindings.executor import InflightBatchingStats, IterationStats


class _StubRequest:
    """Stub LlmRequest exposing only the accessors ``_update_iter_stats`` reads.

    Attributes:
      py_last_context_chunk: tuple (start, end) — the (begin_compute,
        begin_compute + chunk_size) pair cached by ``_update_request_states``
        before state mutation. ``start`` is the primary source of
        ``num_ctx_kv_tokens``. Set to None for decode/paused reqs.
      context_current_position: fallback source consulted when
        ``py_last_context_chunk`` is None.
      _num_tokens: return value for ``get_num_tokens()``, used for decode
        and paused (preempted-decode) requests.
    """

    def __init__(
        self,
        *,
        context_chunk_size: int = 0,
        context_current_position: int = 0,
        num_tokens: int = 0,
        is_attention_dp_dummy: bool = False,
        is_cuda_graph_dummy: bool = False,
        is_dummy_request: bool = False,
    ):
        self.context_current_position = context_current_position
        # py_last_context_chunk = (begin_compute, end_compute). For a fresh
        # prefill, begin_compute == 0; for continuations, == prev position.
        if context_chunk_size > 0:
            self.py_last_context_chunk = (
                context_current_position,
                context_current_position + context_chunk_size,
            )
        else:
            self.py_last_context_chunk = None
        self._num_tokens = num_tokens
        self.is_attention_dp_dummy = is_attention_dp_dummy
        self.is_cuda_graph_dummy = is_cuda_graph_dummy
        self.is_dummy_request = is_dummy_request

    @property
    def is_dummy(self) -> bool:
        return self.is_attention_dp_dummy or self.is_cuda_graph_dummy or self.is_dummy_request

    def get_num_tokens(self, beam: int = 0) -> int:
        return self._num_tokens


class _StubScheduledBatch:
    def __init__(self, context_reqs=None, gen_reqs=None, paused_reqs=None):
        self.context_requests = list(context_reqs or [])
        self.generation_requests = list(gen_reqs or [])
        self.paused_requests = list(paused_reqs or [])

    @property
    def num_context_requests(self):
        return len(self.context_requests)

    @property
    def num_generation_requests(self):
        return len(self.generation_requests)


class _StubQueueItem:
    _next_id = 0

    def __init__(
        self,
        input_token_ids,
        *,
        is_normal_request: bool = True,
        request_type=None,
    ):
        # request_type defaults to REQUEST_TYPE_CONTEXT_AND_GENERATION,
        # matching the executor::Request constructor default. Tests that
        # exercise disagg routing (REQUEST_TYPE_CONTEXT_ONLY /
        # REQUEST_TYPE_GENERATION_ONLY) pass it explicitly.
        if request_type is None:
            from tensorrt_llm.bindings.executor import RequestType

            request_type = RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        self.request = types.SimpleNamespace(
            input_token_ids=input_token_ids,
            request_type=request_type,
        )
        self.is_normal_request = is_normal_request
        # Stable id so the populate block's drift-warning log has a value
        # to reference even if a future branch exercises item.id.
        _StubQueueItem._next_id += 1
        self.id = _StubQueueItem._next_id


def _build_fake_self(queued_items, iter_states, *, enable_attention_dp=False):
    """Minimal 'self' for ``PyExecutor._update_iter_stats(self, ...)``.

    Stubs only the ``self.*`` attributes the method actually reads:

    Setup reads (run before per-request aggregation):
      * ``max_num_active_requests``, ``iter_counter`` — scalars
      * ``executor_request_queue.get_request_queue_size()`` — for
        top-level ``num_queued_requests`` (not one of the fields
        exercised here)
      * ``resource_manager.resource_managers.get(...)`` — returns None so
        the KV-cache-stats block is skipped entirely
      * ``drafter`` — None (spec-decode block no-ops when
        ``stats.specdec_stats`` is None, which is the default on a fresh
        ``IterationStats``)

    Per-request aggregation reads:
      * ``executor_request_queue.get_request_queue().queue`` — source for
        ``num_queued_context_requests`` / ``num_queued_ctx_tokens``
      * ``model_engine.iter_states`` — stubbed but not read by the
        request-aggregate fields under test here; the regression test
        ``test_num_ctx_kv_tokens_ignores_iter_states_side_channel``
        verifies the populate block does not read it
    """
    fake = MagicMock()
    fake.max_num_active_requests = 64
    fake.iter_counter = 1
    fake.executor_request_queue.get_request_queue_size.return_value = len(queued_items)
    fake.executor_request_queue.get_request_queue.return_value.queue = queued_items
    fake.resource_manager.resource_managers.get.return_value = None
    fake.drafter = None
    fake.model_engine = types.SimpleNamespace(iter_states=iter_states)
    fake.enable_attention_dp = enable_attention_dp
    return fake


def _invoke_update_iter_stats(
    scheduled_batch, queued_items, *, num_ctx_tokens, enable_attention_dp=False
):
    """Call real ``PyExecutor._update_iter_stats`` unbound; return the stats.

    Patches ``torch.cuda.mem_get_info`` so the method can run on hosts
    without a live CUDA context (the call is unconditional for
    ``gpu_mem_usage`` but the value is not consumed by the fields
    under test).

    Parameters
    ----------
    scheduled_batch : _StubScheduledBatch
    queued_items    : list[_StubQueueItem]
    num_ctx_tokens  : int | None
        If int, wired into ``model_engine.iter_states = {"num_ctx_tokens": num_ctx_tokens}``.
        If None, ``iter_states`` is set to None. The populate block under
        test does not consume this value; it is plumbed so regression
        tests can verify the side channel remains unread.
    """
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    iter_states = None if num_ctx_tokens is None else {"num_ctx_tokens": num_ctx_tokens}

    fake_self = _build_fake_self(queued_items, iter_states, enable_attention_dp=enable_attention_dp)

    stats = IterationStats()
    # The method reads ``stats.inflight_batching_stats.*`` unconditionally;
    # the default on a fresh IterationStats is None, so we allocate one.
    stats.inflight_batching_stats = InflightBatchingStats()

    with patch(
        "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.mem_get_info",
        return_value=(1 << 30, 1 << 30),
    ):
        PyExecutor._update_iter_stats(
            fake_self,
            stats,
            iter_latency_ms=10.0,
            num_completed_requests=0,
            scheduled_batch=scheduled_batch,
            micro_batch_id=0,
        )
    return stats


# ---------------------------------------------------------------------------
# Populate tests: call the real ``_update_iter_stats`` and assert on the
# inflight_batching_stats request-aggregate fields it populates.
# These are the fields Dynamo publishes as forward-pass metrics.
# ---------------------------------------------------------------------------


def test_empty_iteration():
    stats = _invoke_update_iter_stats(_StubScheduledBatch(), [], num_ctx_tokens=0)
    ifb = stats.inflight_batching_stats
    assert ifb.num_context_requests == 0
    assert ifb.num_gen_requests == 0
    assert ifb.num_paused_requests == 0
    assert ifb.num_ctx_kv_tokens == 0
    assert ifb.num_gen_kv_tokens == 0
    assert ifb.num_queued_context_requests == 0
    assert ifb.num_queued_ctx_tokens == 0
    assert ifb.num_queued_gen_requests == 0
    assert ifb.num_queued_gen_kv_tokens == 0
    assert ifb.num_paused_kv_tokens == 0


def test_prefill_only_no_prefix_cache():
    # Two fresh prefill requests: prompts of 100 and 200 tokens, chunk size
    # == full prompt (no chunked prefill). No prefix cache hits.
    reqs = [
        _StubRequest(context_chunk_size=100, context_current_position=0),
        _StubRequest(context_chunk_size=200, context_current_position=0),
    ]
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=reqs), [], num_ctx_tokens=300
    )
    ifb = stats.inflight_batching_stats
    assert ifb.num_context_requests == 2
    assert ifb.num_ctx_kv_tokens == 0  # py_last_context_chunk[0] == 0 for both


def test_prefill_with_prefix_cache_hit():
    # Prompt 1000 tokens; 256 already in prefix cache (prepopulatedPromptLen).
    # Chunk size = remaining = 744. py_last_context_chunk = (256, 1000);
    # start=256 is the precomputed-tokens count.
    reqs = [
        _StubRequest(context_chunk_size=744, context_current_position=256),
    ]
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=reqs), [], num_ctx_tokens=744
    )
    ifb = stats.inflight_batching_stats
    assert ifb.num_context_requests == 1
    assert ifb.num_ctx_kv_tokens == 256


def test_chunked_prefill_continuation():
    # Chunked prefill: 3-chunk request, each chunk 512. This is step 2:
    # chunk size 512, previously computed 512 (== context_current_position).
    # py_last_context_chunk = (512, 1024); start=512.
    reqs = [
        _StubRequest(context_chunk_size=512, context_current_position=512),
    ]
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=reqs), [], num_ctx_tokens=512
    )
    ifb = stats.inflight_batching_stats
    assert ifb.num_context_requests == 1
    assert ifb.num_ctx_kv_tokens == 512


def test_decode_only():
    # Two decode requests: 1024 total context and 2048 total context.
    reqs = [
        _StubRequest(num_tokens=1024),
        _StubRequest(num_tokens=2048),
    ]
    stats = _invoke_update_iter_stats(_StubScheduledBatch(gen_reqs=reqs), [], num_ctx_tokens=0)
    ifb = stats.inflight_batching_stats
    assert ifb.num_gen_requests == 2
    assert ifb.num_gen_kv_tokens == 3072
    assert ifb.num_context_requests == 0
    assert ifb.num_ctx_kv_tokens == 0


def test_mixed_prefill_and_decode():
    ctx = [_StubRequest(context_chunk_size=128, context_current_position=0)]
    gen = [_StubRequest(num_tokens=500), _StubRequest(num_tokens=700)]
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=ctx, gen_reqs=gen), [], num_ctx_tokens=128
    )
    ifb = stats.inflight_batching_stats
    assert ifb.num_context_requests == 1
    assert ifb.num_gen_requests == 2
    assert ifb.num_gen_kv_tokens == 1200


def test_queued_context_requests_from_request_queue():
    items = [
        _StubQueueItem(input_token_ids=list(range(256))),
        _StubQueueItem(input_token_ids=list(range(1024))),
    ]
    stats = _invoke_update_iter_stats(_StubScheduledBatch(), items, num_ctx_tokens=0)
    ifb = stats.inflight_batching_stats
    assert ifb.num_queued_context_requests == 2
    assert ifb.num_queued_ctx_tokens == 1280


def test_queued_filters_non_normal_requests():
    # Shutdown / cancel / control items should be ignored.
    items = [
        _StubQueueItem(input_token_ids=list(range(100)), is_normal_request=False),
        _StubQueueItem(input_token_ids=list(range(50))),
    ]
    stats = _invoke_update_iter_stats(_StubScheduledBatch(), items, num_ctx_tokens=0)
    ifb = stats.inflight_batching_stats
    assert ifb.num_queued_context_requests == 1
    assert ifb.num_queued_ctx_tokens == 50


def test_queued_routes_by_request_type():
    # Disaggregated serving: a decode engine receives
    # REQUEST_TYPE_GENERATION_ONLY items that await KV transfer from a
    # prefill engine before starting decode. They belong in the
    # queued-gen counters, not the queued-context counters. A prefill
    # engine sees REQUEST_TYPE_CONTEXT_ONLY items, which are real
    # queued-prefill work and count in the context counters alongside
    # default-type (CONTEXT_AND_GENERATION) items.
    from tensorrt_llm.bindings.executor import RequestType

    items = [
        # Non-disagg default: queued-context work, len=100.
        _StubQueueItem(
            input_token_ids=list(range(100)),
            request_type=RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION,
        ),
        # Disagg prefill side: queued-context work, len=200.
        _StubQueueItem(
            input_token_ids=list(range(200)),
            request_type=RequestType.REQUEST_TYPE_CONTEXT_ONLY,
        ),
        # Disagg decode side: queued-gen work, len=512.
        _StubQueueItem(
            input_token_ids=list(range(512)),
            request_type=RequestType.REQUEST_TYPE_GENERATION_ONLY,
        ),
        # Disagg decode side: queued-gen work, len=1024.
        _StubQueueItem(
            input_token_ids=list(range(1024)),
            request_type=RequestType.REQUEST_TYPE_GENERATION_ONLY,
        ),
    ]
    stats = _invoke_update_iter_stats(_StubScheduledBatch(), items, num_ctx_tokens=0)
    ifb = stats.inflight_batching_stats
    # Two context-flavoured items -> queued-context counters.
    assert ifb.num_queued_context_requests == 2
    assert ifb.num_queued_ctx_tokens == 300  # 100 + 200
    # Two generation-only items -> queued-gen counters.
    assert ifb.num_queued_gen_requests == 2
    assert ifb.num_queued_gen_kv_tokens == 1536  # 512 + 1024


def test_paused_decode_requests():
    paused = [
        _StubRequest(num_tokens=300),
        _StubRequest(num_tokens=800),
    ]
    stats = _invoke_update_iter_stats(_StubScheduledBatch(paused_reqs=paused), [], num_ctx_tokens=0)
    ifb = stats.inflight_batching_stats
    assert ifb.num_paused_requests == 2
    assert ifb.num_paused_kv_tokens == 1100


def test_dummy_filtering_on_kv_token_fields():
    """Verify dummy requests are excluded from KV-token-weighted counters."""
    # Dummy-padding added by Attention-DP or CUDA graph capture must not
    # contribute to the KV-token-weighted fields under test
    # (num_ctx_kv_tokens, num_gen_kv_tokens, num_paused_kv_tokens).
    # The existing count fields (num_context_requests / num_gen_requests /
    # num_paused_requests) are set directly from ``scheduled_batch``
    # properties earlier in _update_iter_stats, so they DO include dummies.
    # This asymmetry is by design for the new KV-token fields: Dynamo
    # wants "real work only" for its autoscaling signal.
    ctx = [
        _StubRequest(
            context_chunk_size=100,
            context_current_position=0,
            is_attention_dp_dummy=True,
        ),
        _StubRequest(
            context_chunk_size=300,
            context_current_position=75,
            is_cuda_graph_dummy=True,
        ),
        _StubRequest(context_chunk_size=200, context_current_position=50),
    ]
    gen = [
        _StubRequest(num_tokens=1024, is_attention_dp_dummy=True),
        _StubRequest(num_tokens=2048, is_cuda_graph_dummy=True),
    ]
    paused = [
        _StubRequest(num_tokens=500, is_attention_dp_dummy=True),
        _StubRequest(num_tokens=700, is_cuda_graph_dummy=True),
    ]
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=ctx, gen_reqs=gen, paused_reqs=paused),
        [],
        num_ctx_tokens=300,
    )
    ifb = stats.inflight_batching_stats
    # Count fields include dummies (populated directly from scheduled_batch).
    assert ifb.num_context_requests == 3
    assert ifb.num_gen_requests == 2
    assert ifb.num_paused_requests == 2
    # KV-token-weighted new fields filter dummies.
    assert ifb.num_ctx_kv_tokens == 50  # only the non-dummy's start
    assert ifb.num_gen_kv_tokens == 0  # dummy gen filtered
    assert ifb.num_paused_kv_tokens == 0  # dummy paused filtered


def test_attention_dp_dummy_filtering_on_count_fields():
    """Verify Attention-DP mode excludes dummy padding from rank-local counts."""
    # Under attention-DP, the rank-local payload emitted for each rank must
    # exclude ADP and CUDA graph dummy padding from request counts too.
    ctx = [
        _StubRequest(
            context_chunk_size=100,
            context_current_position=0,
            is_attention_dp_dummy=True,
        ),
        _StubRequest(
            context_chunk_size=150,
            context_current_position=25,
            is_cuda_graph_dummy=True,
        ),
        _StubRequest(context_chunk_size=200, context_current_position=50),
    ]
    gen = [
        _StubRequest(num_tokens=1024, is_attention_dp_dummy=True),
        _StubRequest(num_tokens=1536, is_cuda_graph_dummy=True),
        _StubRequest(num_tokens=2048),
    ]
    paused = [
        _StubRequest(num_tokens=500, is_attention_dp_dummy=True),
        _StubRequest(num_tokens=600, is_cuda_graph_dummy=True),
        _StubRequest(num_tokens=700),
    ]

    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=ctx, gen_reqs=gen, paused_reqs=paused),
        [],
        num_ctx_tokens=300,
        enable_attention_dp=True,
    )
    ifb = stats.inflight_batching_stats
    assert ifb.num_context_requests == 1
    assert ifb.num_gen_requests == 1
    assert ifb.num_paused_requests == 1
    assert ifb.num_scheduled_requests == 2
    assert ifb.num_ctx_kv_tokens == 50
    assert ifb.num_gen_kv_tokens == 2048
    assert ifb.num_paused_kv_tokens == 700


def test_full_mixed_iteration():
    # Realistic scenario: 3 prefill (1 fresh, 2 continuing chunks), 4 decode,
    # 2 preempted, 3 queued.
    ctx = [
        _StubRequest(context_chunk_size=1024, context_current_position=0),
        _StubRequest(context_chunk_size=512, context_current_position=1024),
        _StubRequest(context_chunk_size=256, context_current_position=768),
    ]
    gen = [_StubRequest(num_tokens=n) for n in (500, 1500, 2500, 3500)]
    paused = [_StubRequest(num_tokens=n) for n in (400, 900)]
    qitems = [_StubQueueItem(input_token_ids=list(range(n))) for n in (256, 512, 1024)]

    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=ctx, gen_reqs=gen, paused_reqs=paused),
        qitems,
        num_ctx_tokens=1024 + 512 + 256,
    )

    ifb = stats.inflight_batching_stats
    assert ifb.num_context_requests == 3
    # py_last_context_chunk[0] per req = 0, 1024, 768
    assert ifb.num_ctx_kv_tokens == 0 + 1024 + 768
    assert ifb.num_gen_requests == 4
    assert ifb.num_gen_kv_tokens == 500 + 1500 + 2500 + 3500
    assert ifb.num_queued_context_requests == 3
    assert ifb.num_queued_ctx_tokens == 256 + 512 + 1024
    assert ifb.num_paused_requests == 2
    assert ifb.num_paused_kv_tokens == 400 + 900


def test_num_ctx_kv_tokens_ignores_iter_states_side_channel():
    """Regression guard: num_ctx_kv_tokens must not read iter_states.

    Under overlap scheduling the ``model_engine.iter_states["num_ctx_tokens"]``
    side channel is rewritten by the current iteration's forward step
    before the previous batch's stats are emitted, so consuming it would
    report the wrong iteration's count. This test pushes a deliberately
    wrong value into ``iter_states`` and confirms the field is still
    computed from per-request ``py_last_context_chunk``. If someone
    re-introduces the side-channel read, this test fails.
    """
    ctx = [_StubRequest(context_chunk_size=744, context_current_position=256)]
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=ctx), [], num_ctx_tokens=99999
    )
    # py_last_context_chunk = (256, 1000); precomputed = start = 256.
    # If the side channel were consulted, we'd see 99999.
    ifb = stats.inflight_batching_stats
    assert ifb.num_context_requests == 1
    assert ifb.num_ctx_kv_tokens == 256


# ---------------------------------------------------------------------------
# Attention-DP fanout tests: completed rank-local payloads are carried by the
# next ADP allgather, then rank 0 appends one row per ADP rank.
# ---------------------------------------------------------------------------


def _make_adp_iteration_stats(
    *,
    iter_id=7,
    num_context_requests=0,
    num_ctx_tokens=0,
    num_ctx_kv_tokens=0,
    num_gen_requests=0,
    num_gen_kv_tokens=0,
    num_paused_requests=0,
    num_paused_kv_tokens=0,
    num_queued_context_requests=0,
    num_queued_ctx_tokens=0,
    num_queued_gen_requests=0,
    num_queued_gen_kv_tokens=0,
):
    ifb = InflightBatchingStats()
    ifb.num_context_requests = num_context_requests
    ifb.num_ctx_tokens = num_ctx_tokens
    ifb.num_ctx_kv_tokens = num_ctx_kv_tokens
    ifb.num_gen_requests = num_gen_requests
    ifb.num_gen_kv_tokens = num_gen_kv_tokens
    ifb.num_paused_requests = num_paused_requests
    ifb.num_paused_kv_tokens = num_paused_kv_tokens
    ifb.num_scheduled_requests = num_context_requests + num_gen_requests
    ifb.num_queued_context_requests = num_queued_context_requests
    ifb.num_queued_ctx_tokens = num_queued_ctx_tokens
    ifb.num_queued_gen_requests = num_queued_gen_requests
    ifb.num_queued_gen_kv_tokens = num_queued_gen_kv_tokens

    stats = IterationStats()
    stats.iter = iter_id
    stats.inflight_batching_stats = ifb
    return stats


def _build_adp_stats_buffer(pending_stats, *, is_rank0=True):
    buffer = ADPIterStatsBuffer()
    buffer.queue(
        pending_stats,
        ["req-stats"] if is_rank0 else None,
        kv_iter_stats={0: "pending-kv"} if is_rank0 else None,
        is_rank0=is_rank0,
    )
    return buffer


def test_attention_dp_fanout_emits_rank_local_rows_with_rank0_queue():
    """Verify ADP fanout emits one rank-local row per rank with queue on rank 0."""
    # Rank 0 emits one stats row per ADP rank. Scheduled fields stay
    # rank-local so FPM can reveal load imbalance, while queued fields remain
    # rank-0-owned because the executor request queue lives on rank 0.
    rank0_stats = _make_adp_iteration_stats(
        iter_id=9,
        num_context_requests=1,
        num_ctx_tokens=100,
        num_ctx_kv_tokens=10,
        num_gen_requests=2,
        num_gen_kv_tokens=20,
        num_paused_requests=1,
        num_paused_kv_tokens=5,
        num_queued_context_requests=7,
        num_queued_ctx_tokens=700,
        num_queued_gen_requests=8,
        num_queued_gen_kv_tokens=800,
    )
    rank0_stats.iter_latency_ms = 12.5
    buffer = _build_adp_stats_buffer(rank0_stats)
    rank0_state = RankState(rank=0, iter_stats=buffer.next_payload())
    rank1_state = RankState(
        rank=1,
        iter_stats=RankIterStatsPayload(
            has_iter_stats=1,
            iter_stats_iter=9,
            num_context_requests=3,
            num_ctx_tokens=300,
            num_ctx_kv_tokens=30,
            num_gen_requests=4,
            num_gen_kv_tokens=40,
            num_paused_requests=2,
            num_paused_kv_tokens=25,
        ),
    )

    records = buffer.finalize([rank0_state, rank1_state], is_rank0=True)

    assert len(records) == 2

    rank0_record = records[0]
    rank0_row = rank0_record.stats
    rank0_ifb = rank0_row.inflight_batching_stats
    assert rank0_record.attention_dp_rank == 0
    assert rank0_row.iter_latency_ms == 12.5
    assert rank0_ifb.num_context_requests == 1
    assert rank0_ifb.num_ctx_tokens == 100
    assert rank0_ifb.num_ctx_kv_tokens == 10
    assert rank0_ifb.num_gen_requests == 2
    assert rank0_ifb.num_gen_kv_tokens == 20
    assert rank0_ifb.num_paused_requests == 1
    assert rank0_ifb.num_paused_kv_tokens == 5
    assert rank0_ifb.num_scheduled_requests == 3
    assert rank0_ifb.num_queued_context_requests == 7
    assert rank0_ifb.num_queued_ctx_tokens == 700
    assert rank0_ifb.num_queued_gen_requests == 8
    assert rank0_ifb.num_queued_gen_kv_tokens == 800
    assert rank0_record.req_stats == ["req-stats"]
    assert rank0_record.kv_iter_stats == {0: "pending-kv"}

    rank1_record = records[1]
    rank1_row = rank1_record.stats
    rank1_ifb = rank1_row.inflight_batching_stats
    assert rank1_record.attention_dp_rank == 1
    assert rank1_row.iter_latency_ms == 12.5
    assert rank1_ifb.num_context_requests == 3
    assert rank1_ifb.num_ctx_tokens == 300
    assert rank1_ifb.num_ctx_kv_tokens == 30
    assert rank1_ifb.num_gen_requests == 4
    assert rank1_ifb.num_gen_kv_tokens == 40
    assert rank1_ifb.num_paused_requests == 2
    assert rank1_ifb.num_paused_kv_tokens == 25
    assert rank1_ifb.num_scheduled_requests == 7
    # Expected to be zero/None because queued/request/KV stats are reported
    # only on rank 0.
    assert rank1_ifb.num_queued_context_requests == 0
    assert rank1_ifb.num_queued_ctx_tokens == 0
    assert rank1_ifb.num_queued_gen_requests == 0
    assert rank1_ifb.num_queued_gen_kv_tokens == 0
    assert rank1_record.req_stats is None
    assert rank1_record.kv_iter_stats is None

    assert buffer._payloads == {}
    assert buffer.next_payload() is None


def test_attention_dp_fanout_waits_for_complete_matching_payloads():
    """Verify ADP fanout waits until every rank reports the rank-0 iteration."""
    rank0_stats = _make_adp_iteration_stats(iter_id=9, num_context_requests=1)
    buffer = _build_adp_stats_buffer(rank0_stats)
    rank0_state = RankState(rank=0, iter_stats=buffer.next_payload())
    missing_rank1_state = RankState(rank=1)

    records = buffer.finalize([rank0_state, missing_rank1_state], is_rank0=True)

    assert records == []
    assert buffer.next_payload() is rank0_state.iter_stats

    mismatched_rank1_state = RankState(
        rank=1,
        iter_stats=RankIterStatsPayload(
            has_iter_stats=1,
            iter_stats_iter=8,
            num_context_requests=2,
        ),
    )

    records = buffer.finalize([rank0_state, mismatched_rank1_state], is_rank0=True)

    assert records == []
    assert buffer.next_payload() is rank0_state.iter_stats


def test_attention_dp_fanout_buffers_multiple_pending_payloads():
    """Verify ADP fanout keeps multiple pending iterations ordered by iter id."""
    rank0_stats_9 = _make_adp_iteration_stats(iter_id=9, num_context_requests=1)
    rank0_stats_10 = _make_adp_iteration_stats(iter_id=10, num_context_requests=2)
    buffer = _build_adp_stats_buffer(rank0_stats_9)

    buffer.queue(
        rank0_stats_10,
        ["req-stats-10"],
        kv_iter_stats={0: "pending-kv-10"},
        is_rank0=True,
    )

    assert sorted(buffer._payloads) == [9, 10]
    assert buffer.next_payload().iter_stats_iter == 9


def test_attention_dp_fanout_clears_when_rank0_stats_object_missing():
    """Verify ADP fanout drops stale state if rank-0 stats are unavailable."""
    rank0_stats = _make_adp_iteration_stats(iter_id=9, num_context_requests=1)
    buffer = _build_adp_stats_buffer(rank0_stats)
    rank0_state = RankState(rank=0, iter_stats=buffer.next_payload())
    rank1_state = RankState(
        rank=1,
        iter_stats=RankIterStatsPayload(
            has_iter_stats=1,
            iter_stats_iter=9,
            num_context_requests=2,
        ),
    )
    buffer._rank0_iter_stats.pop(9)

    records = buffer.finalize([rank0_state, rank1_state], is_rank0=True)

    assert records == []
    assert buffer._payloads == {}
    assert buffer.next_payload() is None
    assert buffer._rank0_req_stats == {}
    assert buffer._rank0_kv_iter_stats == {}


def test_attention_dp_fanout_aligns_non_rank0_to_rank0_iter():
    """Verify non-rank0 buffers synthesize zero payloads to align to rank 0."""
    rank1_stats = _make_adp_iteration_stats(iter_id=10, num_context_requests=3)
    buffer = _build_adp_stats_buffer(rank1_stats, is_rank0=False)
    rank0_state = RankState(
        rank=0,
        iter_stats=RankIterStatsPayload(
            has_iter_stats=1,
            iter_stats_iter=9,
            num_context_requests=1,
        ),
    )
    rank1_state = RankState(rank=1, iter_stats=buffer.next_payload())

    records = buffer.finalize([rank0_state, rank1_state], is_rank0=False)

    assert records == []
    next_payload = buffer.next_payload()
    assert next_payload.iter_stats_iter == 9
    assert next_payload.num_context_requests == 0
    assert 9 in buffer._synthetic_iters
    assert 10 in buffer._payloads


def test_attention_dp_fanout_copy_lists_cover_iteration_stats_fields():
    """Guard the hardcoded field lists used when cloning rank-0 stats."""
    actual_fields = {
        field
        for field in dir(IterationStats())
        if not field.startswith("_") and field != "to_json_str"
    }
    copied_fields = (
        set(_ITERATION_STATS_SCALAR_FIELDS)
        | set(_ITERATION_STATS_OPTIONAL_FIELDS)
        | {"inflight_batching_stats"}
    )

    assert copied_fields == actual_fields


# ---------------------------------------------------------------------------
# Serializer test: confirm the C++ NLOHMANN serializer exposes every new
# InflightBatchingStats member under its expected camelCase key, nested
# inside the ``inflightBatchingStats`` object. Distinct values per field
# so a cross-wiring bug (e.g. swapped keys) fails the assertion rather
# than round-tripping silently.
# ---------------------------------------------------------------------------


def test_to_json_str_roundtrip_includes_new_inflight_batching_stats_fields():
    """Confirm the C++ NLOHMANN serializer emits every new InflightBatchingStats key.

    Every new request-aggregate field must appear in the JSON dict under
    its expected camelCase key, nested inside ``inflightBatchingStats``.
    """
    import json as _json

    ifb = InflightBatchingStats()
    ifb.num_scheduled_requests = 12
    ifb.num_context_requests = 5
    ifb.num_gen_requests = 7
    ifb.num_paused_requests = 3
    ifb.num_ctx_tokens = 2048
    ifb.micro_batch_id = 4
    ifb.avg_num_decoded_tokens_per_iter = 1.25
    ifb.num_ctx_kv_tokens = 256
    ifb.num_gen_kv_tokens = 9000
    ifb.num_queued_context_requests = 11
    ifb.num_queued_ctx_tokens = 4096
    ifb.num_queued_gen_requests = 6
    ifb.num_queued_gen_kv_tokens = 2345
    ifb.num_paused_kv_tokens = 1500

    stats = IterationStats()
    stats.inflight_batching_stats = ifb

    d = _json.loads(stats.to_json_str())
    ifb_d = d["inflightBatchingStats"]
    # Existing keys still round-trip.
    assert ifb_d["numScheduledRequests"] == 12
    assert ifb_d["numContextRequests"] == 5
    assert ifb_d["numGenRequests"] == 7
    assert ifb_d["numPausedRequests"] == 3
    assert ifb_d["numCtxTokens"] == 2048
    # New keys round-trip under the expected camelCase.
    assert ifb_d["numCtxKvTokens"] == 256
    assert ifb_d["numGenKvTokens"] == 9000
    assert ifb_d["numQueuedContextRequests"] == 11
    assert ifb_d["numQueuedCtxTokens"] == 4096
    assert ifb_d["numQueuedGenRequests"] == 6
    assert ifb_d["numQueuedGenKvTokens"] == 2345
    assert ifb_d["numPausedKvTokens"] == 1500
