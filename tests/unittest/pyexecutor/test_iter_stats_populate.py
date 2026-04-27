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
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

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


def _build_fake_self(queued_items, iter_states):
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
    return fake


def _invoke_update_iter_stats(scheduled_batch, queued_items, *, num_ctx_tokens):
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

    fake_self = _build_fake_self(queued_items, iter_states)

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


def test_attention_dp_dummy_filtering_on_kv_token_fields():
    # Dummy-padding added by ``_pad_attention_dp_dummy_request`` must not
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
        _StubRequest(context_chunk_size=200, context_current_position=50),
    ]
    gen = [_StubRequest(num_tokens=1024, is_attention_dp_dummy=True)]
    paused = [_StubRequest(num_tokens=500, is_attention_dp_dummy=True)]
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=ctx, gen_reqs=gen, paused_reqs=paused),
        [],
        num_ctx_tokens=300,
    )
    ifb = stats.inflight_batching_stats
    # Count fields include dummies (populated directly from scheduled_batch).
    assert ifb.num_context_requests == 2
    assert ifb.num_gen_requests == 1
    assert ifb.num_paused_requests == 1
    # KV-token-weighted new fields filter dummies.
    assert ifb.num_ctx_kv_tokens == 50  # only the non-dummy's start
    assert ifb.num_gen_kv_tokens == 0  # dummy gen filtered
    assert ifb.num_paused_kv_tokens == 0  # dummy paused filtered


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
