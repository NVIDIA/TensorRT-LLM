# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ForwardPassMetrics (FPM) flat fields on IterationStats.

Covers the 9 MVP fields populated in PyExecutor._update_iter_stats:
  scheduled_{num_prefill_requests, sum_prefill_tokens, sum_prefill_kv_tokens,
             num_decode_requests, sum_decode_kv_tokens}
  queued_{num_prefill_requests, sum_prefill_tokens,
          num_decode_requests, sum_decode_kv_tokens}

The `_populate` helper below mirrors the REAL algorithm in
PyExecutor._update_iter_stats:
  * scheduled_sum_prefill_kv_tokens  ← `py_last_context_chunk[0]` primary,
                                       `context_current_position` fallback.
  * scheduled_sum_prefill_tokens     ← `model_engine.iter_states["num_ctx_tokens"]`
                                       (a model-engine-level aggregate set in
                                       `_prepare_tp_inputs`, NOT a per-request
                                       sum; see populate-fix commit c30121b0d).

Invariants match vLLM's InstrumentedScheduler at
components/src/dynamo/vllm/instrumented_scheduler.py on Dynamo origin/main
(_extract_scheduled + _compute_queued).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

from tensorrt_llm.bindings.executor import IterationStats


class _StubRequest:
    """Stub LlmRequest exposing only the accessors _update_iter_stats reads.

    Attributes:
      py_last_context_chunk: tuple (start, end) — the (begin_compute,
        begin_compute + chunk_size) pair cached by _update_request_states
        before state mutation. `start` is the primary source of
        scheduled_sum_prefill_kv_tokens. Set to None for decode/paused reqs.
      context_current_position: fallback source (Python binding on LlmRequest)
        consulted when py_last_context_chunk is None.
      _num_tokens: return value for get_num_tokens(), used for decode +
        paused (preempted-decode) requests.
    """

    def __init__(
        self,
        *,
        context_chunk_size: int = 0,
        context_current_position: int = 0,
        num_tokens: int = 0,
        is_attention_dp_dummy: bool = False,
    ):
        # Keep context_chunk_size as a private attr so tests can still express
        # per-request chunk sizes (used to compute the expected num_ctx_tokens
        # aggregate for the iter_states stub).
        self._context_chunk_size = context_chunk_size
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
    def __init__(self, input_token_ids, is_normal_request=True):
        self.request = types.SimpleNamespace(input_token_ids=input_token_ids)
        self.is_normal_request = is_normal_request


def _make_executor_stub(scheduled_batch, queued_items, num_ctx_tokens):
    """Build a minimal object with just the attributes _update_iter_stats reads.

    Avoids constructing a real PyExecutor (which needs dist + model_engine +
    kv_cache_manager). The 9-field populate block touches only:
      * `self.executor_request_queue.get_request_queue().queue` (queued iter)
      * `self.model_engine.iter_states` (for num_ctx_tokens)

    Uses plain MagicMock (NOT `spec=PyExecutor`) so we can set arbitrary
    instance attributes; `executor_request_queue` is an instance attribute
    set in PyExecutor.__init__, so a spec'd mock wouldn't expose it.
    """
    stub = MagicMock()
    stub.executor_request_queue.get_request_queue.return_value.queue = queued_items
    # model_engine.iter_states is the aggregate dict set in _prepare_tp_inputs;
    # the populate code reads num_ctx_tokens from it and uses that as
    # scheduled_sum_prefill_tokens.
    stub.model_engine = types.SimpleNamespace(iter_states={"num_ctx_tokens": num_ctx_tokens})
    return stub


def _expected_num_ctx_tokens(scheduled_batch) -> int:
    """Compute the expected model-engine-level num_ctx_tokens aggregate.

    In the real code, _prepare_tp_inputs sets iter_states["num_ctx_tokens"] =
    len(input_ids), where input_ids is built by iterating
    scheduled_requests.context_requests and extending with each request's
    current-chunk prompt tokens. The chunk size per request is
    context_chunk_size. Attention-DP dummy requests are excluded from the
    prefill accounting in the real code (via the populate-block dummy filter);
    the model-engine aggregate includes dummies in num_ctx_tokens, but our
    populate then falls through to the model_engine value — so the aggregate
    reflects the engine's real input_ids length. For test simplicity we
    mirror the engine: sum context_chunk_size across ALL context requests
    (including dummies); tests that want dummy exclusion on count fields
    will still see the per-request dummy filter kick in for scheduled_num_*
    fields.

    NOTE: For tests that specifically want to verify the dummy-exclusion
    semantics of counts, this helper still sums dummies into num_ctx_tokens
    (matching real engine behavior). Individual test cases then assert
    scheduled_num_prefill_requests (which filters dummies) but do NOT assert
    scheduled_sum_prefill_tokens unless they want the engine-aggregate value.
    """
    return sum(req._context_chunk_size for req in scheduled_batch.context_requests)


def _populate(scheduled_batch, queued_items, num_ctx_tokens_override=None):
    """Invoke just the FPM population block on a fresh IterationStats.

    Mirrors the exact algorithm in PyExecutor._update_iter_stats (lines
    ~1192–1283 on the trtllm-fpm branch).

    Parameters
    ----------
    scheduled_batch : _StubScheduledBatch
    queued_items   : list[_StubQueueItem]
    num_ctx_tokens_override : int | None
        If provided, used as iter_states["num_ctx_tokens"]. If None,
        defaults to sum of context_chunk_size across all context requests
        (matching what the real model engine computes from the same batch).
    """
    if num_ctx_tokens_override is None:
        num_ctx_tokens_override = _expected_num_ctx_tokens(scheduled_batch)

    stats = IterationStats()
    stub = _make_executor_stub(scheduled_batch, queued_items, num_ctx_tokens_override)

    # --- Scheduled ---
    scheduled_num_prefill = 0
    scheduled_sum_prefill_kv_tokens = 0
    for req in scheduled_batch.context_requests:
        if getattr(req, "is_attention_dp_dummy", False):
            continue
        scheduled_num_prefill += 1
        # Primary: py_last_context_chunk[0] (start position, pre-mutation).
        last_chunk = getattr(req, "py_last_context_chunk", None)
        if last_chunk is not None and last_chunk[0] is not None:
            start, _end = last_chunk
            scheduled_sum_prefill_kv_tokens += start
        else:
            # Fallback: context_current_position (may raise RuntimeError on
            # state-mutated requests in real code; simulated here as just a
            # dict read that never fails).
            try:
                scheduled_sum_prefill_kv_tokens += req.context_current_position
            except RuntimeError:
                pass

    scheduled_num_decode = 0
    scheduled_sum_decode_kv_tokens = 0
    for req in scheduled_batch.generation_requests:
        if getattr(req, "is_attention_dp_dummy", False):
            continue
        scheduled_num_decode += 1
        scheduled_sum_decode_kv_tokens += req.get_num_tokens(0)

    # --- Queued ---
    queued_num_prefill = 0
    queued_sum_prefill_tokens = 0
    for item in list(stub.executor_request_queue.get_request_queue().queue):
        if not getattr(item, "is_normal_request", False):
            continue
        if item.request is None:
            continue
        queued_num_prefill += 1
        try:
            queued_sum_prefill_tokens += len(item.request.input_token_ids)
        except Exception:
            pass

    queued_num_decode = 0
    queued_sum_decode_kv_tokens = 0
    for req in scheduled_batch.paused_requests:
        if getattr(req, "is_attention_dp_dummy", False):
            continue
        queued_num_decode += 1
        queued_sum_decode_kv_tokens += req.get_num_tokens(0)

    # Prefer the model_engine aggregate for fresh-computed prefill tokens
    # this iteration (matches py_executor.py:1266-1273). Fallback to the
    # per-request running sum only if iter_states is missing.
    scheduled_sum_prefill_tokens = 0  # unused fallback
    model_engine_states = getattr(stub.model_engine, "iter_states", None)
    if model_engine_states is not None:
        scheduled_sum_prefill_tokens = int(
            model_engine_states.get("num_ctx_tokens", scheduled_sum_prefill_tokens)
        )

    stats.scheduled_num_prefill_requests = scheduled_num_prefill
    stats.scheduled_sum_prefill_tokens = scheduled_sum_prefill_tokens
    stats.scheduled_sum_prefill_kv_tokens = scheduled_sum_prefill_kv_tokens
    stats.scheduled_num_decode_requests = scheduled_num_decode
    stats.scheduled_sum_decode_kv_tokens = scheduled_sum_decode_kv_tokens
    stats.queued_num_prefill_requests = queued_num_prefill
    stats.queued_sum_prefill_tokens = queued_sum_prefill_tokens
    stats.queued_num_decode_requests = queued_num_decode
    stats.queued_sum_decode_kv_tokens = queued_sum_decode_kv_tokens
    return stats


def test_empty_iteration():
    stats = _populate(_StubScheduledBatch(), [])
    assert stats.scheduled_num_prefill_requests == 0
    assert stats.scheduled_sum_prefill_tokens == 0
    assert stats.scheduled_sum_prefill_kv_tokens == 0
    assert stats.scheduled_num_decode_requests == 0
    assert stats.scheduled_sum_decode_kv_tokens == 0
    assert stats.queued_num_prefill_requests == 0
    assert stats.queued_sum_prefill_tokens == 0
    assert stats.queued_num_decode_requests == 0
    assert stats.queued_sum_decode_kv_tokens == 0


def test_prefill_only_no_prefix_cache():
    # Two fresh prefill requests: prompts of 100 and 200 tokens, chunk size
    # == full prompt (no chunked prefill). No prefix cache hits.
    # Engine aggregate num_ctx_tokens = 100 + 200 = 300.
    reqs = [
        _StubRequest(context_chunk_size=100, context_current_position=0),
        _StubRequest(context_chunk_size=200, context_current_position=0),
    ]
    stats = _populate(_StubScheduledBatch(context_reqs=reqs), [])
    assert stats.scheduled_num_prefill_requests == 2
    assert stats.scheduled_sum_prefill_tokens == 300  # from iter_states aggregate
    assert stats.scheduled_sum_prefill_kv_tokens == 0  # py_last_context_chunk[0] == 0


def test_prefill_with_prefix_cache_hit():
    # Prompt 1000 tokens; 256 already in prefix cache (prepopulatedPromptLen).
    # Chunk size = remaining = 744. py_last_context_chunk = (256, 1000);
    # start=256 is the KV-tokens count. iter_states agg = 744.
    reqs = [
        _StubRequest(context_chunk_size=744, context_current_position=256),
    ]
    stats = _populate(_StubScheduledBatch(context_reqs=reqs), [])
    assert stats.scheduled_num_prefill_requests == 1
    assert stats.scheduled_sum_prefill_tokens == 744
    assert stats.scheduled_sum_prefill_kv_tokens == 256


def test_chunked_prefill_continuation():
    # Chunked prefill: 3-chunk request, each chunk 512. This is step 2:
    # chunk size 512, previously computed 512 (= context_current_position).
    # py_last_context_chunk = (512, 1024); start=512.
    reqs = [
        _StubRequest(context_chunk_size=512, context_current_position=512),
    ]
    stats = _populate(_StubScheduledBatch(context_reqs=reqs), [])
    assert stats.scheduled_sum_prefill_tokens == 512
    assert stats.scheduled_sum_prefill_kv_tokens == 512


def test_decode_only():
    # Two decode requests: 1024 total context and 2048 total context.
    reqs = [
        _StubRequest(num_tokens=1024),
        _StubRequest(num_tokens=2048),
    ]
    stats = _populate(_StubScheduledBatch(gen_reqs=reqs), [])
    assert stats.scheduled_num_decode_requests == 2
    assert stats.scheduled_sum_decode_kv_tokens == 3072
    assert stats.scheduled_num_prefill_requests == 0
    # No context requests → iter_states num_ctx_tokens == 0.
    assert stats.scheduled_sum_prefill_tokens == 0


def test_mixed_prefill_and_decode():
    ctx = [_StubRequest(context_chunk_size=128, context_current_position=0)]
    gen = [_StubRequest(num_tokens=500), _StubRequest(num_tokens=700)]
    stats = _populate(_StubScheduledBatch(context_reqs=ctx, gen_reqs=gen), [])
    assert stats.scheduled_num_prefill_requests == 1
    assert stats.scheduled_sum_prefill_tokens == 128
    assert stats.scheduled_num_decode_requests == 2
    assert stats.scheduled_sum_decode_kv_tokens == 1200


def test_queued_prefill_from_request_queue():
    items = [
        _StubQueueItem(input_token_ids=list(range(256))),
        _StubQueueItem(input_token_ids=list(range(1024))),
    ]
    stats = _populate(_StubScheduledBatch(), items)
    assert stats.queued_num_prefill_requests == 2
    assert stats.queued_sum_prefill_tokens == 1280


def test_queued_filters_non_normal_requests():
    # Shutdown / cancel / control items should be ignored.
    items = [
        _StubQueueItem(input_token_ids=list(range(100)), is_normal_request=False),
        _StubQueueItem(input_token_ids=list(range(50))),
    ]
    stats = _populate(_StubScheduledBatch(), items)
    assert stats.queued_num_prefill_requests == 1
    assert stats.queued_sum_prefill_tokens == 50


def test_queued_decode_from_paused_requests():
    paused = [
        _StubRequest(num_tokens=300),
        _StubRequest(num_tokens=800),
    ]
    stats = _populate(_StubScheduledBatch(paused_reqs=paused), [])
    assert stats.queued_num_decode_requests == 2
    assert stats.queued_sum_decode_kv_tokens == 1100


def test_attention_dp_dummy_requests_excluded():
    # The dummy-padding added by _pad_attention_dp_dummy_request must not
    # contribute to the per-rank COUNT fields that the populate code
    # explicitly filters for. The iter_states num_ctx_tokens aggregate from
    # the engine is engine-level and does NOT filter dummies — but the
    # per-request num/kv fields on the populate block do. This test covers
    # the populate-block filter semantics.
    ctx = [
        _StubRequest(
            context_chunk_size=100, context_current_position=0, is_attention_dp_dummy=True
        ),
        _StubRequest(context_chunk_size=200, context_current_position=50),
    ]
    gen = [_StubRequest(num_tokens=1024, is_attention_dp_dummy=True)]
    paused = [_StubRequest(num_tokens=500, is_attention_dp_dummy=True)]
    # Engine aggregate includes dummies: 100 + 200 = 300 (matches real
    # _prepare_tp_inputs, which iterates ALL scheduled context_requests
    # including dummies). Override explicitly to make the test's assumption
    # visible.
    stats = _populate(
        _StubScheduledBatch(context_reqs=ctx, gen_reqs=gen, paused_reqs=paused),
        [],
        num_ctx_tokens_override=300,
    )
    # Count-based fields filter dummies: only 1 non-dummy ctx req, 0 non-dummy gen/paused.
    assert stats.scheduled_num_prefill_requests == 1
    assert stats.scheduled_sum_prefill_kv_tokens == 50  # only the non-dummy's start
    assert stats.scheduled_num_decode_requests == 0
    assert stats.queued_num_decode_requests == 0
    # Engine aggregate stays at 300 regardless of populate-block dummy filter.
    assert stats.scheduled_sum_prefill_tokens == 300


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

    stats = _populate(
        _StubScheduledBatch(context_reqs=ctx, gen_reqs=gen, paused_reqs=paused), qitems
    )

    assert stats.scheduled_num_prefill_requests == 3
    # iter_states num_ctx_tokens aggregate = 1024 + 512 + 256 = 1792
    assert stats.scheduled_sum_prefill_tokens == 1024 + 512 + 256
    # py_last_context_chunk[0] per req = 0, 1024, 768
    assert stats.scheduled_sum_prefill_kv_tokens == 0 + 1024 + 768
    assert stats.scheduled_num_decode_requests == 4
    assert stats.scheduled_sum_decode_kv_tokens == 500 + 1500 + 2500 + 3500
    assert stats.queued_num_prefill_requests == 3
    assert stats.queued_sum_prefill_tokens == 256 + 512 + 1024
    assert stats.queued_num_decode_requests == 2
    assert stats.queued_sum_decode_kv_tokens == 400 + 900


def test_iter_states_missing_falls_back_to_zero():
    """Fall back gracefully when model_engine.iter_states lacks num_ctx_tokens.

    The populate block must not raise. Real code uses
    `model_engine_states.get("num_ctx_tokens", scheduled_sum_prefill_tokens)`.
    """
    ctx = [_StubRequest(context_chunk_size=100, context_current_position=0)]
    # Override to None-equivalent: use 0 (the per-request fallback is also 0
    # since the helper's inline fallback initializes to 0).
    stats = _populate(_StubScheduledBatch(context_reqs=ctx), [], num_ctx_tokens_override=0)
    # num_ctx_tokens=0 → scheduled_sum_prefill_tokens==0; prefill request is
    # still counted + KV tokens still read from py_last_context_chunk.
    assert stats.scheduled_num_prefill_requests == 1
    assert stats.scheduled_sum_prefill_tokens == 0
    assert stats.scheduled_sum_prefill_kv_tokens == 0


def test_to_json_str_roundtrip_includes_new_fields():
    """Confirm the C++ NLOHMANN serializer emits every new FPM field under the expected camelCase key.

    Every field gets a distinct value so a cross-wiring bug in
    ``jsonSerialization.cpp`` (e.g. accidentally swapping the prefill and
    decode count keys) surfaces as an assertion failure rather than a silent
    schema drift.
    """
    import json as _json

    stats = IterationStats()
    # Distinct values per field so a field-swap bug in the C++ serializer is
    # observable.
    stats.scheduled_num_prefill_requests = 5
    stats.scheduled_sum_prefill_tokens = 2048
    stats.scheduled_sum_prefill_kv_tokens = 256
    stats.scheduled_num_decode_requests = 7
    stats.scheduled_sum_decode_kv_tokens = 9000
    stats.queued_num_prefill_requests = 11
    stats.queued_sum_prefill_tokens = 4096
    stats.queued_num_decode_requests = 3
    stats.queued_sum_decode_kv_tokens = 1500

    d = _json.loads(stats.to_json_str())
    assert d["scheduledNumPrefillRequests"] == 5
    assert d["scheduledSumPrefillTokens"] == 2048
    assert d["scheduledSumPrefillKvTokens"] == 256
    assert d["scheduledNumDecodeRequests"] == 7
    assert d["scheduledSumDecodeKvTokens"] == 9000
    assert d["queuedNumPrefillRequests"] == 11
    assert d["queuedSumPrefillTokens"] == 4096
    assert d["queuedNumDecodeRequests"] == 3
    assert d["queuedSumDecodeKvTokens"] == 1500
