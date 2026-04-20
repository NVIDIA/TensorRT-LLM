# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ForwardPassMetrics (FPM) flat fields on IterationStats.

Covers the 9 MVP fields populated in ``PyExecutor._update_iter_stats``:
  scheduled_{num_prefill_requests, sum_prefill_tokens, sum_prefill_kv_tokens,
             num_decode_requests, sum_decode_kv_tokens}
  queued_{num_prefill_requests, sum_prefill_tokens,
          num_decode_requests, sum_decode_kv_tokens}

These tests invoke the REAL ``PyExecutor._update_iter_stats`` as an unbound
call against a minimal fake ``self``. This catches any drift in the
production populate block — including refactors, renamed attributes, or
unit changes — that a duplicated test shim would miss.

Invariants match vLLM's InstrumentedScheduler at
``components/src/dynamo/vllm/instrumented_scheduler.py`` on Dynamo origin/main
(``_extract_scheduled`` + ``_compute_queued``).
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
        ``scheduled_sum_prefill_kv_tokens``. Set to None for decode/paused reqs.
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
    def __init__(self, input_token_ids, is_normal_request=True):
        self.request = types.SimpleNamespace(input_token_ids=input_token_ids)
        self.is_normal_request = is_normal_request


def _build_fake_self(queued_items, iter_states):
    """Minimal 'self' for ``PyExecutor._update_iter_stats(self, ...)``.

    Stubs only the ``self.*`` attributes the method actually reads:

    General path (runs before the FPM block):
      * ``max_num_active_requests``, ``iter_counter`` — scalars
      * ``executor_request_queue.get_request_queue_size()`` — for
        ``num_queued_requests`` (NOT one of the 9 FPM fields)
      * ``resource_manager.resource_managers.get(...)`` — returns None so the
        KV-cache-stats block is skipped entirely
      * ``drafter`` — None (specdec block no-ops when ``stats.specdec_stats``
        is None anyway, which is the default on a fresh ``IterationStats``)

    FPM block:
      * ``executor_request_queue.get_request_queue().queue`` — source for
        ``queued_num_prefill_requests`` / ``queued_sum_prefill_tokens``
      * ``model_engine.iter_states`` — source for
        ``scheduled_sum_prefill_tokens`` (the model-engine aggregate computed
        before state mutation)
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
    ``gpu_mem_usage`` but the value is not read by the FPM block).

    Parameters
    ----------
    scheduled_batch : _StubScheduledBatch
    queued_items    : list[_StubQueueItem]
    num_ctx_tokens  : int | None
        If int, passed via ``iter_states = {"num_ctx_tokens": num_ctx_tokens}``.
        If None, ``iter_states`` is set to None to exercise the fallback
        path (legacy engine / missing aggregate).
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
# Populate tests: call the REAL ``_update_iter_stats`` and assert on the 9
# FPM fields it populates.
# ---------------------------------------------------------------------------


def test_empty_iteration():
    stats = _invoke_update_iter_stats(_StubScheduledBatch(), [], num_ctx_tokens=0)
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
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=reqs), [], num_ctx_tokens=300
    )
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
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=reqs), [], num_ctx_tokens=744
    )
    assert stats.scheduled_num_prefill_requests == 1
    assert stats.scheduled_sum_prefill_tokens == 744
    assert stats.scheduled_sum_prefill_kv_tokens == 256


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
    assert stats.scheduled_sum_prefill_tokens == 512
    assert stats.scheduled_sum_prefill_kv_tokens == 512


def test_decode_only():
    # Two decode requests: 1024 total context and 2048 total context.
    reqs = [
        _StubRequest(num_tokens=1024),
        _StubRequest(num_tokens=2048),
    ]
    stats = _invoke_update_iter_stats(_StubScheduledBatch(gen_reqs=reqs), [], num_ctx_tokens=0)
    assert stats.scheduled_num_decode_requests == 2
    assert stats.scheduled_sum_decode_kv_tokens == 3072
    assert stats.scheduled_num_prefill_requests == 0
    # No context requests → iter_states num_ctx_tokens == 0.
    assert stats.scheduled_sum_prefill_tokens == 0


def test_mixed_prefill_and_decode():
    ctx = [_StubRequest(context_chunk_size=128, context_current_position=0)]
    gen = [_StubRequest(num_tokens=500), _StubRequest(num_tokens=700)]
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=ctx, gen_reqs=gen), [], num_ctx_tokens=128
    )
    assert stats.scheduled_num_prefill_requests == 1
    assert stats.scheduled_sum_prefill_tokens == 128
    assert stats.scheduled_num_decode_requests == 2
    assert stats.scheduled_sum_decode_kv_tokens == 1200


def test_queued_prefill_from_request_queue():
    items = [
        _StubQueueItem(input_token_ids=list(range(256))),
        _StubQueueItem(input_token_ids=list(range(1024))),
    ]
    stats = _invoke_update_iter_stats(_StubScheduledBatch(), items, num_ctx_tokens=0)
    assert stats.queued_num_prefill_requests == 2
    assert stats.queued_sum_prefill_tokens == 1280


def test_queued_filters_non_normal_requests():
    # Shutdown / cancel / control items should be ignored.
    items = [
        _StubQueueItem(input_token_ids=list(range(100)), is_normal_request=False),
        _StubQueueItem(input_token_ids=list(range(50))),
    ]
    stats = _invoke_update_iter_stats(_StubScheduledBatch(), items, num_ctx_tokens=0)
    assert stats.queued_num_prefill_requests == 1
    assert stats.queued_sum_prefill_tokens == 50


def test_queued_decode_from_paused_requests():
    paused = [
        _StubRequest(num_tokens=300),
        _StubRequest(num_tokens=800),
    ]
    stats = _invoke_update_iter_stats(_StubScheduledBatch(paused_reqs=paused), [], num_ctx_tokens=0)
    assert stats.queued_num_decode_requests == 2
    assert stats.queued_sum_decode_kv_tokens == 1100


def test_attention_dp_dummy_requests_excluded():
    # Dummy-padding added by ``_pad_attention_dp_dummy_request`` must not
    # contribute to the per-rank COUNT fields that the populate code
    # explicitly filters for. The iter_states num_ctx_tokens aggregate from
    # the engine is engine-level and does NOT filter dummies.
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
    # Engine aggregate includes dummies: 100 + 200 = 300 (matches real
    # ``_prepare_tp_inputs``, which iterates ALL scheduled context_requests
    # including dummies).
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=ctx, gen_reqs=gen, paused_reqs=paused),
        [],
        num_ctx_tokens=300,
    )
    # Count-based fields filter dummies: only 1 non-dummy ctx req, 0 non-dummy gen/paused.
    assert stats.scheduled_num_prefill_requests == 1
    assert stats.scheduled_sum_prefill_kv_tokens == 50  # only the non-dummy's start
    assert stats.scheduled_num_decode_requests == 0
    assert stats.queued_num_decode_requests == 0
    # Engine aggregate stays at 300 regardless of the populate-block dummy filter.
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

    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=ctx, gen_reqs=gen, paused_reqs=paused),
        qitems,
        num_ctx_tokens=1024 + 512 + 256,  # iter_states num_ctx_tokens aggregate
    )

    assert stats.scheduled_num_prefill_requests == 3
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
    """Fall back gracefully when ``model_engine.iter_states`` is missing.

    The populate block must not raise. Real code uses
    ``getattr(self.model_engine, 'iter_states', None)`` and then
    ``model_engine_states.get('num_ctx_tokens', scheduled_sum_prefill_tokens)``;
    when iter_states itself is None, the override is skipped entirely.
    """
    ctx = [_StubRequest(context_chunk_size=100, context_current_position=0)]
    stats = _invoke_update_iter_stats(
        _StubScheduledBatch(context_reqs=ctx), [], num_ctx_tokens=None
    )
    # iter_states is None → scheduled_sum_prefill_tokens stays at the
    # per-request running sum, which is 0 for a fresh prefill (no fallback
    # from context_current_position triggered for fresh chunks).
    assert stats.scheduled_num_prefill_requests == 1
    assert stats.scheduled_sum_prefill_tokens == 0
    assert stats.scheduled_sum_prefill_kv_tokens == 0


# ---------------------------------------------------------------------------
# Serializer test: confirm the C++ NLOHMANN serializer exposes every new
# FPM field under its expected camelCase key. Distinct values per field so a
# cross-wiring bug (e.g. swapped prefill/decode count keys) fails the
# assertion rather than round-tripping silently.
# ---------------------------------------------------------------------------


def test_to_json_str_roundtrip_includes_new_fields():
    """Confirm the C++ NLOHMANN serializer emits every new FPM field under the expected camelCase key."""
    import json as _json

    stats = IterationStats()
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
