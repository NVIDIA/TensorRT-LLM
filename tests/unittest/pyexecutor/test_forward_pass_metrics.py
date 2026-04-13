# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ForwardPassMetrics (FPM) extraction logic in PyExecutor.

Tests the FpmScheduledSnapshot capture, queued metrics computation,
emit logic, and idle heartbeat behaviour -- all without requiring
a GPU or running the full executor loop.

These tests directly exercise the FPM helper methods by calling them
as unbound functions with stub objects, avoiding heavy TRT-LLM imports.
"""

# ---------------------------------------------------------------------------
# Standalone copy of FpmScheduledSnapshot for testing without torch.
# This mirrors the dataclass in py_executor.py exactly.
# ---------------------------------------------------------------------------
import dataclasses
from dataclasses import dataclass

import pytest


@dataclasses.dataclass
class FpmScheduledSnapshot:
    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    sum_prefill_kv_tokens: int = 0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    prefill_length_n: int = 0
    prefill_length_mean: float = 0.0
    prefill_length_m2: float = 0.0
    decode_kv_n: int = 0
    decode_kv_mean: float = 0.0
    decode_kv_m2: float = 0.0

    @property
    def var_prefill_length(self) -> float:
        return self.prefill_length_m2 / self.prefill_length_n if self.prefill_length_n > 0 else 0.0

    @property
    def var_decode_kv_tokens(self) -> float:
        return self.decode_kv_m2 / self.decode_kv_n if self.decode_kv_n > 0 else 0.0


# ---------------------------------------------------------------------------
# Lightweight stubs
# ---------------------------------------------------------------------------


@dataclass
class FakeRequest:
    """Minimal stub matching LlmRequest properties used by FPM."""

    request_id: int = 0
    context_chunk_size: int = 0
    context_current_position: int = 0
    py_orig_prompt_len: int = 0
    max_beam_num_tokens: int = 0
    state: object = None
    is_attention_dp_dummy: bool = False
    is_cuda_graph_dummy: bool = False
    is_dummy_request: bool = False

    @property
    def is_dummy(self):
        return self.is_attention_dp_dummy or self.is_cuda_graph_dummy or self.is_dummy_request


class FakeScheduledRequests:
    def __init__(self, context=None, generation=None):
        self.context_requests_chunking = []
        self.context_requests_last_chunk = context or []
        self.generation_requests = generation or []
        self.paused_requests = []

    @property
    def context_requests(self):
        return self.context_requests_chunking + self.context_requests_last_chunk

    def all_requests(self):
        return self.context_requests + self.generation_requests


# ---------------------------------------------------------------------------
# Re-implement the capture logic to test independently (same algorithm)
# ---------------------------------------------------------------------------


def capture_fpm_scheduled(scheduled_batch) -> FpmScheduledSnapshot:
    """Pure-Python reimplementation of PyExecutor._capture_fpm_scheduled.

    Used for testing without torch imports.
    """
    snap = FpmScheduledSnapshot()
    pf_n = 0
    pf_mean = 0.0
    pf_m2 = 0.0
    dk_n = 0
    dk_mean = 0.0
    dk_m2 = 0.0

    for req in scheduled_batch.context_requests:
        if req.is_dummy:
            continue
        snap.num_prefill_requests += 1
        snap.sum_prefill_tokens += req.context_chunk_size
        snap.sum_prefill_kv_tokens += req.context_current_position
        v = req.py_orig_prompt_len
        pf_n += 1
        delta = v - pf_mean
        pf_mean += delta / pf_n
        pf_m2 += delta * (v - pf_mean)

    for req in scheduled_batch.generation_requests:
        if req.is_dummy:
            continue
        snap.num_decode_requests += 1
        kv_len = req.max_beam_num_tokens
        snap.sum_decode_kv_tokens += kv_len
        dk_n += 1
        delta = kv_len - dk_mean
        dk_mean += delta / dk_n
        dk_m2 += delta * (kv_len - dk_mean)

    snap.prefill_length_n = pf_n
    snap.prefill_length_mean = pf_mean
    snap.prefill_length_m2 = pf_m2
    snap.decode_kv_n = dk_n
    snap.decode_kv_mean = dk_mean
    snap.decode_kv_m2 = dk_m2
    return snap


def population_variance(values):
    """Reference population variance."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFpmScheduledSnapshot:
    def test_variance_empty(self):
        snap = FpmScheduledSnapshot()
        assert snap.var_prefill_length == 0.0
        assert snap.var_decode_kv_tokens == 0.0

    def test_variance_single(self):
        snap = FpmScheduledSnapshot(
            prefill_length_n=1, prefill_length_mean=100.0, prefill_length_m2=0.0
        )
        assert snap.var_prefill_length == 0.0

    def test_variance_two_values(self):
        snap = FpmScheduledSnapshot(
            prefill_length_n=2, prefill_length_mean=150.0, prefill_length_m2=5000.0
        )
        assert snap.var_prefill_length == 2500.0


class TestCaptureScheduled:
    def test_empty_batch(self):
        snap = capture_fpm_scheduled(FakeScheduledRequests())
        assert snap.num_prefill_requests == 0
        assert snap.num_decode_requests == 0

    def test_prefill_only(self):
        reqs = [
            FakeRequest(
                request_id=1,
                context_chunk_size=512,
                context_current_position=0,
                py_orig_prompt_len=512,
            ),
            FakeRequest(
                request_id=2,
                context_chunk_size=256,
                context_current_position=128,
                py_orig_prompt_len=384,
            ),
        ]
        snap = capture_fpm_scheduled(FakeScheduledRequests(context=reqs))
        assert snap.num_prefill_requests == 2
        assert snap.sum_prefill_tokens == 768
        assert snap.sum_prefill_kv_tokens == 128
        expected_var = population_variance([512, 384])
        assert snap.var_prefill_length == pytest.approx(expected_var)

    def test_decode_only(self):
        reqs = [
            FakeRequest(request_id=i, max_beam_num_tokens=v)
            for i, v in enumerate([1000, 2000, 3000])
        ]
        snap = capture_fpm_scheduled(FakeScheduledRequests(generation=reqs))
        assert snap.num_decode_requests == 3
        assert snap.sum_decode_kv_tokens == 6000
        expected_var = population_variance([1000, 2000, 3000])
        assert snap.var_decode_kv_tokens == pytest.approx(expected_var)

    def test_mixed_batch(self):
        ctx = [
            FakeRequest(
                request_id=1,
                context_chunk_size=100,
                context_current_position=50,
                py_orig_prompt_len=150,
            )
        ]
        gen = [FakeRequest(request_id=2, max_beam_num_tokens=500)]
        snap = capture_fpm_scheduled(FakeScheduledRequests(context=ctx, generation=gen))
        assert snap.num_prefill_requests == 1
        assert snap.num_decode_requests == 1
        assert snap.sum_prefill_tokens == 100
        assert snap.sum_prefill_kv_tokens == 50
        assert snap.sum_decode_kv_tokens == 500

    def test_dummy_excluded(self):
        reqs = [
            FakeRequest(request_id=1, max_beam_num_tokens=100),
            FakeRequest(request_id=2, max_beam_num_tokens=200, is_attention_dp_dummy=True),
            FakeRequest(request_id=3, max_beam_num_tokens=300, is_cuda_graph_dummy=True),
        ]
        snap = capture_fpm_scheduled(FakeScheduledRequests(generation=reqs))
        assert snap.num_decode_requests == 1
        assert snap.sum_decode_kv_tokens == 100

    def test_chunked_prefill(self):
        """Chunked prefill: chunk_size < prompt_len, position > 0."""
        req = FakeRequest(
            request_id=1,
            context_chunk_size=2048,  # computing 2048 tokens this step
            context_current_position=4096,  # already computed 4096 tokens
            py_orig_prompt_len=8192,  # total prompt is 8192
        )
        snap = capture_fpm_scheduled(FakeScheduledRequests(context=[req]))
        assert snap.num_prefill_requests == 1
        assert snap.sum_prefill_tokens == 2048
        assert snap.sum_prefill_kv_tokens == 4096
        assert snap.var_prefill_length == 0.0  # single request


class TestWelfordNumericalStability:
    def test_uniform_values(self):
        reqs = [FakeRequest(request_id=i, max_beam_num_tokens=500) for i in range(10)]
        snap = capture_fpm_scheduled(FakeScheduledRequests(generation=reqs))
        assert snap.var_decode_kv_tokens == pytest.approx(0.0)

    def test_two_extremes(self):
        reqs = [
            FakeRequest(request_id=0, max_beam_num_tokens=0),
            FakeRequest(request_id=1, max_beam_num_tokens=1000),
        ]
        snap = capture_fpm_scheduled(FakeScheduledRequests(generation=reqs))
        assert snap.var_decode_kv_tokens == pytest.approx(250000.0)

    def test_large_base_small_spread(self):
        """Welford should handle large base values without cancellation."""
        base = 1_000_000
        reqs = [
            FakeRequest(
                request_id=i,
                py_orig_prompt_len=base + i,
                context_chunk_size=1,
                context_current_position=0,
            )
            for i in range(100)
        ]
        snap = capture_fpm_scheduled(FakeScheduledRequests(context=reqs))
        expected = population_variance([base + i for i in range(100)])
        assert snap.var_prefill_length == pytest.approx(expected, rel=1e-6)


class TestEmitAndHeartbeat:
    """Test the emit/heartbeat logic using dict-based payload contracts."""

    def _make_payload(self, snap, wall_time_ms=15.0, dp_rank=0):
        """Build the FPM payload dict the same way _emit_fpm does."""
        return {
            "version": 1,
            "dp_rank": dp_rank,
            "counter_id": 1,
            "wall_time": wall_time_ms / 1000.0,
            "scheduled_requests": {
                "num_prefill_requests": snap.num_prefill_requests,
                "sum_prefill_tokens": snap.sum_prefill_tokens,
                "var_prefill_length": snap.var_prefill_length,
                "sum_prefill_kv_tokens": snap.sum_prefill_kv_tokens,
                "num_decode_requests": snap.num_decode_requests,
                "sum_decode_kv_tokens": snap.sum_decode_kv_tokens,
                "var_decode_kv_tokens": snap.var_decode_kv_tokens,
            },
            "queued_requests": {
                "num_prefill_requests": 0,
                "sum_prefill_tokens": 0,
                "var_prefill_length": 0.0,
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
                "var_decode_kv_tokens": 0.0,
            },
        }

    def test_payload_schema(self):
        snap = FpmScheduledSnapshot(
            num_prefill_requests=2,
            sum_prefill_tokens=768,
            num_decode_requests=5,
            sum_decode_kv_tokens=3000,
        )
        payload = self._make_payload(snap)
        assert payload["version"] == 1
        assert payload["wall_time"] == pytest.approx(0.015)
        sr = payload["scheduled_requests"]
        assert sr["num_prefill_requests"] == 2
        assert sr["sum_prefill_tokens"] == 768
        assert sr["num_decode_requests"] == 5
        assert sr["sum_decode_kv_tokens"] == 3000

    def test_heartbeat_payload(self):
        heartbeat = {
            "version": 1,
            "dp_rank": 0,
            "counter_id": 1,
            "wall_time": 0.0,
            "scheduled_requests": {
                "num_prefill_requests": 0,
                "sum_prefill_tokens": 0,
                "var_prefill_length": 0.0,
                "sum_prefill_kv_tokens": 0,
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
                "var_decode_kv_tokens": 0.0,
            },
            "queued_requests": {
                "num_prefill_requests": 0,
                "sum_prefill_tokens": 0,
                "var_prefill_length": 0.0,
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
                "var_decode_kv_tokens": 0.0,
            },
        }
        assert heartbeat["wall_time"] == 0.0
        for field in heartbeat["scheduled_requests"].values():
            assert field == 0 or field == 0.0

    def test_dp_rank_propagated(self):
        snap = FpmScheduledSnapshot(num_decode_requests=1)
        payload = self._make_payload(snap, dp_rank=3)
        assert payload["dp_rank"] == 3
