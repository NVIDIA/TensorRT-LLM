# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the extended perf time-events capture path.

Covers the Python-only additions behind ``TRTLLM_PERF_TIME_EVENTS_PATH`` /
``capture_extended``:

* ``PerfMetricsManager.append_step_metrics`` merges the per-iteration
  batch-context dict + per-request token counts (and does NOT when
  ``capture_extended`` is off -- guards the existing output shape).
* ``PyExecutor._compute_iter_batch_context`` builds the shared per-iteration
  dict from a ``ScheduledRequests``-like object.
* The off-critical-path per-rank writer enqueues one record per finished
  request, writes a ``time_events_rank{N}_pid{P}.jsonl`` line, and no-ops when
  disabled.

Run with:
    python -m pytest tests/unittest/_torch/executor/test_perf_metrics_events.py -v
"""

import glob
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# The manager + request types pull in the torch stack (CPU is enough; no GPU).
pytest.importorskip("torch")

from tensorrt_llm._torch.pyexecutor.llm_request import PerfTimingInfo
from tensorrt_llm._torch.pyexecutor.perf_metrics_manager import (
    PERF_TIME_EVENTS_PATH_ENV,
    PerfMetricsManager,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


def _make_request(
    *,
    is_gen_only,
    py_decoding_iter,
    iter_batch_context,
    context_chunk_size=None,
    context_remaining_length=0,
    py_draft_tokens=None,
):
    """Minimal duck-typed LlmRequest for append_step_metrics."""
    perf = PerfTimingInfo()
    perf.forward_start_time = 1.0
    perf.forward_end_time = 1.5
    perf.sample_start_time = 1.5
    perf.sample_end_time = 1.6
    perf.iter_batch_context = iter_batch_context
    return SimpleNamespace(
        py_perf_timing=perf,
        py_decoding_iter=py_decoding_iter,
        is_generation_only_request=lambda: is_gen_only,
        context_remaining_length=context_remaining_length,
        context_chunk_size=context_chunk_size,
        py_last_context_chunk=None,
        py_draft_tokens=py_draft_tokens,
    )


@pytest.fixture(autouse=True)
def _clear_env():
    """Keep the env master switch out of these explicit-arg tests."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(PERF_TIME_EVENTS_PATH_ENV, None)
        yield


class TestAppendStepMetricsExtended:
    def test_ctx_merges_batch_context_and_token_count(self):
        mgr = PerfMetricsManager(enabled=True, capture_extended=True)
        batch_ctx = {
            "iter_counter": 7,
            "iter_batch_size": 4,
            "num_ctx_requests": 1,
            "num_gen_requests": 3,
            "context_token_number": 128,
            "generation_token_number": 3,
            "num_capacity_fitting": 5,
            "num_scheduled": 4,
        }
        req = _make_request(
            is_gen_only=False,
            py_decoding_iter=0,
            iter_batch_context=batch_ctx,
            context_chunk_size=128,
        )

        mgr.append_step_metrics(req, iter_counter=7)

        assert len(req.py_perf_timing.ctx_chunk_metrics) == 1
        metric = req.py_perf_timing.ctx_chunk_metrics[0]
        # Batch-context fields merged in.
        for key, val in batch_ctx.items():
            assert metric[key] == val
        # Per-request ctx token count.
        assert metric["req_context_token_number"] == 128
        # Base timing fields still present.
        assert metric["forward_start_time"] == 1.0

    def test_gen_merges_batch_context_and_token_count(self):
        mgr = PerfMetricsManager(enabled=True, capture_extended=True)
        batch_ctx = {
            "iter_counter": 9,
            "iter_batch_size": 2,
            "num_ctx_requests": 0,
            "num_gen_requests": 2,
            "context_token_number": 0,
            "generation_token_number": 2,
        }
        req = _make_request(
            is_gen_only=True, py_decoding_iter=5, iter_batch_context=batch_ctx, py_draft_tokens=None
        )

        mgr.append_step_metrics(req, iter_counter=9)

        assert len(req.py_perf_timing.step_metrics) == 1
        metric = req.py_perf_timing.step_metrics[0]
        assert metric["num_gen_requests"] == 2
        # 1 emitted token + 0 draft tokens.
        assert metric["req_generation_token_number"] == 1
        assert metric["iter"] == 5

    def test_gen_token_count_includes_draft(self):
        mgr = PerfMetricsManager(enabled=True, capture_extended=True)
        req = _make_request(
            is_gen_only=True,
            py_decoding_iter=3,
            iter_batch_context={"iter_counter": 1},
            py_draft_tokens=[10, 11, 12],
        )
        mgr.append_step_metrics(req, iter_counter=1)
        metric = req.py_perf_timing.step_metrics[0]
        assert metric["req_generation_token_number"] == 1 + 3

    def test_disabled_extended_leaves_shape_untouched(self):
        # capture_extended off -> none of the new keys, even with a context dict.
        mgr = PerfMetricsManager(enabled=True, capture_extended=False)
        req = _make_request(
            is_gen_only=True,
            py_decoding_iter=2,
            iter_batch_context={"iter_batch_size": 4},
            py_draft_tokens=None,
        )
        mgr.append_step_metrics(req, iter_counter=2)
        metric = req.py_perf_timing.step_metrics[0]
        assert "iter_batch_size" not in metric
        assert "req_generation_token_number" not in metric
        # Base keys unchanged.
        assert set(metric) >= {"forward_start_time", "forward_end_time", "token_time", "iter"}


class TestComputeIterBatchContext:
    def test_counts_and_tokens(self):
        ctx_reqs = [
            SimpleNamespace(context_chunk_size=64, py_last_context_chunk=None),
            SimpleNamespace(context_chunk_size=32, py_last_context_chunk=None),
        ]
        gen_reqs = [
            SimpleNamespace(num_draft_tokens=0),
            SimpleNamespace(num_draft_tokens=2),
        ]
        scheduled = SimpleNamespace(
            num_context_requests=2,
            num_generation_requests=2,
            batch_size=4,
            context_requests=ctx_reqs,
            generation_requests=gen_reqs,
        )
        fake_self = SimpleNamespace(iter_counter=11)

        ctx = PyExecutor._compute_iter_batch_context(fake_self, scheduled, num_fitting_reqs=6)

        assert ctx["iter_counter"] == 11
        assert ctx["iter_batch_size"] == 4
        assert ctx["num_ctx_requests"] == 2
        assert ctx["num_gen_requests"] == 2
        assert ctx["context_token_number"] == 64 + 32
        assert ctx["generation_token_number"] == (1 + 0) + (1 + 2)
        assert ctx["num_capacity_fitting"] == 6
        assert ctx["num_scheduled"] == 4

    def test_starvation_omitted_when_no_fitting_count(self):
        scheduled = SimpleNamespace(
            num_context_requests=0,
            num_generation_requests=1,
            batch_size=1,
            context_requests=[],
            generation_requests=[SimpleNamespace(num_draft_tokens=0)],
        )
        fake_self = SimpleNamespace(iter_counter=1)
        ctx = PyExecutor._compute_iter_batch_context(fake_self, scheduled)
        assert "num_capacity_fitting" not in ctx
        assert "num_scheduled" not in ctx


class TestPerRankWriter:
    def _make_response(self, request_id, time_breakdown_metrics):
        return SimpleNamespace(
            request_id=request_id,
            result=SimpleNamespace(time_breakdown_metrics=time_breakdown_metrics),
        )

    def test_writes_one_jsonl_line_per_request(self, tmp_path):
        with patch.dict(os.environ, {PERF_TIME_EVENTS_PATH_ENV: str(tmp_path)}):
            # env alone force-enables capture + extended.
            mgr = PerfMetricsManager(enabled=False)
            assert mgr.enabled and mgr.capture_extended
            tbm = {"step_metrics": [{"forward_start_time": 1.0}]}
            resp = self._make_response(42, tbm)
            mgr.maybe_write_request_events(resp, rank=3, ctx_request_id=7)
            mgr.close()  # drains + joins the daemon writer

        files = glob.glob(str(tmp_path / "time_events_rank3_pid*.jsonl"))
        assert len(files) == 1
        assert f"pid{os.getpid()}" in os.path.basename(files[0])
        with open(files[0]) as f:
            lines = [ln for ln in f if ln.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["request_id"] == 42
        assert rec["rank"] == 3
        assert rec["ctx_request_id"] == 7
        assert rec["time_breakdown_metrics"] == tbm

    def test_no_write_when_metrics_absent(self, tmp_path):
        with patch.dict(os.environ, {PERF_TIME_EVENTS_PATH_ENV: str(tmp_path)}):
            mgr = PerfMetricsManager(enabled=False)
            # Final-response gate: no time_breakdown_metrics -> nothing enqueued.
            resp = self._make_response(1, None)
            mgr.maybe_write_request_events(resp, rank=0)
            mgr.close()
        assert glob.glob(str(tmp_path / "*.jsonl")) == []

    def test_no_write_when_disabled(self, tmp_path):
        # env unset -> capture_extended off -> writer never starts, no file.
        os.environ.pop(PERF_TIME_EVENTS_PATH_ENV, None)
        mgr = PerfMetricsManager(enabled=True, capture_extended=False)
        resp = self._make_response(1, {"step_metrics": []})
        mgr.maybe_write_request_events(resp, rank=0)
        mgr.close()  # safe no-op when no writer thread
        assert glob.glob(str(tmp_path / "*.jsonl")) == []
