# Phase 4: Metrics Output — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-request and per-iteration metrics to sim mode, exposed via `sim_config._clock.metrics` Python API.

**Architecture:** New `sim_metrics.py` with dataclasses and pure metrics function. Extend `SimClock` with recording methods. Hook `SimModelEngine.forward()` for iteration records and `SimSampler.update_requests()` for per-request token timestamps. Pass clock reference to SimSampler.

**Tech Stack:** Python, dataclasses, pytest. Container: `docker exec trtllm-hisim-dev3`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `tensorrt_llm/_torch/pyexecutor/sim_metrics.py` | **Create** | `SimRequestStats`, `SimIterationRecord`, `calc_sim_metrics()`, `percentile()` |
| `tensorrt_llm/_torch/pyexecutor/sim_clock.py` | Modify | Add `_iterations`, `_request_stats`, recording methods, `metrics` property, `write_metrics()` |
| `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py` | Modify | Call `clock.record_iteration()` after `clock.step()` |
| `tensorrt_llm/_torch/pyexecutor/sim_sampler.py` | Modify | Accept clock, call `clock.register_request()` + `clock.record_token()` |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` | Modify | Pass clock to `SimSampler(clock=clock)` |
| `tests/unittest/sim/test_sim_metrics.py` | **Create** | Unit tests for dataclasses + calc_sim_metrics |
| `tests/unittest/sim/test_sim_clock.py` | Modify | Tests for recording methods |
| `slop/test_sim.py` | Modify | E2e metrics assertions: constant + AIC + TP=2 |

---

### Task 1: sim_metrics.py — dataclasses + calc_sim_metrics + tests

**Files:**
- Create: `tensorrt_llm/_torch/pyexecutor/sim_metrics.py`
- Create: `tests/unittest/sim/test_sim_metrics.py`

- [ ] **Step 1: Write test_sim_metrics.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ... (standard license header)
"""Tests for sim_metrics."""

import pytest
from tensorrt_llm._torch.pyexecutor.sim_metrics import (
    SimRequestStats, SimIterationRecord, calc_sim_metrics, percentile)


class TestSimRequestStats:

    def test_ttft(self):
        s = SimRequestStats(request_id=1, input_length=10)
        s.gen_token_times = [0.010, 0.015, 0.020]
        assert s.ttft_s == pytest.approx(0.010)

    def test_itl(self):
        s = SimRequestStats(request_id=1, input_length=10)
        s.gen_token_times = [0.010, 0.015, 0.020]
        assert s.itl_s == [pytest.approx(0.005), pytest.approx(0.005)]

    def test_tpot(self):
        s = SimRequestStats(request_id=1, input_length=10)
        s.gen_token_times = [0.010, 0.015, 0.020]
        assert s.tpot_s == pytest.approx(0.005)

    def test_e2e(self):
        s = SimRequestStats(request_id=1, input_length=10)
        s.gen_token_times = [0.010, 0.015, 0.020]
        assert s.e2e_s == pytest.approx(0.020)

    def test_empty_tokens(self):
        s = SimRequestStats(request_id=1, input_length=10)
        assert s.ttft_s == 0.0
        assert s.itl_s == []
        assert s.tpot_s == 0.0
        assert s.e2e_s == 0.0

    def test_single_token(self):
        s = SimRequestStats(request_id=1, input_length=10)
        s.gen_token_times = [0.010]
        assert s.ttft_s == pytest.approx(0.010)
        assert s.itl_s == []
        assert s.tpot_s == 0.0
        assert s.e2e_s == pytest.approx(0.010)


class TestSimIterationRecord:

    def test_fields(self):
        r = SimIterationRecord(
            iteration=1, sim_time_s=0.010,
            predicted_duration_s=0.010,
            num_context_requests=1, num_context_tokens=128,
            num_generation_requests=0)
        assert r.iteration == 1
        assert r.predicted_duration_s == 0.010


class TestPercentile:

    def test_p50(self):
        assert percentile([1, 2, 3, 4, 5], 50) == pytest.approx(3.0)

    def test_p0(self):
        assert percentile([1, 2, 3], 0) == pytest.approx(1.0)

    def test_p100(self):
        assert percentile([1, 2, 3], 100) == pytest.approx(3.0)

    def test_single_element(self):
        assert percentile([42], 50) == pytest.approx(42.0)

    def test_empty_returns_zero(self):
        assert percentile([], 50) == 0.0


class TestCalcSimMetrics:

    def _make_stats(self):
        """Helper: 1 request, prefill=10ms, 7 decodes at 5ms each."""
        s = SimRequestStats(request_id=0, input_length=5, output_length=8)
        # Prefill at t=0.010, then decodes at 0.015, 0.020, ..., 0.045
        s.gen_token_times = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045]
        return {0: s}

    def test_ttft(self):
        m = calc_sim_metrics(self._make_stats(), [])
        assert m["mean_ttft_ms"] == pytest.approx(10.0)

    def test_tpot(self):
        m = calc_sim_metrics(self._make_stats(), [])
        assert m["mean_tpot_ms"] == pytest.approx(5.0)

    def test_itl(self):
        m = calc_sim_metrics(self._make_stats(), [])
        assert m["mean_itl_ms"] == pytest.approx(5.0)

    def test_e2e(self):
        m = calc_sim_metrics(self._make_stats(), [])
        assert m["mean_e2e_latency_ms"] == pytest.approx(45.0)

    def test_throughput(self):
        m = calc_sim_metrics(self._make_stats(), [])
        # 8 tokens / 0.045s ≈ 177.78
        assert m["output_throughput"] == pytest.approx(177.78, rel=0.01)

    def test_completed(self):
        m = calc_sim_metrics(self._make_stats(), [])
        assert m["completed"] == 1
        assert m["total_output"] == 8
        assert m["total_input"] == 5

    def test_empty(self):
        m = calc_sim_metrics({}, [])
        assert m["completed"] == 0
        assert m["output_throughput"] == 0.0
```

- [ ] **Step 2: Write sim_metrics.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ... (standard license header)
"""Metrics dataclasses and computation for simulation mode."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class SimRequestStats:
    """Per-request timing data collected during simulation."""
    request_id: int
    input_length: int
    output_length: int = 0
    created_time: float = 0.0
    gen_token_times: List[float] = field(default_factory=list)

    @property
    def ttft_s(self) -> float:
        if not self.gen_token_times:
            return 0.0
        return self.gen_token_times[0] - self.created_time

    @property
    def itl_s(self) -> List[float]:
        t = self.gen_token_times
        return [t[i] - t[i - 1] for i in range(1, len(t))]

    @property
    def tpot_s(self) -> float:
        itl = self.itl_s
        return sum(itl) / len(itl) if itl else 0.0

    @property
    def e2e_s(self) -> float:
        if not self.gen_token_times:
            return 0.0
        return self.gen_token_times[-1] - self.created_time


@dataclass
class SimIterationRecord:
    """Per-iteration data recorded during simulation."""
    iteration: int
    sim_time_s: float
    predicted_duration_s: float
    num_context_requests: int
    num_context_tokens: int
    num_generation_requests: int


def percentile(data: List[float], pct: float) -> float:
    """Compute percentile without numpy. Returns 0.0 for empty data."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def calc_sim_metrics(request_stats: dict, iterations: list) -> dict:
    """Compute HiSim-compatible metrics from per-request timing.

    Args:
        request_stats: dict of request_id -> SimRequestStats
        iterations: list of SimIterationRecord (unused for now, reserved)

    Returns:
        Dict with TTFT, TPOT, ITL, e2e, throughput metrics in milliseconds.
    """
    if not request_stats:
        return {
            "completed": 0, "total_input": 0, "total_output": 0,
            "duration": 0.0, "request_throughput": 0.0,
            "input_throughput": 0.0, "output_throughput": 0.0,
            "mean_ttft_ms": 0.0, "p50_ttft_ms": 0.0,
            "p95_ttft_ms": 0.0, "p99_ttft_ms": 0.0,
            "mean_tpot_ms": 0.0, "p50_tpot_ms": 0.0,
            "p95_tpot_ms": 0.0, "p99_tpot_ms": 0.0,
            "mean_itl_ms": 0.0, "p50_itl_ms": 0.0,
            "p95_itl_ms": 0.0, "p99_itl_ms": 0.0,
            "mean_e2e_latency_ms": 0.0, "p95_e2e_latency_ms": 0.0,
            "p99_e2e_latency_ms": 0.0,
        }

    stats = list(request_stats.values())
    ttfts = [s.ttft_s for s in stats if s.gen_token_times]
    tpots = [s.tpot_s for s in stats if s.itl_s]
    itls = [lat for s in stats for lat in s.itl_s]
    e2es = [s.e2e_s for s in stats if s.gen_token_times]

    total_input = sum(s.input_length for s in stats)
    total_output = sum(s.output_length for s in stats)
    duration_s = max(
        (s.gen_token_times[-1] for s in stats if s.gen_token_times),
        default=1e-9)

    _mean = lambda vals: sum(vals) / len(vals) if vals else 0.0

    return {
        "completed": len(stats),
        "total_input": total_input,
        "total_output": total_output,
        "duration": duration_s,
        "request_throughput": len(stats) / duration_s,
        "input_throughput": total_input / duration_s,
        "output_throughput": total_output / duration_s,
        "mean_ttft_ms": _mean(ttfts) * 1000,
        "p50_ttft_ms": percentile(ttfts, 50) * 1000,
        "p95_ttft_ms": percentile(ttfts, 95) * 1000,
        "p99_ttft_ms": percentile(ttfts, 99) * 1000,
        "mean_tpot_ms": _mean(tpots) * 1000,
        "p50_tpot_ms": percentile(tpots, 50) * 1000,
        "p95_tpot_ms": percentile(tpots, 95) * 1000,
        "p99_tpot_ms": percentile(tpots, 99) * 1000,
        "mean_itl_ms": _mean(itls) * 1000,
        "p50_itl_ms": percentile(itls, 50) * 1000,
        "p95_itl_ms": percentile(itls, 95) * 1000,
        "p99_itl_ms": percentile(itls, 99) * 1000,
        "mean_e2e_latency_ms": _mean(e2es) * 1000,
        "p95_e2e_latency_ms": percentile(e2es, 95) * 1000,
        "p99_e2e_latency_ms": percentile(e2es, 99) * 1000,
    }
```

- [ ] **Step 3: Run tests**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/test_sim_metrics.py -v'`
Expected: ALL PASSED

- [ ] **Step 4: Commit**

```bash
git add tensorrt_llm/_torch/pyexecutor/sim_metrics.py tests/unittest/sim/test_sim_metrics.py
git commit -s -m "feat: Add sim_metrics with SimRequestStats, SimIterationRecord, calc_sim_metrics"
```

---

### Task 2: Extend SimClock with recording methods

**Files:**
- Modify: `tensorrt_llm/_torch/pyexecutor/sim_clock.py`
- Modify: `tests/unittest/sim/test_sim_clock.py`

- [ ] **Step 1: Add tests for recording methods to test_sim_clock.py**

Append to `tests/unittest/sim/test_sim_clock.py`:

```python
from tensorrt_llm._torch.pyexecutor.sim_metrics import SimIterationRecord, SimRequestStats


class TestSimClockRecording:

    def test_record_iteration(self):
        clock = SimClock()
        clock.step(0.010)
        clock.record_iteration(0.010, num_ctx_req=1, num_ctx_tokens=128, num_gen_req=0)
        assert len(clock.iterations) == 1
        rec = clock.iterations[0]
        assert rec.iteration == 1
        assert rec.sim_time_s == pytest.approx(0.010)
        assert rec.predicted_duration_s == pytest.approx(0.010)
        assert rec.num_context_requests == 1

    def test_register_request(self):
        clock = SimClock()
        clock.register_request(42, input_length=10)
        assert 42 in clock.request_stats
        assert clock.request_stats[42].input_length == 10

    def test_record_token(self):
        clock = SimClock()
        clock.register_request(1, input_length=5)
        clock.step(0.010)
        clock.record_token(1)
        assert clock.request_stats[1].gen_token_times == [pytest.approx(0.010)]
        assert clock.request_stats[1].output_length == 1

    def test_record_multiple_tokens(self):
        clock = SimClock()
        clock.register_request(1, input_length=5)
        clock.step(0.010)
        clock.record_token(1)
        clock.step(0.005)
        clock.record_token(1)
        clock.step(0.005)
        clock.record_token(1)
        stats = clock.request_stats[1]
        assert len(stats.gen_token_times) == 3
        assert stats.gen_token_times == [pytest.approx(0.010), pytest.approx(0.015), pytest.approx(0.020)]
        assert stats.output_length == 3

    def test_metrics_property(self):
        clock = SimClock()
        clock.register_request(1, input_length=5)
        clock.step(0.010)
        clock.record_iteration(0.010, 1, 128, 0)
        clock.record_token(1)
        for _ in range(7):
            clock.step(0.005)
            clock.record_iteration(0.005, 0, 0, 1)
            clock.record_token(1)
        clock.request_stats[1].output_length = 8
        m = clock.metrics
        assert m["mean_ttft_ms"] == pytest.approx(10.0)
        assert m["mean_tpot_ms"] == pytest.approx(5.0)
        assert m["completed"] == 1

    def test_reset_clears_recordings(self):
        clock = SimClock()
        clock.register_request(1, input_length=5)
        clock.step(0.010)
        clock.record_iteration(0.010, 1, 128, 0)
        clock.record_token(1)
        clock.reset()
        assert clock.iterations == []
        assert clock.request_stats == {}
```

- [ ] **Step 2: Extend SimClock implementation**

Replace the entire `sim_clock.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ... (standard license header)
"""Simulated clock for simulation mode."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sim_metrics import SimIterationRecord, SimRequestStats


class SimClock:
    """Accumulates predicted iteration times and per-request/per-iteration data."""

    def __init__(self):
        self._total_time_s: float = 0.0
        self._num_iterations: int = 0
        self._iterations: list[SimIterationRecord] = []
        self._request_stats: dict[int, SimRequestStats] = {}

    def step(self, duration_s: float) -> None:
        self._total_time_s += duration_s
        self._num_iterations += 1

    def record_iteration(self, predicted_duration_s: float,
                         num_ctx_req: int, num_ctx_tokens: int,
                         num_gen_req: int) -> None:
        from .sim_metrics import SimIterationRecord
        self._iterations.append(SimIterationRecord(
            iteration=self._num_iterations,
            sim_time_s=self._total_time_s,
            predicted_duration_s=predicted_duration_s,
            num_context_requests=num_ctx_req,
            num_context_tokens=num_ctx_tokens,
            num_generation_requests=num_gen_req))

    def register_request(self, request_id: int, input_length: int,
                         created_time: float = 0.0) -> None:
        from .sim_metrics import SimRequestStats
        if request_id not in self._request_stats:
            self._request_stats[request_id] = SimRequestStats(
                request_id=request_id,
                input_length=input_length,
                created_time=created_time)

    def record_token(self, request_id: int) -> None:
        stats = self._request_stats[request_id]
        stats.gen_token_times.append(self._total_time_s)
        stats.output_length += 1

    @property
    def total_time_s(self) -> float:
        return self._total_time_s

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    @property
    def iterations(self) -> list:
        return self._iterations

    @property
    def request_stats(self) -> dict:
        return self._request_stats

    @property
    def metrics(self) -> dict:
        from .sim_metrics import calc_sim_metrics
        return calc_sim_metrics(self._request_stats, self._iterations)

    def write_metrics(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
        with open(os.path.join(output_dir, "request.jsonl"), "w") as f:
            for s in self._request_stats.values():
                f.write(json.dumps({
                    "request_id": s.request_id,
                    "input_length": s.input_length,
                    "output_length": s.output_length,
                    "created_time": s.created_time,
                    "gen_token_times": s.gen_token_times,
                    "ttft_ms": s.ttft_s * 1000,
                    "tpot_ms": s.tpot_s * 1000,
                    "e2e_ms": s.e2e_s * 1000,
                }) + "\n")
        with open(os.path.join(output_dir, "iteration.jsonl"), "w") as f:
            for r in self._iterations:
                f.write(json.dumps({
                    "iteration": r.iteration,
                    "sim_time_s": r.sim_time_s,
                    "predicted_duration_s": r.predicted_duration_s,
                    "num_context_requests": r.num_context_requests,
                    "num_context_tokens": r.num_context_tokens,
                    "num_generation_requests": r.num_generation_requests,
                }) + "\n")

    def reset(self) -> None:
        self._total_time_s = 0.0
        self._num_iterations = 0
        self._iterations = []
        self._request_stats = {}
```

- [ ] **Step 3: Run tests**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/test_sim_clock.py -v'`
Expected: ALL PASSED (5 original + 6 new = 11)

- [ ] **Step 4: Commit**

```bash
git add tensorrt_llm/_torch/pyexecutor/sim_clock.py tests/unittest/sim/test_sim_clock.py
git commit -s -m "feat: Extend SimClock with per-iteration and per-request recording"
```

---

### Task 3: Hook SimModelEngine + SimSampler to record data

**Files:**
- Modify: `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py:114-120`
- Modify: `tensorrt_llm/_torch/pyexecutor/sim_sampler.py:28-60`
- Modify: `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py:307`

- [ ] **Step 1: Add record_iteration call in SimModelEngine.forward()**

In `sim_model_engine.py`, after the `clock.step()` + logger.debug block (lines 114-120), add:

```python
            if self.clock is not None:
                self.clock.step(predicted_time)
                self.clock.record_iteration(
                    predicted_time, num_ctx_requests,
                    num_ctx_tokens, num_gen_tokens)
                logger.debug(
                    "[SimModelEngine] iter=%d predicted=%.3fms total=%.3fms",
                    self.clock.num_iterations,
                    predicted_time * 1000,
                    self.clock.total_time_s * 1000)
```

(Replace the existing `if self.clock is not None:` block.)

- [ ] **Step 2: Add clock to SimSampler**

In `sim_sampler.py`, modify the class to accept and use clock:

```python
class SimSampler(Sampler):
    """Sampler that generates dummy tokens and advances request state."""

    DUMMY_TOKEN_ID = 0

    def __init__(self, clock=None):
        self._clock = clock

    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs: dict, num_context_logits_prefix_sum: list,
                     resource_manager=None):
        all_requests = (scheduled_requests.context_requests +
                        scheduled_requests.generation_requests)
        return SampleState(requests=all_requests)

    def update_requests(self, state: SampleState, resource_manager=None):
        for request in state.requests:
            if request.is_generation_complete_state:
                continue

            # Register request on first encounter
            if (self._clock is not None
                    and request.request_id not in self._clock.request_stats):
                self._clock.register_request(
                    request.request_id,
                    input_length=request.orig_prompt_len)

            request.add_new_token(self.DUMMY_TOKEN_ID, 0)
            request.py_decoding_iter += 1

            # Record token timestamp on the simulated clock
            if self._clock is not None:
                self._clock.record_token(request.request_id)

            num_generated = request.get_num_tokens(0) - request.orig_prompt_len
            if num_generated >= request.max_new_tokens:
                request.state = LlmRequestState.GENERATION_COMPLETE
                request.set_finished_reason(FinishReason.LENGTH, 0)

    def is_generation_model(self) -> bool:
        return True
```

- [ ] **Step 3: Pass clock to SimSampler in py_executor_creator.py**

In `_create_sim_py_executor`, change line 307 from:

```python
    sampler = SimSampler()
```

to:

```python
    sampler = SimSampler(clock=clock)
```

- [ ] **Step 4: Run all unit tests**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/ -v'`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add tensorrt_llm/_torch/pyexecutor/sim_model_engine.py tensorrt_llm/_torch/pyexecutor/sim_sampler.py tensorrt_llm/_torch/pyexecutor/py_executor_creator.py
git commit -s -m "feat: Hook SimModelEngine and SimSampler to record metrics data"
```

---

### Task 4: E2e tests with constant + AIC + TP=2 metrics verification

**Files:**
- Modify: `slop/test_sim.py`

- [ ] **Step 1: Rewrite e2e test with metrics assertions**

```python
"""Smoke test for simulation mode metrics.

Run with: python3 slop/test_sim.py
"""
import os

os.environ["TRTLLM_LOG_LEVEL"] = "WARNING"

AIC_SYSTEMS_DIR = "/code/slop/aiconfigurator/src/aiconfigurator/systems"
MODEL_PATH = "/code/slop/models/TinyLlama-1.1B-Chat-v1.0"


def test_constant_metrics():
    """Constant predictor: exact metric values verifiable."""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    print("\n=== Constant Predictor Metrics ===", flush=True)
    sim_config = SimConfig(predictor=PredictorConfig(
        constant_prefill_time_ms=10.0, constant_decode_time_ms=5.0))
    llm = LLM(MODEL_PATH, sim_config=sim_config)
    llm.generate(["Hello world"], sampling_params=SamplingParams(max_tokens=8))

    clock = sim_config._clock
    assert clock is not None
    m = clock.metrics

    print(f"TTFT: {m['mean_ttft_ms']:.1f}ms", flush=True)
    print(f"TPOT: {m['mean_tpot_ms']:.1f}ms", flush=True)
    print(f"E2E:  {m['mean_e2e_latency_ms']:.1f}ms", flush=True)
    print(f"Throughput: {m['output_throughput']:.1f} tok/s", flush=True)

    assert abs(m["mean_ttft_ms"] - 10.0) < 0.1
    assert abs(m["mean_tpot_ms"] - 5.0) < 0.1
    assert abs(m["mean_e2e_latency_ms"] - 45.0) < 0.1
    assert m["output_throughput"] > 170
    assert m["completed"] == 1
    assert m["total_output"] == 8

    # Per-request
    assert len(clock.request_stats) == 1
    stats = list(clock.request_stats.values())[0]
    assert len(stats.gen_token_times) == 8
    assert stats.input_length > 0

    # Per-iteration
    assert len(clock.iterations) == 8

    print("CONSTANT METRICS OK", flush=True)


def test_aic_metrics():
    """AIC predictor: structural checks + cross-consistency."""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    print("\n=== AIC Predictor Metrics ===", flush=True)
    sim_config = SimConfig(predictor=PredictorConfig(
        name="aiconfigurator", device_name="h100_sxm",
        backend_version="1.2.0rc5", database_path=AIC_SYSTEMS_DIR))
    llm = LLM(MODEL_PATH, sim_config=sim_config)
    llm.generate(["Hello world"], sampling_params=SamplingParams(max_tokens=8))

    clock = sim_config._clock
    assert clock is not None
    m = clock.metrics

    print(f"TTFT: {m['mean_ttft_ms']:.2f}ms", flush=True)
    print(f"TPOT: {m['mean_tpot_ms']:.2f}ms", flush=True)
    print(f"E2E:  {m['mean_e2e_latency_ms']:.2f}ms", flush=True)
    print(f"Throughput: {m['output_throughput']:.1f} tok/s", flush=True)

    assert m["completed"] == 1
    assert m["total_output"] == 8
    assert m["mean_ttft_ms"] > 0
    assert m["mean_tpot_ms"] > 0
    assert m["mean_ttft_ms"] > m["mean_tpot_ms"], \
        f"Prefill should be slower than decode: TTFT={m['mean_ttft_ms']:.2f} <= TPOT={m['mean_tpot_ms']:.2f}"

    # Cross-check: e2e ≈ ttft + 7 * tpot
    expected_e2e = m["mean_ttft_ms"] + 7 * m["mean_tpot_ms"]
    assert abs(m["mean_e2e_latency_ms"] - expected_e2e) < 0.1, \
        f"E2E mismatch: {m['mean_e2e_latency_ms']:.2f} vs expected {expected_e2e:.2f}"

    # Per-iteration: prefill vs decode should differ
    iters = clock.iterations
    assert len(iters) == 8
    prefill_time = iters[0].predicted_duration_s
    decode_time = iters[1].predicted_duration_s
    assert prefill_time != decode_time, \
        f"AIC should differentiate prefill vs decode"
    print(f"Prefill iter: {prefill_time*1000:.2f}ms, Decode iter: {decode_time*1000:.2f}ms",
          flush=True)

    print("AIC METRICS OK", flush=True)


def test_aic_tp2_metrics():
    """AIC TP=2: metrics structure same, values differ from TP=1."""
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    print("\n=== AIC TP=2 Metrics ===", flush=True)
    sim_tp2 = SimConfig(predictor=PredictorConfig(
        name="aiconfigurator", device_name="h100_sxm",
        backend_version="1.2.0rc5", database_path=AIC_SYSTEMS_DIR))
    llm = LLM(MODEL_PATH, sim_config=sim_tp2, tensor_parallel_size=2)
    llm.generate(["Hello world"], sampling_params=SamplingParams(max_tokens=8))

    m = sim_tp2._clock.metrics
    assert m["completed"] == 1
    assert m["total_output"] == 8
    assert m["mean_ttft_ms"] > 0
    assert m["mean_tpot_ms"] > 0

    # TP=2 should differ from TP=1 (we don't have tp1 in scope here,
    # but we verify the structure is correct and values are plausible)
    print(f"TP=2 TTFT: {m['mean_ttft_ms']:.2f}ms", flush=True)
    print(f"TP=2 TPOT: {m['mean_tpot_ms']:.2f}ms", flush=True)

    print("AIC TP=2 METRICS OK", flush=True)


def main():
    test_constant_metrics()
    test_aic_metrics()
    test_aic_tp2_metrics()
    print("\n=== ALL TESTS PASSED ===", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run e2e in container**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 slop/test_sim.py'`
Expected: `ALL TESTS PASSED`

- [ ] **Step 3: Run full unit test suite**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/ -v'`
Expected: ALL PASSED

- [ ] **Step 4: Commit**

```bash
git add slop/test_sim.py
git commit -s -m "feat: E2e metrics tests with constant, AIC, and TP=2 verification"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] SimRequestStats with ttft_s, itl_s, tpot_s, e2e_s → Task 1
- [x] SimIterationRecord → Task 1
- [x] calc_sim_metrics() → Task 1
- [x] percentile() without numpy → Task 1
- [x] SimClock recording methods → Task 2
- [x] SimClock.metrics property → Task 2
- [x] SimClock.write_metrics() → Task 2
- [x] Hook SimModelEngine.forward → Task 3
- [x] Hook SimSampler.update_requests → Task 3
- [x] Pass clock to SimSampler → Task 3
- [x] Constant predictor exact values → Task 4
- [x] AIC structural checks + cross-consistency → Task 4
- [x] AIC TP=2 verification → Task 4

**Placeholder scan:** None found.

**Type consistency:** `SimRequestStats.gen_token_times` (list[float]), `SimClock.record_token(request_id)`, `SimClock.register_request(request_id, input_length)`, `SimClock.record_iteration(predicted_duration_s, num_ctx_req, num_ctx_tokens, num_gen_req)` — all consistent across tasks.
