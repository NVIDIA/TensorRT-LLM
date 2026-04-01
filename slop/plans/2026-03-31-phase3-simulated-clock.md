# Phase 3: Simulated Clock — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `time.sleep()` in simulation mode with a `SimClock` that accumulates predicted times, so simulation runs at CPU speed.

**Architecture:** New `SimClock` class accumulates predicted durations. `SimModelEngine.forward()` calls `clock.step()` instead of `time.sleep()`. Clock instance stored on `sim_config._clock` for programmatic access after `generate()`.

**Tech Stack:** Python, Pydantic, pytest. Container: `docker exec trtllm-hisim-dev3`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `tensorrt_llm/_torch/pyexecutor/sim_clock.py` | **Create** | `SimClock` — accumulates predicted iteration times |
| `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py` | Modify | Use `SimClock.step()` instead of `time.sleep()` |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` | Modify | Create clock, pass to engine, store on config |
| `tensorrt_llm/llmapi/sim_config.py` | Modify | Add `_clock` field to `SimConfig` |
| `tests/unittest/sim/test_sim_clock.py` | **Create** | Unit tests for `SimClock` |
| `slop/test_sim.py` | Modify | Clock-based e2e assertions |

---

### Task 1: SimClock class + unit tests

**Files:**
- Create: `tensorrt_llm/_torch/pyexecutor/sim_clock.py`
- Create: `tests/unittest/sim/test_sim_clock.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unittest/sim/test_sim_clock.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for SimClock."""

import pytest

from tensorrt_llm._torch.pyexecutor.sim_clock import SimClock


class TestSimClock:

    def test_initial_state(self):
        clock = SimClock()
        assert clock.total_time_s == 0.0
        assert clock.num_iterations == 0

    def test_step_accumulates(self):
        clock = SimClock()
        clock.step(0.01)
        clock.step(0.01)
        clock.step(0.01)
        assert clock.total_time_s == pytest.approx(0.03)
        assert clock.num_iterations == 3

    def test_reset(self):
        clock = SimClock()
        clock.step(0.05)
        clock.step(0.05)
        clock.reset()
        assert clock.total_time_s == 0.0
        assert clock.num_iterations == 0

    def test_step_zero_duration(self):
        clock = SimClock()
        clock.step(0.0)
        assert clock.total_time_s == 0.0
        assert clock.num_iterations == 1

    def test_step_fractional(self):
        clock = SimClock()
        clock.step(0.013)  # typical AIC prediction
        clock.step(0.0014)
        assert clock.total_time_s == pytest.approx(0.0144)
        assert clock.num_iterations == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/test_sim_clock.py -v'`
Expected: FAIL with `ModuleNotFoundError: No module named 'tensorrt_llm._torch.pyexecutor.sim_clock'`

- [ ] **Step 3: Write SimClock implementation**

Create `tensorrt_llm/_torch/pyexecutor/sim_clock.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Simulated clock for simulation mode."""


class SimClock:
    """Accumulates predicted iteration times for simulation mode.

    Not a singleton — owned by SimModelEngine as an instance attribute.
    Phase 4 will extend this to record per-iteration breakdown.
    """

    def __init__(self):
        self._total_time_s: float = 0.0
        self._num_iterations: int = 0

    def step(self, duration_s: float) -> None:
        """Advance clock by one iteration's predicted duration."""
        self._total_time_s += duration_s
        self._num_iterations += 1

    @property
    def total_time_s(self) -> float:
        return self._total_time_s

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    def reset(self) -> None:
        self._total_time_s = 0.0
        self._num_iterations = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/test_sim_clock.py -v'`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add tensorrt_llm/_torch/pyexecutor/sim_clock.py tests/unittest/sim/test_sim_clock.py
git commit -s -m "feat: Add SimClock class for simulated time accumulation"
```

---

### Task 2: Add `_clock` field to SimConfig

**Files:**
- Modify: `tensorrt_llm/llmapi/sim_config.py:79-89`

- [ ] **Step 1: Write the failing test**

Add to `tests/unittest/sim/test_sim_config.py`:

```python
def test_sim_config_clock_field_default_none():
    """SimConfig._clock defaults to None and is excluded from serialization."""
    from tensorrt_llm.llmapi.sim_config import SimConfig
    config = SimConfig()
    assert config._clock is None
    # _clock should not appear in serialized output
    dumped = config.model_dump()
    assert '_clock' not in dumped
    assert 'clock' not in dumped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/test_sim_config.py::test_sim_config_clock_field_default_none -v'`
Expected: FAIL — `AttributeError: 'SimConfig' object has no attribute '_clock'`

- [ ] **Step 3: Add `_clock` field to SimConfig**

In `tensorrt_llm/llmapi/sim_config.py`, add to the `SimConfig` class after the `predictor` field:

```python
from typing import Any, Literal, Optional

class SimConfig(StrictBaseModel):
    """Simulation mode configuration.

    When set on TorchLlmArgs, enables GPU-free simulation: the real
    scheduler runs but model forward is replaced with predicted timing.
    """

    predictor: PredictorConfig = Field(
        default_factory=PredictorConfig,
        description="Time predictor configuration.")

    _clock: Any = Field(default=None, exclude=True)
```

Note: `Any` is imported from `typing`. The underscore prefix + `exclude=True` keeps this out of Pydantic serialization and signals internal use.

**Important Pydantic detail**: In `StrictBaseModel` (Pydantic v2), underscore-prefixed fields declared with `Field()` need `model_config = ConfigDict(...)` to allow them. Check if `StrictBaseModel` already allows this. If Pydantic rejects `_clock` as a private attribute, use `clock` (no underscore) with `exclude=True` instead, or use Pydantic's `PrivateAttr`:

```python
from pydantic import PrivateAttr

class SimConfig(StrictBaseModel):
    predictor: PredictorConfig = Field(
        default_factory=PredictorConfig,
        description="Time predictor configuration.")

    _clock: Any = PrivateAttr(default=None)
```

`PrivateAttr` is the idiomatic Pydantic v2 way to store internal state that isn't part of the schema.

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/test_sim_config.py -v'`
Expected: ALL PASSED (existing tests + new one)

- [ ] **Step 5: Commit**

```bash
git add tensorrt_llm/llmapi/sim_config.py tests/unittest/sim/test_sim_config.py
git commit -s -m "feat: Add _clock private attribute to SimConfig"
```

---

### Task 3: Wire SimClock into SimModelEngine

**Files:**
- Modify: `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py`

- [ ] **Step 1: Modify SimModelEngine to accept and use SimClock**

Replace the `time.sleep` logic in `sim_model_engine.py`. The full updated file:

In the constructor, add `clock=None` parameter:

```python
def __init__(self, llm_args: TorchLlmArgs, vocab_size: int,
             max_num_sequences: int, time_predictor=None, clock=None):
    # ... existing attribute assignments ...
    self.time_predictor = time_predictor
    self.clock = clock
```

In `forward()`, replace:

```python
# OLD:
import time
# ...
if predicted_time > 0:
    time.sleep(predicted_time)
```

with:

```python
# NEW:
if self.clock is not None:
    self.clock.step(predicted_time)
    logger.debug(
        "[SimModelEngine] iter=%d predicted=%.3fms total=%.3fms",
        self.clock.num_iterations,
        predicted_time * 1000,
        self.clock.total_time_s * 1000)
```

Also remove `import time` from the top of the file (line 21).

- [ ] **Step 2: Run existing unit tests to verify nothing breaks**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/ -v'`
Expected: ALL PASSED (SimClock tests + existing predictor/config tests)

- [ ] **Step 3: Commit**

```bash
git add tensorrt_llm/_torch/pyexecutor/sim_model_engine.py
git commit -s -m "feat: Replace time.sleep with SimClock.step in SimModelEngine"
```

---

### Task 4: Wire SimClock in py_executor_creator

**Files:**
- Modify: `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py:276-298`

- [ ] **Step 1: Create SimClock and pass to SimModelEngine**

In `_create_sim_py_executor()`, after the predictor factory (line 294) and before the `SimModelEngine` creation (line 296), add:

```python
from .sim_clock import SimClock

clock = SimClock()
```

Then modify the `SimModelEngine` constructor call (line 296-297) from:

```python
model_engine = SimModelEngine(llm_args, vocab_size, max_num_sequences,
                               time_predictor=predictor)
```

to:

```python
model_engine = SimModelEngine(llm_args, vocab_size, max_num_sequences,
                               time_predictor=predictor, clock=clock)
```

Then after `py_executor.start_worker()` (line 378), store the clock on the config:

```python
py_executor.start_worker()
sim_config._clock = clock
logger.info("[SimMode] PyExecutor created in simulation mode (clock enabled)")
```

Replace the existing log line at line 379.

- [ ] **Step 2: Run unit tests**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/ -v'`
Expected: ALL PASSED

- [ ] **Step 3: Commit**

```bash
git add tensorrt_llm/_torch/pyexecutor/py_executor_creator.py
git commit -s -m "feat: Wire SimClock into sim executor creation"
```

---

### Task 5: Update e2e test with clock assertions

**Files:**
- Modify: `slop/test_sim.py`

- [ ] **Step 1: Update e2e test**

Replace `slop/test_sim.py` with clock-based assertions:

```python
"""Smoke test for simulation mode with SimConfig."""
import os
import time

os.environ["TRTLLM_LOG_LEVEL"] = "WARNING"

AIC_SYSTEMS_DIR = "/code/slop/aiconfigurator/src/aiconfigurator/systems"


def test_constant_predictor():
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    MODEL_PATH = "/code/slop/models/TinyLlama-1.1B-Chat-v1.0"

    print("\n=== Constant Predictor ===", flush=True)
    sim_config = SimConfig(predictor=PredictorConfig(
        constant_prefill_time_ms=10.0, constant_decode_time_ms=5.0))

    llm = LLM(MODEL_PATH, sim_config=sim_config)

    start = time.monotonic()
    output = llm.generate(["Hello world"],
                          sampling_params=SamplingParams(max_tokens=8))
    wall_clock_ms = (time.monotonic() - start) * 1000

    token_ids = output[0].outputs[0].token_ids
    print(f"Tokens: {token_ids}", flush=True)
    print(f"Wall-clock: {wall_clock_ms:.0f}ms", flush=True)
    assert len(token_ids) == 8
    assert output[0].outputs[0].finish_reason == "length"

    # Clock assertions
    clock = sim_config._clock
    assert clock is not None, "SimClock not attached to sim_config"
    print(f"Predicted time: {clock.total_time_s * 1000:.1f}ms", flush=True)
    print(f"Iterations: {clock.num_iterations}", flush=True)

    # 1 prefill (10ms) + 8 decodes (5ms each = 40ms) = 50ms
    assert clock.total_time_s > 0
    assert abs(clock.total_time_s - 0.050) < 0.001, \
        f"Expected ~50ms, got {clock.total_time_s * 1000:.1f}ms"
    assert clock.num_iterations == 9, \
        f"Expected 9 iterations, got {clock.num_iterations}"

    print("CONSTANT OK", flush=True)


def test_aiconfigurator_predictor():
    from tensorrt_llm.llmapi import LLM, SamplingParams
    from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

    MODEL_PATH = "/code/slop/models/TinyLlama-1.1B-Chat-v1.0"

    print("\n=== AIConfigurator Predictor ===", flush=True)
    sim_config = SimConfig(predictor=PredictorConfig(
        name="aiconfigurator",
        device_name="h100_sxm",
        backend_version="1.2.0rc5",
        database_path=AIC_SYSTEMS_DIR))

    llm = LLM(MODEL_PATH, sim_config=sim_config)

    start = time.monotonic()
    output = llm.generate(["Hello world"],
                          sampling_params=SamplingParams(max_tokens=8))
    wall_clock_ms = (time.monotonic() - start) * 1000

    token_ids = output[0].outputs[0].token_ids
    print(f"Tokens: {token_ids}", flush=True)
    print(f"Wall-clock: {wall_clock_ms:.0f}ms", flush=True)
    assert len(token_ids) == 8
    assert output[0].outputs[0].finish_reason == "length"

    # Clock assertions
    clock = sim_config._clock
    assert clock is not None, "SimClock not attached to sim_config"
    print(f"Predicted time: {clock.total_time_s * 1000:.1f}ms", flush=True)
    print(f"Iterations: {clock.num_iterations}", flush=True)

    assert clock.total_time_s > 0, "AIC should predict positive time"
    assert clock.num_iterations == 9, \
        f"Expected 9 iterations, got {clock.num_iterations}"

    print("AIC OK", flush=True)


def main():
    test_constant_predictor()
    test_aiconfigurator_predictor()
    print("\n=== ALL TESTS PASSED ===", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run e2e test in container**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 slop/test_sim.py'`
Expected: `ALL TESTS PASSED` with predicted times printed and no `time.sleep` delays in the iteration loop.

- [ ] **Step 3: Run full unit test suite**

Run: `docker exec trtllm-hisim-dev3 bash -c 'cd /code && python3 -m pytest tests/unittest/sim/ -v'`
Expected: ALL PASSED (should be ~43 tests: 38 existing + 5 SimClock + 1 config)

- [ ] **Step 4: Commit**

```bash
git add slop/test_sim.py
git commit -s -m "feat: Update e2e tests with SimClock assertions"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] SimClock class in own file → Task 1
- [x] SimModelEngine uses clock.step() instead of time.sleep() → Task 3
- [x] Wire clock in py_executor_creator → Task 4
- [x] SimConfig._clock field → Task 2
- [x] Unit tests for SimClock → Task 1
- [x] E2e test with clock assertions → Task 5
- [x] Verifiable end-state (constant predictor: 50ms, 9 iterations) → Task 5

**Placeholder scan:** None found.

**Type consistency:** `SimClock.step()`, `SimClock.total_time_s`, `SimClock.num_iterations`, `SimClock.reset()` — consistent across all tasks. `sim_config._clock` — consistent in Tasks 2, 4, 5.
