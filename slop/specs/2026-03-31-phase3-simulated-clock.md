# Phase 3: Simulated Clock — Design Spec

**Builds on**: Phase 0 + Phase 1 + Phase 2 (`slop/specs/phase{0,1,2}-what-was-built.md`)

## Problem

Phase 2's `SimModelEngine.forward()` calls `time.sleep(predicted_time)` to simulate batch execution time. This pollutes wall-clock measurements — AIC predicts ~13ms for TinyLlama on H100, but real wall-clock is ~205ms because framework overhead dominates. Metrics from Phase 4 need a virtual clock to be meaningful.

## Goal

Replace `time.sleep()` with a `SimClock` that accumulates predicted times. Simulation completes at CPU speed (no sleeping). Total simulated time is readable programmatically after `generate()` returns.

## Scope

- **Batch/offline mode only** — all requests arrive at t=0. No staggered arrival timing.
- **Instance-owned clock** — not a singleton. `SimClock` is owned by `SimModelEngine`.
- **Minimal surface** — `total_time_s` and `num_iterations`. No per-iteration breakdown (Phase 4).

## Verifiable End-State

```python
from tensorrt_llm.llmapi import LLM, SamplingParams
from tensorrt_llm.llmapi.sim_config import SimConfig, PredictorConfig

sim_config = SimConfig(predictor=PredictorConfig(
    constant_prefill_time_ms=10.0, constant_decode_time_ms=5.0))

llm = LLM(model, sim_config=sim_config)

start = time.monotonic()
output = llm.generate(["Hello world"],
                      sampling_params=SamplingParams(max_tokens=8))
wall_clock_ms = (time.monotonic() - start) * 1000

clock = sim_config._clock
# Predicted time: 1 prefill (10ms) + 8 decodes (40ms) = 50ms
assert clock.total_time_s == pytest.approx(0.050, abs=0.001)
assert clock.num_iterations == 9
# Wall-clock dominated by startup, not sleeping — no time.sleep in hot path
```

### Ralph Loop

Execute inline. After each task, verify in container (`docker exec trtllm-hisim-dev3`). Loop until the verifiable end-state above passes.

## Components

### 1. `SimClock` (new file)

**Create**: `tensorrt_llm/_torch/pyexecutor/sim_clock.py`

```python
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

**Design rationale**: HiSim uses a class-level singleton (`StateManager`) because the hook intercepts multiple unrelated code paths. Our architecture is simpler — `SimModelEngine.forward()` is the only writer. An instance keeps things testable and avoids global state.

### 2. `SimModelEngine` changes

**Modify**: `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py`

- Constructor: accept `clock: SimClock` parameter, store as `self.clock`
- `forward()`: replace `time.sleep(predicted_time)` with `self.clock.step(predicted_time)`
- Remove `import time`
- Log iteration info: `logger.debug("[SimModelEngine] iter=%d, predicted=%.3fms, total=%.3fms", ...)`

### 3. Wire clock through `_create_sim_py_executor`

**Modify**: `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py`

- Create `SimClock()` instance
- Pass to `SimModelEngine(... clock=clock)`
- Store on `sim_config._clock = clock` so caller can read results after `generate()`
- Log summary after executor creation: `"[SimMode] Clock initialized"`

### 4. `SimConfig._clock` attribute

**Modify**: `tensorrt_llm/llmapi/sim_config.py`

- Add `_clock: Any = Field(default=None, exclude=True)` to `SimConfig`
  - `exclude=True` keeps it out of Pydantic serialization
  - Underscore prefix signals internal use
  - Type is `Any` to avoid importing `SimClock` in the config layer

### 5. Unit tests

**Create**: `tests/unittest/sim/test_sim_clock.py`

- `test_initial_state` — total=0, iterations=0
- `test_step_accumulates` — step(0.01) x3 → total=0.03, iterations=3
- `test_reset` — step, reset, verify zeroed
- `test_step_zero_duration` — step(0.0) increments iterations but not time

**Modify**: `tests/unittest/sim/test_sim_predictor.py`

- Update any tests that depend on `time.sleep` behavior (if any)

**Modify**: `slop/test_sim.py`

- Replace wall-clock timing assertions with `sim_config._clock` assertions
- Add assertion: `sim_config._clock.total_time_s > 0`
- Add assertion: `sim_config._clock.num_iterations == 9` (1 prefill + 8 decodes)

## File Map

| File | Action |
|------|--------|
| `tensorrt_llm/_torch/pyexecutor/sim_clock.py` | **Create** — `SimClock` class |
| `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py` | Modify — use `SimClock.step()` instead of `time.sleep()` |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` | Modify — create clock, pass to engine, store on config |
| `tensorrt_llm/llmapi/sim_config.py` | Modify — add `_clock` field |
| `tests/unittest/sim/test_sim_clock.py` | **Create** — SimClock unit tests |
| `slop/test_sim.py` | Modify — clock-based assertions |

## What Changes vs. Phase 2

| Before (Phase 2) | After (Phase 3) |
|---|---|
| `time.sleep(predicted_time)` in `forward()` | `self.clock.step(predicted_time)` |
| Wall-clock ~205ms (AIC) or ~54ms (constant) | Wall-clock <5ms for sim iterations (CPU speed) |
| No programmatic access to predicted time | `sim_config._clock.total_time_s` |
| No iteration count | `sim_config._clock.num_iterations` |

## Out of Scope

- Per-iteration breakdown records (Phase 4 — extend `SimClock` with a list)
- Online/streaming request arrival timing (Phase 5)
- Metrics file output — `metrics.json`, `request.jsonl`, `iteration.jsonl` (Phase 4)
- Multi-GPU simulation (Phase 6)

## Dependencies

- No new external dependencies
- Only internal imports within `tensorrt_llm/_torch/pyexecutor/`

## Injection Seams (Phase 4+)

1. **`SimClock.step()`** — Phase 4 can extend to also append per-iteration records
2. **`sim_config._clock`** — Phase 4 metrics writer reads from this after completion
3. **`SimClock` constructor** — Phase 4 can accept a `record_iterations=True` flag
