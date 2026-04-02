# Phase 4: What Was Built

**Date**: 2026-04-02
**Branch**: `venky/hisim-port`
**Commit**: `9560d2038`

## What Changed

| File | Change |
|------|--------|
| `tensorrt_llm/_torch/pyexecutor/sim_metrics.py` | **Created** — `SimRequestStats`, `SimIterationRecord`, `calc_sim_metrics()`, `percentile()` |
| `tensorrt_llm/_torch/pyexecutor/sim_clock.py` | Extended with `_iterations`, `_request_stats`, recording methods, `metrics` property, `write_metrics()` |
| `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py` | Added `clock.record_iteration()` after `clock.step()` |
| `tensorrt_llm/_torch/pyexecutor/sim_sampler.py` | Added `__init__(clock=None)`, `register_request()`, `record_token()` in `update_requests()` |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` | Pass `clock=clock` to `SimSampler()` |
| `tests/unittest/sim/test_sim_metrics.py` | **Created** — 19 tests |
| `tests/unittest/sim/test_sim_clock.py` | Added 6 recording tests |
| `slop/test_sim.py` | Full metrics assertions: constant + AIC + TP=2 |

## Key Discovery: TTFT Includes First Decode

PyExecutor's sampler is NOT called during the prefill iteration for context
requests. The first `update_requests()` call happens on the first decode
iteration. So for constant predictor (10ms prefill, 5ms decode):

- **TTFT = 15ms** (prefill + first decode), not 10ms
- This matches real inference: the first *generation* token arrives after
  prefill completes AND the first decode step runs

## Metrics Output

```python
clock = sim_config._clock
m = clock.metrics
# Returns dict with:
# mean_ttft_ms, p50_ttft_ms, p95_ttft_ms, p99_ttft_ms
# mean_tpot_ms, p50_tpot_ms, p95_tpot_ms, p99_tpot_ms
# mean_itl_ms, p50_itl_ms, p95_itl_ms, p99_itl_ms
# mean_e2e_latency_ms, p95_e2e_latency_ms, p99_e2e_latency_ms
# completed, total_input, total_output, duration
# request_throughput, input_throughput, output_throughput
```

## Verification Results

| Test | TTFT | TPOT | E2E | Throughput |
|------|------|------|-----|-----------|
| Constant (10/5ms) | 15.0ms | 4.3ms | 45.0ms | 177.8 tok/s |
| AIC H100 TP=1 | 3.19ms | 1.21ms | 11.66ms | 686.2 tok/s |
| AIC H100 TP=2 | 2.91ms | 1.03ms | — | — |

## Test Counts

- Unit tests: 82 (19 new sim_metrics + 6 new sim_clock recording + 57 existing)
- E2e tests: 3 (constant metrics, AIC metrics, AIC TP=2 metrics)
