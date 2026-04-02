# Phase 4: Metrics Output — Design Spec

**Builds on**: Phases 0-3.5
**Date**: 2026-04-02

## Problem

SimClock accumulates total predicted time and iteration count, but there's
no per-request or per-iteration breakdown. Users can't compute TTFT, TPOT,
ITL, throughput, or compare serving configurations.

## Goal

Extend SimClock to record per-iteration and per-request timing. Expose
metrics via a Python API (`sim_config._clock.metrics`) and optional file
output (`metrics.json`, `request.jsonl`, `iteration.jsonl`).

## Scope

- Batch/offline mode only (all requests arrive at t=0)
- Python API first, file output as convenience method
- HiSim-compatible metric definitions (TTFT, TPOT, ITL, e2e, throughput)
- No numpy dependency — use stdlib `statistics` for percentiles
- Verify with both constant predictor (exact values) and AIC (structural checks)

## Data Model

### SimRequestStats

```python
@dataclass
class SimRequestStats:
    request_id: int
    input_length: int
    output_length: int = 0
    created_time: float = 0.0
    gen_token_times: list[float] = field(default_factory=list)

    @property
    def ttft_s(self) -> float:
        """Time to first token (seconds)."""
        return self.gen_token_times[0] - self.created_time if self.gen_token_times else 0.0

    @property
    def itl_s(self) -> list[float]:
        """Inter-token latencies (seconds)."""
        times = self.gen_token_times
        return [times[i] - times[i-1] for i in range(1, len(times))]

    @property
    def tpot_s(self) -> float:
        """Time per output token (seconds). Mean of ITL."""
        itl = self.itl_s
        return sum(itl) / len(itl) if itl else 0.0

    @property
    def e2e_s(self) -> float:
        """End-to-end latency (seconds)."""
        return self.gen_token_times[-1] - self.created_time if self.gen_token_times else 0.0
```

### SimIterationRecord

```python
@dataclass
class SimIterationRecord:
    iteration: int
    sim_time_s: float
    predicted_duration_s: float
    num_context_requests: int
    num_context_tokens: int
    num_generation_requests: int
```

### SimClock Extensions

```python
class SimClock:
    # Existing
    _total_time_s: float
    _num_iterations: int

    # New
    _iterations: list[SimIterationRecord]
    _request_stats: dict[int, SimRequestStats]

    def record_iteration(self, predicted_duration_s, num_ctx_req,
                         num_ctx_tokens, num_gen_req): ...

    def register_request(self, request_id, input_length,
                         created_time=0.0): ...

    def record_token(self, request_id): ...

    @property
    def metrics(self) -> dict: ...

    @property
    def request_stats(self) -> dict[int, SimRequestStats]: ...

    @property
    def iterations(self) -> list[SimIterationRecord]: ...

    def write_metrics(self, output_dir: str) -> None: ...
```

## Recording Flow

```
SimModelEngine.forward(scheduled_requests)
  ├── predicted_time = predictor.predict(batch)
  ├── clock.step(predicted_time)
  ├── clock.record_iteration(predicted_time, num_ctx, ...)  ← NEW
  └── return dummy logits

SimSampler.update_requests(state)
  ├── For each request:
  │   ├── if new: clock.register_request(id, prompt_len)   ← NEW
  │   ├── request.add_new_token(0, 0)
  │   ├── clock.record_token(request_id)                    ← NEW
  │   └── check max_tokens → GENERATION_COMPLETE
```

### Batch time attribution

All requests in a batch get the same clock timestamp for their token.
This is correct: every request in a decode batch waited the full batch
time for its next token. Matches HiSim's behavior.

## Metrics Computation

`calc_sim_metrics()` is a pure function taking `request_stats` dict:

```python
def calc_sim_metrics(request_stats, iterations) -> dict:
    ttfts = [s.ttft_s for s in request_stats.values()]
    tpots = [s.tpot_s for s in request_stats.values() if s.itl_s]
    itls = [l for s in request_stats.values() for l in s.itl_s]
    e2es = [s.e2e_s for s in request_stats.values()]

    total_output = sum(s.output_length for s in request_stats.values())
    total_input = sum(s.input_length for s in request_stats.values())
    duration_s = max((s.gen_token_times[-1] for s in request_stats.values()
                      if s.gen_token_times), default=1e-9)

    return {
        "completed": len(request_stats),
        "total_input": total_input,
        "total_output": total_output,
        "duration": duration_s,
        "request_throughput": len(request_stats) / duration_s,
        "output_throughput": total_output / duration_s,
        "input_throughput": total_input / duration_s,
        "mean_ttft_ms": mean(ttfts) * 1000,
        "p50_ttft_ms": percentile(ttfts, 50) * 1000,
        "p95_ttft_ms": percentile(ttfts, 95) * 1000,
        "p99_ttft_ms": percentile(ttfts, 99) * 1000,
        "mean_tpot_ms": mean(tpots) * 1000,
        "p50_tpot_ms": percentile(tpots, 50) * 1000,
        "p95_tpot_ms": percentile(tpots, 95) * 1000,
        "p99_tpot_ms": percentile(tpots, 99) * 1000,
        "mean_itl_ms": mean(itls) * 1000,
        "p50_itl_ms": percentile(itls, 50) * 1000,
        "p95_itl_ms": percentile(itls, 95) * 1000,
        "p99_itl_ms": percentile(itls, 99) * 1000,
        "mean_e2e_latency_ms": mean(e2es) * 1000,
        "p95_e2e_latency_ms": percentile(e2es, 95) * 1000,
        "p99_e2e_latency_ms": percentile(e2es, 99) * 1000,
    }
```

Percentile helper uses sorted list + interpolation (no numpy).

## File Output

`clock.write_metrics(output_dir)` produces:
- `metrics.json` — `json.dumps(clock.metrics, indent=2)`
- `request.jsonl` — one JSON line per request: `{request_id, input_length, output_length, created_time, gen_token_times, ttft_ms, tpot_ms, e2e_ms}`
- `iteration.jsonl` — one JSON line per iteration: `{iteration, sim_time_s, predicted_duration_s, num_context_requests, num_context_tokens, num_generation_requests}`

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tensorrt_llm/_torch/pyexecutor/sim_metrics.py` | **Create** | `SimRequestStats`, `SimIterationRecord`, `calc_sim_metrics()`, percentile helper |
| `tensorrt_llm/_torch/pyexecutor/sim_clock.py` | Modify | Add `_iterations`, `_request_stats`, recording methods, `metrics` property, `write_metrics()` |
| `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py` | Modify | Call `clock.record_iteration()` after `clock.step()` |
| `tensorrt_llm/_torch/pyexecutor/sim_sampler.py` | Modify | Call `clock.register_request()` + `clock.record_token()`, pass clock reference |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` | Modify | Pass clock to SimSampler |
| `tests/unittest/sim/test_sim_metrics.py` | **Create** | Unit tests for SimRequestStats, SimIterationRecord, calc_sim_metrics |
| `tests/unittest/sim/test_sim_clock.py` | Modify | Tests for new recording methods |
| `slop/test_sim.py` | Modify | Metrics assertions with constant + AIC + TP=2 |

## Verifiable End-State

### Constant predictor (exact values)

```
mean_ttft_ms ≈ 10.0 (1 prefill iteration)
mean_tpot_ms ≈ 5.0 (each decode iteration)
mean_e2e_latency_ms ≈ 45.0 (10 + 7*5)
output_throughput ≈ 177.8 (8 tokens / 0.045s)
completed == 1
total_output == 8
len(iterations) == 8
len(request_stats) == 1
len(gen_token_times) == 8
```

### AIC predictor (structural checks)

```
mean_ttft_ms > 0
mean_tpot_ms > 0
mean_ttft_ms > mean_tpot_ms (prefill > decode, typical for small models)
e2e ≈ ttft + 7 * tpot (cross-check consistency)
prefill_iteration.predicted_duration != decode_iteration.predicted_duration
completed == 1, total_output == 8
```

### AIC TP=2 (proves TP flows through)

```
tp2.mean_ttft_ms != tp1.mean_ttft_ms
same schema, same iteration count
```

## Design Decisions

1. **No numpy**: Use stdlib for percentiles. Avoids dependency for sim-only feature.
2. **SimSampler gets clock reference**: Passed via constructor, not global state.
3. **`register_request` on first encounter**: SimSampler registers when it first sees a request, reading `orig_prompt_len` for input_length.
4. **`record_token` uses current clock time**: After `SimModelEngine.forward()` advances the clock, `SimSampler.update_requests()` records the new time for each request's token.
5. **Derived properties on SimRequestStats**: TTFT, ITL, TPOT, e2e are computed from `gen_token_times`, not stored separately. Single source of truth.

## Out of Scope

- Request arrival modeling (Phase 4.5)
- Streaming metrics (online mode)
- Mixed batch timing fidelity
- KV cache transfer latency
- Multi-request batch tests (would need online mode for meaningful TTFT distribution)
