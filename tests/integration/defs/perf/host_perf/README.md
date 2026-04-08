# Host Performance Regression Tests

## Purpose

These tests detect **host (CPU) performance regressions** in the PyExecutor
pipeline using a two-layer approach:

- **Layer 1 (E2E)**: Run real models with `trtllm-serve` on host-overhead-dominant
  workloads. Standard metrics (ITL, TPOT, throughput) catch regressions.
- **Layer 2 (Module)**: Isolated benchmarks of individual modules (scheduler,
  sampler, resource manager). Pinpoint *which* module regressed.

## Layer 1: E2E Tests

### Why these workloads detect host regressions

| Factor | Choice | Effect |
|--------|--------|--------|
| Model size | Small (8B-16B) | Fast GPU kernels, host overhead exposed |
| Batch size | Small (1-32) | GPU not saturated, host scheduling overhead visible |
| Sequence length | Short (ISL=128, OSL=128-256) | High iteration rate, many scheduling cycles |
| GPU count | 1 | No communication overhead |

### Models

- **DeepSeek-V3-Lite (FP8)**: MLA attention + MoE architecture. Exercises
  attention DP scheduling and expert routing host paths.
- **Llama-3.1-8B (FP16)**: Dense model baseline. Covers core scheduler/sampler
  overhead without MoE/MLA-specific paths.

### E2E Metrics

- **Mean ITL**: Inter-token latency — most direct proxy for per-iteration host overhead
- **Mean TPOT**: Time per output token — includes host + GPU per token
- **P99 ITL**: Catches outlier iterations (GC pauses, scheduling spikes)
- **Token throughput**: Overall throughput sanity check

### Running E2E tests

```bash
# Run all E2E host perf tests
pytest tests/integration/defs/perf/host_perf/test_host_perf.py -v \
    --output-dir ./host_perf_results

# Run a specific config
pytest tests/integration/defs/perf/host_perf/test_host_perf.py -v \
    -k "v3lite_fp8_bs1" --output-dir ./host_perf_results
```

Requires: GPU access (1 GPU), `LLM_MODELS_ROOT` set to model weights directory.

## Layer 2: Module Tests

### Scheduler (`test_module_scheduler.py`)

Benchmarks `schedule_request()` latency at various batch sizes. Runs entirely
on CPU — no GPU or model weights required. Tests the Python-side overhead of
the `SimpleUnifiedScheduler` (capacity check + micro-batch scheduling).

```bash
pytest tests/integration/defs/perf/host_perf/test_module_scheduler.py -v -s
```

### Module Metrics

- **Mean latency (µs)**: Average per-call cost
- **P50/P99 latency (µs)**: Distribution characteristics
- **Calls/sec**: Throughput under stress

## Adding new configs

### E2E configs
1. Create or edit a `host_perf_*.yaml` file in this directory
2. Follow the existing YAML format (see `host_perf_deepseek_v3_lite.yaml`)
3. Keep configs host-overhead-dominant: small batch, short sequences, small models
4. Test cases are auto-discovered from YAML files

### Module tests
1. Create a `test_module_<name>.py` file
2. Set up minimal real objects with synthetic workload state
3. Time the target function in a tight loop (1000+ calls)
4. Report mean/P50/P99 latency per call
