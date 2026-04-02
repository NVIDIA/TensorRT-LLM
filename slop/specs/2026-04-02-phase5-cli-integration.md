# Phase 5: CLI Integration — Design Spec

**Builds on**: Phases 0-4
**Date**: 2026-04-02

## Problem

Sim mode requires writing Python code to use. Engineers comparing serving
configurations need a CLI tool that loads a dataset, runs simulation, and
reports metrics — matching `trtllm-bench throughput`'s output format for
direct comparison.

## Goal

Add `--sim` and `--sim-config` flags to `trtllm-bench throughput` that run
simulation mode and produce output in the same format as a real benchmark.

## User Experience

```bash
# Quick sim with constant predictor defaults
trtllm-bench -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 throughput \
  --dataset data.jsonl --sim --tp 2

# Full AIC sim with config file
trtllm-bench -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 throughput \
  --dataset data.jsonl --sim --sim-config sim.yaml --tp 2

# With JSON output (same flags as normal throughput)
trtllm-bench -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 throughput \
  --dataset data.jsonl --sim --report_json report.json --request_json requests.json
```

## Architecture

When `--sim` is set, the throughput command bypasses `async_benchmark()` and
instead:

1. Creates `SimConfig` from `--sim-config` YAML or constant predictor defaults
2. Injects `sim_config` into LLM kwargs
3. Calls `llm.generate()` synchronously with all dataset requests
4. Reads `sim_config._clock.metrics` for results
5. Prints formatted table with `[SIM]` banner
6. Writes JSON reports via existing `--report_json` / `--request_json` flags

```
throughput_command()
  ├── Parse args (--sim detected)
  ├── Load dataset (same as normal — create_dataset_from_stream)
  ├── Build SimConfig (from YAML or defaults)
  ├── Create LLM with sim_config in kwargs
  │   └── Reuses: --tp, --max_batch_size, --kv_cache_free_gpu_mem_fraction
  ├── run_sim_benchmark(llm, requests, sim_config, sampling_params)
  │   ├── llm.generate(all_prompts, sampling_params)
  │   ├── sim_config._clock.metrics → metrics dict
  │   └── return metrics
  ├── Print formatted table with [SIM] banner
  └── Write --report_json / --request_json if requested
```

## CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sim` | flag | False | Enable simulation mode |
| `--sim-config` | Path | None | Path to YAML with predictor config |

### sim-config YAML format

```yaml
predictor:
  name: aiconfigurator
  device_name: h100_sxm
  backend_version: 1.2.0rc5
  # Optional:
  database_path: /path/to/systems
  prefill_scale_factor: 1.0
  decode_scale_factor: 1.0
```

When `--sim` without `--sim-config`: uses constant predictor (10ms prefill,
5ms decode) as a quick-start default.

## Output Format

Same table layout as normal throughput with `[SIM]` banner:

```
===========================================================
= SIM THROUGHPUT BENCHMARK RESULTS
===========================================================
[METADATA]
  Model:                  TinyLlama/TinyLlama-1.1B-Chat-v1.0
  TP:                     2
  Predictor:              aiconfigurator (h100_sxm)
  Num Requests:           100
  Avg ISL:                128
  Avg OSL:                32

[RESULTS]
  Total Output Tokens:    3200
  Simulated Duration (s): 1.234
  Request Throughput (req/s):  81.0
  Output Throughput (tok/s):   2593.2
  Mean TTFT (ms):         3.19
  P95 TTFT (ms):          4.12
  P99 TTFT (ms):          5.01
  Mean TPOT (ms):         1.21
  P95 TPOT (ms):          1.45
  P99 TPOT (ms):          1.52
  Mean E2E Latency (ms):  40.8
```

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tensorrt_llm/bench/benchmark/throughput.py` | Modify | Add `--sim`, `--sim-config` flags; branch to sim path |
| `tensorrt_llm/bench/benchmark/sim_benchmark.py` | **Create** | `run_sim_benchmark()`, `load_sim_config()`, `print_sim_report()`, `extract_real_timings()` |
| `slop/test_bench_sim.py` | **Create** | E2e CLI test script |

## Verifiable End-State (3 Tiers)

### Tier 1: Sim mode runs and produces output

```bash
# Prepare dataset
trtllm-bench -m /code/slop/models/TinyLlama-1.1B-Chat-v1.0 prepare-dataset \
  --stdout token-norm-dist --num-requests 100 --avg-isl 128 --avg-osl 32 > /tmp/data.jsonl

# Sim run (constant predictor)
trtllm-bench -m /code/slop/models/TinyLlama-1.1B-Chat-v1.0 throughput \
  --dataset /tmp/data.jsonl --sim --report_json /tmp/sim_report.json

# Verify:
# - Exits 0
# - [SIM] banner printed
# - /tmp/sim_report.json written with metrics > 0
# - completed == 100
```

### Tier 2: AIC sim with TP=2

```bash
# sim.yaml for AIC
cat > /tmp/sim_aic.yaml << 'EOF'
predictor:
  name: aiconfigurator
  device_name: h100_sxm
  backend_version: 1.2.0rc5
  database_path: /code/slop/aiconfigurator/src/aiconfigurator/systems
EOF

trtllm-bench -m /code/slop/models/TinyLlama-1.1B-Chat-v1.0 throughput \
  --dataset /tmp/data.jsonl --sim --sim-config /tmp/sim_aic.yaml --tp 2 \
  --report_json /tmp/sim_aic_report.json

# Verify:
# - Exits 0, metrics plausible
# - TTFT > 0, TPOT > 0, throughput > 0
# - completed == 100
```

### Tier 3: Calibrated constant predictor vs real silicon

```bash
# Step 1: Real run on RTX 3090 Ti with iteration logging
trtllm-bench -m /code/slop/models/TinyLlama-1.1B-Chat-v1.0 throughput \
  --dataset /tmp/data.jsonl --backend pytorch \
  --iteration_log /tmp/real_iterations.jsonl \
  --report_json /tmp/real_report.json

# Step 2: Extract coarse prefill/decode times from iteration log
python3 /code/slop/calibrate_sim.py \
  --iteration-log /tmp/real_iterations.jsonl \
  --output /tmp/calibrated_sim.yaml

# Step 3: Sim run with calibrated constant times
trtllm-bench -m /code/slop/models/TinyLlama-1.1B-Chat-v1.0 throughput \
  --dataset /tmp/data.jsonl --sim --sim-config /tmp/calibrated_sim.yaml \
  --report_json /tmp/sim_calibrated_report.json

# Step 4: Compare
python3 /code/slop/compare_reports.py \
  --real /tmp/real_report.json \
  --sim /tmp/sim_calibrated_report.json

# Expected:
# - completed: exact match (100 == 100)
# - total_output: exact match (same dataset)
# - TTFT: within ~30% (constant predictor is coarse)
# - TPOT: within ~30%
# - throughput: within ~30%
```

### Tier 3 Helper Scripts

**`slop/calibrate_sim.py`** — Parses `--iteration_log` JSONL from a real
run, computes mean prefill and decode latencies, writes a constant
predictor sim.yaml:

```python
# Reads iteration.jsonl lines, classifies as prefill (has context requests)
# or decode (generation only), computes mean latency for each, outputs YAML.
```

**`slop/compare_reports.py`** — Loads two report.json files, prints
side-by-side comparison with % difference for each metric.

## Design Decisions

1. **`--sim` on throughput, not a new subcommand** — sim is a mode of the
   throughput benchmark, not a separate tool. Same dataset, same config.
2. **Bypass async_benchmark** — sim is synchronous batch. Running through
   async just to discard wall-clock stats adds complexity.
3. **Same output format** — enables direct comparison with real runs.
4. **Constant predictor as default** — instant value without AIC setup.
   `--sim-config` for full control.
5. **Calibration script is a helper, not core** — lives in `slop/`, not
   in the package. Tier 3 is a validation exercise, not a feature.

## What Stays the Same

- Dataset loading (`create_dataset_from_stream`)
- Tokenizer initialization
- `--tp`, `--pp`, `--max_batch_size`, `--kv_cache_free_gpu_mem_fraction`
- `--report_json`, `--request_json` flag names
- `--num_requests`, `--warmup` (warmup skipped in sim mode)

## Out of Scope

- `trtllm-bench latency --sim` (latency command, future)
- `trtllm-serve --sim` (serving mode, future)
- `--request-rate` / online arrival modeling (Phase 6)
- Automatic AIC database discovery (user provides path)
