# Phase 5: What Was Built

**Date**: 2026-04-02
**Branch**: `venky/hisim-port`
**Commit**: `2a7695027`

## What Changed

| File | Change |
|------|--------|
| `tensorrt_llm/bench/benchmark/sim_benchmark.py` | **Created** — `load_sim_config()`, `run_sim_benchmark()`, `print_sim_report()`, `write_sim_report()`, `write_sim_request_report()` |
| `tensorrt_llm/bench/benchmark/throughput.py` | Added `--sim` and `--sim-config` flags; sim mode branch bypasses `async_benchmark` |
| `slop/calibrate_sim.py` | **Created** — Extract prefill/decode times from real iteration log |
| `slop/compare_reports.py` | **Created** — Side-by-side comparison of real vs sim report.json |
| `slop/test_bench_sim.py` | **Created** — 3-tier CLI e2e test |

## Usage

```bash
# Quick sim (constant predictor defaults)
trtllm-bench -m model throughput --dataset data.jsonl --sim

# AIC sim with TP=2
trtllm-bench -m model throughput --dataset data.jsonl --sim \
  --sim-config sim.yaml --tp 2

# Calibrated sim vs real
trtllm-bench -m model throughput --dataset data.jsonl --backend pytorch \
  --iteration_log real_iters.jsonl --report_json real.json
python3 slop/calibrate_sim.py --iteration-log real_iters.jsonl --output cal.yaml
trtllm-bench -m model throughput --dataset data.jsonl --sim --sim-config cal.yaml \
  --report_json sim.json
python3 slop/compare_reports.py --real real.json --sim sim.json
```

## Key Discovery: Iteration Log Format

TRT-LLM's `--iteration_log` writes Python repr format (single quotes, `None`),
not JSON. The `calibrate_sim.py` script handles both formats by normalizing
quotes before `json.loads()`. Field names are camelCase (`iterLatencyMS`,
`inflightBatchingStats.numContextRequests`).

## Verification Results (3 Tiers)

| Tier | Test | Result |
|------|------|--------|
| 1 | Constant predictor, 10 requests | 1523.8 tok/s, TTFT=33ms, TPOT=4.77ms |
| 2 | AIC H100 TP=2, 10 requests | 6662.4 tok/s, TTFT=5.54ms, TPOT=1.22ms |
| 3 | Calibrated vs real RTX 3090 Ti | Real: prefill=131.58ms, decode=7.63ms. Sim matches structure. |

## Test Counts

- Unit tests: 82 passing
- CLI e2e: 3 tiers passing
