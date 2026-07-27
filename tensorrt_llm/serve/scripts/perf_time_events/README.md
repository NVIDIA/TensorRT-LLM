<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Perf Time Events

End-to-end per-request perf timeline capture for TensorRT-LLM, built for
debugging scheduling stalls / gen bubbles in disaggregated (ctx ⇄ gen) serving.

There are two pieces:

1. **Live per-rank capture** (load-bearing) — set one environment variable and
   every executor rank writes its own `time_events_rank{N}_pid{P}.jsonl` file as
   requests finish. No HTTP scrape, no `--save-request-time-breakdown`, no
   teardown race.
2. **Offline aggregator** (this package, a convenience) — stitches the per-rank
   files (and the disaggregation KV-transfer CSVs) into a single combined JSON
   and, optionally, an interactive HTML timeline.

## 1. Capturing (the master switch)

Set `TRTLLM_PERF_TIME_EVENTS_PATH` to an **output directory** on every worker
(both ctx and gen servers for a disagg run):

```bash
export TRTLLM_PERF_TIME_EVENTS_PATH=/tmp/perf_events
# Optional but recommended for disagg — the KV-transfer sub-event CSVs:
export TRTLLM_KVCACHE_TIME_OUTPUT_PATH=/tmp/kv_csv
```

Setting `TRTLLM_PERF_TIME_EVENTS_PATH` alone:

- **Force-enables capture** regardless of `return_perf_metrics` / `LLM` args.
- Turns on the **extended per-iteration batch context** — each per-iteration
  `step_metrics` / `ctx_chunk_metrics` entry gains:
  `iter_counter`, `iter_batch_size`, `num_ctx_requests`, `num_gen_requests`,
  `context_token_number`, `generation_token_number`, plus the per-request
  `req_context_token_number` / `req_generation_token_number`, and the
  per-iteration starvation counters `num_capacity_fitting` / `num_scheduled`.
- Starts a **daemon writer thread per rank** that drains an in-process queue and
  writes the JSONL file. The executor loop only builds a dict and does a
  non-blocking `queue.put_nowait`, so capture stays off the critical path.

After the run:

```
/tmp/perf_events/time_events_rank0_pid12345.jsonl   # one JSON object per finished request
/tmp/perf_events/time_events_rank1_pid12345.jsonl
...
/tmp/kv_csv/<instance>_<rank>.csv                    # KV-transfer task rows (native transceiver)
/tmp/kv_csv/<instance>_<rank>_gen_transfer_summary.csv
```

Each JSONL record is `{request_id, rank, ctx_request_id, time_breakdown_metrics}`
— the same `time_breakdown_metrics` shape the `time_breakdown` tool already
understands, so a single per-rank file can be fed straight to that tool too.

## 2. Aggregating (optional)

```bash
python -m tensorrt_llm.serve.scripts.perf_time_events \
    --event-dir /tmp/perf_events \
    --kv-csv-dir /tmp/kv_csv \
    -o /tmp/perf_events/combined.json \
    --html /tmp/perf_events/timeline.html
```

Arguments (all optional; env-derived defaults shown):

| Flag | Default | Purpose |
|---|---|---|
| `--event-dir` | `$TRTLLM_PERF_TIME_EVENTS_PATH` | Directory of per-rank `time_events_*.jsonl`. |
| `--kv-csv-dir` | `$TRTLLM_KVCACHE_TIME_OUTPUT_PATH` | Directory of KV-transfer CSVs to join in. |
| `--perf-json` | — | An additional `/perf_metrics` JSON dump (aggregated single-server runs). |
| `-o`, `--output` | `perf_time_events.combined.json` | Combined JSON output path. |
| `--html [PATH]` | off | Also emit the interactive HTML timeline (reuses `time_breakdown`). |

The combined JSON is `{records, unjoined_kv_events, match_stats}`. Each record
gains, where available:

- `kv_transfer_events` / `kv_gen_summary` / `kv_cpp_events` — KV rows joined by
  `request_id` / `ctx_request_id` (native join key `unique_rid`; C++ join key
  `RequestID`). Rows that never matched a request are reported under
  `unjoined_kv_events` with a match-rate WARNING.
- `derived` — `inter_step_gaps` / `inter_chunk_gaps` (idle time between
  consecutive forward passes) and `starved` (per-iteration
  `num_capacity_fitting − num_scheduled`).

`--html` is the only path that pulls in `plotly` (lazily, via the sibling
`time_breakdown` package); the parse / merge / JSON path is pure stdlib.

## Known limitations (Python-only capture)

- **Starvation is a per-iteration count**, not a per-request attribution: the
  Python scheduler only exposes `num_fitting_reqs`, not which requests were
  capacity-fit but not micro-batch scheduled.
- **Pipeline parallelism**: `_executor_loop_pp` does not record extended fields,
  so batch-context keys are absent under PP.
- **ctx ⇄ gen correlation** across the two disagg servers relies on the request's
  disagg `ctx_request_id` when exposed; otherwise the ctx-side and gen-side
  records live in separate per-rank files keyed by their own request ids.

See `docs/source/developer-guide/perf-analysis.md` and `kv-transfer.md` for the
broader perf-analysis workflow.
