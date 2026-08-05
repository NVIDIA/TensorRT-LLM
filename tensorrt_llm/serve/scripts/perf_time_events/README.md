<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Perf Time Events

End-to-end per-request perf timeline capture for TensorRT-LLM, built for
debugging scheduling stalls / gen bubbles in disaggregated (ctx ⇄ gen) serving.

Capture is **per event, not per request**: every process appends **one JSONL
line per lifecycle event, flushed as the event happens**. This is the key
property — a request that never completes (a KV-transfer / fill-gate livelock,
a HangDetector `MPI_Abort` at 300 s) still leaves its *partial* timeline on
disk, and the last line it wrote localizes exactly where it wedged. The older
"one compound record emitted at completion" model captured **nothing** for
precisely the hung requests you most want to see.

Only the **gen-init request's lifecycle** is recorded. Per-decode-step events
are **not** emitted (no dense `step_metrics` series), so the volume is a handful
of lines per request rather than hundreds.

There are two pieces:

1. **Live per-event capture** (load-bearing) — set one environment variable per
   process and each one appends flat event lines to its own file as events fire.
   No HTTP scrape, no teardown race.
2. **Offline compiler** (this package) — reads every per-process file, pivots the
   flat event stream **long → wide**, and emits **one combined record per
   request** (plus an optional latency aggregate and HTML timeline).

## 1. Capturing (per-process master switches)

Each participating process is gated on its own directory env var; unset ⇒ that
process's writer is inert (a no-op enqueue, zero overhead, no file). All four
name an **output directory** (symmetric with the KV logger's
`TRTLLM_KVCACHE_TIME_OUTPUT_PATH`), and files are named `{prefix}_pid{P}.jsonl`
so processes sharing a filesystem never collide.

```bash
# On every worker (both ctx and gen servers for a disagg run):
export TRTLLM_PERF_TIME_EVENTS_PATH=/tmp/perf_events
# On the disaggregated router/orchestrator (OpenAIDisaggServer):
export TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH=/tmp/router_events
# On the benchmark client (benchmark_serving load generator):
export TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH=/tmp/client_events
# Optional but recommended for disagg — the KV-transfer sub-event CSVs:
export TRTLLM_KVCACHE_TIME_OUTPUT_PATH=/tmp/kv_csv
```

Setting `TRTLLM_PERF_TIME_EVENTS_PATH` on a worker **force-enables capture**
regardless of `return_perf_metrics` / `LLM` args, and starts a **daemon writer
thread per rank** that drains an in-process queue. The executor loop only builds
a dict and does a non-blocking `queue.put_nowait` (drop-on-full), so capture
stays off the critical path. The router and client use the same stdlib writer
(`tensorrt_llm/serve/perf_time_events_writer.py`) and stay **torch-free** — they
already hold the steady clock and pass `t` in, so the writer never imports it.

### The flat line schema

Every worker and router line is one flat envelope, identical across roles:

```json
{"role": "gen", "event": "gen_first_token", "request_id": 42,
 "ctx_request_id": "abc", "rank": 3, "t": 12345.678, "pid": 12345}
```

| Field | Meaning |
|---|---|
| `role` | `router`, `ctx`, `gen`, or `client`. |
| `event` | Lifecycle event name (see below). |
| `request_id` | Joins the TP ranks of one worker request (worker); the router-local sequence (router). |
| `ctx_request_id` | Cross-process join key: how the compiler stitches ctx ⇄ gen ⇄ router. `null` until the ctx round-trip assigns it (and on the gen-only path). |
| `rank` | Global worker rank (router/client use `0`). |
| `t` | `steady_clock` seconds (`get_steady_clock_now_in_seconds()`); same clock the C++ `/perf_metrics` scalars use. |
| `pid` | Emitting process — disambiguates router-local `request_id`s across processes. |

Router lines additionally carry `disagg_request_id` / `ctx_server` /
`gen_server` provenance.

### Event set

**ctx worker** (`role=ctx`): `ctx_arrival`, `ctx_first_scheduled`,
`ctx_first_token`, `ctx_ready_sent`.

**gen worker** (`role=gen`, gen-init lifecycle only): `gen_arrival`,
`gen_init_scheduled`, `gen_kv_transfer_start`, `gen_kv_transfer_end`,
`gen_first_scheduled`, `gen_first_token`, `gen_last_token`.

**router** (`role=router`): `arrival`, `ctx_dispatch`, `gen_dispatch`,
`first_token`, `resp_done`.

**client** (`role=client`): the benchmark client writes **one compound
`source=client` record per request** (send / first-token / completion times +
`ttft` / `latency` / `output_tokens`), not per-event lines. It is a post-hoc
batch write, immune to server-side hangs, and preserves the vLLM `ttft` / `e2e`
/ `tpot` path. It has no shared request id or clock epoch with the server, so it
is surfaced standalone (never joined onto a worker record).

After the run:

```
/tmp/perf_events/time_events_rank0_pid12345.jsonl    # flat event lines, ctx/gen
/tmp/perf_events/time_events_rank1_pid12345.jsonl
/tmp/router_events/disagg_router_pid200.jsonl        # flat router event lines
/tmp/client_events/client_pid300.jsonl               # compound client records
/tmp/kv_csv/<instance>_<rank>.csv                     # KV-transfer task rows
/tmp/kv_csv/<instance>_<rank>_gen_transfer_summary.csv
```

## 2. Compiling (long → wide)

```bash
python -m tensorrt_llm.serve.scripts.perf_time_events \
    --event-dir  /tmp/perf_events \
    --router-dir /tmp/router_events \
    --client-dir /tmp/client_events \
    --kv-csv-dir /tmp/kv_csv \
    --events-jsonl /tmp/perf_events/combined_time_events.jsonl \
    -o   /tmp/perf_events/combined.json \
    -a   /tmp/perf_events/agg.jsonl
```

Arguments (all optional; env-derived defaults shown):

| Flag | Default | Purpose |
|---|---|---|
| `--event-dir` | `$TRTLLM_PERF_TIME_EVENTS_PATH` | Directory of per-rank `time_events_*.jsonl`. |
| `--router-dir` | `$TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH` | Directory of `disagg_router_*.jsonl` (joined by `ctx_request_id`). |
| `--client-dir` | `$TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH` | Directory of `client_*.jsonl` (standalone timeline). |
| `--kv-csv-dir` | `$TRTLLM_KVCACHE_TIME_OUTPUT_PATH` | Directory of KV-transfer CSVs to join in. |
| `--perf-json` | — | An additional `/perf_metrics` JSON dump (aggregated single-server runs); feeds the canonical-12 cross-worker spans + `--html`. |
| `--events-jsonl [PATH]` | `combined_time_events.jsonl` | **Primary output**: one line per request. |
| `-o`, `--output` | `perf_time_events.combined.json` | Combined JSON (records + leftovers + match stats). |
| `-a`, `--agg-jsonl [PATH]` | off | Latency aggregate (mean / P50 / P99 per span). |
| `--html [PATH]` | off | Interactive HTML timeline (reuses `time_breakdown`; needs `--perf-json`). |

### Primary output: `combined_time_events.jsonl`

One line per request. The compiler groups events by `request_id`, dedups the TP
ranks (first-seen wins — `parse_event_dir` sorts by filename so `rank0` is
canonical), joins ctx ⇄ gen ⇄ router on `ctx_request_id`, and pivots long → wide
into one record carrying every event timestamp that fired plus the derived
**intra-domain** spans:

```json
{"ctx_request_id": "abc", "gen_request_id": 42, "ctx_worker_request_id": 7,
 "gen_arrival": 200.0, "gen_kv_transfer_start": 200.3, ...,
 "spans": {"gen:kv_transfer_start->end": 0.6,
           "gen:first_token->last_token": 3.5,
           "router:arrival->ctx_dispatch": 0.2, ...}}
```

A request that hung emits its **partial** record — events that never fired are
simply absent (not zero), and a span whose endpoint is missing is omitted rather
than reported as a bogus `0.0`.

### Clock domains (why the spans are intra-domain only)

The steady clocks split into two domains that are NOT NTP-aligned to each other:
domain A = {router, gen worker}, domain B = {ctx worker, client}. Subtracting a
timestamp in one domain from one in the other is meaningless, so each combined
record carries only **within-domain** spans (router chain, ctx chain, gen
chain). The 12 canonical **cross-worker** spans (ctx→gen relay, etc.) are
recovered only from the offset-corrected `--perf-json` dump, where the disagg
server has aligned the clocks; the aggregate flags the two cross-domain disagg
spans with `clock_safe: false` rather than dropping them.

### `-a/--agg-jsonl` latency aggregate

24 rows of `mean / P50 / P99 / min / max / n` — the canonical-12 (from
`--perf-json`), 5 router spans, 4 worker-event spans, and 3 vLLM client metrics
(`ttft` / `e2e` / `tpot`, `tpot = (latency − ttft) / (output_tokens − 1)`). A
span with no data reports `status: not_recorded`.

`--html` is the only path that pulls in `plotly` (lazily, via the sibling
`time_breakdown` package); the parse / merge / JSON path is pure stdlib.

## Known limitations

- **Cross-worker spans need `--perf-json`.** The per-event capture is torch-free
  and clock-split; the offset-corrected ctx→gen relay spans come from the C++
  `/perf_metrics` dump, not from the event lines.
- **Ambiguous `ctx_request_id`.** The gen-only benchmark path hardcodes
  `ctx_request_id=1` for every request; the compiler treats a `ctx_request_id`
  shared by more than one request as **non-joinable** (surfaced under
  `unjoined_router_events`) rather than false-attaching one router row to every
  worker record.
- **Client is standalone.** It has no shared request id or clock epoch with the
  server, so client records are never joined onto request records.

See `docs/source/developer-guide/perf-analysis.md` and `kv-transfer.md` for the
broader perf-analysis workflow.
