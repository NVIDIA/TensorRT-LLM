# Trace Replay via Scaffolding

A practical write-up of the `trace & replay` workflow in TensorRT-LLM
scaffolding. Read it once and walk away with a mental model of the full
pipeline.

## Why Trace Replay Exists

Scaffolding agents such as Coder, IterResearch, and Open Deep Research are
not single-shot generations. A typical run includes:

- multi-turn conversation state,
- MCP tool calls,
- parallel branches,
- and token-heavy context evolution over time.

When performance shifts, the final answer alone does not tell you enough.
You usually need to know:

- What exact execution structure did this request follow?
- How much of wall time came from model generation vs tool waiting?
- How do throughput and latency change across serving setups?

Trace replay addresses this by converting one real agent execution into a
structured event stream, then replaying the same structure against another
endpoint/model configuration with comparable metrics.

## Directory Layout

```
examples/scaffolding/trace_replay/
├── README.md                  -- this file
├── run_trace_replay.py        -- single-trace single-config replay driver
├── metrics.py                 -- replay-result metrics (used by run_trace_replay)
├── trace_example/             -- one ready-to-run example trace
│   └── django__django-16801/
│       ├── django__django-16801.trace.json        (compact)
│       └── django__django-16801.full.trace.json   (full)
├── analysis/                  -- offline KV-cache hit-rate upper-bound analyzer
│   ├── compute_cache_hit_trace.py   CLI: optimal upper-bound hit rate per trace
│   ├── cache_hit.py / aggregation.py / annotate.py / blocks.py /
│   │   branch_summary.py / streams.py / io.py / __init__.py
│   └── README.md
├── pareto/                    -- multi-config Pareto sweep over trtllm-serve
│   ├── trace_replay_client.py             one ladder point: external server client
│   ├── trace_replay_pareto_aggregate.py   N step JSONs -> combined v4 record + PNGs
│   ├── _common.py                         shared helpers
│   └── README.md
└── plots/                     -- plot helpers (loaded dynamically by pareto/)
    ├── plot_trace_replay_token_pareto.py            throughput Pareto PNG
    ├── plot_trace_replay_agent_pareto.py            agent-concurrency Pareto PNG
    ├── plot_trace_replay_job_pareto.py              job-throughput PNG
    ├── plot_trace_replay_session_hit_pareto.py      per-session KV-hit curve
    ├── plot_trace_replay_session_hit_vs_time.py     KV-hit vs session start time
    └── plot_trace_replay_per_call_hit_curves.py     per-LLM-call hit trajectory
```

Four workflows are covered:

| Goal | Entry point |
|---|---|
| Replay one trace once against one serving config | `run_trace_replay.py` |
| Sweep throughput/latency Pareto over a `(B, N, C)` ladder | `pareto/trace_replay_client.py` + `pareto/trace_replay_pareto_aggregate.py` |
| Offline upper bound on KV-cache hit rate (no GPU) | `analysis/compute_cache_hit_trace.py` |

## The Core Design in One Picture

The implementation follows a four-stage pipeline:

1. **Trace capture** during controller execution (`ExecutionTracer`).
2. **Trace artifact export** as compact and full JSON files.
3. **Replay execution** with `ReplayEngine` and `TRTOpenaiWorker`.
4. **Metrics aggregation** into a single JSON report.

Key design choices:

- **Structure-first reproducibility**: preserve event order, branch
  topology, and token budgets.
- **Tool replay decoupling**: replay tool calls as timed waits
  (`duration_ms`) instead of re-executing external tools.
- **Parallel semantics preserved**: branch routing via `branch_path` plus
  `parallel_start`/`parallel_end`.
- **Two trace flavors**: compact for replay/benchmarking, full for
  debugging and forensic analysis.

## Trace: Data Model and Capture Mechanics

### The trace schema

The canonical schema lives in `tensorrt_llm/scaffolding/execution_trace.py`:

- `TraceEvent`
- `ExecutionTrace`

Important event types:

- `message`
- `tool_call`
- `parallel_start`
- `parallel_end`
- `drop_kv_cache`

Critical fields:

- `branch_path` identifies the parallel branch.
- `conversation_id` and `message_index` preserve conversation ordering.
- assistant message events carry `prompt_tokens`, `completion_tokens`,
  `reasoning_tokens`, and `finish_reason`.

### Where tracing is attached

Tracing is wired in controller factories through decorators:

- `with_execution_tracing(...)` to attach `ExecutionTracer`
- `tokenize_trace_scope()` to fill token counts for tokenizable
  non-assistant messages (used in some scaffolds)
- `scaffolding_llm.enable_output_task_collection()` to return tracer
  output via results

Representative integration points:

- `tensorrt_llm/scaffolding/contrib/iter_research/agent.py`
- `tensorrt_llm/scaffolding/contrib/open_deep_research/supervisor.py`
- `tensorrt_llm/scaffolding/contrib/Coder/coder.py`

### How `ExecutionTracer` records events

The core implementation is in `tensorrt_llm/scaffolding/task_collection.py`
(`ExecutionTracer`).

It works in phases:

- **`before_yield`**: records pre-existing context messages
  (system/user/tool) for each `ChatTask`.
- **`after_yield`**: records produced assistant events plus generation
  metadata (tokens, duration, tool calls).
- **parallel handling**: emits `parallel_start`/`parallel_end` and rewrites
  child event paths onto sub-branches.

Token accounting support:

- `tokenize_trace_scope()` creates `TokenizeTask` items and uses the
  generation worker's `/tokenize` endpoint.
- `_correct_system_tokens(...)` then adjusts first system-token counts
  using assistant prompt-token evidence (to account for system-side
  injected context such as tool definitions).

### Compact vs full trace files

`ExecutionTrace.save(path, full=False)` exports:

- **compact trace** (`*.trace.json`): replay-focused structure + token
  metadata, excluding verbose payload.
- **full trace** (`*.full.trace.json`): includes message content, LLM
  request snapshots, tool arguments/results, stdio, and turn-level timing.

Load path for replay is `ExecutionTrace.load(...)`.

### Trace export in example runners

The pattern is consistent across example runners:

1. `tracer = result.task_collections.get("execution_tracer")`
2. `trace = tracer.export_trace()`
3. export both compact and full files

You can see this in:

- `examples/scaffolding/contrib/Coder/run_coder.py`
- `examples/scaffolding/contrib/iter_research/run_iter_research.py`
- `examples/scaffolding/contrib/open_deep_research/run_open_deep_research.py`

## Replay: What Gets Reconstructed and How

### Replay entrypoints

- `run_trace_replay.py`: one trace, one config, one report — the simplest
  thing that works. Use this to sanity-check a trace, or to compare two
  serving configs on a fixed workload.
- `pareto/trace_replay_client.py`: one ladder point of a multi-config
  sweep. Replays the same trace at `N` total sessions with `C` in flight,
  designed to be called repeatedly by an outer driver as `B`/`N`/`C` step
  through a ladder. Writes a single step JSON per invocation.

Both share the same core (`ReplayEngine` + `TRTOpenaiWorker`); they differ
in how they wrap the run (single launch vs `asyncio.Semaphore`-gated
multi-session burst) and what they aggregate.

Flow for `run_trace_replay.py`:

1. parse CLI args,
2. load compact trace via `ExecutionTrace.load(...)`,
3. build `openai.AsyncOpenAI` + `TRTOpenaiWorker`,
4. replay with `ReplayEngine(...).launch_trace(...)`,
5. compute and write metrics report.

### Queue-based branch routing

Replay core is in `tensorrt_llm/scaffolding/replay.py`:

- `ReplayEngine`
- `QueueManager`
- `QueueExecutor`

Design pattern:

- one queue/executor per branch path,
- `parallel_start` allocates child queues for `parent_path + (i,)` after
  draining the parent queue (so the fork happens after the parent's
  fork-producing generation is fully committed, not while it is still in
  flight),
- `parallel_end` closes/waits/cleans child queues,
- non-parallel events are routed by their `branch_path`.

This preserves:

- strict in-branch ordering,
- and branch-level concurrency semantics.

### Event replay semantics

`QueueExecutor` handles each event category differently:

- **`tool_call`**: simulated via `asyncio.sleep(duration_ms / 1000)` (no
  real external tool call).
- **`message` with role `system`/`user`/`tool`**: synthetic token IDs
  generated from recorded token counts and appended to conversation
  context. System messages are cached by `event.system_prompt_id` (or
  `conv:<conv_id>` for untagged templates) so multiple conversations using
  the same template share a token-id prefix — mirroring a real prefix
  cache hit.
- **`message` with role `assistant`**:
  - build `GenerationTask.input_tokens` from accumulated conversation
    segments,
  - set `max_tokens` to recorded `completion_tokens`,
  - run real generation through worker,
  - strip leading reasoning tokens (`reasoning_tokens`) before writing
    content tokens back into context,
  - record per-call client-side wall-clock start/end for downstream
    steady-state-window analytics.

### Worker behavior during replay

The default replay worker is `TRTOpenaiWorker` in
`tensorrt_llm/scaffolding/worker.py`.

In replay, assistant generation is driven by `GenerationTask` through
OpenAI-compatible APIs, with `ignore_eos=True`, using trace-derived token
budgets as the envelope for generation.

## Metrics: What You Get After Replay

### Single-trace report (`metrics.py`)

`run_trace_replay.py` writes the simple report. Output includes:

- trace structure summaries (event/role/tool counts),
- tool-time totals from trace metadata,
- run-level timing and duration stats,
- throughput metrics (`output_tps_aggregate`, `output_tps_per_gpu`,
  per-user TPS),
- assistant-level token details (trace completion budget vs replay output
  lengths).

Output JSON also includes run schema, timestamps, host/runtime metadata,
CLI argv, replay endpoint settings, and trace file metadata.

### Multi-config Pareto record (`pareto/`)

`trace_replay_client.py` emits a per-ladder-point **step JSON**
(`schema = trace_replay_pareto_frontier.step.v4`).
`trace_replay_pareto_aggregate.py` concatenates `runs[]` from N step JSONs
into a combined `trace_replay_pareto_frontier.v4` record and delegates to
`../plots/` to write the Pareto PNGs.

The per-run row carries:

- the three load knobs (`max_batch_size`, `total_sessions`, `concurrency`)
- whole-burst throughput and TPOT/intvty (`output_tps_aggregate_full_burst`,
  `full_burst_*_tpot_ms`, `full_burst_*_intvty`)
- **refill-sustained "steady-state" window**: starts at the C+1 admission
  (first refill), ends at the first completion after the last admission;
  only LLM calls fully contained in the window count. Fields:
  `steady_state_window` (audit dict), `output_tps_aggregate_steady_state`,
  `steady_state_*_tpot_ms`, `steady_state_*_intvty`,
  `total_output_tokens_steady_state`. Headline unsuffixed fields
  (`output_tps_aggregate`, `median_tpot_ms`, ...) point at steady-state
  when valid (`N > C`); for saturation sweeps (`N <= C`) consult the
  `*_full_burst` mirrors.
- per-session admission/end offsets, per-LLM-call timing offsets,
  `steady_state_included` flag on every detail entry.
- offline KV-cache hit-rate upper bound (`optimal_cache_hit`) loaded from
  `<trace_dir>/*.cachehit.json` if present — stamped by the aggregator
  onto each run row. Engine-measured per-session rates are derived on
  demand at plot time from `replay_assistant_generations_detail`
  (`num_reused_blocks` / `num_missed_blocks` per assistant LLM call), so
  no measured-rate aggregate is cached on the row.

### KV-cache hit-rate analysis (`analysis/`)

`compute_cache_hit_trace.py` is a pure offline simulator: it assumes an
infinite, non-evicting cache and walks the trace in `QueueExecutor` order,
scoring each assistant prompt against a token-level radix tree of
synthetic IDs. Output:

- `<name>.cachehit.json` (summary + per-request stats + rollups by branch /
  depth / system-prompt UUID) with all hit-rate fields prefixed
  ``optimal_`` so the offline upper-bound nature is unambiguous
  (`optimal_overall_cache_block_hit_rate`,
  `optimal_cache_block_hit_rate` per request, …).
- `<name>.trace.cachehit.json` (annotated trace with the same `optimal_*`
  fields stitched onto each scored assistant event).

Defaults mirror real TRT-LLM behavior (`tokens_per_block=32`,
`--decode-kv-reuse`, `--cot-pollutes-cache`). See `analysis/README.md`
for the full flag/schema reference.

Pair `optimal_cache_hit` (offline UB) with the on-demand engine-measured
per-session rate at plot time to read how much of the ideal block-reuse
the runtime actually captures at any `(B, N, C)` point.

### Plots (`plots/`)

Loaded dynamically by `pareto/trace_replay_pareto_aggregate.py` after the
combined record is written. Each `write_*_png_from_json_file(...)` accepts
the v4 JSON path and writes a sibling PNG. Available helpers:

| Module | Output | What it shows |
|---|---|---|
| `plot_trace_replay_token_pareto` | `<stem>_throughput_pareto.png` | token throughput vs median intvty |
| `plot_trace_replay_agent_pareto` | `<stem>_agent_pareto.png` | agent concurrency vs intvty |
| `plot_trace_replay_job_pareto` | `<stem>_job_pareto.png` | agent-sessions/hour vs intvty |
| `plot_trace_replay_session_hit_pareto` | `<stem>_session_hit_pareto.png` | per-session KV-cache block hit-rate curve |
| `plot_trace_replay_session_hit_vs_time` | `<stem>_session_hit_vs_time.png` | session KV-hit vs session start offset |
| `plot_trace_replay_per_call_hit_curves` | `<stem>_per_call_hit_curves.png` | per-LLM-call hit trajectory across sessions |

## Usage: End-to-End Workflow

### 1) Generate traces from a scaffold run

Run a scaffold with `--enable_tracing`; it emits both compact and full
traces.

```bash
python examples/scaffolding/contrib/iter_research/run_iter_research.py \
  --config examples/scaffolding/contrib/iter_research/config.yaml \
  --enable_tracing
```

```bash
python examples/scaffolding/contrib/open_deep_research/run_open_deep_research.py \
  --config examples/scaffolding/contrib/open_deep_research/config.yaml \
  --enable_tracing
```

```bash
python examples/scaffolding/contrib/Coder/run_coder.py \
    --base_url http://localhost:8000/v1 \
    --model Qwen3/Qwen3-30B-A3B \
    --apiary_url http://172.17.0.1:8080 \
    --mcp_url http://127.0.0.1:8083/sse \
    --image ubuntu:22.04 \
    --prompt "Implement a thread-safe LRU cache in Python" \
    --max_iterations 50 \
    --enable_tracing
```

### 2) Replay the compact trace (single-config)

```bash
python examples/scaffolding/trace_replay/run_trace_replay.py \
  examples/scaffolding/trace_replay/trace_example/django__django-16801/django__django-16801.trace.json \
  --model "Qwen3-30B-A3B" \
  --openai-base-url "http://127.0.0.1:8000/v1"
```

You can replay your own generated traces the same way:

```bash
python examples/scaffolding/trace_replay/run_trace_replay.py \
  iter_research_trace_20260422_125644/iter_research.trace.json \
  --model "Qwen3-30B-A3B" \
  --openai-base-url "http://127.0.0.1:8000/v1"
```

Useful options:

- `--openai-api-key` (or `OPENAI_API_KEY`)
- `--tensor-parallel-size` (used for per-GPU normalization)
- `--output-json` (custom report path)

Default output filename:

- `<trace_base>_<model>_replay_statistics_<YYYYMMDD_HHMMSS>.json`

### 3) Sweep a `(B, N, C)` Pareto ladder

Start `trtllm-serve` outside the client (the driver script handles
lifecycle), then run one ladder point:

```bash
python examples/scaffolding/trace_replay/pareto/trace_replay_client.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen3-235B-A22B \
  --trace_dir .../traces/swebench/django__django-14787 \
  --total_sessions 32 --concurrency 16 --max_batch_size 16 \
  --ladder_index 1 --ladder_step 16 \
  --tensor_parallel_size 4 --moe_expert_parallel_size 4 \
  --output_json .../step16.json
```

After running every ladder step, aggregate:

```bash
python examples/scaffolding/trace_replay/pareto/trace_replay_pareto_aggregate.py \
  --step_jsons out/step8.json out/step16.json out/step32.json \
  --trace_dir .../traces/swebench/django__django-14787 \
  --output_json out/django__django-14787_Qwen3-235B-A22B_tp4_ep4.json
```

The combined JSON and the Pareto PNGs land in the same directory as
`--output_json`. The reference Slurm driver that orchestrates server
lifecycle + the full ladder is `exps/drivers/run_trace_pareto_server.sh`.

See `pareto/README.md` for the full client/aggregator reference.

### 4) Compute KV-cache hit rates

Offline ideal upper bound from a trace (no GPU needed):

```bash
# Single trace dir
python examples/scaffolding/trace_replay/analysis/compute_cache_hit_trace.py \
  path/to/<task>/

# Whole dataset
python examples/scaffolding/trace_replay/analysis/compute_cache_hit_trace.py \
  path/to/dataset/

# Single trace file
python examples/scaffolding/trace_replay/analysis/compute_cache_hit_trace.py \
  path/to/some.trace.json
```

Engine-measured per-session hit rates are computed directly from each
step JSON's `replay_assistant_generations_detail` (`num_reused_blocks` /
`num_missed_blocks` per call) at plot time — no extra CLI step is
required.

See `analysis/README.md` for flags and output schema.

## Practical Boundaries and Gotchas

- Replay is **structure-faithful**, not text-faithful; generated wording
  may differ from original trace output.
- Tool calls are not re-run; they are timing-simulated from historical
  `duration_ms`.
- Assistant replay requires `completion_tokens > 0`; otherwise replay
  will fail for that turn.
- Parallel trace topology must be valid (properly paired `parallel_start`
  / `parallel_end`).
- Full traces can be very large; use compact traces for routine replay
  benchmarking.
- In Pareto sweeps the steady-state window is only valid when
  `total_sessions > concurrency`. For saturation sweeps (`N == C`) the
  unsuffixed headline fields (`output_tps_aggregate`, `median_tpot_ms`,
  ...) are `None`; use the `*_full_burst` mirrors.
- The two cache-hit numbers serve different purposes:
  `optimal_cache_hit` is an offline upper bound; the engine-measured
  per-session block hit rate (derived on demand at plot time from
  `num_reused_blocks` / `num_missed_blocks`) is the actual achieved
  rate. Their gap reflects scheduling / eviction losses, not a bug.

---

If you want to scale this into multi-session ladder-style replay, the
current building blocks (`ReplayEngine` + `compute_replay_run_metrics(...)`
+ the `pareto/` driver pair) keep metric definitions consistent across
single-config and ladder-style runs.
