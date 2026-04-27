# Trace Replay via Scaffolding

This document is a practical, write-up of the `trace & replay` workflow in TensorRT-LLM scaffolding.

The goal is to read this once and walk away with a mental model of the full pipeline.

## Why Trace Replay Exists

Scaffolding agents such as Coder, IterResearch, and Open Deep Research are not single-shot generations. A typical run includes:

- multi-turn conversation state,
- MCP tool calls,
- parallel branches,
- and token-heavy context evolution over time.

When performance shifts, the final answer alone does not tell you enough. You usually need to know:

- What exact execution structure did this request follow?
- How much of wall time came from model generation vs tool waiting?
- How do throughput and latency change across serving setups?

Trace replay addresses this by converting one real agent execution into a structured event stream, then replaying the same structure against another endpoint/model configuration with comparable metrics.

## The Core Design in One Picture

The implementation follows a four-stage pipeline:

1. **Trace capture** during controller execution (`ExecutionTracer`).
2. **Trace artifact export** as compact and full JSON files.
3. **Replay execution** with `ReplayEngine` and `TRTOpenaiWorker`.
4. **Metrics aggregation** into a single JSON report.

Key design choices:

- **Structure-first reproducibility**: preserve event order, branch topology, and token budgets.
- **Tool replay decoupling**: replay tool calls as timed waits (`duration_ms`) instead of re-executing external tools.
- **Parallel semantics preserved**: branch routing via `branch_path` plus `parallel_start`/`parallel_end`.
- **Two trace flavors**: compact for replay/benchmarking, full for debugging and forensic analysis.

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
- assistant message events carry `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, and `finish_reason`.

### Where tracing is attached

Tracing is wired in controller factories through decorators:

- `with_execution_tracing(...)` to attach `ExecutionTracer`
- `tokenize_trace_scope()` to fill token counts for tokenizable non-assistant messages (used in some scaffolds)
- `scaffolding_llm.enable_output_task_collection()` to return tracer output via results

Representative integration points:

- `tensorrt_llm/scaffolding/contrib/iter_research/agent.py`
- `tensorrt_llm/scaffolding/contrib/open_deep_research/supervisor.py`
- `tensorrt_llm/scaffolding/contrib/Coder/coder.py`

### How `ExecutionTracer` records events

The core implementation is in `tensorrt_llm/scaffolding/task_collection.py` (`ExecutionTracer`).

It works in phases:

- **`before_yield`**: records pre-existing context messages (system/user/tool) for each `ChatTask`.
- **`after_yield`**: records produced assistant events plus generation metadata (tokens, duration, tool calls).
- **parallel handling**: emits `parallel_start`/`parallel_end` and rewrites child event paths onto sub-branches.

Token accounting support:

- `tokenize_trace_scope()` creates `TokenizeTask` items and uses the generation worker’s `/tokenize` endpoint.
- `_correct_system_tokenss(...)` then adjusts first system-token counts using assistant prompt-token evidence (to account for system-side injected context such as tool definitions).

### Compact vs full trace files

`ExecutionTrace.save(path, full=False)` exports:

- **compact trace** (`*.trace.json`): replay-focused structure + token metadata, excluding verbose payload.
- **full trace** (`*.full.trace.json`): includes message content, LLM request snapshots, tool arguments/results, stdio, and turn-level timing.

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

### Replay entrypoint

Replay is launched by:

- `examples/scaffolding/trace_replay/run_trace_replay.py`

Flow:

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
- `parallel_start` allocates child queues for `parent_path + (i,)`,
- `parallel_end` closes/waits/cleans child queues,
- non-parallel events are routed by their `branch_path`.

This preserves:

- strict in-branch ordering,
- and branch-level concurrency semantics.

### Event replay semantics

`QueueExecutor` handles each event category differently:

- **`tool_call`**: simulated via `asyncio.sleep(duration_ms / 1000)` (no real external tool call).
- **`message` with role `system`/`user`/`tool`**: synthetic token IDs generated from recorded token counts and appended to conversation context.
- **`message` with role `assistant`**:
  - build `GenerationTask.input_tokens` from accumulated conversation segments,
  - set `max_tokens` to recorded `completion_tokens`,
  - run real generation through worker,
  - strip leading reasoning tokens (`reasoning_tokens`) before writing content tokens back into context.

### Worker behavior during replay

The default replay worker is `TRTOpenaiWorker` in `tensorrt_llm/scaffolding/worker.py`.

In replay, assistant generation is driven by `GenerationTask` through OpenAI-compatible APIs, with `ignore_eos=True`, using trace-derived token budgets as the envelope for generation.

## Metrics: What You Get After Replay

Metrics logic is implemented in `examples/scaffolding/trace_replay/metrics.py`.

Output includes:

- trace structure summaries (event/role/tool counts),
- tool-time totals from trace metadata,
- run-level timing and duration stats,
- throughput metrics (`output_tps_aggregate`, `output_tps_per_gpu`, per-user TPS),
- assistant-level token details (trace completion budget vs replay output lengths).

Output JSON from `run_trace_replay.py` also includes:

- run schema and timestamps,
- host/runtime metadata,
- CLI argv and replay endpoint settings,
- trace file metadata (name, size, mtime).

## Usage: End-to-End Workflow

### 1) Generate traces from a scaffold run

Run a scaffold with `--enable_tracing`; it emits both compact and full traces.

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

### 2) Replay the compact trace

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

## Practical Boundaries and Gotchas

- Replay is **structure-faithful**, not text-faithful; generated wording may differ from original trace output.
- Tool calls are not re-run; they are timing-simulated from historical `duration_ms`.
- Assistant replay requires `completion_tokens > 0`; otherwise replay will fail for that turn.
- Parallel trace topology must be valid (properly paired `parallel_start`/`parallel_end`).
- Full traces can be very large; use compact traces for routine replay benchmarking.

---

If you want to scale this into multi-session ladder-style replay, the current building blocks (`ReplayEngine` + `compute_replay_run_metrics(...)`) are already a strong foundation and keep metric definitions consistent.
