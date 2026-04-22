# Trace Replay Quick Start

This directory contains scripts for replaying saved scaffolding traces and exporting replay metrics.

## Single-Trace Replay (via `trtllm-serve`)

Use `run_trace_replay.py` to replay one `.trace.json` file through `TRTOpenaiWorker` (OpenAI-compatible API)
and write a timestamped JSON report.

### Input Example

Use this trace folder as input:

- `examples/scaffolding/trace_replay/trace_example/django__django-16801`

### Command Example

From repository root:

```bash
python examples/scaffolding/trace_replay/run_trace_replay.py \
  examples/scaffolding/trace_replay/trace_example/django__django-16801/django__django-16801.trace.json \
  --model "Qwen3-30B-A3B" \
  --openai-base-url "http://127.0.0.1:8000/v1" 
```

python examples/scaffolding/trace_replay/run_trace_replay.py \
  iter_research_trace_20260422_125644/iter_research.trace.json \
  --model "Qwen3-30B-A3B" \
  --openai-base-url "http://127.0.0.1:8000/v1" 

/home/scratch.kleinc_gpu/tekit/open_deep_research_trace_20260422_142252/open_deep_research.trace.json

python examples/scaffolding/trace_replay/run_trace_replay.py \
  open_deep_research_trace_20260422_142252/open_deep_research.trace.json\
  --model "Qwen3-30B-A3B" \
  --openai-base-url "http://127.0.0.1:8000/v1" 

By default, output JSON is written next to the input trace with a timestamped name:
`<trace_base>_replay_<model>_<YYYYMMDD_HHMMSS>.json`.
The script prints the final JSON path after completion.

## What Gets Recorded in the Output JSON

`run_trace_replay.py` writes a structured report that includes:

- Run metadata: schema version, UTC start/end time, full CLI argv.
- Host metadata: hostname, cwd, python version, pid.
- Replay endpoint config: OpenAI base URL, model name, tensor parallel size.
- Trace metadata: trace id, event count, file stats (name/size/mtime), token summaries, role/event/tool counts.
- Replay metrics: wall time, per-session durations, token throughput metrics, per-user and per-GPU throughput, and generation details for assistant turns.

## Prerequisites

- A running `trtllm-serve` endpoint with OpenAI-compatible API.
- The `--model` name must match a model exposed by that endpoint.
- If required by your endpoint, set an API key with `--openai-api-key` or `OPENAI_API_KEY`.
