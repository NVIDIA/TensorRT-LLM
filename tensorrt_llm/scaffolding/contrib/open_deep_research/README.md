# Open Deep Research

Open Deep Research agent built on TensorRT-LLM Scaffolding. It uses the same MCP stack as IterResearch (Tavily, Google Scholar, webpage fetch, Python sandbox via `coder_mcp`), while preserving a Planner-Executor multi-agent design.

## Design Overview

**Open Deep Research** follows a Planner-Executor architecture:

- **Supervisor (Planner)**: Accepts user input, generates a research brief, delegates tasks to Researchers, and synthesizes the final report.
- **Researcher (Executor)**: Receives a topic, performs multiple rounds of tool-augmented research, then summarizes and compresses findings.

### Frontend (Control Flow)

Controllers orchestrate the workflow:

| Controller | Description |
|------------|-------------|
| `Supervisor` | Entry controller for the workflow |
| `BriefController` | Generates the research brief from user input |
| `ResearchPlanningController` | Plans and delegates topics to sub-agents |
| `Researcher` | Sub-agent that researches assigned topics |
| `ChatWithMCPController` | Handles MCP-based tool calling |
| `Compressor` | Compresses search outputs and reflections |
| `FinalReportController` | Synthesizes the final report |

### Backend (Workers)

Workers serve generation and tool requests:

| Worker | Description |
|--------|-------------|
| `TRTOpenaiWorker` | LLM generation via TensorRT-LLM OpenAI-compatible endpoint |
| `MCPWorker` | MCP tool calling |

### Modularity

Each controller/worker can evolve independently. For example:

- Swap in a different report-synthesis controller without changing the rest of the pipeline.
- Add another model endpoint by implementing a worker similar to `TRTOpenaiWorker`.

## Prerequisites

1. Serve a chat model at an OpenAI-compatible `base_url` (for example, `trtllm-serve`).
2. If code execution is needed, start `coder_mcp.py` (Apiary-backed), then MCP servers.
3. Start each MCP server with the same `config.yaml` (same layout as `examples/scaffolding/contrib/iter_research/config.yaml`).
4. Fill required API keys in `config.yaml` (Tavily, Jina/Scraper for fetch, Google Scholar credentials, and so on).

Example MCP startup commands:

```bash
cd examples/scaffolding/mcp/tavily_search && uv run tavily_search.py \
    --config examples/scaffolding/contrib/open_deep_research/config.yaml
```

```bash
cd examples/scaffolding/mcp/google_scholar && uv run google_scholar.py \
    --config examples/scaffolding/contrib/open_deep_research/config.yaml
```

```bash
cd examples/scaffolding/mcp/fetch_webpage && uv run fetch_webpage.py \
    --config examples/scaffolding/contrib/open_deep_research/config.yaml
```

```bash
python examples/scaffolding/mcp/coder/coder_mcp.py \
    --apiary-url http://172.17.0.1:8080 \
    --default-image python:3.12-slim \
    --port 8086
```

## Single Prompt Run

Use the runnable example entrypoint:

```bash
python examples/scaffolding/contrib/open_deep_research/run_open_deep_research.py \
  --config examples/scaffolding/contrib/open_deep_research/config.yaml \
  --base_url http://localhost:8000/v1 \
  --model Qwen3/Qwen3-30B-A3B \
  --enable_tracing
```

You can override `--prompt` for custom questions. Add `--enable_tracing` or `--enable_query_collector` as needed. When both CLI flags and config file values are set, CLI flags take precedence.

## Tracing (`--enable_tracing`)

`--enable_tracing` captures execution traces for a single run.

### Output directory behavior

- If `--trace_output_dir` is set, traces are written there.
- Otherwise, a timestamped directory is created: `open_deep_research_trace_<YYYYMMDD_HHMMSS>`.
- The chosen output directory is printed during execution.

### Trace files

- `open_deep_research.trace.json`: compact trace for replay/analysis.
- `open_deep_research.full.trace.json`: full trace with complete payloads.

### Example with explicit trace directory

```bash
python examples/scaffolding/contrib/open_deep_research/run_open_deep_research.py \
  --config examples/scaffolding/contrib/open_deep_research/config.yaml \
  --base_url http://localhost:8000/v1 \
  --model Qwen3/Qwen3-30B-A3B \
  --enable_tracing \
  --trace_output_dir ./open_deep_research_trace_manual
```

## Trace Replay

For replaying traces, use `run_deep_research_replay.py` in the example directory, or the general replay tooling under `examples/scaffolding/trace_replay`.

## Acknowledgments

This implementation follows the design of [Open Deep Research](https://github.com/langchain-ai/open_deep_research) by LangChain.
