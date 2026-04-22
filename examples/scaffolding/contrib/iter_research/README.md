# IterResearch (scaffolding example)

Multi-tool research agent using TensorRT-LLM scaffolding: Tavily search, Google Scholar, webpage fetch, and an optional Python sandbox via `coder_mcp`. Configuration lives in `config.yaml` (API keys, MCP host/ports, `base_url`, model name, turn/token limits).

```bash
cd examples/scaffolding/mcp/tavily_search && uv run tavily_search.py \
    --config examples/scaffolding/contrib/iter_research/config.yaml
```
```bash
cd examples/scaffolding/mcp/google_scholar && uv run google_scholar.py \
    --config examples/scaffolding/contrib/iter_research/config.yaml
```
```bash
cd examples/scaffolding/mcp/fetch_webpage && uv run fetch_webpage.py \
    --config examples/scaffolding/contrib/iter_research/config.yaml
```
```bash
python examples/scaffolding/mcp/coder/coder_mcp.py \
    --apiary-url http://172.17.0.1:8080 \
    --default-image python:3.12-slim \
    --port 8086
```

## Prerequisites

1. Serve a compatible chat model (for example with `trtllm-serve`) so `base_url` in `config.yaml` points at a running OpenAI-compatible API.
2. Start the Python sandbox MCP if you need code execution: run `coder_mcp.py` (Apiary-backed), then the MCP servers—see comments in `config.yaml` for host/port wiring.
3. Start the MCP tools you need (same `config.yaml` passed to each server, for example `uv run … --config …/config.yaml`).
4. Fill in search/scraper keys in `config.yaml` where required.

## Single question

```bash
python examples/scaffolding/contrib/iter_research/run_iter_research.py \
  --config examples/scaffolding/contrib/iter_research/config.yaml \
  --enable_tracing
```

Use `--question "…"` to override the default prompt. Outputs and traces depend on the flags you pass.

## Tracing (`--enable_tracing`)

`--enable_tracing` turns on execution trace capture for one IterResearch run.

### Output directory behavior

- If `--trace_output_dir` is set, traces are written there.
- Otherwise, the script creates a timestamped folder: `iter_research_trace_<YYYYMMDD_HHMMSS>`.
- The script prints the final trace output directory at runtime.

### Files written when tracing is enabled

- `iter_research.trace.json`: compact trace for replay-oriented workflows.
- `iter_research.full.trace.json`: full trace with richer event payloads.

### Optional explicit output directory

```bash
python examples/scaffolding/contrib/iter_research/run_iter_research.py \
  --config examples/scaffolding/contrib/iter_research/config.yaml \
  --enable_tracing \
  --trace_output_dir ./iter_research_trace_manual
```

The generated `.trace.json` files can be consumed by replay scripts under
`examples/scaffolding/trace_replay`.
