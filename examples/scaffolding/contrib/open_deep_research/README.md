# Open Deep Research (scaffolding example)

Research agent built on TensorRT-LLM scaffolding with the same MCP stack as IterResearch (Tavily, Google Scholar, webpage fetch, optional Python sandbox via `coder_mcp`). Configuration and ports live in `config.yaml` (same layout as `examples/scaffolding/contrib/iter_research/config.yaml`). Batch runs use JSONL benchmarks under `search_datasets/` (`browsecomp`, `hle`).

## Prerequisites

1. Serve a chat model at an OpenAI-compatible `base_url` (for example `trtllm-serve`).
2. If you need code execution, start `coder_mcp.py` (Apiary-backed), then the MCP servers (see comments in `config.yaml`).
3. Start the MCP tools you need, passing the **same** `config.yaml` to each server, for example `uv run … --config …/config.yaml`.
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

4. Fill in API keys in `config.yaml` (Tavily, Jina/Scraper for fetch, Google keys for scholar, etc.).

## Single prompt

```bash
python examples/scaffolding/contrib/open_deep_research/run_open_deep_research.py \
  --config examples/scaffolding/contrib/open_deep_research/config.yaml \
  --base_url http://localhost:8000/v1 \
  --model Qwen3/Qwen3-30B-A3B \
  --enable_tracing
```

Override `--prompt` for your own question. Add `--enable_tracing` or `--enable_query_collector` as needed. CLI flags override values from `--config` when both are set.

## Tracing (`--enable_tracing`)

`--enable_tracing` enables execution trace capture for a single Open Deep Research run.

### Output directory behavior

- If `--trace_output_dir` is provided, outputs are written to that directory.
- Otherwise, the script creates a timestamped directory:
  `open_deep_research_trace_<YYYYMMDD_HHMMSS>`.
- The script prints the chosen trace output directory during execution.

### Files written when tracing is enabled

- `open_deep_research.trace.json`: compact trace for replay and analysis.
- `open_deep_research.full.trace.json`: full trace with complete event payloads.

### Example with explicit trace directory

```bash
python examples/scaffolding/contrib/open_deep_research/run_open_deep_research.py \
  --config examples/scaffolding/contrib/open_deep_research/config.yaml \
  --base_url http://localhost:8000/v1 \
  --model Qwen3/Qwen3-30B-A3B \
  --enable_tracing \
  --trace_output_dir ./open_deep_research_trace_manual
```

## Trace replay

For replaying recorded traces, see `run_deep_research_replay.py` in this directory,
or use the general scripts under `examples/scaffolding/trace_replay`.
