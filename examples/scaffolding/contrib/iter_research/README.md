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
       --config examples/scaffolding/contrib/iter_research/config.yaml --enable_statistics --enable_tracing
```

Use `--question "…"` to override the default prompt. Outputs and traces depend on the flags you pass.

## Benchmark runs and evaluation

Runs write under an `iter_research_output`-style directory with `preds.json` and per-task trajectories. Evaluation uses an LLM judge (`--judge_model`); point `--base_url` at whichever API serves that judge.

```bash
python examples/scaffolding/contrib/iter_research/run_iter_research_dataset.py \
  --config examples/scaffolding/contrib/iter_research/config.yaml \
  --model Qwen3-235B-A22B \
  --dataset hle \
  --batch_id 1 \
  --batch_size 100 \
  --base_url http://localhost:8000/v1 \
  --enable_statistics
python examples/scaffolding/contrib/iter_research/evaluate_iter_research_dataset.py \
  --dataset hle \
  --batch_size 100 \
  --batch_id 1 \
  --base_url http://localhost:8000/v1 \
  --judge_model Qwen3/Qwen3-235B-A22B

```
