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
python examples/scaffolding/contrib/open_deep_research/run_deep_research.py \
  --config examples/scaffolding/contrib/open_deep_research/config.yaml \
  --base_url http://localhost:8000/v1 \
  --model Qwen3/Qwen3-30B-A3B \
  --enable_statistics
```

Override `--prompt` for your own question. Add `--enable_tracing` or `--enable_query_collector` as needed. CLI flags override values from `--config` when both are set.

## Benchmark batch run

Writes under `./open_deep_research_output` by default (see `run_open_deep_research_dataset.py` for the exact folder naming and artifacts).

```bash
python examples/scaffolding/contrib/open_deep_research/run_open_deep_research_dataset.py \
  --config examples/scaffolding/contrib/open_deep_research/config.yaml \
  --model Qwen3-235B-A22B \
  --dataset hle \
  --batch_id 1 \
  --batch_size 100 \
  --base_url http://localhost:8000/v1 \
  --enable_statistics

```

## Evaluation (LLM-as-judge)

Finds the latest matching run for the given dataset/batch keys (or pass `--preds_path` to a specific `preds.json`). Produces `eval_summary_<run_id>.json` next to `preds.json`. The judge uses `--base_url` and `--judge_model` (not the agent `--model` from the run).

```bash
python examples/scaffolding/contrib/open_deep_research/evaluate_open_deep_research_dataset.py \
  --dataset hle \
  --batch_id 1 \
  --batch_size 100 \
  --base_url http://localhost:8000/v1 \
  --judge_model Qwen3/Qwen3-235B-A22B

```

## Trace replay

For replaying recorded traces, see `run_deep_research_replay.py` in this directory.
