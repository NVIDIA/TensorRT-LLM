# Tree-of-Thought Research (scaffolding example)

Research agent built on TensorRT-LLM scaffolding that explores multiple
reasoning branches (tree of thought) using the same MCP stack as IterResearch
and Open Deep Research: Tavily search, webpage fetch, and an optional Python
sandbox via `coder_mcp`.

Configuration (API keys, MCP host/ports, `base_url`, model name, turn/token
limits) uses the shared example config
`examples/scaffolding/contrib/open_deep_research/config.example.yaml`.
IterResearch, Tree-of-Thought Research, and Open Deep Research all share the
same layout, so there is no per-agent config file — copy the example to your
own `config.yaml` and fill in the keys.

## Prerequisites

1. Serve a compatible chat model (for example with `trtllm-serve`) so `base_url`
   points at a running OpenAI-compatible API.
2. If you need code execution, start `coder_mcp.py` (Apiary-backed), then the
   MCP servers.
3. Start the MCP tools you need, passing the **same** config to each server, for
   example `uv run … --config …/config.example.yaml`.
4. Fill in search/scraper keys in your `config.yaml`.

## Single prompt

```bash
python examples/scaffolding/contrib/tree_of_thought_research/run_tot_research.py \
  --config examples/scaffolding/contrib/open_deep_research/config.example.yaml \
  --enable_tracing
```

Use `--prompt "…"` to override the default question. Tree-shape knobs such as
`--max_depth`, `--num_thoughts_per_step`, `--branch_factor`, and
`--complete_score_threshold` control the search; CLI flags override values from
`--config` when both are set.

## Tracing (`--enable_tracing`)

`--enable_tracing` turns on execution trace capture for one run. Outputs go to
`--trace_output_dir` if provided, otherwise to a timestamped
`tot_research_trace_<YYYYMMDD_HHMMSS>` directory. The generated `.trace.json`
files can be consumed by the replay scripts under
`examples/scaffolding/trace_replay`.
