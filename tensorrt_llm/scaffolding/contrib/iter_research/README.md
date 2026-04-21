# IterResearch (TensorRT-LLM Scaffolding)

This package is a **reference integration** of **IterResearch** with the TensorRT-LLM **Scaffolding** stack: multi-turn tool use, a **Visit** sub-controller for web retrieval, and an OpenAI-compatible inference backend (e.g. `trtllm-serve`), all wired through **MCP over SSE**.

## Overview

IterResearch frames long-horizon research as **Markovian state reconstruction**: each turn rebuilds the prompt from only the **current report**, the **last tool call**, and the **last observation**, so many interactions fit in a bounded context. This codebase maps the paper’s tool surface to four MCP servers (SerpAPI, Jina/ScraperAPI, Apiary sandbox). `run_iter_research.py` starts `MCPWorker` and `TRTOpenaiWorker` and drives `IterResearchController`.

## Repository layout

```text
tensorrt_llm/scaffolding/contrib/iter_research/
├── __init__.py          # Exports controllers and ScaffoldingLlm factories
├── agent.py             # IterResearchController, VisitController, VisitTask
├── prompts.py           # System/user prompts and tool templates
├── utils.py             # Tool JSON, tag parsing, format checks
└── README.md            # This file

examples/scaffolding/contrib/iter_research/
├── config.yaml          # SerpAPI, Jina, MCP ports, Apiary gateway, runner settings
└── run_iter_research.py # Entry: load config, MCP SSE URLs, ScaffoldingLlm, question

examples/scaffolding/mcp/          # Four MCP servers (SSE) used by this flow
├── google_search/     # SerpAPI web search (ports from config.yaml)
├── google_scholar/    # SerpAPI Scholar + web search
├── fetch_webpage/     # Jina Reader / ScraperAPI for HTML/PDF-style fetch paths
└── coder/
    └── coder_mcp.py            # Apiary-backed MCP (read_file/shell/exec/python_interpreter/...)
```

### MCP tools under `examples/scaffolding/mcp`

| Directory | MCP tool name | Role |
|-----------|---------------|------|
| `google_search/` | `google_search` | Google web search via SerpAPI; returns formatted hits. |
| `google_scholar/` | `google_scholar` | Scholar plus web search in parallel; merged output. |
| `fetch_webpage/` | `fetch_webpage` | Fetches URL content (implementation chooses HTML/Markdown/PDF paths). |
| `coder/` | `python_interpreter` (among others) | Executes Python code inside a per-client Apiary sandbox session (shared with `read_file`, `shell`, `exec`, ...). |

The runner builds `http://<host>:<port>/sse` for each service from `mcp_tools.*` and `mcp_client_host` in `config.yaml`. `coder_mcp.py` talks to the Apiary daemon directly via `ApiarySessionMux` — no standalone Python gateway is needed.

## Docker-based startup

Assume **Apiary** and the **TensorRT-LLM** environment run in **separate** containers (or hosts). The TRT-LLM side must reach Apiary’s HTTP API—pass a host or `host.docker.internal` address that resolves from that container to `coder_mcp.py` via `--apiary-url`.

### Container 1: Apiary (sandbox daemon)

1. **Clone the repository**

   ```bash
   git clone https://github.com/Boreas618/apiary.git
   cd apiary
   ```


2. **Enter the Apiary compose service (publish port 8080)**

   ```bash
   docker compose -f docker-compose.dev.yml run --rm -p 8080:8080 apiary bash
   ```
   

3. **Inside the container: install and run Apiary**

   ```bash
   cargo install --path .
   apiary init
   apiary daemon --bind 0.0.0.0:8080
   ```
5. **Load a Python rootfs for Apiary sandboxes at runtime**

    Run the following script in the host:

   ```bash
   JOB_ID=$(curl -sS -X POST http://127.0.0.1:8080/api/v1/images \
    -H "Content-Type: application/json" \
    -d '{"images":["python:3.12-slim"]}' | jq -r '.job_id')

    # 2. Poll until the job is terminal
    while true; do
    STATE=$(curl -sS "http://127.0.0.1:8080/api/v1/images/jobs/${JOB_ID}" | jq -r '.state')
    echo "state=$STATE"
    [[ "$STATE" == "running" ]] || break
    sleep 2
    done
   ```


   Leave this process running. On the TRT-LLM side, `apiary_url` must point at this listener (host and port).

### Container 2: TensorRT-LLM (model, MCP, gateway, runner)

The following steps start **after** you are inside a TensorRT-LLM dev or inference image (build and shell steps omitted; see the [installation docs](https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/installation)). Commands assume the **repository root** as the working directory; edit `config.yaml` for API keys, model path, `base_url`, and networking.

1. **Start the OpenAI-compatible server (example) with terminal 1**

   ```bash
   trtllm-serve serve /path/to/your/model \
       --max_num_tokens 32768 \
       --kv_cache_free_gpu_memory_fraction 0.8
   ```

2. **Start the MCP servers (shared config) with terminals 2-5**

   For each server, `cd` into its directory first and then `uv run` the script. Pass the repository-root-relative `--config` path (adjust the relative prefix to wherever the repo root is from your shell, e.g. `../../../..`).

   ```bash
   # Terminal 2 — google_search
   cd examples/scaffolding/mcp/google_search
   uv run google_search.py \
       --config ../../../../examples/scaffolding/contrib/iter_research/config.yaml

   # Terminal 3 — google_scholar
   cd examples/scaffolding/mcp/google_scholar
   uv run google_scholar.py \
       --config ../../../../examples/scaffolding/contrib/iter_research/config.yaml

   # Terminal 4 — fetch_webpage
   cd examples/scaffolding/mcp/fetch_webpage
   uv run fetch_webpage.py \
       --config ../../../../examples/scaffolding/contrib/iter_research/config.yaml

   # Terminal 5 — coder_mcp (serves python_interpreter, need to set port from config.yaml manually)
    python examples/scaffolding/mcp/coder/coder_mcp.py \
        --apiary-url http://172.17.0.1:8080 \
        --default-image python:3.12-slim \
        --port 8086
   ```

   The `coder_mcp` server provides the `python_interpreter` tool (plus `read_file`, `shell`, `exec`, ...) backed directly by the Apiary daemon — no standalone Python gateway is required. Point `--apiary-url` at **Container 1**’s `apiary daemon`, and ensure the runner reaches this SSE endpoint at `mcp_tools.python_interpreter` (host/port in `config.yaml`).

3. **Run the IterResearch example with terminal 6**

   ```bash
   python examples/scaffolding/contrib/iter_research/run_iter_research.py \
       --config examples/scaffolding/contrib/iter_research/config.yaml --enable_statistics --enable_tracing
   ```

   Optional flags: `--enable_statistics`, `--enable_tracing`, `--question "..."`; see `run_iter_research.py --help`.

## Configuration (summary)

- **`config.yaml`**: Central place for SerpAPI/Jina keys, `mcp_tools` ports, and runner `base_url` / `model`. The `python_interpreter` tool is served by `coder_mcp.py` and talks to Apiary directly, so the former `apiary_python_gateway` / `sandbox_endpoint` settings are no longer used by this flow (Apiary URL is passed via `coder_mcp.py --apiary-url`). Do not commit real secrets to a public repository.
- **Networking**: If MCP processes and the runner are not all on `127.0.0.1`, set `mcp_client_host` and/or per-service `mcp_tools.*.host` so the runner can reach every `/sse` endpoint.

## Acknowledgement

This code is a **reproduction** of the paper **IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling**. 

Paper (arXiv): <https://arxiv.org/abs/2511.07327>
