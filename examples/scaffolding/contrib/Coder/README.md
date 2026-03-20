# Scaffolding Coder Agent with Apiary Integration

Agentic coding system built on the TensorRT-LLM scaffolding framework.
The Coder agent uses an LLM for reasoning and planning, and executes code
modifications inside isolated Apiary sandboxes via an MCP (Model Context
Protocol) server.

## Architecture

```
┌─────────────┐    OpenAI API     ┌──────────────────┐
│  LLM Server │◄─────────────────►│  ScaffoldingLlm  │
│ (trtllm-serve│                  │                  │
│  or OpenAI)  │                  │  ┌────────────┐  │
└─────────────┘                   │  │   Coder    │  │
                                  │  │ Controller │  │
                                  │  └─────┬──────┘  │
                                  │        │         │
                                  │  ApiaryMCPWorker │
                                  └────────┼─────────┘
                                           │ SSE (per-scope)
                                  ┌────────▼─────────┐
                                  │   coder_mcp      │
                                  │   MCP Server      │
                                  │  (Starlette+SSE) │
                                  └────────┬─────────┘
                                           │ REST API
                                  ┌────────▼─────────┐
                                  │  Apiary Daemon   │
                                  │  (sandbox mgmt)  │
                                  └──────────────────┘
```

**Three services** must be running:

| Service | Role | Default Address |
|---------|------|-----------------|
| **Apiary daemon** | Manages sandboxed Linux environments | `http://127.0.0.1:8080` |
| **coder_mcp server** | Translates MCP tool calls into sandbox commands | `http://0.0.0.0:8083/sse` |
| **LLM server** | Serves the model via OpenAI-compatible API | `http://localhost:8000/v1` |

### Key Concepts

- **ExecutionScope**: Each request (and each parallel branch) gets a unique
  scope ID.  `ApiaryMCPWorker` opens a separate SSE connection per scope,
  giving every agent instance its own sandbox.
- **MCP Tools**: `read_file`, `list_dir`, `grep_files`, `apply_patch`, `shell`,
  `exec`, `update_plan`, `think`, `complete_task`.
- **Codex-style patches**: The `apply_patch` tool accepts structured patches
  (`*** Add File`, `*** Delete File`, `*** Update File` with `@@` hunks).

## Prerequisites

```bash
# 1. TensorRT-LLM (the repo you're in)
pip install -e .

# 2. coder_mcp server
pip install -e examples/scaffolding/contrib/Coder/coder_mcp

# 3. (SWE-bench only) apiary_swebench library
pip install -e /path/to/apiary-integration/apiary/swebench

# 4. (SWE-bench only, HuggingFace datasets)
pip install datasets
```

## Quick Start — Single Task

### Step 1: Build and start the Apiary daemon

Apiary is a Rust-based sandbox pool that provides namespace-isolated Linux
environments.  Each sandbox gets its own OverlayFS, user/mount/PID namespaces,
seccomp, and cgroup limits.

**System requirements**: Linux kernel 5.11+, cgroups v2 with delegation,
`uidmap` package.  Recommended: Ubuntu 22.04+ / Fedora 36+ / Debian 12+.

#### Option A: Docker (recommended)

The Apiary repo includes a `Dockerfile` with all prerequisites preconfigured.
The stock `docker-compose.yml` does not expose ports, so you need to add a port
mapping to make the daemon reachable from the host (where `coder_mcp` runs).

```bash
cd /path/to/apiary

# Start the container with port 8080 forwarded to the host.
# The daemon must bind to 0.0.0.0 inside the container.
docker compose run --rm -p 8080:8080 apiary bash
```

Inside the container, build and start the daemon:

```bash
cargo build --release

# Create a base rootfs
mkdir -p rootfs
CID=$(docker create ubuntu:jammy) \
  && docker export "$CID" | tar -xf - --exclude='dev/*' -C rootfs \
  && docker rm "$CID" > /dev/null

# Initialize the sandbox pool
apiary init --base-image ./rootfs --min-sandboxes 10 --max-sandboxes 40

# Start the daemon — bind to 0.0.0.0 so the port forward works
apiary daemon --bind 0.0.0.0:8080
```

Alternatively, add the port mapping permanently to `docker-compose.yml`:

```yaml
services:
  apiary:
    ports:
      - "8080:8080"
    # ... rest of existing config
```

#### Option B: Native build

```bash
cd /path/to/apiary-integration/apiary

# Enable rootless user namespaces (if not already)
sudo sysctl -w kernel.unprivileged_userns_clone=1

# Build
cargo build --release

# Create a base rootfs
mkdir -p rootfs
CID=$(docker create ubuntu:jammy) \
  && docker export "$CID" | tar -xf - --exclude='dev/*' -C rootfs \
  && docker rm "$CID" > /dev/null

# Initialize the sandbox pool
apiary init --base-image ./rootfs --min-sandboxes 10 --max-sandboxes 40

# Start the daemon
apiary daemon --bind 127.0.0.1:8080
```

#### Verify the daemon

```bash
curl -s http://172.17.0.1:8080/healthz
# {"status":"ok"}

curl -s http://127.0.0.1:8080/api/v1/status
# {"min_sandboxes":10,"max_sandboxes":40,"available":10,...}
```

The daemon exposes the following REST API used by `coder_mcp`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/api/v1/status` | GET | Pool status and counters |
| `/api/v1/sessions` | POST | Create a persistent session (reserves a sandbox) |
| `/api/v1/sessions/:id` | DELETE | Close session and release sandbox |
| `/api/v1/tasks` | POST | Execute a command in a session |

### Step 2: Start the LLM server

```bash
trtllm-serve <hf_model> --port 8000
```

Or use any OpenAI-compatible endpoint.

### Step 3: Start the coder_mcp server

```bash
python -m coder_mcp \
    --backend apiary \
    --apiary-url http://172.17.0.1:8080 \
    --port 8083
```

Key flags:

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--apiary-url` | `APIARY_URL` | `http://127.0.0.1:8080` | Apiary daemon URL |
| `--apiary-token` | `APIARY_API_TOKEN` | (none) | Bearer token for Apiary |
| `--mcp-token` | `MCP_AUTH_TOKEN` | (none) | Bearer token clients must present |
| `--working-dir` | `APIARY_WORKING_DIR` | `/workspace` | Default sandbox working dir |
| `--idle-timeout` | | `300` | Seconds before idle sandbox is reaped |
| `--port` | | `8083` | SSE bind port |
| `--debug` | | off | Enable ASGI debug mode |

### Step 4: Run the Coder agent

```bash
python examples/scaffolding/contrib/Coder/run_coder.py \
    --base_url http://localhost:8000/v1 \
    --model Qwen3/Qwen3-30B-A3B \
    --mcp_url http://0.0.0.0:8083/sse \
    --prompt "Implement a thread-safe LRU cache in Python" \
    --max_iterations 50
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--base_url` | `http://localhost:8000/v1` | LLM server endpoint |
| `--model` | `Qwen3/Qwen3-30B-A3B` | Model name |
| `--mcp_url` | `http://0.0.0.0:8083/sse` | coder_mcp SSE endpoint |
| `--max_mcp_connections` | `200` | Max concurrent sandbox connections |
| `--prompt` | (built-in example) | Task prompt |
| `--max_tokens` | `16384` | Max generation tokens |
| `--max_iterations` | `50` | Max tool-calling iterations |
| `--enable_statistics` | off | Print detailed metrics |
| `--enable_query_collector` | off | Dump query info to `query_result.json` |

## SWE-bench Evaluation

Runs the Coder agent against SWE-bench instances in parallel, each in its own
Apiary sandbox with a pre-exported Docker rootfs.

### Additional setup

```bash
pip install -e /path/to/apiary-integration/apiary/swebench
pip install datasets  # only for HuggingFace dataset loading
```

### Run

```bash
python examples/scaffolding/contrib/Coder/run_swebench.py \
    --dataset lite \
    --base_url http://localhost:8000/v1 \
    --model Qwen3/Qwen3-30B-A3B \
    --mcp_url http://0.0.0.0:8083/sse \
    --rootfs_cache_dir /tmp/apiary_rootfs \
    --output_dir ./swebench_output \
    --max_parallel_requests 16
```

The `--dataset` flag accepts:

| Value | HuggingFace Dataset |
|-------|---------------------|
| `full` | `princeton-nlp/SWE-Bench` |
| `verified` | `princeton-nlp/SWE-Bench_Verified` |
| `lite` | `princeton-nlp/SWE-Bench_Lite` |
| `multimodal` | `princeton-nlp/SWE-Bench_Multimodal` |
| `multilingual` | `swe-bench/SWE-Bench_Multilingual` |
| `/path/to/file.json` | Local JSON/JSONL |

Outputs:
- `swebench_output/preds.json` — predictions for SWE-bench evaluation
- `swebench_output/<instance_id>/<instance_id>.traj.json` — per-instance trajectory

## Benchmarks

The Coder benchmark is integrated into the scaffolding benchmark harness and
measures throughput and latency under configurable concurrency.

```bash
python -m examples.scaffolding.benchmarks \
    --model_dir /path/to/model \
    --base_url http://localhost:8000/v1 \
    --enable_coder \
    --coder_concurrency 16 \
    --coder_prompt_num 8 \
    --coder_max_iterations 50 \
    --mcp_url http://0.0.0.0:8083/sse \
    --enable_statistics
```

Coder-specific benchmark flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--enable_coder` | off | Enable the Coder benchmark |
| `--coder_concurrency` | `32` | Number of concurrent requests |
| `--coder_prompt_num` | `8` | Number of prompts to run |
| `--coder_max_iterations` | `50` | Max tool-calling iterations per request |
| `--coder_max_connections` | `200` | Max Apiary sandbox connections |
| `--mcp_url` | `http://0.0.0.0:8083/sse` | coder_mcp SSE endpoint |
| `--coder_rate` | `1.0` | Poisson arrival rate (req/s) in rate mode |

## Programmatic Usage

```python
import asyncio
from openai import AsyncOpenAI
from tensorrt_llm.scaffolding import ApiaryMCPWorker, TRTOpenaiWorker
from tensorrt_llm.scaffolding.contrib.Coder import create_coder_scaffolding_llm

async def main():
    client = AsyncOpenAI(api_key="tensorrt_llm", base_url="http://localhost:8000/v1")
    generation_worker = TRTOpenaiWorker(client, "Qwen3/Qwen3-30B-A3B")
    mcp_worker = ApiaryMCPWorker("http://0.0.0.0:8083/sse")

    llm = create_coder_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=16384,
        max_iterations=50,
    )

    result = llm.generate_async("Add input validation to the API handlers")
    output = await result.aresult()
    print(output.outputs[0].text)

    await mcp_worker.async_shutdown()
    llm.shutdown()

asyncio.run(main())
```

## Module Structure

```
tensorrt_llm/scaffolding/
├── execution_scope.py        # Per-request/branch scope tracking
├── scaffolding_llm.py        # Main orchestration engine
├── worker.py                 # Worker implementations (incl. ApiaryMCPWorker)
├── controller.py             # Controller base and ChatWithMCPController
├── task.py                   # Task types (ChatTask, MCPCallTask, etc.)
├── task_collection.py        # Metrics, tracing, KV cache hints
├── replay.py                 # Trace-driven replay for debugging
└── contrib/Coder/
    ├── coder.py              # Coder controller + factory
    ├── swebench.py           # SWE-bench controller + factory
    ├── tools.py              # OpenAI-style tool definitions
    └── prompts.py            # System prompts

examples/scaffolding/contrib/Coder/
├── run_coder.py              # Single-task CLI runner
├── run_swebench.py           # SWE-bench batch runner
└── coder_mcp/                # MCP server package
    └── src/coder_mcp/
        ├── __main__.py       # CLI entry point
        ├── server.py         # Composition root (wires backend + tools + app)
        ├── app.py            # Starlette ASGI app (SSE transport)
        ├── tools.py          # MCP tool implementations
        ├── patch.py          # Codex-style patch parser
        ├── plan.py           # Per-client plan state
        └── backends/
            ├── __init__.py   # SandboxBackend protocol + registry
            └── apiary.py     # Apiary REST backend implementation

examples/scaffolding/benchmarks/
├── __main__.py               # Benchmark harness entry point
└── coder_benchmark.py        # Coder-specific benchmark logic
```

## Troubleshooting

**Connection refused to coder_mcp**: Ensure the MCP server is running and the
`--mcp_url` matches the server's `--host`/`--port`.

**Apiary session errors**: Check that the Apiary daemon is reachable at the
configured `--apiary-url`.  The MCP server auto-recreates sessions on 404.

**`ImportError: apiary_swebench`**: Install the library from the
apiary-integration repo: `pip install -e /path/to/apiary-integration/apiary/swebench`

**Sandbox idle reaping**: By default, sandboxes with no connected SSE client
are reaped after 300 seconds (`--idle-timeout`).  Increase this for long-running
tasks.

**Execution tracing**: Pass `--enable_tracing` to `run_swebench.py` or use
`enable_tracing=True` in the factory function to record execution traces for
later replay via `TraceReplayScaffoldingLlm`.
