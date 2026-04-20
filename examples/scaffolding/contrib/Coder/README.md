# Scaffolding Coder Agent

Agentic coding system built on the TensorRT-LLM Scaffolding framework. The Coder agent uses an LLM for reasoning and planning, and executes filesystem and shell operations inside isolated Apiary sandboxes through a dedicated MCP server.

## Architecture

```text
LLM server <-> ScaffoldingLlm <-> Coder / SWEBenchCoder
                                 |
                                 v
                           ApiaryMCPWorker
                                 |
                                 v
                         examples/.../coder_mcp.py
                                 |
                                 v
                       apiary_client.ApiarySessionMux
                                 |
                                 v
                           Apiary daemon / sessions
```

Three services are involved:

- Apiary daemon: manages sandbox sessions, image registry (populated at runtime via HTTP), and command execution
- `coder_mcp.py`: exposes the Coder tool surface over MCP SSE
- LLM server: OpenAI-compatible endpoint used by Scaffolding

## Tooling Model

The Coder agent expects these MCP tools:

- `read_file`
- `list_dir`
- `grep_files`
- `exec`
- `shell`
- `update_plan`
- `think`
- `complete_task`

File edits go through `shell` (e.g. `sed -i`, `cat <<'EOF' > path` heredocs, `tee`).

`ApiaryMCPWorker` opens one SSE connection per Scaffolding execution scope, so parallel branches naturally get isolated sandboxes.

## Prerequisites

```bash
# Apiary Python bindings (shared with coder_mcp, runners, and SWE-bench helpers)
pip install /path/to/apiary/bindings/python
```

For HuggingFace SWE-bench datasets, install the `swebench` extra:

```bash
pip install '/path/to/apiary/bindings/python[swebench]'
```

## Start Apiary


The recommended deployment method is the Apiary container shipped under `/apiary/docker-compose.yml`.

### Container (recommended)

```bash
cd /path/to/apiary

# Build and start the container; 8080 is exposed on the host by default.
docker compose up -d
```

The container's entrypoint runs `apiary init && apiary daemon --bind 0.0.0.0:8080`, leaving you with an empty pool ready to accept image registrations from clients.

Useful environment variables (see the Apiary README for the full list):

| Variable | Default | Purpose |
|---|---|---|
| `APIARY_PORT` | `8080` | Host port to publish |
| `APIARY_BIND` | `0.0.0.0:8080` | Bind address inside the container |
| `APIARY_API_TOKEN` | (empty) | Bearer token for API auth (empty disables auth) |
| `APIARY_MAX_SANDBOXES` | `40` | Pool concurrency cap |
| `APIARY_LAYERS_DIR` | `/var/lib/apiary/layers` | Layer cache (named volume) |
| `APIARY_OVERLAY_DIR` | `/var/lib/apiary/overlays` | Overlay scratch (named volume) |

Verify from the host:

```bash
curl -s http://172.17.0.1:8080/healthz       # {"status":"ok"}
curl -s http://172.17.0.1:8080/api/v1/status # pool counters + registered_images
```

### Native install

Only needed when Docker is unavailable. Requires Linux 5.11+, cgroups v2 with delegation, and the `uidmap` package.

```bash
cd /path/to/apiary
cargo build --release

apiary init --max-sandboxes 40
apiary daemon --bind 0.0.0.0:8080
```

The Coder runners (`run_coder.py`, `run_swebench.py`, the benchmark) then register the images they need via `POST /api/v1/images` before they dispatch any work.

## Start the MCP Server

```bash
python examples/scaffolding/mcp/coder/coder_mcp.py \
    --apiary-url http://172.17.0.1:8080 \
    --default-image ubuntu:22.04 \
    --port 8083
```

Key flags:

- `--apiary-url` — Apiary daemon URL
- `--apiary-token` — bearer token for daemon auth
- `--mcp-token` — bearer token required on the SSE endpoint
- `--default-image` — fallback Docker image for sandbox sessions when an SSE client omits the `image` query parameter (must already be registered with the daemon)
- `--working-dir` — default sandbox working directory
- `--idle-timeout` — idle session reap timeout in seconds

Per-request image selection works through the SSE `image` query parameter. `ApiaryMCPWorker.set_scope_params(..., image=...)` is how the Scaffolding runners select the correct sandbox image for each request. The image is expected to be registered with the daemon already; the runners listed below do that for you.

## Run a Single Coder Task

```bash
python examples/scaffolding/contrib/Coder/run_coder.py \
    --base_url http://localhost:8000/v1 \
    --model Qwen3/Qwen3-30B-A3B \
    --apiary_url http://172.17.0.1:8080 \
    --mcp_url http://127.0.0.1:8083/sse \
    --image ubuntu:22.04 \
    --prompt "Implement a thread-safe LRU cache in Python" \
    --max_iterations 50
```

The runner registers `--image` with the Apiary daemon on startup (via `AsyncApiary`) and only dispatches the request once the image is loaded.

Important flags:

- `--image`: Docker image used for the request's sandbox (auto-registered)
- `--apiary_url` / `--apiary_token`: How to reach the Apiary daemon for image registration (defaults to `$APIARY_URL` / `$APIARY_API_TOKEN`)
- `--mcp_url`: `coder_mcp.py` SSE endpoint
- `--max_mcp_connections`: Max concurrent SSE / sandbox connections

## Run SWE-bench

The runner resolves the SWE-bench image set from the dataset, registers all unique images with the Apiary daemon (with per-image progress logging), and only then starts dispatching agent requests.

```bash
python examples/scaffolding/contrib/Coder/run_swebench.py \
    --dataset lite \
    --split dev \
    --apiary_url http://172.17.0.1:8080 \
    --base_url http://localhost:8000/v1 \
    --model Qwen3/Qwen3-30B-A3B \
    --mcp_url http://0.0.0.0:8083/sse \
    --output_dir ./swebench_output \
    --max_parallel_requests 16
```

Useful flags:

- `--apiary_url` / `--apiary_token`: Apiary daemon target
- `--apiary_load_timeout`: Bound the wait for image registration (default: no timeout — large SWE-bench splits can take a while on a cold cache)

If the daemon is unreachable the runner aborts immediately with a clear error before any LLM work is dispatched. Per-image failures are logged and surfaced as warnings; only those instances are affected — the rest of the batch still runs against the images that did load.

If you want to pre-load the image set out of band (for example to share a warmed-up daemon across multiple runs), use the `apiary-load-swebench` CLI shipped with `apiary-client[swebench]`:

```bash
apiary-load-swebench --apiary-url http://172.17.0.1:8080 --dataset lite
```

The runner is idempotent: already-loaded images are reported as `alreadypresent` and skip the load pipeline.

Outputs:

- `swebench_output/preds.json`: Predictions for SWE-bench evaluation
- `swebench_output/<instance_id>/<instance_id>.traj.json`: Per-instancetrajectory
- `swebench_output/<instance_id>/<instance_id>.trace.json`: Optional execution trace when `--enable_tracing` is enabled

## Run the Coder Benchmark

The benchmark harness passes a Docker image name per request through the shared `scope_params` path used by `ApiaryMCPWorker`, and the harness itself registers that image with the Apiary daemon before any benchmark request is dispatched.

```bash
python -m examples.scaffolding.benchmarks \
    --model_dir /path/to/model \
    --base_url http://localhost:8000/v1 \
    --enable_coder \
    --coder_image ubuntu:22.04 \
    --apiary_url http://172.17.0.1:8080 \
    --coder_concurrency 16 \
    --coder_prompt_num 8 \
    --coder_max_iterations 50 \
    --mcp_url http://0.0.0.0:8083/sse
```

Coder-specific flags:

- `--coder_image`: Docker image used for benchmark sandboxes (auto-registered)
- `--apiary_url` / `--apiary_token`: Apiary daemon target for image registration (defaults to `$APIARY_URL` / `$APIARY_API_TOKEN`)
- `--coder_concurrency`: Number of concurrent requests
- `--coder_prompt_num`: Number of prompts to run
- `--coder_max_iterations`: Max tool-calling iterations per request
- `--coder_max_connections`: Max concurrent Apiary sandbox connections

## Programmatic Usage

```python
import asyncio
from apiary_client import AsyncApiary
from openai import AsyncOpenAI
from tensorrt_llm.scaffolding import ApiaryMCPWorker, TRTOpenaiWorker
from tensorrt_llm.scaffolding.contrib.Coder import create_coder_scaffolding_llm


async def main() -> None:
    image = "ubuntu:22.04"

    # 1. Register the image with the Apiary daemon (idempotent).
    async with AsyncApiary(
        apiary_url="http://172.17.0.1:8080",
        images=[image],
    ) as apiary:
        assert image in apiary.loaded, f"Apiary failed to load {image}"

    # 2. Now run the Coder agent against the registered image.
    client = AsyncOpenAI(
        api_key="tensorrt_llm",
        base_url="http://localhost:8000/v1",
    )
    generation_worker = TRTOpenaiWorker(client, "Qwen3/Qwen3-30B-A3B")
    mcp_worker = ApiaryMCPWorker("http://0.0.0.0:8083/sse")

    llm = create_coder_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=16384,
        max_iterations=50,
    )

    result = llm.generate_async("Add input validation to the API handlers")
    mcp_worker.set_scope_params(result.id, image=image)
    output = await result.aresult()
    print(output.outputs[0].text)

    await mcp_worker.async_shutdown()
    llm.shutdown()
    generation_worker.shutdown()


asyncio.run(main())
```
