# Coder MCP Server

`CoderMCP` is an Apiary-backed MCP server for the Coder agent.

## Tools

### File Tools
- `read_file`
- `list_dir`
- `grep_files`
- `apply_patch`

### Shell Tools
- `shell`
- `shell_command`

### Planning Tools
- `update_plan`
- `think`
- `complete_task`

## Architecture

Each MCP client connects to `/sse` and gets a `client_id`. That `client_id`
selects a persistent Apiary sandbox session, so filesystem and shell state are
preserved across tool calls for that client.

```
ScaffoldingLlm -> MCPWorker -> CoderMCP -> Apiary daemon -> sandbox session
```

## Running the Server

```bash
# Default: runs on 0.0.0.0:8083 and uses APIARY_URL/APIARY_WORKING_DIR
python coder_mcp.py

# Custom host/port
python coder_mcp.py --host 127.0.0.1 --port 9000

# Explicit Apiary config
python coder_mcp.py \
    --apiary-url http://127.0.0.1:8080 \
    --working-dir /workspace \
    --idle-timeout 300
```

## Environment Variables

- `APIARY_URL` - Apiary daemon URL
- `APIARY_API_TOKEN` - Bearer token for Apiary authentication
- `APIARY_WORKING_DIR` - Default working directory inside the sandbox
- `MCP_AUTH_TOKEN` - Optional bearer token required on the MCP SSE endpoint

## Using with TensorRT-LLM Scaffolding

```python
from tensorrt_llm.scaffolding.worker import MCPWorker
from tensorrt_llm.scaffolding.contrib.Coder import create_coder_scaffolding_llm

mcp_worker = MCPWorker.init_with_urls(
    ["http://localhost:8083/sse?client_id=my-session"]
)
await mcp_worker.init_in_asyncio_event_loop()

coder = create_coder_scaffolding_llm(
    generation_worker=generation_worker,
    mcp_worker=mcp_worker,
)

result = coder.generate("Add a hello world function to main.py")
print(result.text)
```

## Protocol

The server uses SSE transport for MCP communication:
- SSE endpoint: `/sse`
- Message endpoint: `/messages/`

Pass `client_id` as an SSE query parameter to bind a caller to a specific
sandbox session:

```text
http://localhost:8083/sse?client_id=my-session
```
