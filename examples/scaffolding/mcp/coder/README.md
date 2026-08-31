# Coder MCP Server

`coder_mcp.py` is the MCP SSE server backing the Scaffolding Coder agent. Each tool call runs inside an isolated Apiary sandbox session.

## Tool Surface

The server exposes the tools expected by `tensorrt_llm/scaffolding/contrib/Coder/tools.py`:

- `read_file`
- `list_dir`
- `grep_files`
- `exec`
- `shell`
- `update_plan`
- `think`
- `complete_task`

File edits go through `shell` (e.g. `sed -i`, `cat <<'EOF' > path` heredocs, `tee`).

## Dependencies

Install the shared Apiary Python bindings before running the server:

```bash
pip install /path/to/apiary/bindings/python
pip install -e examples/scaffolding/mcp/coder
```

`apiary-client` supplies:

- `ApiarySessionMux` for per-client session management
- `Apiary` / `AsyncApiary` for runtime image registration
- `apiary-mcp` reference MCP server
- `apiary-load-swebench` CLI (and `apiary_client.swebench.resolve`) for
  SWE-bench image-list generation

## Running the Server

Every Apiary session needs a Docker image name. Each SSE request can pick an image via the `image` query parameter; when it doesn't, the server falls back to `--default-image`.

```bash
python examples/scaffolding/mcp/coder/coder_mcp.py \
    --apiary-url http://172.17.0.1:8080 \
    --default-image ubuntu:22.04 \
    --port 8083
```

Useful flags:

- `--apiary-url`: Apiary daemon base URL
- `--apiary-token`: Bearer token for daemon auth
- `--mcp-token`: Bearer token required on the SSE endpoint
- `--default-image`: Fallback Docker image for sessions when an SSE client omits the `image` query parameter (must already be registered with the daemon)
- `--working-dir`: Default sandbox working directory
- `--idle-timeout`: Idle session reap timeout in seconds

## SSE Parameters

The SSE transport lives at `/sse`, with the MCP message endpoint at
`/messages/`.

Supported query parameters:

- `client_id`: Stable client identity for a persistent session
- `image`: Docker image name for this request/session

Example:

```text
http://localhost:8083/sse?client_id=my-session&image=ubuntu:22.04
```

If `image` is omitted, `coder_mcp.py` falls back to `--default-image`.

## Environment Variables

- `APIARY_URL`: Apiary daemon URL
- `APIARY_API_TOKEN`: Bearer token for Apiary authentication
- `APIARY_WORKING_DIR`: Default working directory inside the sandbox
- `MCP_AUTH_TOKEN`: Optional bearer token required on the MCP SSE endpoint
