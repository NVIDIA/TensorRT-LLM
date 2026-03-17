# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCP server for executing Python code in a sandboxed environment.

Start the server:
    export SANDBOX_ENDPOINT=http://127.0.0.1:8080
    uv run python_interpreter.py --port 8086
"""

import asyncio
import logging
import os
import re
from pathlib import Path

import httpx
import uvicorn
import yaml
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

mcp = FastMCP("python_interpreter")


def _load_config(config_path: str | None = None) -> dict:
    """Load configuration from a YAML file.

    Falls back to environment variables for any key not present in the
    YAML file.
    """
    cfg = {}
    if config_path and Path(config_path).is_file():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        LOGGER.info("Loaded config from %s", config_path)

    def _get(yaml_key: str, env_key: str, default: str = "") -> str:
        return str(cfg.get(yaml_key, "") or os.getenv(env_key, default))

    return {
        "sandbox_endpoint": _get("sandbox_endpoint", "SANDBOX_ENDPOINT",
                                 "http://127.0.0.1:8080"),
        "mcp_host": _get("mcp_host", "MCP_HOST", "0.0.0.0"),
        "mcp_port": int(_get("mcp_port", "MCP_PORT", "8086")),
    }


_CFG: dict = {
    "sandbox_endpoint": os.getenv("SANDBOX_ENDPOINT",
                                  "http://127.0.0.1:8080"),
}


def _extract_code(code: str) -> str:
    """Extract code from markdown code blocks if present."""
    triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
    if triple_match:
        return triple_match.group(1)
    code_match = re.search(r'<code>(.*?)</code>', code, re.DOTALL)
    if code_match:
        return code_match.group(1)
    return code


@mcp.tool()
async def python_interpreter(code: str) -> str:
    """Execute Python code in sandboxed environment."""
    code = _extract_code(code)
    if not code.strip():
        return "[PythonInterpreter Error]: Empty code."
    if not _CFG["sandbox_endpoint"]:
        return "[PythonInterpreter Error]: sandbox_endpoint not configured."
    payload = {
        "code": code,
        "language": "python",
        "run_timeout": 50,
    }
    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                resp = await client.post(f"{_CFG["sandbox_endpoint"]}/run",
                                         json=payload,
                                         timeout=60)
                resp.raise_for_status()
                result = resp.json()
                parts = []
                if result.get("stdout"):
                    parts.append(f"stdout:\n{result['stdout']}")
                if result.get("stderr"):
                    parts.append(f"stderr:\n{result['stderr']}")
                exec_time = result.get("execution_time", 0)
                if exec_time >= 49:
                    parts.append(
                        "[PythonInterpreter Error] TimeoutError: "
                        "Execution timed out.")
                output = '\n'.join(parts)
                return output if output.strip() else "Finished execution."
            except Exception as e:
                if attempt == 2:
                    return f"[PythonInterpreter Error]: {e}"
                await asyncio.sleep(1)
    return "[PythonInterpreter Error]: All attempts failed."


def create_starlette_app(mcp_server: Server,
                         *,
                         debug: bool = False) -> Starlette:
    """Create a Starlette app serving the MCP server via SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )
        return Response()

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Python Interpreter MCP SSE-based server")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (API keys, endpoints)")
    parser.add_argument("--host", default=None,
                        help="Host to bind to (overrides config)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to listen on (overrides config)")
    args = parser.parse_args()

    _CFG.update(_load_config(args.config))

    host = args.host or _CFG.get("mcp_host", "0.0.0.0")
    port = args.port or _CFG.get("mcp_port", 8086)

    mcp_server = mcp._mcp_server  # noqa: WPS437
    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=host, port=port)
