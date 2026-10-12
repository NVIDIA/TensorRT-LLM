import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_coder_mcp_module():
    apiary_client = types.ModuleType("apiary_client")
    apiary_client.ApiarySessionMux = type("ApiarySessionMux", (), {})
    apiary_client.TaskResult = type("TaskResult", (), {})
    sys.modules["apiary_client"] = apiary_client

    module_path = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "scaffolding"
        / "mcp"
        / "coder"
        / "coder_mcp.py"
    )
    spec = importlib.util.spec_from_file_location("coder_mcp_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_coder_mcp_requires_client_id():
    coder_mcp = _load_coder_mcp_module()
    downstream_called = False

    async def downstream(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True

    middleware = coder_mcp._ApiarySessionMiddleware(downstream)
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    async def invoke_middleware():
        await middleware(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/mcp",
                "raw_path": b"/mcp",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8083),
            },
            receive,
            send,
        )

    asyncio.run(invoke_middleware())

    assert not downstream_called
    assert messages[0]["status"] == 400
    assert json.loads(messages[1]["body"]) == {"error": "client_id query parameter is required"}
