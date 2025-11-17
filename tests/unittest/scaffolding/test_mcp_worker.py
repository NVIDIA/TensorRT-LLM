import json
import multiprocessing
import time
from typing import Optional

import pytest
import requests
import uvicorn
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Mount, Route

from tensorrt_llm.scaffolding.controller import ChatWithMCPController, NativeGenerationController
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import (
    AssistantMessage,
    ChatTask,
    MCPCallTask,
    SystemMessage,
    TaskStatus,
)
from tensorrt_llm.scaffolding.worker import MCPWorker, Worker

# ============================================================
# MCP Server Definition (based on websearch.py)
# ============================================================

# Initialize FastMCP server for testing
mcp = FastMCP("test_mcp_server")


@mcp.tool()
async def add_numbers(a: int, b: int) -> int:
    return a + b


@mcp.tool()
async def echo_message(message: str) -> str:
    return f"Echo: {message}"


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    sse = SseServerTransport("/messages/")
    print("Creating Starlette app with SSE transport")

    async def handle_sse(request: Request) -> StreamingResponse:
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
        # Return an empty streaming response after SSE connection closes
        return StreamingResponse(iter([]), media_type="text/event-stream")

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


def run_mcp_server(host: str, port: int):
    """Run MCP server in a separate process."""
    print(f"Running MCP server on {host}:{port}")
    mcp_server = mcp._mcp_server  # noqa: WPS437
    starlette_app = create_starlette_app(mcp_server, debug=False)
    uvicorn.run(starlette_app, host=host, port=port, log_level="error")


# ============================================================
# RemoteMCPServer Class (based on RemoteOpenAIServer in openai_server.py)
# ============================================================


class RemoteMCPServer:
    MAX_SERVER_START_WAIT_S = 60  # Maximum time to wait for server startup (seconds)

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, env: Optional[dict] = None) -> None:
        self.host = host
        self.port = port
        self.proc = None

        # Start MCP server in a separate process
        self.proc = multiprocessing.Process(
            target=run_mcp_server, args=(self.host, self.port), daemon=True
        )
        self.proc.start()

        # Wait for server to be ready
        # self._wait_for_server(url=self.sse_url,
        #                      timeout=self.MAX_SERVER_START_WAIT_S)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def terminate(self):
        if self.proc is None:
            return

        # Terminate the process
        self.proc.terminate()
        self.proc.join(timeout=5)

        # Force kill if process is still running
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=5)

        self.proc = None

    def _wait_for_server(self, *, url: str, timeout: float):
        start = time.time()
        while True:
            try:
                # Try to connect to server
                requests.get(url, timeout=1)
                # SSE endpoint should return a response (even if connection will be closed)
                break
            except (requests.ConnectionError, requests.Timeout) as err:
                # Check if process exited unexpectedly
                if self.proc is not None and not self.proc.is_alive():
                    raise RuntimeError("MCP server process exited unexpectedly.") from err

                # Check if timeout occurred
                if time.time() - start > timeout:
                    raise RuntimeError(f"MCP server failed to start in {timeout} seconds.") from err

                # Wait before retrying
                time.sleep(0.5)

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def sse_url(self) -> str:
        return f"{self.url_root}/sse"

    def get_sse_url(self) -> str:
        return self.sse_url


# ============================================================
# Pytest Fixtures (based on test_worker.py pattern)
# ============================================================


@pytest.fixture(scope="module")
def mcp_server():
    # Create and start MCP server (similar to RemoteOpenAIServer usage)
    remote_server = RemoteMCPServer(host="0.0.0.0", port=8080)

    # Yield server instance for test use
    yield remote_server

    # Cleanup: terminate server after tests
    remote_server.terminate()


class FunctionCall:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, name: str, arguments: dict):
        self.function = FunctionCall(name, arguments)


class DummyWorker(Worker):
    async def dummy_handler(self, task: ChatTask):
        if len(task.messages) == 2:
            task.add_message(
                AssistantMessage(
                    content="call add_numbers(5, 3)",
                    tool_calls=[ToolCall("add_numbers", '{"a": 5, "b": 3}')],
                )
            )
            task.finish_reason = "tool_calls"
        elif len(task.messages) == 4:
            task.add_message(
                AssistantMessage(
                    content="Hello MCP!",
                    tool_calls=[ToolCall("echo_message", '{"message": "Hello MCP!"}')],
                )
            )
            task.finish_reason = "tool_calls"
        else:
            task.add_message(AssistantMessage(content="Hello MCP!"))
            task.finish_reason = "stop"

        return TaskStatus.SUCCESS

    task_handlers = {ChatTask: dummy_handler}


# ============================================================
# Test Cases
# ============================================================


@pytest.mark.asyncio
async def test_mcp_worker_add_numbers(mcp_server):
    # 1. Initialize MCPWorker with mcp_server's SSE URL
    sse_url = mcp_server.get_sse_url()
    worker = MCPWorker(urls=[sse_url])

    # 2. Initialize worker in asyncio event loop
    await worker.init_in_asyncio_event_loop()

    try:
        # 3. Build MCPCallTask to call add_numbers tool
        task = MCPCallTask()
        task.tool_name = "add_numbers"
        # args needs to be in JSON string format
        task.args = json.dumps({"a": 5, "b": 3})

        # 4. Run task through worker
        from tensorrt_llm.scaffolding.task import TaskStatus

        status = await worker.run_task(task)

        # 5. Verify results
        assert status == TaskStatus.SUCCESS, f"Task failed with status: {status}"
        assert task.result_str is not None, "Task result_str should not be None"

        # add_numbers returns integer, verify result
        result = int(task.result_str)
        assert result == 8, f"Expected 8, but got {result}"

        print(f"✓ Test passed: add_numbers(5, 3) = {result}")

    finally:
        # 6. Clean up worker resources
        await worker.async_shutdown()


@pytest.mark.asyncio
async def test_mcp_worker_echo_message(mcp_server):
    # Initialize MCPWorker
    sse_url = mcp_server.get_sse_url()
    worker = MCPWorker(urls=[sse_url])

    # Initialize in event loop
    await worker.init_in_asyncio_event_loop()

    try:
        # Create task to call echo_message
        task = MCPCallTask()
        task.tool_name = "echo_message"
        task.args = json.dumps({"message": "Hello MCP!"})

        # Run task
        status = await worker.run_task(task)

        # Verify results
        assert status == TaskStatus.SUCCESS, f"Task failed with status: {status}"
        assert task.result_str is not None, "Task result_str should not be None"
        assert "Echo: Hello MCP!" in task.result_str, f"Unexpected output: {task.result_str}"

        print(f"✓ Test passed: echo_message returned '{task.result_str}'")

    finally:
        await worker.async_shutdown()


@pytest.mark.asyncio
async def test_mcp_worker_multiple_calls(mcp_server):
    import json

    from tensorrt_llm.scaffolding.task import MCPCallTask, TaskStatus
    from tensorrt_llm.scaffolding.worker import MCPWorker

    # Initialize MCPWorker
    sse_url = mcp_server.get_sse_url()
    worker = MCPWorker(urls=[sse_url])

    await worker.init_in_asyncio_event_loop()

    try:
        # First call: add_numbers(10, 20)
        task1 = MCPCallTask()
        task1.tool_name = "add_numbers"
        task1.args = json.dumps({"a": 10, "b": 20})

        status1 = await worker.run_task(task1)
        assert status1 == TaskStatus.SUCCESS
        result1 = int(task1.result_str)
        assert result1 == 30, f"Expected 30, but got {result1}"

        # Second call: add_numbers(100, 200)
        task2 = MCPCallTask()
        task2.tool_name = "add_numbers"
        task2.args = json.dumps({"a": 100, "b": 200})

        status2 = await worker.run_task(task2)
        assert status2 == TaskStatus.SUCCESS
        result2 = int(task2.result_str)
        assert result2 == 300, f"Expected 300, but got {result2}"

        # Third call: echo_message
        task3 = MCPCallTask()
        task3.tool_name = "echo_message"
        task3.args = json.dumps({"message": "Multiple calls test"})

        status3 = await worker.run_task(task3)
        assert status3 == TaskStatus.SUCCESS
        assert "Echo: Multiple calls test" in task3.result_str

        print("✓ Test passed: Multiple calls successful")
        print(f"  - Call 1: add_numbers(10, 20) = {result1}")
        print(f"  - Call 2: add_numbers(100, 200) = {result2}")
        print(f"  - Call 3: echo_message returned '{task3.result_str}'")

    finally:
        await worker.async_shutdown()


@pytest.mark.asyncio
async def test_scaffolding_with_chat_mcp_controller(mcp_server):
    # Initialize workers
    sse_url = mcp_server.get_sse_url()
    mcp_worker = MCPWorker(urls=[sse_url])
    await mcp_worker.init_in_asyncio_event_loop()

    dummy_worker = DummyWorker()

    # Create controllers
    native_generation_controller = NativeGenerationController(
        sampling_params={
            "max_tokens": 50,
            "temperature": 0.7,
        }
    )

    system_prompts = [SystemMessage("You are a helpful assistant.")]
    add_number_tool = {
        "name": "add_numbers",
        "description": "Add two numbers together",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        },
    }
    echo_message_tool = {
        "name": "echo_message",
        "description": "Echo back the input message",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
        },
    }
    tools = [add_number_tool, echo_message_tool]
    chat_mcp_controller = ChatWithMCPController(
        generation_controller=native_generation_controller,
        system_prompts=system_prompts,
        max_iterations=5,
        tools=tools,
    )

    # Create workers dictionary
    workers = {
        NativeGenerationController.WorkerTag.GENERATION: dummy_worker,
        ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
    }

    # Create ScaffoldingLlm
    scaffolding_llm = ScaffoldingLlm(
        chat_mcp_controller,
        workers=workers,
    )

    try:
        # Run generation
        future = scaffolding_llm.generate_async("Please help me calculate something")
        result = await future.aresult()

        # Verify results
        assert isinstance(result.outputs[0].text, str) and len(result.outputs[0].text) > 0, (
            "Output should be a non-empty string"
        )

        print("✓ Test passed: ChatWithMCPController returned output")
        print(f"  Output: {result.outputs[0].text}")

    finally:
        # Clean up
        await mcp_worker.async_shutdown()
        scaffolding_llm.shutdown(shutdown_workers=False)  # Don't shutdown workers again


# ============================================================
# Optional: Run this file directly to start the MCP server
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run test MCP SSE-based server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8090, help="Port to listen on")
    args = parser.parse_args()

    mcp_server = mcp._mcp_server  # noqa: WPS437
    starlette_app = create_starlette_app(mcp_server, debug=True)

    print(f"Starting MCP server on {args.host}:{args.port}")
    uvicorn.run(starlette_app, host=args.host, port=args.port)
