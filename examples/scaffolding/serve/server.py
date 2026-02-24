r"""Agent Server - Unified server for remote agent execution.

This is the single server process that combines scaffolding, TensorRT-LLM
(via OpenAI-compatible API), and tool call relay. Clients connect via a
single WebSocket and the server handles everything:

    Client <--WebSocket--> Agent Server (scaffolding + trtllm + tool relay)

No separate MCP server is needed. The server directly relays tool calls
to the client for local execution through the same WebSocket connection.

Protocol (JSON over WebSocket at /ws):
    Client -> Server:
        {"type": "agent_request", "request_id": "...",
         "application": "coder|deep_research", "prompt": "..."}

    Server -> Client:
        {"type": "tool_call", "request_id": "...", "call_id": "...",
         "tool_name": "...", "arguments": {...}}

    Client -> Server:
        {"type": "tool_result", "call_id": "...", "result": "..."}

    Server -> Client:
        {"type": "status", "request_id": "...", "message": "..."}
        {"type": "agent_result", "request_id": "...", "output": "...",
         "error": null}

Usage:
    python -m examples.scaffolding.serve.server \\
        --base_url http://localhost:8000/v1 --model your-model
"""

import argparse
import asyncio
import json
import logging
import traceback
import uuid
from typing import Dict

import aiohttp
from aiohttp import web
from openai import AsyncOpenAI

from tensorrt_llm.scaffolding.contrib.Coder import create_coder_scaffolding_llm
from tensorrt_llm.scaffolding.contrib.open_deep_research import (
    create_open_deep_research_scaffolding_llm,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm, current_scaffolding_result
from tensorrt_llm.scaffolding.task import MCPCallTask, TaskStatus
from tensorrt_llm.scaffolding.worker import TRTOpenaiWorker, Worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent_server")


class ClientRelayMCPWorker(Worker):
    """Server-wide worker that routes MCP tool calls to client WebSockets.

    A single instance is shared across **all** WebSocket connections and
    ``ScaffoldingLlm`` instances.  Routing works without a shared
    registry: the server attaches ``ws`` and ``client_request_id``
    directly to :pyattr:`ScaffoldingResult.metadata` after calling
    ``generate_async``, and ``call_handler`` reads them from the
    :data:`current_scaffolding_result` context variable.

    Because ``generate_async`` returns synchronously and metadata
    assignment is a plain Python statement (no yield point), the
    metadata is guaranteed to be set before the event loop can start
    processing the request.

    Data flow::

        Server:
            result = llm.generate_async(prompt)
            result.metadata["ws"] = ws                  # sync
            result.metadata["client_request_id"] = rid  # sync
            await result.aresult()                      # first yield

        ScaffoldingLlm._handle_single_request:
            current_scaffolding_result.set(request.result)
                -> propagated to all child asyncio tasks

        ChatWithMCPController -> MCPCallTask
            -> call_handler() reads current_scaffolding_result
            -> reads ws from result.metadata["ws"]
            -> WebSocket send to client
            -> Client executes tool locally
            -> WebSocket receive -> handle_tool_result()
            -> MCPCallTask.result_str set
            -> Controller continues
    """

    # Default timeout for a single tool call (seconds).  If the client
    # does not respond within this window the call is treated as failed.
    DEFAULT_TOOL_CALL_TIMEOUT = 300  # 5 minutes

    def __init__(self, tool_call_timeout: float = DEFAULT_TOOL_CALL_TIMEOUT):
        # call_id -> (ws, Future) for pending tool results
        self._pending_calls: Dict[str, tuple] = {}
        self._lock = asyncio.Lock()
        self._tool_call_timeout = tool_call_timeout

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cancel_pending_for_connection(
        self,
        ws: web.WebSocketResponse,
    ):
        """Cancel **all** pending tool calls tied to *ws*.

        Called as a safety-net when a WebSocket disconnects so that no
        futures are leaked.
        """
        async with self._lock:
            to_cancel = [cid for cid, (w, _) in self._pending_calls.items() if w is ws]
            for cid in to_cancel:
                _, future = self._pending_calls.pop(cid)
                if not future.done():
                    future.cancel()

    # ------------------------------------------------------------------
    # Task handler
    # ------------------------------------------------------------------

    async def call_handler(self, task: MCPCallTask) -> TaskStatus:
        """Route a tool call to the correct client WebSocket.

        Routing data is read directly from
        ``current_scaffolding_result.metadata`` — no shared registry
        lookup is needed.
        """
        result = current_scaffolding_result.get(None)
        if result is None:
            logger.error("MCPCallTask has no current_scaffolding_result in async context")
            task.result_str = "Error: No scaffolding result in async context"
            return TaskStatus.SUCCESS

        ws = result.metadata.get("ws")
        client_request_id = result.metadata.get("client_request_id")

        if ws is None:
            logger.error(f"No WebSocket in result.metadata for result_id={result.id}")
            task.result_str = "Error: No WebSocket in result metadata for routing"
            return TaskStatus.SUCCESS

        if ws.closed:
            logger.error(f"WebSocket closed for result_id={result.id}")
            task.result_str = f"Error: WebSocket closed for result {result.id}"
            return TaskStatus.SUCCESS

        call_id = str(uuid.uuid4())
        tool_name = task.tool_name

        # Parse args – they arrive as a JSON string from the controller
        if isinstance(task.args, str):
            try:
                tool_args = json.loads(task.args)
            except json.JSONDecodeError:
                tool_args = {"raw": task.args}
        else:
            tool_args = task.args or {}

        # Create a future resolved when the client sends back the result
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        async with self._lock:
            self._pending_calls[call_id] = (ws, future)

        message = {
            "type": "tool_call",
            "request_id": client_request_id,
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": tool_args,
        }

        logger.info(f"Relaying tool call to client: {tool_name}({json.dumps(tool_args)[:200]}...)")

        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send tool call to client: {e}")
            task.result_str = f"Error: Failed to relay tool call: {e}"
            async with self._lock:
                self._pending_calls.pop(call_id, None)
            return TaskStatus.SUCCESS

        # Wait for client response with a timeout so that a hung client
        # cannot block the scaffolding pipeline indefinitely.
        try:
            tool_result = await asyncio.wait_for(
                future,
                timeout=self._tool_call_timeout,
            )
            task.result_str = tool_result
            logger.info(
                f"Received tool result for {tool_name}"
                f" (call_id={call_id}): {str(tool_result)[:200]}..."
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Tool call timed out after {self._tool_call_timeout}s: "
                f"{tool_name} (call_id={call_id})"
            )
            task.result_str = f"Error: Tool call timed out after {self._tool_call_timeout}s"
        except asyncio.CancelledError:
            logger.warning(f"Tool call cancelled: {tool_name} (call_id={call_id})")
            task.result_str = "Error: Tool call cancelled"
        except Exception as e:
            logger.error(f"Error waiting for tool result: {e}")
            task.result_str = f"Error: Tool execution failed: {e}"
        finally:
            async with self._lock:
                self._pending_calls.pop(call_id, None)

        return TaskStatus.SUCCESS

    # ------------------------------------------------------------------
    # Inbound tool results from clients
    # ------------------------------------------------------------------

    async def handle_tool_result(self, call_id: str, result: str):
        """Resolve a pending tool call future with the client's result.

        Called by the WebSocket handler when the client sends a
        ``tool_result`` message.
        """
        async with self._lock:
            entry = self._pending_calls.get(call_id)

        if entry is not None:
            _, future = entry
            if not future.done():
                future.set_result(result)
        else:
            logger.warning(f"Received result for unknown or completed call_id: {call_id}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def async_shutdown(self):
        """Cancel all pending tool calls on server shutdown."""
        async with self._lock:
            for call_id, (_, future) in self._pending_calls.items():
                if not future.done():
                    future.cancel()
            self._pending_calls.clear()

    task_handlers = {MCPCallTask: call_handler}


def _create_agent_llm(
    application: str,
    generation_worker: Worker,
    relay_worker: ClientRelayMCPWorker,
    args: argparse.Namespace,
) -> ScaffoldingLlm:
    """Create a ScaffoldingLlm configured for the specified agent type.

    This is an internal helper called by :func:`acquire_agent_llm`.
    It **must** be called from within an async context on the server's
    event loop so that the ``ScaffoldingLlm`` reuses the same loop
    instead of spinning up a dedicated thread.

    Args:
        application: ``"coder"`` or ``"deep_research"``.
        generation_worker: Worker for LLM generation
            (``TRTOpenaiWorker``).
        relay_worker: Worker for tool calls
            (``ClientRelayMCPWorker``).
        args: Command line arguments with model configuration.

    Returns:
        Configured ``ScaffoldingLlm`` instance.

    Raises:
        ValueError: If *application* is not recognized.
    """
    if application == "coder":
        return create_coder_scaffolding_llm(
            generation_worker,
            relay_worker,
            max_tokens=args.max_tokens,
            max_iterations=args.max_iterations,
            max_parallel_requests=args.max_parallel_requests,
            enable_statistics=args.enable_statistics,
        )
    elif application == "deep_research":
        return create_open_deep_research_scaffolding_llm(
            generation_worker,
            relay_worker,
            max_tokens=args.max_tokens,
            max_parallel_requests=args.max_parallel_requests,
            enable_statistics=args.enable_statistics,
        )
    else:
        raise ValueError(
            f"Unknown agent type: {application}. Supported types: coder, deep_research"
        )


# Lock that serialises lazy ScaffoldingLlm creation so that the first
# two concurrent requests for the same application don't both create one.
_agent_llms_lock = asyncio.Lock()


async def acquire_agent_llm(
    app: web.Application,
    application: str,
) -> ScaffoldingLlm:
    """Return (and lazily create) the server-wide ``ScaffoldingLlm``.

    The ``ScaffoldingLlm`` **must** be created from an async context
    running on the server's event loop so that it reuses that loop
    (``own_loop=False``).  Lazy creation avoids paying startup cost for
    agent types that are never requested.
    """
    agent_llms: Dict[str, ScaffoldingLlm] = app["agent_llms"]
    if application in agent_llms:
        return agent_llms[application]

    async with _agent_llms_lock:
        # Double-check after acquiring the lock.
        if application not in agent_llms:
            agent_llms[application] = _create_agent_llm(
                application,
                app["generation_worker"],
                app["relay_worker"],
                app["args"],
            )
            logger.info(f"Created server-wide ScaffoldingLlm for application={application}")
        return agent_llms[application]


async def process_agent_request(
    ws: web.WebSocketResponse,
    app: web.Application,
    data: dict,
    request_id: str,
):
    """Process a single agent request through ScaffoldingLlm.

    Uses the **server-wide** ``ScaffoldingLlm`` (lazily created on first
    use) and the **server-wide** ``ClientRelayMCPWorker``.  Routing of
    tool calls to the correct WebSocket is handled via
    :pyattr:`ScaffoldingResult.metadata`: the server attaches the
    WebSocket and client request ID to the result object immediately
    after ``generate_async`` returns (synchronous – no yield point),
    and ``call_handler`` reads them from the
    :data:`current_scaffolding_result` context variable.
    """
    application = data.get("application", "coder")
    prompt = data["prompt"]

    await ws.send_json(
        {
            "type": "status",
            "request_id": request_id,
            "message": f"Starting {application} agent...",
        }
    )

    try:
        llm = await acquire_agent_llm(app, application)

        logger.info(f"Processing {application} request (id={request_id}): {prompt[:100]}...")

        # generate_async returns synchronously.  Setting metadata is
        # a plain attribute write (no yield point).  The event loop
        # cannot start processing the queued request until the first
        # await below, so call_handler is guaranteed to see the
        # metadata when it reads current_scaffolding_result.
        scaffolding_result = llm.generate_async(prompt)
        scaffolding_result.metadata["ws"] = ws
        scaffolding_result.metadata["client_request_id"] = request_id

        await scaffolding_result.aresult()

        output_text = (
            scaffolding_result.outputs[0].text
            if scaffolding_result.outputs and scaffolding_result.outputs[0].text
            else None
        )

        await ws.send_json(
            {
                "type": "agent_result",
                "request_id": request_id,
                "output": output_text,
                "error": None,
            }
        )

        logger.info(f"Request {request_id} completed successfully")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Request {request_id} failed: {error_msg}\n{traceback.format_exc()}")
        try:
            await ws.send_json(
                {
                    "type": "agent_result",
                    "request_id": request_id,
                    "output": None,
                    "error": error_msg,
                }
            )
        except Exception:
            pass  # Client may have disconnected


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Handle a single client WebSocket connection.

    All connections share the **server-wide** ``ClientRelayMCPWorker``
    and ``ScaffoldingLlm`` instances (one per application, lazily
    created).  Per-request routing works via
    ``ScaffoldingResult.metadata``: the server attaches ``ws`` and
    ``client_request_id`` to the result object, and the relay worker
    reads them from the ``current_scaffolding_result`` context variable.

    Tool calls from the scaffolding pipeline are sent to the client
    through this WebSocket, and the client's tool results flow back
    through the same connection.
    """
    ws = web.WebSocketResponse(max_msg_size=50 * 1024 * 1024)
    await ws.prepare(request)

    relay_worker: ClientRelayMCPWorker = request.app["relay_worker"]
    client_info = request.remote or "unknown"

    logger.info(f"Client connected: {client_info}")

    active_tasks: set = set()

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {msg.data[:200]}")
                    continue

                msg_type = data.get("type")

                if msg_type == "agent_request":
                    if "prompt" not in data:
                        await ws.send_json(
                            {
                                "type": "error",
                                "message": ("Missing 'prompt' field in agent_request"),
                            }
                        )
                        continue

                    request_id = data.get(
                        "request_id",
                        str(uuid.uuid4()),
                    )

                    # Process in background so we can keep receiving
                    # messages (tool_result messages need to arrive
                    # while processing is in progress).
                    task = asyncio.create_task(
                        process_agent_request(
                            ws,
                            request.app,
                            data,
                            request_id,
                        )
                    )
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)

                elif msg_type == "tool_result":
                    call_id = data.get("call_id")
                    result = data.get("result", "")
                    if call_id:
                        await relay_worker.handle_tool_result(call_id, result)
                    else:
                        logger.warning("Received tool_result without call_id")

                else:
                    logger.warning(f"Unknown message type: {msg_type}")

            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                break

    except Exception as e:
        logger.error(f"Error handling client {client_info}: {e}", exc_info=True)

    finally:
        # Cancel any active request processing tasks.
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

        # Broad safety-net: cancel every pending tool call still tied
        # to this WebSocket (individual process_agent_request tasks
        # already try to unregister in their own finally blocks, but
        # cancellation might have prevented that).
        await relay_worker.cancel_pending_for_connection(ws)

        logger.info(f"Client session ended: {client_info}")

    return ws


def create_app(args: argparse.Namespace) -> web.Application:
    """Create the aiohttp application with the WebSocket endpoint.

    Server-wide singletons created here:

    * ``generation_worker`` – :class:`TRTOpenaiWorker` shared by all
      ``ScaffoldingLlm`` instances.
    * ``relay_worker`` – :class:`ClientRelayMCPWorker` shared by all
      connections; routes tool calls via ``ScaffoldingResult.metadata``.
    * ``agent_llms`` – ``Dict[str, ScaffoldingLlm]``, lazily populated
      on the server's event loop via :func:`acquire_agent_llm`.

    Args:
        args: Parsed command line arguments.

    Returns:
        Configured ``aiohttp.web.Application``.
    """
    openai_client = AsyncOpenAI(
        api_key=args.openai_api_key,
        base_url=args.base_url,
    )
    generation_worker = TRTOpenaiWorker(
        openai_client,
        args.model,
        kv_cache_hint_enabled=args.kv_cache_hint_enabled,
    )

    # Server-wide relay worker – routing data lives on each
    # ScaffoldingResult.metadata, so no per-connection registration.
    relay_worker = ClientRelayMCPWorker()

    app = web.Application()
    app["generation_worker"] = generation_worker
    app["relay_worker"] = relay_worker
    app["agent_llms"] = {}  # lazily populated on first use
    app["args"] = args
    app.router.add_get("/ws", websocket_handler)

    async def on_shutdown(app_: web.Application):
        """Shut down all server-wide ScaffoldingLlm instances."""
        for application, llm in app_["agent_llms"].items():
            logger.info(f"Shutting down server-wide ScaffoldingLlm for application={application}")
            llm.shutdown()
        await app_["relay_worker"].async_shutdown()

    app.on_shutdown.append(on_shutdown)

    return app


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agent Server - single process combining scaffolding, "
            "TensorRT-LLM generation, and tool call relay. Clients "
            "connect via WebSocket; tool calls are relayed back to the "
            "client for local execution through the same connection."
        ),
    )

    # Server binding
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Port to listen on (default: 8090)",
    )

    # OpenAI API connection (for LLM generation)
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="tensorrt_llm",
        help="API key for the OpenAI-compatible server",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the OpenAI-compatible server",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-20b",
        help="Model name to use for generation",
    )

    # Agent settings
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=131072,
        help="Maximum tokens for generation (default: 131072)",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="[Coder only] Maximum tool-calling iterations (default: 50)",
    )
    parser.add_argument(
        "--max_parallel_requests",
        type=int,
        default=16,
        help="Maximum parallel requests per client (default: 16)",
    )
    parser.add_argument(
        "--enable_statistics",
        action="store_true",
        help="Enable task metrics statistics",
    )
    parser.add_argument(
        "--kv_cache_hint_enabled",
        action="store_true",
        help="Enable KV cache hints",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    logger.info("Agent Server configuration:")
    logger.info(f"  Listen: ws://{args.host}:{args.port}/ws")
    logger.info(f"  OpenAI API: {args.base_url}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Max tokens: {args.max_tokens}")
    logger.info(f"  Max iterations (coder): {args.max_iterations}")

    app = create_app(args)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
