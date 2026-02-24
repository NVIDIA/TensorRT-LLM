r"""Agent Client - Connect to an Agent Server for remote task execution.

Usage:
    python -m examples.scaffolding.serve.client \\
        --server ws://gpu-server:8090/ws \\
        --application coder \\
        --prompt "Implement a hello world function in Python" \\
        --working_dir /path/to/your/project

    python -m examples.scaffolding.serve.client \\
        --server ws://gpu-server:8090/ws \\
        --application deep_research \\
        --prompt "Analyze the impact of quantum computing on cryptography"

Tool calls from the server are executed locally on this machine.
No tensorrt_llm dependency required.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Set

import aiohttp

from .tools import ToolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("agent_client")


class AgentClient:
    """WebSocket client that connects to an Agent Server.

    Sends agent requests to the server and executes tool calls locally
    using a :class:`ToolExecutor` instance. All communication flows
    through a single WebSocket connection.

    Args:
        server_url: WebSocket URL of the agent server
            (e.g. ``ws://host:8090/ws``).
        working_dir: Working directory for local tool execution.
            Defaults to the current working directory.
    """

    def __init__(
        self,
        server_url: str,
        working_dir: str = "",
    ):
        self.server_url = server_url
        self.tools = ToolExecutor(working_dir or os.getcwd())

    async def run(self, application: str, prompt: str) -> int:
        """Connect to the Agent Server and run an agent task.

        Opens a WebSocket connection, sends an ``agent_request``, and
        processes messages until the server sends an ``agent_result`` or
        the connection is lost.

        Args:
            application: Type of agent (``"coder"`` or ``"deep_research"``).
            prompt: The task prompt for the agent.

        Returns:
            Exit code: 0 for success, 1 for failure.
        """
        logger.info(f"Working directory: {self.tools.working_dir}")
        logger.info(f"Connecting to agent server: {self.server_url}")

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.server_url, max_msg_size=50 * 1024 * 1024) as ws:
                logger.info("Connected to agent server")

                request_id = str(uuid.uuid4())
                request = {
                    "type": "agent_request",
                    "request_id": request_id,
                    "application": application,
                    "prompt": prompt,
                }

                logger.info(f"Sending {application} request: {prompt[:200]}...")
                await ws.send_json(request)

                return await self._process_messages(ws)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    async def _handle_tool_call(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        data: dict,
    ) -> None:
        """Execute a tool call locally and send the result back.

        Args:
            ws: WebSocket connection to send the result on.
            data: Tool call message from the server containing
                ``tool_name``, ``arguments``, and ``call_id``.
        """
        tool_name = data["tool_name"]
        arguments = data["arguments"]
        call_id = data["call_id"]

        logger.info(f"  [Tool] Executing: {tool_name}({json.dumps(arguments)[:200]}...)")

        result = await self.tools.execute(tool_name, arguments)

        logger.info(f"  [Tool] Result ({tool_name}): {str(result)[:200]}...")

        await ws.send_json(
            {
                "type": "tool_result",
                "call_id": call_id,
                "result": result,
            }
        )

    async def _process_messages(
        self,
        ws: aiohttp.ClientWebSocketResponse,
    ) -> int:
        """Process WebSocket messages until agent result or disconnect.

        Handles tool call dispatching (in background tasks for
        concurrency), status updates, and the final result.

        Args:
            ws: Active WebSocket connection.

        Returns:
            Exit code: 0 for success, 1 for failure.
        """
        pending_tool_tasks: Set[asyncio.Task] = set()

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from server: {msg.data[:200]}")
                    continue

                msg_type = data.get("type")

                if msg_type == "tool_call":
                    # Execute in background so the message loop can
                    # keep receiving (e.g. more tool_call messages).
                    task = asyncio.create_task(self._handle_tool_call(ws, data))
                    pending_tool_tasks.add(task)
                    task.add_done_callback(pending_tool_tasks.discard)

                elif msg_type == "status":
                    logger.info(f"[Status] {data.get('message', '')}")

                elif msg_type == "agent_result":
                    # Wait for any in-flight tool tasks to finish
                    if pending_tool_tasks:
                        await asyncio.gather(*pending_tool_tasks, return_exceptions=True)
                    return self._print_result(data)

                elif msg_type == "error":
                    logger.error(f"Server error: {data.get('message', 'Unknown')}")
                    return 1

                else:
                    logger.warning(f"Unknown message type from server: {msg_type}")

            elif msg.type in (
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                logger.error("WebSocket connection lost")
                return 1

        logger.warning("Connection closed before receiving agent result")
        return 1

    @staticmethod
    def _print_result(data: dict) -> int:
        """Print the agent result and return an exit code.

        Args:
            data: The ``agent_result`` message payload.

        Returns:
            0 if the agent succeeded, 1 if it reported an error.
        """
        output = data.get("output")
        error = data.get("error")

        separator = "=" * 60

        if error:
            print(f"\n{separator}")
            print("Agent Error:")
            print(separator)
            print(error)
            print(separator)
            return 1
        else:
            print(f"\n{separator}")
            print("Agent Result:")
            print(separator)
            print(output or "(no output)")
            print(separator)
            return 0


# =====================================================================
# CLI
# =====================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Client for Scaffolding Agent Serving Stack"),
    )
    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8090/ws",
        help=("WebSocket URL of the Agent Server (default: ws://localhost:8090/ws)"),
    )
    parser.add_argument(
        "--application",
        type=str,
        default="coder",
        choices=["coder", "deep_research"],
        help="Type of application to use (default: coder)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt/task for the agent",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default=None,
        help=("Working directory for local tool execution (default: current directory)"),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    client = AgentClient(
        server_url=args.server,
        working_dir=args.working_dir or "",
    )
    exit_code = asyncio.run(client.run(args.application, args.prompt))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
