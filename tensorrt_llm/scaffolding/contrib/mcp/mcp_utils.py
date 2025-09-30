import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

load_dotenv()  # load environment variables from .env


class MCPClient:

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def list_tools(self):
        response = await self.session.list_tools()
        return response

    async def call_tool(self, tool_name, tool_args):
        result = await self.session.call_tool(tool_name, tool_args)
        return result

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        streams_context = sse_client(url=server_url)
        streams = await self.exit_stack.enter_async_context(streams_context)

        session_context = ClientSession(*streams)
        self.session = await self.exit_stack.enter_async_context(session_context
                                                                 )

        # Initialize session
        await self.session.initialize()

    async def cleanup(self):
        """Properly clean up all registered async resources."""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print(
            "Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>"
        )
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
