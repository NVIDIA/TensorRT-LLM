import os

import uvicorn
from brave import Brave
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

# Initialize FastMCP server for Weather tools (SSE)
mcp = FastMCP("websearch")

# Load environment variables
load_dotenv()


@mcp.tool()
async def websearch(query: str) -> str:
    """Web search, fetch information from the internet
    Args:
        query:  string of what you want to search
    """
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
    brave = Brave(BRAVE_API_KEY)
    print(f"brave apikey {BRAVE_API_KEY }")
    search_results = brave.search(q=query, raw=True)
    return f"{search_results}"


def create_starlette_app(mcp_server: Server,
                         *,
                         debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provided mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
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

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse

    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port',
                        type=int,
                        default=8082,
                        help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)
