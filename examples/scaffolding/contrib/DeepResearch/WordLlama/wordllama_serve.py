import uvicorn
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
#from tavily import TavilyClient
from wordllama import WordLlama
import json
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

query_keys = []
query_dict = {}
wl = WordLlama.load()

# Initialize FastMCP server for Weather tools (SSE)
# Fake tavily_search
mcp = FastMCP("tavily_search")


@mcp.tool()
async def tavily_search(query: str) -> str:
    sim_key = wl.key(query)

    # Most similar candidate
    best_candidate = max(query_keys, key=sim_key)

    return query_dict[best_candidate]


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provided mcp server with SSE."""
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
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse

    parser = argparse.ArgumentParser(description="Run MCP SSE-based server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082, help="Port to listen on")
    parser.add_argument("--query_file", type=str, default="query_result.json", help="File containing query keys")
    args = parser.parse_args()

    with open(args.query_file, "r") as file:
        query_dict = json.load(file)
        query_keys = list(query_dict.keys())

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)