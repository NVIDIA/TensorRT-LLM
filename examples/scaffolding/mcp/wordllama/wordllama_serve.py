import asyncio
import json
import logging
import time

import uvicorn
from mcp.server.fastmcp import FastMCP

# from tavily import TavilyClient
from wordllama import WordLlama

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

query_keys = []
query_dict = {}
wl = WordLlama.load()

# Default minimum latency for simulating real search (in seconds)
DEFAULT_LATENCY = 1.5

# Fake tavily_search
mcp = FastMCP("tavily_search")


def _sync_search(query: str) -> str:
    """Synchronous search operation - runs in thread pool to avoid blocking event loop."""
    sim_key = wl.key(query)
    best_candidate = max(query_keys, key=sim_key)
    return query_dict[best_candidate]


@mcp.tool()
async def web_search(query: str) -> str:
    start_time = time.monotonic()

    # Run CPU-bound operations in thread pool to avoid blocking the event loop
    result = await asyncio.to_thread(_sync_search, query)

    # Ensure minimum latency (simulate real search latency)
    elapsed = time.monotonic() - start_time
    remaining = DEFAULT_LATENCY - elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MCP Streamable HTTP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082, help="Port to listen on")
    parser.add_argument(
        "--query_file", type=str, default="query_result.json", help="File containing query keys"
    )
    args = parser.parse_args()

    with open(args.query_file, "r") as file:
        query_dict = json.load(file)
        query_keys = list(query_dict.keys())

    starlette_app = mcp.streamable_http_app()

    uvicorn.run(starlette_app, host=args.host, port=args.port)
