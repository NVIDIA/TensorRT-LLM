"""MCP server: Google web search via SerpAPI."""

import asyncio
import logging
import os
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

mcp = FastMCP("google_search")


def _mcp_bind_from_cfg(cfg: dict, service: str, default_port: int) -> tuple[str, int]:
    svc = (cfg.get("mcp_tools") or {}).get(service) or {}
    return str(svc.get("host", "0.0.0.0")), int(svc.get("port", default_port))


def _load_config(config_path: str | None = None) -> dict:
    cfg = {}
    if config_path and Path(config_path).is_file():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        LOGGER.info("Loaded config from %s", config_path)

    def _get(yaml_key: str, env_key: str, default: str = "") -> str:
        return str(cfg.get(yaml_key, "") or os.getenv(env_key, default))

    mcp_host, mcp_port = _mcp_bind_from_cfg(cfg, "google_search", 8083)
    return {
        "search_api_key": _get("search_api_key", "SEARCH_API_KEY"),
        "search_api_url": _get("search_api_url", "SEARCH_API_URL", "https://serpapi.com/search"),
        "mcp_host": mcp_host,
        "mcp_port": mcp_port,
    }


_CFG: dict = {
    "search_api_key": os.getenv("SEARCH_API_KEY", ""),
    "search_api_url": os.getenv("SEARCH_API_URL", "https://serpapi.com/search"),
}


async def _search_single(client: httpx.AsyncClient, query: str) -> str:
    contains_chinese = any("\u4e00" <= c <= "\u9fff" for c in query)
    params = {
        "api_key": _CFG["search_api_key"],
        "q": query,
        "num": 10,
        "hl": "zh-CN" if contains_chinese else "en",
        "gl": "CN" if contains_chinese else "US",
    }
    for attempt in range(3):
        try:
            resp = await client.get(_CFG["search_api_url"], params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()
            organic = results.get("organic_results", [])
            if not organic:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return f"No results found for '{query}'."
            snippets = []
            for idx, page in enumerate(organic, 1):
                title = page.get("title", "No title")
                link = page.get("link", "")
                snippet = page.get("snippet", "")
                date = page.get("date", "")
                source = page.get("source", "")
                date_str = f"\nDate published: {date}" if date else ""
                source_str = f"\nSource: {source}" if source else ""
                snippet_str = f"\n{snippet}" if snippet else ""
                snippets.append(f"{idx}. [{title}]({link}){date_str}{source_str}{snippet_str}")
            return (
                f"A Google search for '{query}' found "
                f"{len(snippets)} results:\n\n## Web Results\n" + "\n\n".join(snippets)
            )
        except Exception as e:
            if attempt == 2:
                return f"Search failed for '{query}': {e}"
            await asyncio.sleep(2)
    return f"No results found for '{query}'."


@mcp.tool()
async def google_search(query: list[str]) -> str:
    """Web search via SerpAPI. Returns formatted results for each query."""
    if not _CFG["search_api_key"]:
        return "[google_search] Error: search_api_key not configured."
    async with httpx.AsyncClient() as client:
        tasks = [_search_single(client, q) for q in query]
        results = await asyncio.gather(*tasks)
    return "\n=======\n".join(results)


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
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

    parser = argparse.ArgumentParser(description="Google Search MCP SSE-based server")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yaml (API keys, endpoints)"
    )
    parser.add_argument("--host", default=None, help="Host to bind to (overrides config)")
    parser.add_argument(
        "--port", type=int, default=None, help="Port to listen on (overrides config)"
    )
    args = parser.parse_args()

    _CFG.update(_load_config(args.config))

    host = args.host or _CFG.get("mcp_host", "0.0.0.0")
    port = args.port or _CFG.get("mcp_port", 8083)

    mcp_server = mcp._mcp_server  # noqa: WPS437
    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=host, port=port)
