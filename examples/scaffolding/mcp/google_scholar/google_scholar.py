"""MCP server: Google Scholar + web search via SerpAPI."""

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

mcp = FastMCP("google_scholar")


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

    mcp_host, mcp_port = _mcp_bind_from_cfg(cfg, "google_scholar", 8084)
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


async def _scholar_single(client: httpx.AsyncClient, query: str) -> str:
    params = {
        "api_key": _CFG["search_api_key"],
        "engine": "google_scholar",
        "q": query,
        "num": 10,
        "hl": "en",
    }
    for attempt in range(3):
        try:
            resp = await client.get(_CFG["search_api_url"], params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json()
            organic = results.get("organic_results", [])
            if not organic:
                return f"No scholar results for '{query}'."
            snippets = []
            for idx, page in enumerate(organic, 1):
                title = page.get("title", "No title")
                link = page.get("link", "no available link")
                snippet = page.get("snippet", "")
                pub_info = page.get("publication_info", {}).get("summary", "")
                cited_by = page.get("inline_links", {}).get("cited_by", {}).get("total", "")
                resources = page.get("resources", [])
                pdf_link = ""
                for res in resources:
                    if res.get("file_format") == "PDF":
                        pdf_link = res.get("link", "")
                        break
                link_info = f"pdfUrl: {pdf_link}" if pdf_link else f"link: {link}"
                pub_str = f"\npublicationInfo: {pub_info}" if pub_info else ""
                cite_str = f"\ncitedBy: {cited_by}" if cited_by else ""
                snippet_str = f"\n{snippet}" if snippet else ""
                snippets.append(f"{idx}. [{title}]({link_info}){pub_str}{cite_str}{snippet_str}")
            return (
                f"A Google Scholar search for '{query}' found "
                f"{len(snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(snippets)
            )
        except Exception as e:
            if attempt == 2:
                return f"Scholar search failed for '{query}': {e}"
            await asyncio.sleep(1)
    return f"No scholar results for '{query}'."


@mcp.tool()
async def google_scholar(query: list[str]) -> str:
    """Scholar search via SerpAPI + web search combined results."""
    if not _CFG["search_api_key"]:
        return "[google_scholar] Error: search_api_key not configured."
    async with httpx.AsyncClient() as client:
        scholar_tasks = [_scholar_single(client, q) for q in query]
        search_tasks = [_search_single(client, q) for q in query]
        all_results = await asyncio.gather(*(scholar_tasks + search_tasks))
    n = len(query)
    combined = []
    for i in range(n):
        combined.append(f"{all_results[i]}\n\n{all_results[n + i]}")
    return "\n=======\n".join(combined)


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

    parser = argparse.ArgumentParser(description="Google Scholar MCP SSE-based server")
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
    port = args.port or _CFG.get("mcp_port", 8084)

    mcp_server = mcp._mcp_server  # noqa: WPS437
    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=host, port=port)
