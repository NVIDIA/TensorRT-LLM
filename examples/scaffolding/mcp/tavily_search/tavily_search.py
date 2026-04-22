# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server: web search via Tavily API."""

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

mcp = FastMCP("tavily_search")


def _resolve_config_path(config_path: str) -> Path | None:
    """Resolve ``--config`` to an existing file.

    ``uv run --directory <mcp_dir>`` sets *cwd* to that directory, so a path
    relative to the repo root (e.g. ``examples/scaffolding/contrib/...``) does
    not resolve from cwd. Try cwd first, then each ancestor of this file.
    """
    raw = Path(config_path).expanduser()
    if raw.is_file():
        return raw.resolve()
    if raw.is_absolute():
        return None
    rel = Path(config_path)
    here = Path(__file__).resolve().parent
    seen: set[Path] = set()
    for base in (Path.cwd(), here, *here.parents):
        base = base.resolve()
        if base in seen:
            continue
        seen.add(base)
        cand = (base / rel).resolve()
        if cand.is_file():
            return cand
    return None


def _mcp_bind_from_cfg(cfg: dict, service: str, default_port: int) -> tuple[str, int]:
    svc = (cfg.get("mcp_tools") or {}).get(service) or {}
    return str(svc.get("host", "0.0.0.0")), int(svc.get("port", default_port))


def _load_config(config_path: str | None = None) -> dict:
    cfg = {}
    if config_path:
        resolved = _resolve_config_path(config_path)
        if resolved is None:
            LOGGER.warning(
                "Config file not found: %s (cwd=%s). API keys fall back to environment variables.",
                config_path,
                Path.cwd(),
            )
        else:
            with open(resolved) as f:
                cfg = yaml.safe_load(f) or {}
            LOGGER.info("Loaded config from %s", resolved)

    def _get(yaml_key: str, env_key: str, default: str = "") -> str:
        return str(cfg.get(yaml_key, "") or os.getenv(env_key, default))

    mcp_host, mcp_port = _mcp_bind_from_cfg(cfg, "tavily_search", 8087)
    return {
        "search_api_key": _get("tavily_api_key", "TAVILY_API_KEY"),
        "search_api_url": _get("tavily_api_url", "TAVILY_API_URL", "https://api.tavily.com/search"),
        "mcp_host": mcp_host,
        "mcp_port": mcp_port,
    }


_CFG: dict = {
    "search_api_key": os.getenv("TAVILY_API_KEY", ""),
    "search_api_url": os.getenv("TAVILY_API_URL", "https://api.tavily.com/search"),
}


async def _search_single(client: httpx.AsyncClient, query: str) -> str:
    payload = {
        "api_key": _CFG["search_api_key"],
        "query": query,
        "max_results": 10,
    }

    for attempt in range(3):
        try:
            resp = await client.post(
                _CFG["search_api_url"],
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json()
            organic = results.get("results", [])
            if not organic:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return f"No results found for '{query}'."
            snippets = []
            for idx, page in enumerate(organic, 1):
                title = page.get("title", "No title")
                link = page.get("url", "")
                snippet = page.get("content", "")
                date = page.get("published_date", "")
                source = page.get("source", "")
                date_str = f"\nDate published: {date}" if date else ""
                source_str = f"\nSource: {source}" if source else ""
                snippet_str = f"\n{snippet}" if snippet else ""
                snippets.append(f"{idx}. [{title}]({link}){date_str}{source_str}{snippet_str}")
            return (
                f"A Tavily search for '{query}' found "
                f"{len(snippets)} results:\n\n## Web Results\n" + "\n\n".join(snippets)
            )
        except Exception as e:
            if attempt == 2:
                return f"Search failed for '{query}': {e}"
            await asyncio.sleep(2)
    return f"No results found for '{query}'."


@mcp.tool()
async def tavily_search(query: list[str]) -> str:
    """Web search via Tavily API. Returns formatted results for each query."""
    if not _CFG["search_api_key"]:
        return "[tavily_search] Error: search_api_key not configured."
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

    parser = argparse.ArgumentParser(description="Tavily Search MCP SSE-based server")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yaml (API keys, endpoints)"
    )
    parser.add_argument("--host", default=None, help="Host to bind to (overrides config)")
    parser.add_argument(
        "--port", type=int, default=None, help="Port to listen on (overrides config)"
    )
    args = parser.parse_args()

    _CFG.update(_load_config(args.config))

    _has_search_key = bool(str(_CFG.get("search_api_key", "")).strip())
    _key_status = "detected" if _has_search_key else "not detected (empty or missing)"
    if _has_search_key:
        LOGGER.info(
            "Tavily API key (TAVILY_API_KEY / config tavily_api_key): %s",
            _key_status,
        )
    else:
        LOGGER.warning(
            "Tavily API key (TAVILY_API_KEY / config tavily_api_key): %s",
            _key_status,
        )

    host = args.host or _CFG.get("mcp_host", "0.0.0.0")
    port = args.port or _CFG.get("mcp_port", 8087)

    mcp_server = mcp._mcp_server  # noqa: WPS437
    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=host, port=port)
