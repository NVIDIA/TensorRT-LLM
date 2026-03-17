# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCP server for fetching webpage/PDF content via Jina Reader / ScraperAPI.

Start the server:
    export JINA_API_KEY=...
    uv run fetch_webpage.py --port 8085
"""

import asyncio
import logging
import os
import tempfile
import uuid
from datetime import datetime
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

mcp = FastMCP("fetch_webpage")


def _load_config(config_path: str | None = None) -> dict:
    """Load configuration from a YAML file.

    Falls back to environment variables for any key not present in the
    YAML file.
    """
    cfg = {}
    if config_path and Path(config_path).is_file():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        LOGGER.info("Loaded config from %s", config_path)

    def _get(yaml_key: str, env_key: str, default: str = "") -> str:
        return str(cfg.get(yaml_key, "") or os.getenv(env_key, default))

    return {
        "jina_api_key": _get("jina_api_key", "JINA_API_KEY"),
        "scraper_api_key": _get("scraper_api_key", "SCRAPER_API_KEY"),
        "mcp_host": _get("mcp_host", "MCP_HOST", "0.0.0.0"),
        "mcp_port": int(_get("mcp_port", "MCP_PORT", "8085")),
    }


_CFG: dict = {
    "jina_api_key": os.getenv("JINA_API_KEY", ""),
    "scraper_api_key": os.getenv("SCRAPER_API_KEY", ""),
}


async def _fetch_via_jina(client: httpx.AsyncClient, url: str) -> str:
    """Fetch webpage content using Jina Reader API."""
    if not _CFG["jina_api_key"]:
        # Fallback: direct fetch
        try:
            headers = {
                "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            resp = await client.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception:
            return ""
    headers = {"Authorization": f"Bearer {_CFG["jina_api_key"]}"}
    for attempt in range(3):
        try:
            resp = await client.get(f"https://r.jina.ai/{url}",
                                    headers=headers,
                                    timeout=60)
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass
        await asyncio.sleep(2)
    return ""


async def _fetch_via_scraper(client: httpx.AsyncClient,
                             url: str) -> tuple[str, str]:
    """Fetch via ScraperAPI, returns (content_type, content_or_path)."""
    if not _CFG["scraper_api_key"]:
        return ('html', '')
    params = {
        'api_key': _CFG["scraper_api_key"],
        'url': url,
        'device_type': 'desktop',
        'country_code': 'us',
        'output_format': 'markdown',
    }
    for attempt in range(3):
        try:
            resp = await client.get('https://api.scraperapi.com/',
                                    params=params,
                                    timeout=60)
            resp.raise_for_status()
            ct = resp.headers.get('Content-Type', '').lower()
            if 'pdf' in ct:
                tmp_dir = tempfile.mkdtemp(prefix="fetch_webpage_pdf_")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(tmp_dir,
                                    f"doc_{ts}_{uuid.uuid4()}.pdf")
                with open(path, 'wb') as f:
                    f.write(resp.content)
                return ('pdf', path)
            if 'text' in ct:
                return ('html', resp.text)
            return ('other', '')
        except Exception:
            if attempt == 2:
                return ('other', '')
            await asyncio.sleep(1)
    return ('other', '')


def _parse_pdf(pdf_path: str) -> str:
    """Parse a PDF file using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "[PDF Parser Error]: PyMuPDF not installed. pip install pymupdf"
    try:
        doc = fitz.open(pdf_path)
        parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                parts.append(f"## Page {page_num + 1}\n\n{text}")
        doc.close()
        return '\n\n'.join(parts)
    except Exception as e:
        return f"[PDF Parser Error]: {e}"


async def _fetch_single_url(client: httpx.AsyncClient, url: str,
                            parse_type: str) -> str:
    """Fetch content from a single URL."""
    # Try ScraperAPI first
    content_type, content = await _fetch_via_scraper(client, url)
    if content_type == 'pdf':
        return await asyncio.to_thread(_parse_pdf, content)
    if content_type == 'html' and content:
        return content
    # Fallback to Jina
    content = await _fetch_via_jina(client, url)
    if content:
        return content
    return f"[fetch_webpage] Failed to fetch: {url}"


@mcp.tool()
async def fetch_webpage(url: list[str],
                        parse_type: str = "html") -> str:
    """Fetch raw webpage/PDF content via Jina Reader / ScraperAPI."""
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_single_url(client, u, parse_type) for u in url]
        results = await asyncio.gather(*tasks)
    return "\n=======\n".join(results)


def create_starlette_app(mcp_server: Server,
                         *,
                         debug: bool = False) -> Starlette:
    """Create a Starlette app serving the MCP server via SSE."""
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

    parser = argparse.ArgumentParser(
        description="Fetch Webpage MCP SSE-based server")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (API keys, endpoints)")
    parser.add_argument("--host", default=None,
                        help="Host to bind to (overrides config)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to listen on (overrides config)")
    args = parser.parse_args()

    _CFG.update(_load_config(args.config))

    host = args.host or _CFG.get("mcp_host", "0.0.0.0")
    port = args.port or _CFG.get("mcp_port", 8085)

    mcp_server = mcp._mcp_server  # noqa: WPS437
    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=host, port=port)
