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

"""MCP server implementing IterResearch tools.

Provides 4 tools:
- google_search: Web search via SerpAPI
- google_scholar: Scholar search via SerpAPI + web search combined
- fetch_webpage: Fetch raw webpage/PDF content via Jina Reader / ScraperAPI
- python_interpreter: Execute Python code in sandboxed environment

Start the server:
    # Fill in API keys in config.yaml, then:
    uv run iter_research_tools.py --config ../config.yaml

    # Or use environment variables:
    export SEARCH_API_KEY=... JINA_API_KEY=...
    uv run iter_research_tools.py --port 8083
"""

import asyncio
import logging
import os
import re
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

mcp = FastMCP("iter_research_tools")


# ---------------------------------------------------------------------------
# Configuration: YAML file > environment variables > defaults
# ---------------------------------------------------------------------------
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
        "search_api_key": _get("search_api_key", "SEARCH_API_KEY"),
        "search_api_url": _get("search_api_url", "SEARCH_API_URL",
                               "https://serpapi.com/search"),
        "jina_api_key": _get("jina_api_key", "JINA_API_KEY"),
        "scraper_api_key": _get("scraper_api_key", "SCRAPER_API_KEY"),
        "sandbox_endpoint": _get("sandbox_endpoint", "SANDBOX_ENDPOINT",
                                 "http://127.0.0.1:8080"),
        "mcp_host": _get("mcp_host", "MCP_HOST", "0.0.0.0"),
        "mcp_port": int(_get("mcp_port", "MCP_PORT", "8083")),
    }


# Populated at startup in __main__; tools read from this dict.
_CFG: dict = {
    "search_api_key": os.getenv("SEARCH_API_KEY", ""),
    "search_api_url": os.getenv("SEARCH_API_URL",
                                "https://serpapi.com/search"),
    "jina_api_key": os.getenv("JINA_API_KEY", ""),
    "scraper_api_key": os.getenv("SCRAPER_API_KEY", ""),
    "sandbox_endpoint": os.getenv("SANDBOX_ENDPOINT",
                                  "http://127.0.0.1:8080"),
}


# ---------------------------------------------------------------------------
# google_search
# ---------------------------------------------------------------------------
async def _search_single(client: httpx.AsyncClient, query: str) -> str:
    """Perform a single SerpAPI web search."""
    contains_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
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
                snippets.append(
                    f"{idx}. [{title}]({link}){date_str}{source_str}"
                    f"{snippet_str}")
            return (f"A Google search for '{query}' found "
                    f"{len(snippets)} results:\n\n## Web Results\n"
                    + "\n\n".join(snippets))
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


# ---------------------------------------------------------------------------
# google_scholar
# ---------------------------------------------------------------------------
async def _scholar_single(client: httpx.AsyncClient, query: str) -> str:
    """Perform a single SerpAPI Google Scholar search."""
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
                pub_info = page.get("publication_info",
                                    {}).get("summary", "")
                cited_by = (page.get("inline_links",
                                     {}).get("cited_by",
                                             {}).get("total", ""))
                resources = page.get("resources", [])
                pdf_link = ""
                for res in resources:
                    if res.get("file_format") == "PDF":
                        pdf_link = res.get("link", "")
                        break
                link_info = (f"pdfUrl: {pdf_link}"
                             if pdf_link else f"link: {link}")
                pub_str = (f"\npublicationInfo: {pub_info}"
                           if pub_info else "")
                cite_str = (f"\ncitedBy: {cited_by}"
                            if cited_by else "")
                snippet_str = f"\n{snippet}" if snippet else ""
                snippets.append(
                    f"{idx}. [{title}]({link_info}){pub_str}{cite_str}"
                    f"{snippet_str}")
            return (f"A Google Scholar search for '{query}' found "
                    f"{len(snippets)} results:\n\n## Scholar Results\n"
                    + "\n\n".join(snippets))
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


# ---------------------------------------------------------------------------
# fetch_webpage
# ---------------------------------------------------------------------------
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
                tmp_dir = tempfile.mkdtemp(prefix="iter_research_pdf_")
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


# ---------------------------------------------------------------------------
# python_interpreter
# ---------------------------------------------------------------------------
def _extract_code(code: str) -> str:
    """Extract code from markdown code blocks if present."""
    triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
    if triple_match:
        return triple_match.group(1)
    code_match = re.search(r'<code>(.*?)</code>', code, re.DOTALL)
    if code_match:
        return code_match.group(1)
    return code


@mcp.tool()
async def python_interpreter(code: str) -> str:
    """Execute Python code in sandboxed environment."""
    code = _extract_code(code)
    if not code.strip():
        return "[PythonInterpreter Error]: Empty code."
    if not _CFG["sandbox_endpoint"]:
        return "[PythonInterpreter Error]: sandbox_endpoint not configured."
    payload = {
        "code": code,
        "language": "python",
        "run_timeout": 50,
    }
    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                resp = await client.post(f"{_CFG["sandbox_endpoint"]}/run",
                                         json=payload,
                                         timeout=60)
                resp.raise_for_status()
                result = resp.json()
                parts = []
                if result.get("stdout"):
                    parts.append(f"stdout:\n{result['stdout']}")
                if result.get("stderr"):
                    parts.append(f"stderr:\n{result['stderr']}")
                exec_time = result.get("execution_time", 0)
                if exec_time >= 49:
                    parts.append(
                        "[PythonInterpreter Error] TimeoutError: "
                        "Execution timed out.")
                output = '\n'.join(parts)
                return output if output.strip() else "Finished execution."
            except Exception as e:
                if attempt == 2:
                    return f"[PythonInterpreter Error]: {e}"
                await asyncio.sleep(1)
    return "[PythonInterpreter Error]: All attempts failed."


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------
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
        description="IterResearch MCP SSE-based server")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (API keys, endpoints)")
    parser.add_argument("--host", default=None,
                        help="Host to bind to (overrides config)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to listen on (overrides config)")
    args = parser.parse_args()

    # Load config from YAML, falling back to env vars / defaults
    _CFG.update(_load_config(args.config))

    host = args.host or _CFG.get("mcp_host", "0.0.0.0")
    port = args.port or _CFG.get("mcp_port", 8083)

    mcp_server = mcp._mcp_server  # noqa: WPS437
    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=host, port=port)
