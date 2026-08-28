# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Search backends for the ``web_search`` server tool.

Web search is a *server* tool on every API that offers it: the client sends
the tool definition and expects the server to run the query itself and feed
the results back to the model within a single exchange. Each API spells the
tool differently - the Responses API calls it
``web_search``/``web_search_preview``, others use a dated name such as
``web_search_20250305`` - but the work behind them is the same, so it lives
here rather than in the calling endpoint. Today the only caller is the
Responses API (see responses_web_search.py).

Nothing in this module knows which endpoint asked. An endpoint contributes
two things: how it recognises the tool in a request (see WebSearchToolSpec)
and how it renders a completed search back to its own client. Everything
between - provider selection, the query itself, domain filtering and the
per-request budget - is shared.

No provider is enabled by default. Deployments select one through
``TRTLLM_WEB_SEARCH``; until they do, the tool keeps returning the same
"not supported by this server" error as before, so enabling the feature is
always an explicit, auditable decision.

Providers
---------

=========== ============================ ==========================
provider    credentials                  notes
=========== ============================ ==========================
``off``       -                          default; feature disabled
``wikipedia`` none                       official MediaWiki search API;
                                         encyclopaedic scope only, but
                                         reliable and rate-limit friendly
``mojeek``    none                       general web, scraped from HTML.
                                         Unauthenticated use is rate
                                         limited and starts returning 403
                                         under sustained load - fine for
                                         a trial, not for a shared server
``brave``     ``BRAVE_SEARCH_API_KEY``   general web, JSON API
``tavily``    ``TAVILY_API_KEY``         general web, JSON API, LLM-oriented
``searxng``   ``SEARXNG_URL``            self-hosted JSON endpoint
=========== ============================ ==========================

Environment
-----------

``TRTLLM_WEB_SEARCH``              provider name (default ``off``)
``TRTLLM_WEB_SEARCH_MAX_RESULTS``  results per query (default 5)
``TRTLLM_WEB_SEARCH_TIMEOUT_S``    per-query timeout (default 15)
``TRTLLM_WEB_SEARCH_MAX_USES``     hard cap on searches per request
                                   (default 5); a client asking for more is
                                   clamped, so a prompt cannot make the
                                   server issue unbounded outbound traffic
"""

import asyncio
import html
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence
from urllib.parse import quote_plus, urlparse

import aiohttp

from tensorrt_llm.logger import logger

DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_USES = 5
DEFAULT_RETRIES = 2
DEFAULT_RETRY_BACKOFF_S = 0.5

# Mojeek returns plain HTML; results are <a class="title" href="..."> anchors
# followed by a <p class="s"> snippet. Parsed with regexes rather than an HTML
# parser to avoid adding a dependency for one provider.
_MOJEEK_RESULT_RE = re.compile(
    r'<a class="title"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_MOJEEK_SNIPPET_RE = re.compile(r'<p class="s">(?P<snippet>.*?)</p>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")

_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) TensorRT-LLM/web-search"


class WebSearchError(RuntimeError):
    """A search query could not be completed."""


@dataclass
class WebSearchResult:
    url: str
    title: str
    snippet: str = ""
    page_age: Optional[str] = None


@dataclass
class WebSearchConfig:
    provider: str = "off"
    max_results: int = DEFAULT_MAX_RESULTS
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_uses: int = DEFAULT_MAX_USES
    retries: int = DEFAULT_RETRIES
    retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    # Populated from the request's tool definition, not the environment.
    allowed_domains: Sequence[str] = field(default_factory=tuple)
    blocked_domains: Sequence[str] = field(default_factory=tuple)

    @property
    def enabled(self) -> bool:
        return self.provider != "off"


def _env(suffix: str) -> Optional[str]:
    """Read one ``TRTLLM_WEB_SEARCH*`` setting."""
    return os.environ.get(f"TRTLLM_WEB_SEARCH{suffix}")


def _env_float(suffix: str, default: float) -> float:
    raw = _env(suffix)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("web search %s=%r is not a number; using %s", suffix, raw, default)
        return default


def _env_int(suffix: str, default: int) -> int:
    raw = _env(suffix)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("web search %s=%r is not an integer; using %s", suffix, raw, default)
        return default


def load_web_search_config() -> WebSearchConfig:
    """Read the web-search configuration from the environment.

    Called per request so a provider can be switched without a restart; the
    work is a handful of ``os.environ`` lookups.
    """
    provider = (_env("") or "off").strip().lower()
    config = WebSearchConfig(
        provider=provider,
        max_results=_env_int("_MAX_RESULTS", DEFAULT_MAX_RESULTS),
        timeout_s=_env_float("_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        max_uses=_env_int("_MAX_USES", DEFAULT_MAX_USES),
        retries=_env_int("_RETRIES", DEFAULT_RETRIES),
    )
    if provider == "brave":
        config.api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        config.endpoint = "https://api.search.brave.com/res/v1/web/search"
    elif provider == "tavily":
        config.api_key = os.environ.get("TAVILY_API_KEY")
        config.endpoint = "https://api.tavily.com/search"
    elif provider == "searxng":
        config.endpoint = os.environ.get("SEARXNG_URL")
    return config


def validate_web_search_config(config: WebSearchConfig) -> Optional[str]:
    """Return an error string if the selected provider cannot run, else None."""
    if not config.enabled:
        return None
    if config.provider not in _PROVIDERS:
        return (
            f"unknown web search provider {config.provider!r}; expected one of "
            "off, " + ", ".join(sorted(_PROVIDERS))
        )
    if config.provider == "brave" and not config.api_key:
        return "web search provider 'brave' requires BRAVE_SEARCH_API_KEY"
    if config.provider == "tavily" and not config.api_key:
        return "web search provider 'tavily' requires TAVILY_API_KEY"
    if config.provider == "searxng" and not config.endpoint:
        return "web search provider 'searxng' requires SEARXNG_URL"
    return None


def _strip_html(raw: str) -> str:
    return html.unescape(_TAG_RE.sub("", raw)).strip()


def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _domain_matches(domain: str, pattern: str) -> bool:
    pattern = pattern.strip().lower().lstrip(".")
    if not pattern:
        return False
    return domain == pattern or domain.endswith("." + pattern)


def filter_results(
    results: Sequence[WebSearchResult], config: WebSearchConfig
) -> List[WebSearchResult]:
    """Apply the request's allowed/blocked domain lists.

    The two lists are meant to be mutually exclusive; if a caller sends both,
    ``allowed_domains`` wins because it is the more restrictive intent.
    """
    filtered: List[WebSearchResult] = []
    for result in results:
        domain = _domain_of(result.url)
        if not domain:
            continue
        if config.allowed_domains:
            if not any(_domain_matches(domain, p) for p in config.allowed_domains):
                continue
        elif config.blocked_domains:
            if any(_domain_matches(domain, p) for p in config.blocked_domains):
                continue
        filtered.append(result)
    return filtered


async def _fetch(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    **kwargs: Any,
) -> str:
    async with session.request(method, url, **kwargs) as response:
        body = await response.text()
        if response.status >= 400:
            raise WebSearchError(f"search backend returned HTTP {response.status}")
        return body


async def _search_mojeek(
    session: aiohttp.ClientSession, query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    url = f"https://www.mojeek.com/search?q={quote_plus(query)}"
    body = await _fetch(session, "GET", url)
    titles = list(_MOJEEK_RESULT_RE.finditer(body))
    snippets = [m.group("snippet") for m in _MOJEEK_SNIPPET_RE.finditer(body)]
    results: List[WebSearchResult] = []
    for index, match in enumerate(titles):
        snippet = _strip_html(snippets[index]) if index < len(snippets) else ""
        results.append(
            WebSearchResult(
                url=html.unescape(match.group("url")),
                title=_strip_html(match.group("title")),
                snippet=snippet,
            )
        )
    return results


async def _search_brave(
    session: aiohttp.ClientSession, query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    body = await _fetch(
        session,
        "GET",
        config.endpoint,
        params={"q": query, "count": config.max_results},
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": config.api_key or "",
        },
    )
    payload = json.loads(body)
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=_strip_html(item.get("title", "")),
            snippet=_strip_html(item.get("description", "")),
            page_age=item.get("page_age"),
        )
        for item in (payload.get("web", {}).get("results") or [])
    ]


async def _search_tavily(
    session: aiohttp.ClientSession, query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    body = await _fetch(
        session,
        "POST",
        config.endpoint,
        json={
            "api_key": config.api_key,
            "query": query,
            "max_results": config.max_results,
        },
        headers={"Content-Type": "application/json"},
    )
    payload = json.loads(body)
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=_strip_html(item.get("title", "")),
            snippet=_strip_html(item.get("content", "")),
        )
        for item in (payload.get("results") or [])
    ]


async def _search_searxng(
    session: aiohttp.ClientSession, query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    body = await _fetch(
        session,
        "GET",
        config.endpoint.rstrip("/") + "/search",
        params={"q": query, "format": "json"},
    )
    payload = json.loads(body)
    return [
        WebSearchResult(
            url=item.get("url", ""),
            title=_strip_html(item.get("title", "")),
            snippet=_strip_html(item.get("content", "")),
        )
        for item in (payload.get("results") or [])
    ]


async def _search_wikipedia(
    session: aiohttp.ClientSession, query: str, config: WebSearchConfig
) -> List[WebSearchResult]:
    body = await _fetch(
        session,
        "GET",
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": config.max_results,
            "format": "json",
        },
    )
    payload = json.loads(body)
    results = []
    for item in payload.get("query", {}).get("search", []) or []:
        title = item.get("title", "")
        results.append(
            WebSearchResult(
                url="https://en.wikipedia.org/wiki/" + quote_plus(title.replace(" ", "_")),
                title=title,
                snippet=_strip_html(item.get("snippet", "")),
                page_age=item.get("timestamp"),
            )
        )
    return results


_PROVIDERS = {
    "mojeek": _search_mojeek,
    "wikipedia": _search_wikipedia,
    "brave": _search_brave,
    "tavily": _search_tavily,
    "searxng": _search_searxng,
}


async def run_web_search(query: str, config: WebSearchConfig) -> List[WebSearchResult]:
    """Run one search query and return filtered, truncated results.

    Raises ``WebSearchError`` on any transport or backend failure; the caller
    turns that into a ``web_search_tool_result`` error block so the model can
    carry on rather than the whole request failing.
    """
    provider = _PROVIDERS.get(config.provider)
    if provider is None:
        raise WebSearchError(f"web search provider {config.provider!r} is not available")
    if not query or not query.strip():
        raise WebSearchError("web search query is empty")

    timeout = aiohttp.ClientTimeout(total=config.timeout_s)
    # Keyless providers drop connections intermittently - measured around half
    # of requests on one cluster - and a dropped connection is indistinguishable
    # from "no results" once it reaches the model. Retry transport failures so a
    # flaky hop does not silently become a wrong answer.
    last_error: Optional[Exception] = None
    for attempt in range(config.retries + 1):
        if attempt:
            await asyncio.sleep(config.retry_backoff_s * attempt)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, headers={"User-Agent": _USER_AGENT}
            ) as session:
                results = await provider(session, query, config)
            break
        except (aiohttp.ClientError, WebSearchError) as e:
            last_error = e
        except json.JSONDecodeError as e:
            raise WebSearchError(f"search backend returned invalid JSON: {e}") from e
        except (TimeoutError, asyncio.TimeoutError) as e:
            last_error = e
        logger.warning(
            "web search attempt %d/%d failed: %s",
            attempt + 1,
            config.retries + 1,
            last_error,
        )
    else:
        raise WebSearchError(
            f"search backend failed after {config.retries + 1} attempts: {last_error}"
        )

    results = [r for r in results if r.url]
    results = filter_results(results, config)
    return results[: config.max_results]


def results_as_model_text(query: str, results: Sequence[WebSearchResult]) -> str:
    """Render results as the tool-result text handed back to the model."""
    if not results:
        return f'No results found for "{query}".'
    lines = [f'Search results for "{query}":', ""]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}")
        lines.append(f"   URL: {result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet}")
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# The endpoint-neutral seam
# ---------------------------------------------------------------------------


@dataclass
class WebSearchToolSpec:
    """What an endpoint tells this module about a requested web_search tool.

    An API may spell the tool with a version suffix or carry its own limits -
    the Responses API sends ``web_search``/``web_search_preview`` - but all
    any of them contribute is a name and some limits. Reducing a request's
    tool to this keeps the rest of the module from importing an endpoint's
    types.
    """

    name: str
    type: Optional[str] = None
    max_uses: Optional[int] = None
    allowed_domains: Sequence[str] = field(default_factory=tuple)
    blocked_domains: Sequence[str] = field(default_factory=tuple)


def resolve_web_search(
    spec: Optional[WebSearchToolSpec],
) -> tuple[Optional[WebSearchConfig], Optional[str]]:
    """Resolve the server's web-search configuration for one request.

    Returns ``(config, error)``. ``config`` is None when the request did not
    ask for web search. ``error`` is set when it did but the server cannot
    honour it, and the caller should reject the request rather than drop the
    tool: dropping it silently would make the model answer from stale
    parametric knowledge while the client believed a live search had run.

    A client's ``max_uses`` may lower the server's cap but never raise it, so
    a prompt cannot talk the server into unbounded outbound traffic.
    """
    if spec is None:
        return None, None

    config = load_web_search_config()
    if not config.enabled:
        return None, (
            f"server tool {spec.name!r} is not supported by this server: web "
            "search is disabled. Set TRTLLM_WEB_SEARCH to a provider "
            "(wikipedia, mojeek, brave, tavily, searxng) to enable it."
        )

    error = validate_web_search_config(config)
    if error is not None:
        return None, f"server tool {spec.name!r} cannot run: {error}"

    if spec.max_uses is not None and spec.max_uses > 0:
        config.max_uses = min(config.max_uses, spec.max_uses)
    config.allowed_domains = tuple(spec.allowed_domains or ())
    config.blocked_domains = tuple(spec.blocked_domains or ())
    return config, None


@dataclass
class SearchOutcome:
    """One completed search, however it turned out."""

    query: str
    results: List[WebSearchResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_model_text(self) -> str:
        """What the model is shown for this search, on any endpoint."""
        if self.error is not None:
            return f'Web search for "{self.query}" failed: {self.error}'
        return results_as_model_text(self.query, self.results)


class WebSearchSession:
    """Runs the searches for one request and holds the budget.

    The budget belongs here rather than in a caller's loop because every
    endpoint needs the same rule and gets it wrong in the same way: a model
    that keeps asking to search must be told it has run out, not silently
    ignored, or it answers as though the search had returned nothing.
    """

    def __init__(self, config: WebSearchConfig):
        self.config = config
        self.searches_done = 0

    @property
    def exhausted(self) -> bool:
        return self.searches_done >= self.config.max_uses

    @property
    def budget_notice(self) -> str:
        return (
            f"The web search budget for this request is used up "
            f"({self.config.max_uses} searches). Answer with what you "
            f"already have."
        )

    async def run(self, query: str) -> SearchOutcome:
        """Execute one search, counting it against the budget.

        A failure is returned rather than raised: the model is told the search
        failed and can carry on, which is a better answer than a 500.
        """
        if self.exhausted:
            return SearchOutcome(query=query, error="search budget exhausted")
        self.searches_done += 1
        try:
            results = await run_web_search(query, self.config)
        except WebSearchError as exc:
            logger.warning("web search for %r failed: %s", query, exc)
            return SearchOutcome(query=query, error=str(exc))
        return SearchOutcome(query=query, results=list(results))
