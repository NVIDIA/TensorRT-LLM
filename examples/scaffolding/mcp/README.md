# MCP Servers

This directory contains all MCP (Model Context Protocol) servers used by scaffolding examples.
Each subdirectory is a standalone MCP server that can be reused across different projects.

## Available Servers

| Server | Description | Default Port |
|--------|-------------|-------------|
| `coder/` | Apiary-backed file/shell/planning tools for the Coder agent | 8083 |
| `e2b/` | E2B sandbox for running Python code | 8081 |
| `fetch_webpage/` | Fetch raw webpage/PDF content via Jina Reader / ScraperAPI | 8085 |
| `google_search/` | Google web search via SerpAPI | 8083 |
| `google_scholar/` | Google Scholar search via SerpAPI (+ web search combined) | 8084 |
| `tavily_search/` | Tavily web search API | 8087 |
| `weather/` | US National Weather Service alerts and forecasts | 8080 |
| `wordllama/` | Local knowledge base search (fake Tavily replacement) | 8082 |

## Quick Start

Each server can be started independently. For example:

> Note: some servers share the same default port (for example `coder/` and
> `google_search/` both default to `8083`). If you run multiple servers at the
> same time, pass `--port` to override.

### Weather

```bash
cd weather
uv run weather.py
```

### E2B Sandbox

```bash
cd e2b
uv run e2bserver.py
```

### Tavily Search

```bash
cd tavily_search
export TAVILY_API_KEY=<your_api_key>
uv run tavily_search.py
```

### Coder Tools

```bash
cd coder
uv run coder_mcp.py
```

### Google Search

```bash
cd google_search
export SEARCH_API_KEY=<your_serpapi_key>
uv run google_search.py
```

### Google Scholar

```bash
cd google_scholar
export SEARCH_API_KEY=<your_serpapi_key>
uv run google_scholar.py
```

### Fetch Webpage

```bash
cd fetch_webpage
export JINA_API_KEY=<your_jina_key>
uv run fetch_webpage.py
```

### WordLlama (Local KB Search)

```bash
cd wordllama
uv run wordllama_serve.py
```

## Test

```bash
python3 mcptest.py --API_KEY YOUR_API_KEY
```
