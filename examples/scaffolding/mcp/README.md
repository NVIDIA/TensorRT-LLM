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
| `python_interpreter/` | Execute Python code in a sandboxed environment | 8086 |
| `tavily/` | Tavily web search API | 8082 |
| `weather/` | US National Weather Service alerts and forecasts | 8080 |
| `websearch/` | Brave web search | 8082 |
| `wordllama/` | Local knowledge base search (fake Tavily replacement) | 8082 |

## Quick Start

Each server can be started independently. For example:

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

### Web Search (Brave)

```bash
cd websearch
uv run websearch.py
```

### Tavily Search

```bash
cd tavily
export TAVILY_API_KEY=<your_api_key>
uv run travily.py
```

### Coder Tools

```bash
cd coder
python coder_mcp.py
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

### Python Interpreter

```bash
cd python_interpreter
export SANDBOX_ENDPOINT=http://127.0.0.1:8080
uv run python_interpreter.py
```

## Test

```bash
python3 mcptest.py --API_KEY YOUR_API_KEY
```
