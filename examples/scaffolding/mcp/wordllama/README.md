# Fake Tavily MCP

This is a MCP server for search with local knowledge base. It is used to fake the search from web.

## Usage

```
python run_deep_research.py --enable_query_collector

You will see query_result.json and copy query_result.json to this path.

Kill the Travily serve.

uv run wordllama_serve.py

Then you can continue to run run_deep_research.py.
```
