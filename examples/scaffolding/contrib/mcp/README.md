# MCP USAGE

## Step1: Run Servers

### Terminal1:

`cd weather`

`pip install uv`

`uv add "mcp[cli]" httpx openai`

`uv pip install httpx mcp`
`uv init --no-workspace`
`uv run weather.py`



### Terminal2:

`cd e2b`

`pip install uv`

`uv add "mcp[cli]" httpx openai`

`uv pip install e2b_code_interpreter mcp`
`uv init --no-workspace`
`uv run e2bserver.py`



### Terminal3:

`cd websearch`

`pip install uv`

`uv add "mcp[cli]" httpx openai`
`uv pip install brave-search mcp starlette uvicorn`
`uv init --no-workspace`
`uv run websearch.py`





## Step2: Run Test

`python3 mcptest.py --API_KEY YOUR_API_KEY`
