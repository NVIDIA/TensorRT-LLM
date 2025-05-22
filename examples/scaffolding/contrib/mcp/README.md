# MCP USAGE

## Step1: Run Servers

### Terminal1: 

`cd weather` 

`pip install uv`

`uv add "mcp[cli]" httpx openai`

`uv pip install httpx mcp`
`uv init --no-workspace`
`uv run weather.py`

![image-20250520125319448](E:\code\wu1du2\TensorRT-LLM\examples\scaffolding\contrib\mcp\assets\image-20250520125319448.png)


### Terminal2:

`cd e2b`

`pip install uv`

`uv add "mcp[cli]" httpx openai`

`uv pip install e2b_code_interpreter mcp`
`uv init --no-workspace`
`uv run e2bserver.py`

![image-20250520125329071](E:\code\wu1du2\TensorRT-LLM\examples\scaffolding\contrib\mcp\assets\image-20250520125329071.png)

### Terminal3:

`cd websearch`

`pip install uv`

`uv add "mcp[cli]" httpx openai`
`uv pip install brave-search mcp starlette uvicorn`
`uv init --no-workspace`
`uv run websearch.py`

![image-20250520125343632](E:\code\wu1du2\TensorRT-LLM\examples\scaffolding\contrib\mcp\assets\image-20250520125343632.png)



## Step2: Run Test

`python3 mcptest.py --API_KEY YOUR_API_KEY`



'What was the score of the NBA playoffs game 7 between the Thunder and the Nuggets in 2025?'

![image-20250520125509938](E:\code\wu1du2\TensorRT-LLM\examples\scaffolding\contrib\mcp\assets\image-20250520125509938.png)