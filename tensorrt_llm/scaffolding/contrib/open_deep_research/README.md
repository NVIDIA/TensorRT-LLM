# Open Deep Research

This module implements Open Deep Research with Scaffolding to enable joint optimization between multi-agent applications and TensorRT-LLM.

## Design Overview

**Open Deep Research** is an open-source deep research agent built on a multi-agent Planner-Executor architecture:

- **Supervisor (Planner)**: Accepts user input, generates a research brief, delegates tasks to Researchers, and synthesizes the final report once sufficient information has been gathered.
- **Researcher (Executor)**: Receives a research topic, conducts multiple rounds of interaction with external search tools, then summarizes and compresses the findings before returning results.

### Architecture

The frontend-backend decoupling and modular architecture of Scaffolding supports building multi-agent systems efficiently.

#### Frontend (Control Flow)

The frontend encompasses the control flow of the Planner-Executor architecture through `Controller`s:

| Controller | Description |
|------------|-------------|
| `Supervisor` | Entry controller for the entire agent; orchestrates the research workflow |
| `BriefController` | Generates the research brief from user input |
| `ResearchPlanningController` | Plans and delegates research topics to sub-agents |
| `Researcher` | Sub-agent that conducts research on specific topics |
| `ChatWithMCPController` | Handles tool calling for web search (reusable Scaffolding controller) |
| `Compressor` | Compresses search results and model reflections |
| `FinalReportController` | Synthesizes findings into the final report |

#### Backend (Workers)

The backend serves LLM generation and tool call requests through `Worker` instances:

| Worker | Description |
|--------|-------------|
| `TRTOpenaiWorker` | Serves LLM generation requests via TensorRT-LLM OpenAI-compatible endpoint |
| `MCPWorker` | Serves tool calling requests via MCP server |

### Modularity

Scaffolding supports the evolution of individual components independent of other components in the multi-agent system. For example:
- To use a more sophisticated sub-agent for the final report, simply replace the corresponding Controller in that module.
- To support other LLM endpoints (e.g., Anthropic, Google), implement a Worker similar to `TRTOpenaiWorker`.

## Quick Start

### 1. Start TensorRT-LLM Server

```bash
trtllm-serve serve Qwen3/Qwen3-30B-A3B \
    --max_num_tokens 32768 \
    --kv_cache_free_gpu_memory_fraction 0.8 \
    --extra_llm_api_options .extra-llm-api-config.yml \
    --reasoning_parser qwen3 \
    --tool_parser qwen3
```

Create `.extra-llm-api-config.yml` with the following content:

```yaml
resort_policy_config:
    policy_name: "AgentTree"
    policy_args:
        agent_percentage: 0.5
        agent_types: ["agent_deep_research"]
        agent_inflight_seq_num: 8
```

### 2. Start Tavily MCP Server

```bash
cd examples/scaffolding/contrib/open_deep_research/TavilyMCP
export TAVILY_API_KEY=<your_api_key>
uv run travily.py
```

### 3. Run the Example

```bash
cd examples/scaffolding/contrib/open_deep_research
python run_deep_research.py
```

#### Optional Flags

- `--enable_statistics`: Enable token counting and task timing metrics
- `--enable_query_collector`: Enable query collection for debugging
- `--base_url`: Specify a custom TensorRT-LLM server URL (default: `http://localhost:8000/v1`)
- `--model`: Specify a different model (default: `Qwen3/Qwen3-30B-A3B`)

Example with statistics enabled:

```bash
python run_deep_research.py --enable_statistics
```

To enable sub-request marking for detailed tracing:

```bash
ENABLE_SUB_REQUEST_MARKER=1 python run_deep_research.py --enable_statistics
```

## Acknowledgments

This implementation follows the design of [Open Deep Research](https://github.com/langchain-ai/open_deep_research) by LangChain.
