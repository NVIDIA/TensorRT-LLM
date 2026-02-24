# Agent Client-Server

A remote agent execution system that connects lightweight clients to a
GPU-backed agent server over WebSocket. The server runs scaffolding agents
(Coder, Deep Research) powered by TensorRT-LLM, while tool calls are
relayed back to the client for local execution -- allowing the agent to
read files, run commands, and apply patches on the client's machine.

## Architecture

```
┌───────────────────┐          WebSocket          ┌──────────────────────────────┐
│   Agent Client    │◄──────────────────────────► │         Agent Server         │
│                   │                             │                              │
│  • Tool executor  │  agent_request ───────────► │  • Scaffolding controllers   │
│    (filesystem,   │                             │  • TRT-LLM generation        │
│     shell, patch) │  ◄──────────── tool_call    │    (via OpenAI-compat API)   │
│  • No GPU needed  │  tool_result ─────────────► │  • Tool call relay           │
│                   │  ◄──────────── agent_result │                              │
└───────────────────┘                             └──────────────────────────────┘
```

The server holds the LLM and orchestrates the scaffolding pipeline. When
the agent needs to interact with the user's environment (read a file, run
a command, etc.), the tool call is serialized and sent to the client. The
client executes it locally and returns the result through the same
WebSocket connection.

## Module Structure

```
examples/scaffolding/serve/
├── __init__.py     Package marker
├── README.md       This documentation
├── client.py       WebSocket client with local tool execution
├── server.py       WebSocket server with scaffolding + TRT-LLM
└── tools.py        ToolExecutor class (all local tool implementations)
```

## Prerequisites

**Server** (GPU machine):
- TensorRT-LLM installed with scaffolding support
- An OpenAI-compatible inference endpoint running (e.g. `trtllm-serve`)
- Python packages: `aiohttp`, `openai`

**Client** (any machine):
- Python 3.10+
- `aiohttp` (no TensorRT-LLM dependency required)

## Quick Start

### 1. Start the inference endpoint

```bash
trtllm-serve start --model your-model
```

### 2. Launch the Agent Server

```bash
python -m examples.scaffolding.serve.server \
    --base_url http://localhost:8000/v1 \
    --model your-model
```

### 3. Run the Agent Client

```bash
# Coder agent -- reads/writes files, runs commands
python -m examples.scaffolding.serve.client \
    --server ws://gpu-server:8090/ws \
    --agent_type coder \
    --prompt "Implement a hello world function in Python" \
    --working_dir /path/to/your/project

# Deep Research agent
python -m examples.scaffolding.serve.client \
    --server ws://gpu-server:8090/ws \
    --agent_type deep_research \
    --prompt "Analyze the impact of quantum computing on cryptography"
```

## WebSocket Protocol

All messages are JSON over a single WebSocket connection at `/ws`.

### Client → Server

| Message Type     | Fields                               | Description                |
|------------------|--------------------------------------|----------------------------|
| `agent_request`  | `request_id`, `agent_type`, `prompt` | Start an agent task        |
| `tool_result`    | `call_id`, `result`                  | Return a local tool result |

### Server → Client

| Message Type   | Fields                                           | Description                  |
|----------------|--------------------------------------------------|------------------------------|
| `tool_call`    | `request_id`, `call_id`, `tool_name`, `arguments`| Request local tool execution |
| `status`       | `request_id`, `message`                          | Progress update              |
| `agent_result` | `request_id`, `output`, `error`                  | Final agent output           |
| `error`        | `message`                                        | Protocol/validation error    |

### Example Flow

```
Client                              Server
  │                                    │
  │──── agent_request ────────────────►│
  │     {agent_type: "coder",          │
  │      prompt: "add tests"}          │
  │                                    │
  │◄──── status ───────────────────────│
  │     "Starting coder agent..."      │
  │                                    │
  │◄──── tool_call ────────────────────│
  │     {tool_name: "read_file",       │
  │      arguments: {file_path: "..."}}│
  │                                    │
  │──── tool_result ──────────────────►│
  │     {result: "file contents..."}   │
  │                                    │
  │◄──── tool_call ────────────────────│
  │     {tool_name: "apply_patch", ...}│
  │                                    │
  │──── tool_result ──────────────────►│
  │     {result: "Updated file: ..."}  │
  │                                    │
  │◄──── agent_result ─────────────────│
  │     {output: "Task completed..."}  │
  │                                    │
```

## Available Client-Side Tools

The client executes these tools locally on behalf of the agent:

| Tool             | Description                                        |
|------------------|----------------------------------------------------|
| `read_file`      | Read file contents with 1-indexed line numbers     |
| `list_dir`       | List directory contents with type labels           |
| `grep_files`     | Search files with regex patterns                   |
| `apply_patch`    | Apply structured patches (add/delete/update files) |
| `shell`          | Execute a command array via execvp (no shell)      |
| `shell_command`  | Execute a string in the user's default shell       |
| `update_plan`    | Update a task plan with progress tracking          |
| `think`          | Record agent thoughts/reflections                  |
| `complete_task`  | Signal task completion with a summary              |

## Server Configuration

| Flag                        | Default                    | Description                              |
|-----------------------------|----------------------------|------------------------------------------|
| `--host`                    | `0.0.0.0`                 | Host to bind to                          |
| `--port`                    | `8090`                     | Port to listen on                        |
| `--base_url`                | `http://localhost:8000/v1` | OpenAI-compatible API base URL           |
| `--model`                   | `gpt-oss-20b`             | Model name for generation                |
| `--openai_api_key`          | `tensorrt_llm`            | API key for the inference server         |
| `--max_tokens`              | `131072`                   | Maximum tokens per generation            |
| `--max_iterations`          | `50`                       | Max tool-calling iterations (Coder only) |
| `--max_parallel_requests`   | `16`                       | Max parallel requests per client         |
| `--enable_statistics`       | off                        | Enable task metrics                      |
| `--kv_cache_hint_enabled`   | off                        | Enable KV cache hints                    |

## Client Configuration

| Flag             | Default                  | Description                              |
|------------------|--------------------------|------------------------------------------|
| `--server`       | `ws://localhost:8090/ws` | WebSocket URL of the agent server        |
| `--agent_type`   | `coder`                  | Agent type: `coder` or `deep_research`   |
| `--prompt`       | *(required)*             | Task prompt for the agent                |
| `--working_dir`  | current directory        | Working directory for tool execution     |

## Available Agent Applications

### Coder

An AI coding assistant that can read, write, and modify files, run shell
commands, and apply patches to your codebase. Uses a structured planning
approach to break down coding tasks into manageable steps.

### Deep Research

An AI research agent that orchestrates multi-step research workflows with
web search, content analysis, and report generation. Decomposes research
questions into parallel investigation streams for comprehensive analysis.
