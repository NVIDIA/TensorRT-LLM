# Coder MCP Server

MCP (Model Context Protocol) server that exposes Coder agent tools for file system operations, shell commands, and task management.

## Tools

The server exposes the following tools:

### File System Tools
- **read_file** - Read file contents with line numbers
- **list_dir** - List directory contents with type labels
- **grep_files** - Search files for regex patterns
- **apply_patch** - Apply patches to create, update, or delete files

### Shell Tools
- **shell** - Execute shell commands as an array (via execvp)
- **shell_command** - Execute shell commands as a string in user's shell

### Planning/Control Tools
- **update_plan** - Update task plan with progress tracking
- **think** - Record thoughts/reflections
- **complete_task** - Signal task completion

## Usage

### Running the Server

```bash
# Default: runs on 0.0.0.0:8083
python coder_mcp.py

# Custom host/port
python coder_mcp.py --host 127.0.0.1 --port 9000

# Set working directory for file operations
python coder_mcp.py --working-dir /path/to/project
```

### Environment Variables

- `CODER_WORKING_DIRECTORY` - Default working directory for file operations (defaults to current directory)

### Using with TensorRT-LLM Scaffolding

```python
from tensorrt_llm.scaffolding.worker import MCPWorker
from tensorrt_llm.scaffolding.contrib.Coder import create_coder_scaffolding_llm

# Start the Coder MCP server first (in a separate process):
# python coder_mcp.py --port 8083

# Then create the MCP worker and Coder agent
mcp_worker = MCPWorker(urls=["http://localhost:8083/sse"])

coder = create_coder_scaffolding_llm(
    generation_worker=generation_worker,
    mcp_worker=mcp_worker,
)

# Run a coding task
result = coder.generate("Add a hello world function to main.py")
print(result.text)
```

## Protocol

The server uses SSE (Server-Sent Events) transport for MCP communication:
- SSE endpoint: `/sse`
- Message endpoint: `/messages/`
