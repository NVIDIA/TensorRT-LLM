"""Coder agent for the Scaffolding framework.

This module provides a coding agent that can read, write, and modify code
using the TensorRT-LLM Scaffolding framework. It uses MCP (Model Context Protocol)
for tool execution via the CoderMCP server.

Example usage:
    ```python
    from tensorrt_llm.scaffolding.worker import TRTOpenaiWorker, MCPWorker
    from tensorrt_llm.scaffolding.contrib.Coder import create_coder_scaffolding_llm

    # Create workers
    generation_worker = TRTOpenaiWorker(
        base_url="http://localhost:8000/v1",
        api_key="your-api-key",
        model="your-model",
    )

    # Start CoderMCP server first:
    # python examples/scaffolding/mcp/coder/coder_mcp.py --port 8083

    mcp_worker = MCPWorker(urls=["http://localhost:8083/sse"])

    # Create the Coder agent
    coder = create_coder_scaffolding_llm(
        generation_worker=generation_worker,
        mcp_worker=mcp_worker,
    )

    # Run a coding task
    result = coder.generate("Add error handling to the parse_config function")
    print(result.text)
    ```
"""

from .coder import Coder, CoderTask, create_coder_scaffolding_llm
from .tools import (
    ALL_CODER_TOOLS,
    FILE_TOOLS,
    PLANNING_TOOLS,
    SHELL_TOOLS,
    apply_patch_tool,
    complete_task_tool,
    grep_files_tool,
    list_dir_tool,
    read_file_tool,
    shell_command_tool,
    shell_tool,
    think_tool,
    update_plan_tool,
)

__all__ = [
    # Main controller
    "Coder",
    "CoderTask",
    # Factory function
    "create_coder_scaffolding_llm",
    # Tool definitions
    "ALL_CODER_TOOLS",
    "FILE_TOOLS",
    "SHELL_TOOLS",
    "PLANNING_TOOLS",
    "read_file_tool",
    "list_dir_tool",
    "grep_files_tool",
    "apply_patch_tool",
    "update_plan_tool",
    "shell_tool",
    "shell_command_tool",
    "think_tool",
    "complete_task_tool",
]
