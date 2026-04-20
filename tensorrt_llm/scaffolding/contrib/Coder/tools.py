"""Tool definitions for the Coder agent using OpenAIToolDescription format."""

from tensorrt_llm.scaffolding.task import OpenAIToolDescription

# File System Tools
read_file_tool = OpenAIToolDescription(
    name="read_file",
    description=(
        "Reads a local file with 1-indexed line numbers. "
        "Supports slice mode (simple line ranges) and indentation-aware mode "
        "(expands around an anchor line based on indentation levels)."
    ),
    parameters={
        "file_path": {"type": "string", "description": "Absolute path to the file."},
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (1-indexed). Defaults to 1.",
        },
        "limit": {"type": "integer", "description": "Maximum number of lines to return."},
        "mode": {
            "type": "string",
            "description": 'Mode selector: "slice" for simple ranges (default) or '
            '"indentation" to expand around an anchor line.',
        },
    },
)

list_dir_tool = OpenAIToolDescription(
    name="list_dir",
    description=(
        "Lists entries in a local directory with 1-indexed entry numbers and type labels. "
        "Directories are shown first, then files, sorted alphabetically."
    ),
    parameters={
        "dir_path": {"type": "string", "description": "Absolute path to the directory to list."},
        "offset": {
            "type": "integer",
            "description": "Entry number to start listing from (1-indexed). Defaults to 1.",
        },
        "limit": {"type": "integer", "description": "Maximum number of entries to return."},
        "depth": {
            "type": "integer",
            "description": "Maximum directory depth to traverse. Defaults to 1.",
        },
    },
)

grep_files_tool = OpenAIToolDescription(
    name="grep_files",
    description=(
        "Finds files whose contents match a regex pattern. "
        "Results are sorted by modification time (most recent first). "
        "Returns matching lines with line numbers."
    ),
    parameters={
        "pattern": {"type": "string", "description": "Regular expression pattern to search for."},
        "include": {
            "type": "string",
            "description": 'Glob pattern to filter which files are searched (e.g., "*.py" or "*.{ts,tsx}").',
        },
        "path": {
            "type": "string",
            "description": "Directory or file path to search. Defaults to current working directory.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of file paths to return (defaults to 100).",
        },
    },
)

# Planning Tool
update_plan_tool = OpenAIToolDescription(
    name="update_plan",
    description=(
        "Updates the task plan with steps and progress tracking. "
        "At most one step can be in_progress at a time. "
        "Use this for complex, multi-step tasks to track progress."
    ),
    parameters={
        "explanation": {"type": "string", "description": "Optional explanation for plan changes."},
        "plan": {
            "type": "array",
            "description": "The list of steps with their status.",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "string", "description": "Description of the step."},
                    "status": {
                        "type": "string",
                        "description": 'One of: "pending", "in_progress", "completed"',
                    },
                },
                "required": ["step", "status"],
            },
        },
    },
)

# Shell Tools
exec_tool = OpenAIToolDescription(
    name="exec",
    description=(
        "Execute a command array directly via execvp (no shell interpretation). "
        "The command is passed as an array of strings, with the first element being "
        "the program to execute and the rest being arguments. "
        "Always set the workdir parameter."
    ),
    parameters={
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The command to execute as an array of strings.",
        },
        "workdir": {
            "type": "string",
            "description": "The working directory to execute the command in.",
        },
        "timeout_ms": {
            "type": "integer",
            "description": "The timeout for the command in milliseconds.",
        },
    },
)

shell_tool = OpenAIToolDescription(
    name="shell",
    description=(
        "Run a shell command string (pipes, redirects, &&, etc. all work). "
        "Always set the workdir parameter."
    ),
    parameters={
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        },
        "workdir": {
            "type": "string",
            "description": "The working directory to execute the command in.",
        },
        "timeout_ms": {
            "type": "integer",
            "description": "The timeout for the command in milliseconds.",
        },
    },
)

# Think Tool (for reflection/planning)
think_tool = OpenAIToolDescription(
    name="think",
    description=(
        "Use this tool to think about the task, plan your approach, "
        "reflect on results, and decide next steps. "
        "This is for internal reasoning and does not execute any action."
    ),
    parameters={
        "thought": {
            "type": "string",
            "description": "Your thought or reflection about the current task.",
        }
    },
)

# Complete Task Tool
complete_task_tool = OpenAIToolDescription(
    name="complete_task",
    description=(
        "Indicate that you have completed the task. "
        "Call this when you have finished all work and are ready to provide the final response."
    ),
    parameters={
        "summary": {"type": "string", "description": "A brief summary of what was accomplished."}
    },
)

# All tools list for easy access
ALL_CODER_TOOLS = [
    read_file_tool,
    list_dir_tool,
    grep_files_tool,
    update_plan_tool,
    exec_tool,
    shell_tool,
    think_tool,
    complete_task_tool,
]

# Commonly used tool subsets
FILE_TOOLS = [read_file_tool, list_dir_tool, grep_files_tool]
SHELL_TOOLS = [exec_tool, shell_tool]
PLANNING_TOOLS = [update_plan_tool, think_tool, complete_task_tool]
