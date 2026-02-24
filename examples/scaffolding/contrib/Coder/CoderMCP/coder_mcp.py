"""MCP Server for Coder agent tools.

This module implements an MCP server that exposes all Coder tools
(file system operations, shell commands, etc.) via the Model Context Protocol.

Relay Mode (--relay):
    When started with --relay, tool calls are forwarded to a connected
    WebSocket client instead of being executed locally. This enables a
    remote agent architecture where:

        ScaffoldingLlm -> MCPWorker -> This MCP Server -> Client (WebSocket)

    The client executes tools on its local machine and sends results back.
    A WebSocket endpoint is available at /ws for client connections.
"""

import asyncio
import fnmatch
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.websockets import WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Initialize FastMCP server for Coder tools (SSE)
mcp = FastMCP("coder_tools")

# Global working directory (can be set via environment variable)
WORKING_DIRECTORY = os.environ.get("CODER_WORKING_DIRECTORY", os.getcwd())

# Plan state (in-memory for the session)
_current_plan: list[dict] = []
_plan_explanation: Optional[str] = None

# =============================================================================
# Relay Mode State
# =============================================================================

_relay_mode: bool = False
_relay_ws: Optional[WebSocket] = None
_relay_pending: Dict[str, asyncio.Future] = {}
_relay_lock = asyncio.Lock()


async def _relay_tool_call(tool_name: str, arguments: dict) -> str:
    """Forward a tool call to the connected relay client via WebSocket.

    This is called by tool functions when relay mode is active. Instead of
    executing the tool locally, we send the request to the client and wait
    for the result.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments as a dict

    Returns:
        Tool result string from the client
    """
    if _relay_ws is None:
        return "Error: No relay client connected. Start a client with --mcp_relay_url."

    call_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    async with _relay_lock:
        _relay_pending[call_id] = future

    LOGGER.info(f"[Relay] Forwarding tool call to client: {tool_name}")

    try:
        await _relay_ws.send_json(
            {
                "type": "tool_call",
                "call_id": call_id,
                "tool_name": tool_name,
                "arguments": arguments,
            }
        )
    except Exception as e:
        async with _relay_lock:
            _relay_pending.pop(call_id, None)
        return f"Error: Failed to send tool call to relay client: {e}"

    try:
        # Wait for client response with a generous timeout
        result = await asyncio.wait_for(future, timeout=600)
        LOGGER.info(f"[Relay] Received result for {tool_name}: {str(result)[:200]}...")
        return result
    except asyncio.TimeoutError:
        LOGGER.error(f"[Relay] Timeout waiting for {tool_name} result")
        return f"Error: Relay timeout waiting for {tool_name} result (600s)"
    finally:
        async with _relay_lock:
            _relay_pending.pop(call_id, None)


def _resolve_path(path: str) -> str:
    """Resolve a path relative to the working directory if not absolute."""
    if not os.path.isabs(path):
        return os.path.join(WORKING_DIRECTORY, path)
    return path


# =============================================================================
# File System Tools
# =============================================================================


@mcp.tool()
async def read_file(
    file_path: str,
    offset: int = 1,
    limit: Optional[int] = None,
    mode: str = "slice",
) -> str:
    """Read a file with 1-indexed line numbers.

    Args:
        file_path: Absolute path to the file.
        offset: Line number to start reading from (1-indexed). Defaults to 1.
        limit: Maximum number of lines to return.
        mode: Mode selector: "slice" for simple ranges (default).

    Returns:
        File contents with line numbers, or an error message.
    """
    if _relay_mode:
        args = {"file_path": file_path, "offset": offset, "mode": mode}
        if limit is not None:
            args["limit"] = limit
        return await _relay_tool_call("read_file", args)

    if not file_path:
        return "Error: file_path is required"

    file_path = _resolve_path(file_path)

    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"

    if not os.path.isfile(file_path):
        return f"Error: Path is not a file: {file_path}"

    if offset < 1:
        return "Error: offset must be 1 or greater"

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()

        total_lines = len(lines)

        # Simple slice mode
        start_idx = offset - 1
        if limit:
            end_idx = start_idx + limit
        else:
            end_idx = len(lines)

        result_lines = []
        for i in range(start_idx, min(end_idx, len(lines))):
            result_lines.append((i + 1, lines[i]))

        # Format output with line numbers
        output_lines = []
        for line_num, line_content in result_lines:
            output_lines.append(f"{line_num:6d}|{line_content}")

        result = "\n".join(output_lines)
        return f"{result}\n\n[Total lines: {total_lines}, Lines returned: {len(result_lines)}]"

    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def list_dir(
    dir_path: str,
    offset: int = 1,
    limit: Optional[int] = None,
    depth: int = 1,
) -> str:
    """List directory contents with type labels.

    Args:
        dir_path: Absolute path to the directory to list.
        offset: Entry number to start listing from (1-indexed). Defaults to 1.
        limit: Maximum number of entries to return.
        depth: Maximum directory depth to traverse. Defaults to 1.

    Returns:
        Directory listing with entry numbers and types, or an error message.
    """
    if _relay_mode:
        args = {"dir_path": dir_path, "offset": offset, "depth": depth}
        if limit is not None:
            args["limit"] = limit
        return await _relay_tool_call("list_dir", args)

    LOGGER.info(
        f"Tool called: list_dir(dir_path={dir_path!r}, offset={offset}, limit={limit}, depth={depth})"
    )
    if not dir_path:
        return "Error: dir_path is required"

    dir_path = _resolve_path(dir_path)

    if not os.path.exists(dir_path):
        return f"Error: Directory not found: {dir_path}"

    if not os.path.isdir(dir_path):
        return f"Error: Path is not a directory: {dir_path}"

    if offset < 1:
        return "Error: offset must be 1 or greater"

    if depth < 1:
        return "Error: depth must be 1 or greater"

    def get_entry_type(path: Path) -> str:
        if path.is_dir():
            return "dir"
        elif path.is_symlink():
            return "link"
        elif path.is_file():
            return "file"
        else:
            return "other"

    def list_entries(
        path: Path, current_depth: int, max_depth: int, prefix: str = ""
    ) -> list[tuple[str, str]]:
        entries = []
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return [(f"{prefix}(permission denied)", "error")]

        for item in items:
            if item.name.startswith("."):
                continue

            entry_type = get_entry_type(item)
            display_name = f"{prefix}{item.name}"

            if entry_type == "dir":
                display_name += "/"

            entries.append((display_name, entry_type))

            if entry_type == "dir" and current_depth < max_depth:
                sub_entries = list_entries(item, current_depth + 1, max_depth, prefix + "  ")
                entries.extend(sub_entries)

        return entries

    try:
        path = Path(dir_path)
        all_entries = list_entries(path, 1, depth)

        total_entries = len(all_entries)

        start_idx = offset - 1
        if limit:
            end_idx = start_idx + limit
        else:
            end_idx = len(all_entries)

        selected_entries = all_entries[start_idx:end_idx]

        output_lines = []
        for i, (name, entry_type) in enumerate(selected_entries, start=offset):
            output_lines.append(f"{i:6d}. [{entry_type:5s}] {name}")

        result = "\n".join(output_lines)
        return f"{result}\n\n[Total entries: {total_entries}, Entries returned: {len(selected_entries)}]"

    except PermissionError:
        return f"Error: Permission denied: {dir_path}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def grep_files(
    pattern: str,
    include: Optional[str] = None,
    path: Optional[str] = None,
    limit: int = 100,
) -> str:
    """Search files for a regex pattern.

    Args:
        pattern: Regular expression pattern to search for.
        include: Glob pattern to filter which files are searched.
        path: Directory or file path to search. Defaults to working directory.
        limit: Maximum number of file paths to return (defaults to 100).

    Returns:
        Matching files with line numbers and content, or an error message.
    """
    if _relay_mode:
        args = {"pattern": pattern, "limit": limit}
        if include is not None:
            args["include"] = include
        if path is not None:
            args["path"] = path
        return await _relay_tool_call("grep_files", args)

    LOGGER.info(
        f"Tool called: grep_files(pattern={pattern!r}, include={include!r}, path={path!r}, limit={limit})"
    )
    if not pattern:
        return "Error: pattern is required"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    search_path = _resolve_path(path) if path else WORKING_DIRECTORY

    if not os.path.exists(search_path):
        return f"Error: Path not found: {search_path}"

    def matches_glob(filepath: str, glob_pattern: str) -> bool:
        if "{" in glob_pattern and "}" in glob_pattern:
            base, rest = glob_pattern.split("{", 1)
            extensions, suffix = rest.split("}", 1)
            for ext in extensions.split(","):
                full_pattern = f"{base}{ext}{suffix}"
                if fnmatch.fnmatch(filepath, full_pattern):
                    return True
            return False
        return fnmatch.fnmatch(filepath, glob_pattern)

    def search_file(filepath: str) -> list[tuple[int, str]]:
        matches = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, start=1):
                    if regex.search(line):
                        matches.append((i, line.rstrip()))
        except (PermissionError, OSError):
            pass
        return matches

    try:
        matching_files = []

        if os.path.isfile(search_path):
            matches = search_file(search_path)
            if matches:
                mtime = os.path.getmtime(search_path)
                matching_files.append((search_path, mtime, matches))
        else:
            for root, dirs, files in os.walk(search_path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]

                for filename in files:
                    if filename.startswith("."):
                        continue

                    filepath = os.path.join(root, filename)

                    if include and not matches_glob(filename, include):
                        continue

                    matches = search_file(filepath)
                    if matches:
                        try:
                            mtime = os.path.getmtime(filepath)
                            matching_files.append((filepath, mtime, matches))
                        except OSError:
                            continue

                    if len(matching_files) >= limit:
                        break

                if len(matching_files) >= limit:
                    break

        matching_files.sort(key=lambda x: x[1], reverse=True)
        matching_files = matching_files[:limit]

        output_lines = []
        for filepath, mtime, matches in matching_files:
            rel_path = (
                os.path.relpath(filepath, search_path) if os.path.isdir(search_path) else filepath
            )
            output_lines.append(f"\n{rel_path}:")
            for line_num, line_content in matches[:10]:
                if len(line_content) > 200:
                    line_content = line_content[:200] + "..."
                output_lines.append(f"  {line_num}: {line_content}")
            if len(matches) > 10:
                output_lines.append(f"  ... ({len(matches) - 10} more matches)")

        result = "\n".join(output_lines).strip()
        limit_msg = " (limit reached)" if len(matching_files) >= limit else ""
        return f"{result}\n\n[Files matched: {len(matching_files)}{limit_msg}]"

    except PermissionError:
        return f"Error: Permission denied: {search_path}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# Patch Tool
# =============================================================================


@mcp.tool()
async def apply_patch(patch: str) -> str:
    """Apply a patch to files.

    Supports three operations:
    - "*** Add File: <path>" - create a new file (lines prefixed with +)
    - "*** Delete File: <path>" - remove an existing file
    - "*** Update File: <path>" - patch an existing file (with optional "*** Move to:")

    Patch must be wrapped in *** Begin Patch and *** End Patch envelope.

    Args:
        patch: The patch content to apply.

    Returns:
        Result message describing what was changed, or an error message.
    """
    if _relay_mode:
        return await _relay_tool_call("apply_patch", {"patch": patch})

    LOGGER.info(f"Tool called: apply_patch(patch=<{len(patch)} chars>)")
    results = []
    lines = patch.strip().split("\n")

    if not lines or lines[0].strip() != "*** Begin Patch":
        return 'Error: Patch must start with "*** Begin Patch"'
    if lines[-1].strip() != "*** End Patch":
        return 'Error: Patch must end with "*** End Patch"'

    lines = lines[1:-1]
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Add File
        if line.startswith("*** Add File:"):
            filepath = line[len("*** Add File:") :].strip()
            filepath = _resolve_path(filepath)
            i += 1
            content_lines = []
            while i < len(lines) and not lines[i].strip().startswith("***"):
                content_line = lines[i]
                if content_line.startswith("+"):
                    content_lines.append(content_line[1:])
                i += 1
            try:
                if os.path.dirname(filepath):
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w") as f:
                    f.write("\n".join(content_lines))
                results.append(f"Added file: {filepath}")
            except Exception as e:
                return f"Error: Failed to add file {filepath}: {e}"

        # Delete File
        elif line.startswith("*** Delete File:"):
            filepath = line[len("*** Delete File:") :].strip()
            filepath = _resolve_path(filepath)
            i += 1
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    results.append(f"Deleted file: {filepath}")
                else:
                    results.append(f"File already deleted or not found: {filepath}")
            except Exception as e:
                return f"Error: Failed to delete file {filepath}: {e}"

        # Update File
        elif line.startswith("*** Update File:"):
            filepath = line[len("*** Update File:") :].strip()
            filepath = _resolve_path(filepath)
            i += 1
            new_filepath = None

            if i < len(lines) and lines[i].strip().startswith("*** Move to:"):
                new_filepath = lines[i].strip()[len("*** Move to:") :].strip()
                new_filepath = _resolve_path(new_filepath)
                i += 1

            try:
                with open(filepath, "r") as f:
                    file_content = f.read()
                file_lines = file_content.split("\n")
            except Exception as e:
                return f"Error: Failed to read file {filepath}: {e}"

            while i < len(lines) and not lines[i].strip().startswith("*** "):
                hunk_line = lines[i]

                if hunk_line.startswith("@@"):
                    context = hunk_line[2:].strip()
                    i += 1

                    context_idx = None
                    for idx, fl in enumerate(file_lines):
                        if context in fl:
                            context_idx = idx
                            break

                    if context_idx is None:
                        return f'Error: Could not find context "{context}" in {filepath}'

                    current_idx = context_idx + 1
                    while (
                        i < len(lines)
                        and not lines[i].startswith("@@")
                        and not lines[i].strip().startswith("*** ")
                    ):
                        change_line = lines[i]
                        if change_line.startswith("-"):
                            to_remove = change_line[1:]
                            found = False
                            for search_idx in range(
                                context_idx, min(context_idx + 20, len(file_lines))
                            ):
                                if (
                                    search_idx < len(file_lines)
                                    and to_remove in file_lines[search_idx]
                                ):
                                    file_lines.pop(search_idx)
                                    found = True
                                    break
                            if not found:
                                return f'Error: Could not find line to remove: "{to_remove}" in {filepath}'
                        elif change_line.startswith("+"):
                            to_add = change_line[1:]
                            file_lines.insert(current_idx, to_add)
                            current_idx += 1
                        i += 1
                else:
                    i += 1

            try:
                target_path = new_filepath if new_filepath else filepath
                if os.path.dirname(target_path):
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "w") as f:
                    f.write("\n".join(file_lines))

                if new_filepath and os.path.exists(filepath):
                    os.remove(filepath)
                    results.append(f"Updated and moved file: {filepath} -> {new_filepath}")
                else:
                    results.append(f"Updated file: {filepath}")
            except Exception as e:
                return f"Error: Failed to write file {target_path}: {e}"
        else:
            i += 1

    return "; ".join(results) if results else "No changes applied"


# =============================================================================
# Planning Tool
# =============================================================================


@mcp.tool()
async def update_plan(
    plan: str,
    explanation: Optional[str] = None,
) -> str:
    """Update the task plan with steps and progress tracking.

    Args:
        plan: JSON string representing a list of steps. Each step should have
              "step" (description) and "status" (pending/in_progress/completed).
        explanation: Optional explanation for plan changes.

    Returns:
        Formatted plan display, or an error message.
    """
    if _relay_mode:
        args = {"plan": plan}
        if explanation is not None:
            args["explanation"] = explanation
        return await _relay_tool_call("update_plan", args)

    LOGGER.info(f"Tool called: update_plan(plan=<{len(plan)} chars>, explanation={explanation!r})")
    global _current_plan, _plan_explanation

    try:
        plan_list = json.loads(plan)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON for plan: {e}"

    if not plan_list:
        return "Error: plan is required and cannot be empty"

    valid_statuses = {"pending", "in_progress", "completed"}
    in_progress_count = 0

    validated_plan = []
    for i, item in enumerate(plan_list):
        if not isinstance(item, dict):
            return f"Error: Plan item {i + 1} must be an object"

        step = item.get("step")
        status = item.get("status")

        if not step:
            return f"Error: Plan item {i + 1} missing 'step' field"

        if not status:
            return f"Error: Plan item {i + 1} missing 'status' field"

        if status not in valid_statuses:
            return (
                f"Error: Plan item {i + 1} has invalid status '{status}'. "
                "Must be one of: {', '.join(valid_statuses)}"
            )

        if status == "in_progress":
            in_progress_count += 1

        validated_plan.append({"step": step, "status": status})

    if in_progress_count > 1:
        return (
            f"Error: At most one step can be in_progress at a time, but found {in_progress_count}"
        )

    _current_plan = validated_plan
    _plan_explanation = explanation

    output_lines = []
    if explanation:
        output_lines.append(f"Explanation: {explanation}")
        output_lines.append("")

    output_lines.append("Plan:")
    for i, item in enumerate(validated_plan, start=1):
        status_icon = {"pending": "○", "in_progress": "◐", "completed": "●"}.get(
            item["status"], "?"
        )
        output_lines.append(f"  {i}. [{status_icon}] {item['step']} ({item['status']})")

    total = len(validated_plan)
    completed = sum(1 for item in validated_plan if item["status"] == "completed")
    progress = round(completed / total * 100, 1) if total > 0 else 0

    output_lines.append("")
    output_lines.append(f"Progress: {completed}/{total} ({progress}%)")

    return "\n".join(output_lines)


# =============================================================================
# Shell Tools
# =============================================================================


@mcp.tool()
async def shell(
    command: str,
    workdir: Optional[str] = None,
    timeout_ms: Optional[int] = None,
) -> str:
    """Execute a shell command as an array via execvp().

    Args:
        command: JSON string representing the command array (e.g., '["ls", "-la"]').
        workdir: The working directory to execute the command in.
        timeout_ms: The timeout for the command in milliseconds.

    Returns:
        Command output, or an error message.
    """
    if _relay_mode:
        args = {"command": command}
        if workdir is not None:
            args["workdir"] = workdir
        if timeout_ms is not None:
            args["timeout_ms"] = timeout_ms
        return await _relay_tool_call("shell", args)

    LOGGER.info(
        f"Tool called: shell(command={command!r}, workdir={workdir!r}, timeout_ms={timeout_ms})"
    )
    try:
        command_list = json.loads(command)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON for command: {e}"

    if not command_list:
        return "Error: Command array cannot be empty"

    cwd = _resolve_path(workdir) if workdir else WORKING_DIRECTORY

    if not os.path.isdir(cwd):
        return f"Error: Working directory does not exist: {cwd}"

    timeout = timeout_ms / 1000.0 if timeout_ms else None

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            command_list,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        exit_info = f"\n\n[Exit code: {result.returncode}]"
        return (output.strip() if output else "(no output)") + exit_info

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout_ms}ms"
    except FileNotFoundError:
        return f"Error: Command not found: {command_list[0]}"
    except PermissionError:
        return f"Error: Permission denied: {command_list[0]}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def shell_command(
    command: str,
    workdir: Optional[str] = None,
    timeout_ms: Optional[int] = None,
) -> str:
    """Execute a shell command as a string in the user's default shell.

    Args:
        command: The shell script to execute.
        workdir: The working directory to execute the command in.
        timeout_ms: The timeout for the command in milliseconds.

    Returns:
        Command output, or an error message.
    """
    if _relay_mode:
        args = {"command": command}
        if workdir is not None:
            args["workdir"] = workdir
        if timeout_ms is not None:
            args["timeout_ms"] = timeout_ms
        return await _relay_tool_call("shell_command", args)

    LOGGER.info(
        f"Tool called: shell_command(command={command!r}, workdir={workdir!r}, timeout_ms={timeout_ms})"
    )
    if not command:
        return "Error: Command cannot be empty"

    cwd = _resolve_path(workdir) if workdir else WORKING_DIRECTORY

    if not os.path.isdir(cwd):
        return f"Error: Working directory does not exist: {cwd}"

    timeout = timeout_ms / 1000.0 if timeout_ms else None

    shell_prog = "powershell.exe" if sys.platform == "win32" else os.environ.get("SHELL", "/bin/sh")

    if sys.platform == "win32":
        shell_cmd = [shell_prog, "-Command", command]
    else:
        shell_cmd = [shell_prog, "-l", "-c", command]

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            shell_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        exit_info = f"\n\n[Exit code: {result.returncode}]"
        return (output.strip() if output else "(no output)") + exit_info

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout_ms}ms"
    except FileNotFoundError:
        return f"Error: Shell not found: {shell_prog}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# Control Tools
# =============================================================================


@mcp.tool()
async def think(thought: str) -> str:
    """Record a thought or reflection about the current task.

    Use this tool to think about the task, plan your approach,
    reflect on results, and decide next steps.

    Args:
        thought: Your thought or reflection about the current task.

    Returns:
        Confirmation that the thought was recorded.
    """
    if _relay_mode:
        return await _relay_tool_call("think", {"thought": thought})

    LOGGER.info(f"Tool called: think(thought={thought!r})")
    return f"Thought recorded: {thought}"


@mcp.tool()
async def complete_task(summary: str) -> str:
    """Signal that the task has been completed.

    Call this when you have finished all work and are ready to provide
    the final response.

    Args:
        summary: A brief summary of what was accomplished.

    Returns:
        Confirmation that the task is complete.
    """
    if _relay_mode:
        return await _relay_tool_call("complete_task", {"summary": summary})

    LOGGER.info(f"Tool called: complete_task(summary={summary!r})")
    return f"Task completed: {summary}"


# =============================================================================
# Server Setup
# =============================================================================


def create_starlette_app(
    mcp_server: Server,
    *,
    debug: bool = False,
    relay: bool = False,
) -> Starlette:
    """Create a Starlette application that can serve the MCP server with SSE.

    Args:
        mcp_server: The MCP server instance
        debug: Enable Starlette debug mode
        relay: If True, add WebSocket endpoint at /ws for relay clients
    """
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )
        return Response()

    routes = [
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]

    if relay:
        from starlette.routing import WebSocketRoute

        async def relay_ws_endpoint(websocket: WebSocket):
            """WebSocket endpoint for relay clients.

            When relay mode is active, this endpoint accepts a single client
            connection. Tool calls received from MCPWorker via SSE are forwarded
            to the connected client, and results are sent back.
            """
            global _relay_ws

            await websocket.accept()
            _relay_ws = websocket
            LOGGER.info("[Relay] Client connected via WebSocket")

            try:
                while True:
                    data = await websocket.receive_json()
                    msg_type = data.get("type")

                    if msg_type == "tool_result":
                        call_id = data.get("call_id")
                        result = data.get("result", "")

                        async with _relay_lock:
                            future = _relay_pending.get(call_id)

                        if future is not None and not future.done():
                            future.set_result(result)
                        else:
                            LOGGER.warning(f"[Relay] Result for unknown call_id: {call_id}")
                    else:
                        LOGGER.warning(f"[Relay] Unknown message type from client: {msg_type}")

            except WebSocketDisconnect:
                LOGGER.info("[Relay] Client disconnected")
            except Exception as e:
                LOGGER.error(f"[Relay] WebSocket error: {e}")
            finally:
                _relay_ws = None
                # Cancel any pending relay calls
                async with _relay_lock:
                    for call_id, future in _relay_pending.items():
                        if not future.done():
                            future.set_exception(Exception("Relay client disconnected"))
                    _relay_pending.clear()
                LOGGER.info("[Relay] Cleaned up relay state")

        routes.append(WebSocketRoute("/ws", endpoint=relay_ws_endpoint))
        LOGGER.info("[Relay] WebSocket endpoint registered at /ws")

    return Starlette(debug=debug, routes=routes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Coder MCP SSE-based server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8083, help="Port to listen on")
    parser.add_argument(
        "--working-dir",
        default=None,
        help="Working directory for file operations (default: current directory)",
    )
    parser.add_argument(
        "--relay",
        action="store_true",
        help=(
            "Enable relay mode: forward tool calls to a connected WebSocket "
            "client instead of executing locally. A WebSocket endpoint is "
            "available at /ws for client connections."
        ),
    )
    args = parser.parse_args()

    if args.working_dir:
        WORKING_DIRECTORY = args.working_dir
        LOGGER.info(f"Working directory set to: {WORKING_DIRECTORY}")

    if args.relay:
        _relay_mode = True
        LOGGER.info(
            "[Relay] Relay mode enabled. Tool calls will be forwarded to "
            "connected WebSocket client at /ws"
        )

    mcp_server = mcp._mcp_server  # noqa: WPS437

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True, relay=args.relay)

    LOGGER.info(f"Starting Coder MCP server on {args.host}:{args.port}")
    if args.relay:
        LOGGER.info(f"  SSE endpoint: http://{args.host}:{args.port}/sse (for MCPWorker)")
        LOGGER.info(f"  WebSocket endpoint: ws://{args.host}:{args.port}/ws (for relay client)")
    uvicorn.run(starlette_app, host=args.host, port=args.port)
