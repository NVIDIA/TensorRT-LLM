"""Local tool implementations for the Agent Client.

Provides a `ToolExecutor` class that encapsulates all tool implementations
used by the agent client for local execution. Tools include file system
operations, shell execution, patch application, and planning utilities.

All state (working directory, plan) is held within the `ToolExecutor` instance
rather than module-level globals, making it safe to create multiple instances
for different working directories or concurrent sessions.
"""

import asyncio
import fnmatch
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes agent tool calls locally.

    Encapsulates the working directory and plan state, providing
    implementations for all tool types that the agent may invoke.
    Each public async method corresponds to a tool name in the agent
    protocol and can be dispatched via :meth:`execute`.

    Args:
        working_dir: Absolute or relative path used as the root for all
            file and shell operations. Resolved to an absolute path on init.
    """

    AVAILABLE_TOOLS = frozenset(
        {
            "read_file",
            "list_dir",
            "grep_files",
            "apply_patch",
            "shell",
            "shell_command",
            "think",
            "complete_task",
            "update_plan",
        }
    )

    def __init__(self, working_dir: str):
        self.working_dir = os.path.abspath(working_dir)
        self._current_plan: List[dict] = []
        self._plan_explanation: Optional[str] = None

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the working directory if not absolute."""
        if not os.path.isabs(path):
            return os.path.join(self.working_dir, path)
        return path

    async def execute(self, tool_name: str, arguments: dict) -> str:
        """Dispatch and execute a tool by name.

        Args:
            tool_name: Name of the tool (must be in ``AVAILABLE_TOOLS``).
            arguments: Keyword arguments forwarded to the tool method.

        Returns:
            Tool execution result as a string.
        """
        if tool_name not in self.AVAILABLE_TOOLS:
            return (
                f"Error: Unknown tool: {tool_name}. "
                f"Available tools: {', '.join(sorted(self.AVAILABLE_TOOLS))}"
            )

        handler = getattr(self, tool_name)
        try:
            return await handler(**arguments)
        except TypeError as e:
            return f"Error: Invalid arguments for {tool_name}: {e}"
        except Exception as e:
            return f"Error executing {tool_name}: {type(e).__name__}: {e}"

    # =====================================================================
    # File System Tools
    # =====================================================================

    async def read_file(
        self,
        file_path: str,
        offset: int = 1,
        limit: Optional[int] = None,
        mode: str = "slice",
    ) -> str:
        """Read a file with 1-indexed line numbers.

        Args:
            file_path: Path to the file (absolute or relative to working_dir).
            offset: 1-indexed starting line number.
            limit: Maximum number of lines to return, or None for all.
            mode: Read mode (currently only ``"slice"`` is implemented).

        Returns:
            Numbered file contents with summary metadata.
        """
        if not file_path:
            return "Error: file_path is required"

        file_path = self._resolve_path(file_path)

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
            start_idx = offset - 1
            end_idx = start_idx + limit if limit else len(lines)

            result_lines = []
            for i in range(start_idx, min(end_idx, len(lines))):
                result_lines.append((i + 1, lines[i]))

            output_lines = [f"{ln:6d}|{lc}" for ln, lc in result_lines]
            result = "\n".join(output_lines)
            return f"{result}\n\n[Total lines: {total_lines}, Lines returned: {len(result_lines)}]"

        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def list_dir(
        self,
        dir_path: str,
        offset: int = 1,
        limit: Optional[int] = None,
        depth: int = 1,
    ) -> str:
        """List directory contents with type labels.

        Args:
            dir_path: Path to the directory.
            offset: 1-indexed starting entry.
            limit: Maximum entries to return, or None for all.
            depth: Maximum recursion depth (1 = no recursion).

        Returns:
            Formatted directory listing with summary metadata.
        """
        if not dir_path:
            return "Error: dir_path is required"

        dir_path = self._resolve_path(dir_path)

        if not os.path.exists(dir_path):
            return f"Error: Directory not found: {dir_path}"
        if not os.path.isdir(dir_path):
            return f"Error: Path is not a directory: {dir_path}"
        if offset < 1:
            return "Error: offset must be 1 or greater"
        if depth < 1:
            return "Error: depth must be 1 or greater"

        def get_entry_type(p: Path) -> str:
            if p.is_dir():
                return "dir"
            elif p.is_symlink():
                return "link"
            elif p.is_file():
                return "file"
            return "other"

        def list_entries(
            path: Path,
            current_depth: int,
            max_depth: int,
            prefix: str = "",
        ) -> list:
            entries = []
            try:
                items = sorted(
                    path.iterdir(),
                    key=lambda x: (not x.is_dir(), x.name.lower()),
                )
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
                    entries.extend(list_entries(item, current_depth + 1, max_depth, prefix + "  "))
            return entries

        try:
            path = Path(dir_path)
            all_entries = list_entries(path, 1, depth)
            total = len(all_entries)
            start_idx = offset - 1
            end_idx = start_idx + limit if limit else len(all_entries)
            selected = all_entries[start_idx:end_idx]

            output_lines = [
                f"{i:6d}. [{t:5s}] {n}" for i, (n, t) in enumerate(selected, start=offset)
            ]
            result = "\n".join(output_lines)
            return f"{result}\n\n[Total entries: {total}, Entries returned: {len(selected)}]"

        except Exception as e:
            return f"Error: {str(e)}"

    async def grep_files(
        self,
        pattern: str,
        include: Optional[str] = None,
        path: Optional[str] = None,
        limit: int = 100,
    ) -> str:
        """Search files for a regex pattern.

        The actual file-system walk and content scanning is offloaded to
        a thread via :func:`asyncio.to_thread` so that large directory
        trees do not block the event loop.

        Args:
            pattern: Regular expression to search for.
            include: Glob pattern to filter file names (e.g. ``"*.py"``).
            path: Directory or file to search in, defaults to working_dir.
            limit: Maximum number of matching files to return.

        Returns:
            Matching lines grouped by file, sorted by modification time.
        """
        if not pattern:
            return "Error: pattern is required"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        search_path = self._resolve_path(path) if path else self.working_dir
        if not os.path.exists(search_path):
            return f"Error: Path not found: {search_path}"

        def _grep_sync() -> str:
            """Synchronous grep implementation run in a worker thread."""

            def matches_glob(filepath: str, glob_pattern: str) -> bool:
                if "{" in glob_pattern and "}" in glob_pattern:
                    base, rest = glob_pattern.split("{", 1)
                    extensions, suffix = rest.split("}", 1)
                    return any(
                        fnmatch.fnmatch(filepath, f"{base}{ext}{suffix}")
                        for ext in extensions.split(",")
                    )
                return fnmatch.fnmatch(filepath, glob_pattern)

            def search_file(filepath: str) -> list:
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
                for filepath, _, matches in matching_files:
                    rel = (
                        os.path.relpath(filepath, search_path)
                        if os.path.isdir(search_path)
                        else filepath
                    )
                    output_lines.append(f"\n{rel}:")
                    for ln, lc in matches[:10]:
                        truncated = lc[:200] + "..." if len(lc) > 200 else lc
                        output_lines.append(f"  {ln}: {truncated}")
                    if len(matches) > 10:
                        output_lines.append(f"  ... ({len(matches) - 10} more matches)")

                result = "\n".join(output_lines).strip()
                limit_msg = " (limit reached)" if len(matching_files) >= limit else ""
                return f"{result}\n\n[Files matched: {len(matching_files)}{limit_msg}]"

            except Exception as e:
                return f"Error: {str(e)}"

        return await asyncio.to_thread(_grep_sync)

    # =====================================================================
    # Patch Tool
    # =====================================================================

    async def apply_patch(self, patch: str) -> str:
        """Apply a structured patch to files atomically.

        The patch is processed in two phases:

        1. **Validate** – parse every operation and compute the resulting
           file contents in memory.  If *any* operation fails (missing
           context, unreadable file, …) an error is returned and
           **nothing** is written to disk.
        2. **Apply** – write all computed results to disk in one pass.

        Supports three operations within a single patch block:

        - **Add File:** Create a new file with the given content.
        - **Delete File:** Remove an existing file.
        - **Update File:** Apply context-based hunks to modify a file,
          optionally moving it to a new path.

        The patch format uses ``*** Begin Patch`` / ``*** End Patch``
        delimiters, with ``+`` prefixed lines for additions and ``-``
        prefixed lines for removals within ``@@`` context hunks.

        Args:
            patch: The patch string in the custom patch format.

        Returns:
            Summary of applied operations, or an error message.
        """
        lines = patch.strip().split("\n")

        if not lines or lines[0].strip() != "*** Begin Patch":
            return 'Error: Patch must start with "*** Begin Patch"'
        if lines[-1].strip() != "*** End Patch":
            return 'Error: Patch must end with "*** End Patch"'

        lines = lines[1:-1]
        i = 0

        # Phase 1: Parse & validate – collect planned operations.
        # Each entry is a dict describing a single file operation.
        planned_ops: List[dict] = []

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # --- Add File ---
            if line.startswith("*** Add File:"):
                filepath = line[len("*** Add File:") :].strip()
                filepath = self._resolve_path(filepath)
                i += 1
                content_lines = []
                while i < len(lines) and not lines[i].strip().startswith("***"):
                    cl = lines[i]
                    if cl.startswith("+"):
                        content_lines.append(cl[1:])
                    i += 1
                planned_ops.append(
                    {
                        "type": "add",
                        "filepath": filepath,
                        "content": "\n".join(content_lines),
                    }
                )

            # --- Delete File ---
            elif line.startswith("*** Delete File:"):
                filepath = line[len("*** Delete File:") :].strip()
                filepath = self._resolve_path(filepath)
                i += 1
                planned_ops.append(
                    {
                        "type": "delete",
                        "filepath": filepath,
                        "exists": os.path.exists(filepath),
                    }
                )

            # --- Update File ---
            elif line.startswith("*** Update File:"):
                result = self._compute_update_hunk(lines, i)
                if isinstance(result, str):
                    return result  # Validation error – nothing written
                op, i = result
                planned_ops.append(op)

            else:
                i += 1

        if not planned_ops:
            return "No changes applied"

        # Phase 2: Apply – write all changes to disk.
        results = []
        for op in planned_ops:
            try:
                if op["type"] == "add":
                    dirpath = os.path.dirname(op["filepath"])
                    if dirpath:
                        os.makedirs(dirpath, exist_ok=True)
                    with open(op["filepath"], "w") as f:
                        f.write(op["content"])
                    results.append(f"Added file: {op['filepath']}")

                elif op["type"] == "delete":
                    if op["exists"]:
                        os.remove(op["filepath"])
                        results.append(f"Deleted file: {op['filepath']}")
                    else:
                        results.append(f"File already deleted or not found: {op['filepath']}")

                elif op["type"] == "update":
                    target = op["target"]
                    dirpath = os.path.dirname(target)
                    if dirpath:
                        os.makedirs(dirpath, exist_ok=True)
                    with open(target, "w") as f:
                        f.write(op["content"])
                    if op.get("move_from") and os.path.exists(op["move_from"]):
                        os.remove(op["move_from"])
                        results.append(f"Updated and moved: {op['move_from']} -> {target}")
                    else:
                        results.append(f"Updated file: {target}")

            except Exception as e:
                return (
                    f"Error: Failed to apply {op['type']} "
                    f"{op.get('filepath') or op.get('target')}: {e}"
                )

        return "; ".join(results)

    def _compute_update_hunk(
        self,
        lines: List[str],
        i: int,
    ) -> Union[str, tuple]:
        """Parse and validate a single Update File hunk from a patch.

        Computes the resulting file content **in memory** without writing
        to disk, so that the caller can abort the entire patch on error.

        Context matching uses exact-line match first (stripped), then
        falls back to substring containment.  Each ``@@`` hunk resumes
        searching from after the previous hunk's position so that
        repeated patterns in the file do not cause mis-targeting.

        Args:
            lines: All patch lines (between Begin/End markers).
            i: Current line index pointing to the ``*** Update File:`` line.

        Returns:
            On success, a tuple ``(operation_dict, next_line_index)``.
            On error, an error message string.
        """
        line = lines[i].strip()
        filepath = line[len("*** Update File:") :].strip()
        filepath = self._resolve_path(filepath)
        i += 1
        new_filepath = None

        if i < len(lines) and lines[i].strip().startswith("*** Move to:"):
            new_filepath = lines[i].strip()[len("*** Move to:") :].strip()
            new_filepath = self._resolve_path(new_filepath)
            i += 1

        try:
            with open(filepath, "r") as f:
                file_content = f.read()
            file_lines = file_content.split("\n")
        except Exception as e:
            return f"Error: Failed to read file {filepath}: {e}"

        # Track the last matched position so successive hunks search
        # forward rather than always matching the first occurrence.
        search_from = 0

        while i < len(lines) and not lines[i].strip().startswith("*** "):
            hunk_line = lines[i]

            if hunk_line.startswith("@@"):
                context = hunk_line[2:].strip()
                i += 1

                # Find context line: try exact match first, then
                # substring containment.  Search starts from after the
                # last matched hunk to avoid re-matching earlier lines.
                context_idx = None
                for idx in range(search_from, len(file_lines)):
                    if file_lines[idx].strip() == context:
                        context_idx = idx
                        break
                if context_idx is None:
                    for idx in range(search_from, len(file_lines)):
                        if context in file_lines[idx]:
                            context_idx = idx
                            break

                if context_idx is None:
                    return f'Error: Could not find context "{context}" in {filepath}'

                search_from = context_idx + 1
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
                        for si in range(
                            context_idx,
                            min(context_idx + 20, len(file_lines)),
                        ):
                            if si < len(file_lines) and to_remove in file_lines[si]:
                                file_lines.pop(si)
                                found = True
                                break
                        if not found:
                            return (
                                f'Error: Could not find line to remove: "{to_remove}" in {filepath}'
                            )
                    elif change_line.startswith("+"):
                        file_lines.insert(current_idx, change_line[1:])
                        current_idx += 1
                    i += 1
            else:
                i += 1

        target = new_filepath if new_filepath else filepath
        op = {
            "type": "update",
            "filepath": filepath,
            "target": target,
            "content": "\n".join(file_lines),
        }
        if new_filepath:
            op["move_from"] = filepath

        return op, i

    # =====================================================================
    # Shell Tools
    # =====================================================================

    async def _run_subprocess(
        self,
        command: list,
        workdir: Optional[str],
        timeout_ms: Optional[int],
    ) -> str:
        """Run a subprocess and return formatted output.

        Common implementation shared by :meth:`shell` and :meth:`shell_command`.

        Args:
            command: Command as a list of strings (already prepared).
            workdir: Working directory override, or None for
                :attr:`working_dir`.
            timeout_ms: Timeout in milliseconds, or None for no timeout.

        Returns:
            Formatted output string with exit code appended.
        """
        cwd = self._resolve_path(workdir) if workdir else self.working_dir
        if not os.path.isdir(cwd):
            return f"Error: Working directory does not exist: {cwd}"

        timeout = timeout_ms / 1000.0 if timeout_ms else None

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                command,
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
            return f"Error: Command not found: {command[0]}"
        except PermissionError:
            return f"Error: Permission denied: {command[0]}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def shell(
        self,
        command: Union[list, str],
        workdir: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> str:
        """Execute a command as an array via execvp (no shell interpretation).

        Args:
            command: A list of strings (``["ls", "-la"]``) or a JSON string
                encoding such a list (``'["ls", "-la"]'``).
            workdir: Working directory override.
            timeout_ms: Timeout in milliseconds, or None for no timeout.

        Returns:
            Command output with exit code.
        """
        if isinstance(command, list):
            command_list = command
        elif isinstance(command, str):
            try:
                command_list = json.loads(command)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON for command: {command}"
        else:
            return f"Error: Invalid command format: {type(command)}"

        if not command_list:
            return "Error: Command array cannot be empty"

        return await self._run_subprocess(command_list, workdir, timeout_ms)

    async def shell_command(
        self,
        command: str,
        workdir: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> str:
        """Execute a shell command string in the user's default shell.

        On Unix, the command is run via ``$SHELL -l -c <command>``.
        On Windows, it is run via ``powershell.exe -Command <command>``.

        Args:
            command: Shell command string.
            workdir: Working directory override.
            timeout_ms: Timeout in milliseconds, or None for no timeout.

        Returns:
            Command output with exit code.
        """
        if not command:
            return "Error: Command cannot be empty"

        if sys.platform == "win32":
            shell_cmd = ["powershell.exe", "-Command", command]
        else:
            shell_prog = os.environ.get("SHELL", "/bin/sh")
            shell_cmd = [shell_prog, "-l", "-c", command]

        return await self._run_subprocess(shell_cmd, workdir, timeout_ms)

    # =====================================================================
    # Planning & Control Tools
    # =====================================================================

    async def update_plan(
        self,
        plan: Union[list, str],
        explanation: Optional[str] = None,
    ) -> str:
        """Update the task plan with steps and progress tracking.

        Args:
            plan: A list of dicts (``[{"step": "...", "status": "pending"}]``)
                or a JSON string encoding such a list.
            explanation: Optional high-level explanation for the plan.

        Returns:
            Formatted plan display with progress percentage.
        """
        if isinstance(plan, str):
            try:
                plan_list = json.loads(plan)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON for plan: {e}"
        elif isinstance(plan, list):
            plan_list = plan
        else:
            return f"Error: Invalid plan format: {type(plan)}"

        if not plan_list:
            return "Error: plan is required and cannot be empty"

        valid_statuses = {"pending", "in_progress", "completed"}
        in_progress_count = 0
        validated = []

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
                    f"Error: Plan item {i + 1} has invalid status "
                    f"'{status}'. Must be one of: "
                    f"{', '.join(valid_statuses)}"
                )
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"step": step, "status": status})

        if in_progress_count > 1:
            return (
                f"Error: At most one step can be in_progress at a time, "
                f"but found {in_progress_count}"
            )

        self._current_plan = validated
        self._plan_explanation = explanation

        output_lines = []
        if explanation:
            output_lines.extend([f"Explanation: {explanation}", ""])

        output_lines.append("Plan:")
        status_icons = {
            "pending": "\u25cb",
            "in_progress": "\u25d0",
            "completed": "\u25cf",
        }
        for i, item in enumerate(validated, start=1):
            icon = status_icons.get(item["status"], "?")
            output_lines.append(f"  {i}. [{icon}] {item['step']} ({item['status']})")

        total = len(validated)
        completed = sum(1 for item in validated if item["status"] == "completed")
        progress = round(completed / total * 100, 1) if total > 0 else 0
        output_lines.extend(["", f"Progress: {completed}/{total} ({progress}%)"])

        return "\n".join(output_lines)

    async def think(self, thought: str) -> str:
        """Record a thought or reflection about the current task.

        Args:
            thought: The agent's reasoning or reflection.

        Returns:
            Acknowledgement string.
        """
        return f"Thought recorded: {thought}"

    async def complete_task(self, summary: str) -> str:
        """Signal that the task has been completed.

        Args:
            summary: Summary of what was accomplished.

        Returns:
            Completion acknowledgement string.
        """
        return f"Task completed: {summary}"
