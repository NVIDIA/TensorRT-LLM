"""Apiary-backed MCP server for the Scaffolding Coder tools."""

from __future__ import annotations

import argparse
import base64
import contextvars
import hmac
import logging
import os
import re
import shlex
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import httpx
import uvicorn
from apiary_client import ApiarySessionMux, TaskResult
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route


class _ToolFormatError(Exception):
    """Raised by formatters when a tool's raw output can't be parsed."""


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

mcp = FastMCP("coder_tools")

_client_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_client_id",
    default="stdio",
)
_session: ApiarySessionMux
_mcp_auth_token: str | None = os.getenv("MCP_AUTH_TOKEN")

_GREP_LINE_RE = re.compile(r"^(.*?):(\d+):(.*)$")


@dataclass
class PlanState:
    current_plan: list[dict[str, str]] = field(default_factory=list)
    explanation: str | None = None


_plan_states: dict[str, PlanState] = {}


def _cid() -> str:
    return _client_id.get()


def _q(value: str) -> str:
    return shlex.quote(value)


def _current_plan_state() -> PlanState:
    cid = _cid()
    if cid not in _plan_states:
        _plan_states[cid] = PlanState()
    return _plan_states[cid]


def _format_shell_result(result: TaskResult) -> str:
    stdout = result.stdout.rstrip("\n")
    stderr = result.stderr.rstrip("\n")
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if result.timed_out:
        parts.append("[timed out]")
    body = "\n".join(parts) if parts else "(no output)"
    return f"{body}\n\n[Exit code: {result.exit_code}]"


def _format_http_error(error: httpx.HTTPError) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        body = error.response.text.strip()
        try:
            payload = error.response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            body = str(payload["error"])
        return body or str(error)
    return str(error)


def _format_tool_error(error: Exception) -> str:
    if isinstance(error, httpx.HTTPError):
        return f"Error: {_format_http_error(error)}"
    return f"Error: {error}"


def _build_read_file_command(
    file_path: str,
    offset: int,
    limit: int | None,
) -> str:
    start = max(offset, 1)
    line_limit = -1 if limit is None else max(limit, 0)
    return (
        "awk "
        f"-v start={start} "
        f"-v max_lines={line_limit} "
        "'BEGIN { total = 0; printed = 0 } "
        "{ total = NR } "
        "NR >= start && (max_lines < 0 || printed < max_lines) { "
        'printf "%6d|%s\\n", NR, $0; '
        "printed++ "
        "} "
        'END { printf "__CODER_META__\\t%d\\t%d\\n", total, printed }'
        f"' {_q(file_path)}"
    )


def _format_read_result(result: TaskResult) -> str:
    if result.exit_code != 0:
        raise _ToolFormatError(result.stderr.strip() or result.stdout.strip() or "read_file failed")

    lines = result.stdout.splitlines()
    if not lines or not lines[-1].startswith("__CODER_META__\t"):
        raise _ToolFormatError("Malformed read_file response")

    _, total_str, returned_str = lines[-1].split("\t")
    content = "\n".join(lines[:-1]).rstrip("\n")
    footer = f"[Total lines: {int(total_str)}, Lines returned: {int(returned_str)}]"
    return f"{content}\n\n{footer}" if content else footer


def _build_list_dir_command(dir_path: str, depth: int) -> str:
    max_depth = max(depth, 1)
    return f'find {_q(dir_path)} -mindepth 1 -maxdepth {max_depth} -printf "%y\\t%P\\n"'


def _format_list_result(
    result: TaskResult,
    *,
    offset: int,
    limit: int | None,
) -> str:
    if result.exit_code != 0:
        raise _ToolFormatError(result.stderr.strip() or result.stdout.strip() or "list_dir failed")

    entries: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        entry_type, _, name = line.partition("\t")
        if not name:
            continue
        entries.append((entry_type, name))

    entries.sort(key=lambda item: (_entry_sort_key(item[0]), item[1]))

    start_index = max(offset, 1) - 1
    display_offset = start_index + 1
    end_index = None if limit is None else start_index + max(limit, 0)
    page = entries[start_index:end_index]

    rendered = "\n".join(
        f"{display_offset + index:6d}. [{_entry_label(entry_type):5s}] {name}"
        for index, (entry_type, name) in enumerate(page)
    )
    footer = f"[Total entries: {len(entries)}, Entries returned: {len(page)}]"
    return f"{rendered}\n\n{footer}" if rendered else footer


def _entry_sort_key(entry_type: str) -> int:
    if entry_type == "d":
        return 0
    if entry_type == "f":
        return 1
    return 2


def _entry_label(entry_type: str) -> str:
    return {
        "d": "dir",
        "f": "file",
        "l": "link",
    }.get(entry_type, entry_type)


def _build_grep_command(
    pattern: str,
    search_path: str,
    include: str | None,
) -> str:
    parts = [
        "grep -RIn --color=never --binary-files=without-match",
    ]
    if include:
        parts.append(f"--include={_q(include)}")
    parts.extend(
        [
            "--",
            _q(pattern),
            _q(search_path),
            "2>/dev/null",
        ]
    )
    return " ".join(parts)


def _format_grep_result(result: TaskResult, limit: int) -> str:
    if result.exit_code not in {0, 1}:
        raise _ToolFormatError(
            result.stderr.strip() or result.stdout.strip() or "grep_files failed"
        )

    grouped: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
    for line in result.stdout.splitlines():
        match = _GREP_LINE_RE.match(line)
        if match is None:
            continue
        path, line_no, content = match.groups()
        grouped.setdefault(path, []).append((int(line_no), content))

    limited_items = list(grouped.items())[: max(limit, 0)]
    output_lines: list[str] = []
    for path, matches in limited_items:
        output_lines.append(f"{path}:")
        for line_no, content in matches[:10]:
            truncated = content if len(content) <= 200 else content[:200] + "..."
            output_lines.append(f"  {line_no}: {truncated}")
        if len(matches) > 10:
            output_lines.append(f"  ... ({len(matches) - 10} more matches)")
        output_lines.append("")

    body = "\n".join(output_lines).rstrip()
    footer = f"[Files matched: {len(grouped)}]"
    return f"{body}\n\n{footer}" if body else footer


@mcp.tool()
async def read_file(
    file_path: str,
    offset: int = 1,
    limit: int | None = None,
    mode: str = "slice",
) -> str:
    """Read a file with 1-indexed line numbers from the sandbox."""
    _ = mode
    try:
        result = await _session.execute(_cid(), _build_read_file_command(file_path, offset, limit))
        return _format_read_result(result)
    except (httpx.HTTPError, _ToolFormatError) as error:
        return _format_tool_error(error)


@mcp.tool()
async def list_dir(
    dir_path: str,
    offset: int = 1,
    limit: int | None = None,
    depth: int = 1,
) -> str:
    """List directory contents from the sandbox."""
    try:
        result = await _session.shell(_cid(), _build_list_dir_command(dir_path, depth))
        return _format_list_result(result, offset=offset, limit=limit)
    except (httpx.HTTPError, _ToolFormatError) as error:
        return _format_tool_error(error)


@mcp.tool()
async def grep_files(
    pattern: str,
    include: str | None = None,
    path: str | None = None,
    limit: int = 100,
) -> str:
    """Search files for a regex pattern inside the sandbox."""
    try:
        search_path = path or "."
        result = await _session.shell(
            _cid(),
            _build_grep_command(pattern, search_path, include),
        )
        return _format_grep_result(result, limit)
    except (httpx.HTTPError, _ToolFormatError) as error:
        return _format_tool_error(error)


@mcp.tool(name="exec")
async def exec_tool(
    command: list[str],
    workdir: str | None = None,
    timeout_ms: int | None = None,
) -> str:
    """Execute a command array inside the sandbox without shell interpretation."""
    if not command:
        return "Error: command array cannot be empty"
    try:
        result = await _session.execute(
            _cid(),
            shlex.join(command),
            timeout_ms=timeout_ms,
            working_dir=workdir,
        )
        return _format_shell_result(result)
    except httpx.HTTPError as error:
        return _format_tool_error(error)


@mcp.tool()
async def shell(
    command: str,
    workdir: str | None = None,
    timeout_ms: int | None = None,
) -> str:
    """Execute a shell command string inside the sandbox."""
    if not command:
        return "Error: command cannot be empty"
    try:
        result = await _session.shell(
            _cid(),
            command,
            timeout_ms=timeout_ms,
            working_dir=workdir,
        )
        return _format_shell_result(result)
    except httpx.HTTPError as error:
        return _format_tool_error(error)


@mcp.tool()
async def update_plan(
    plan: list[dict[str, str]],
    explanation: str | None = None,
) -> str:
    """Update the per-client plan state."""
    if not plan:
        return "Error: plan is required and cannot be empty"

    valid_statuses = {"pending", "in_progress", "completed"}
    in_progress_count = 0
    validated_plan: list[dict[str, str]] = []

    for index, item in enumerate(plan, start=1):
        step = item.get("step")
        status = item.get("status")
        if not step:
            return f"Error: plan item {index} missing 'step'"
        if status not in valid_statuses:
            allowed = ", ".join(sorted(valid_statuses))
            return (
                f"Error: plan item {index} has invalid status '{status}'. Must be one of: {allowed}"
            )
        if status == "in_progress":
            in_progress_count += 1
        validated_plan.append({"step": step, "status": status})

    if in_progress_count > 1:
        return "Error: at most one plan step can be in_progress at a time"

    state = _current_plan_state()
    state.current_plan = validated_plan
    state.explanation = explanation

    output_lines: list[str] = []
    if explanation:
        output_lines.extend([f"Explanation: {explanation}", ""])
    output_lines.append("Plan:")
    output_lines.extend(
        f"  {index}. [{item['status']}] {item['step']}"
        for index, item in enumerate(validated_plan, start=1)
    )
    completed = sum(1 for item in validated_plan if item["status"] == "completed")
    total = len(validated_plan)
    progress = round(completed / total * 100, 1) if total else 0.0
    output_lines.extend(["", f"Progress: {completed}/{total} ({progress}%)"])
    return "\n".join(output_lines)


def _extract_code(code: str) -> str:
    triple_match = re.search(r"```[^\n]*\n(.+?)```", code, re.DOTALL)
    if triple_match:
        return triple_match.group(1)
    code_match = re.search(r"<code>(.*?)</code>", code, re.DOTALL)
    if code_match:
        return code_match.group(1)
    return code


@mcp.tool()
async def python_interpreter(
    code: str,
    timeout_ms: int | None = None,
    python_exe: str = "python3",
) -> str:
    """Execute Python code in the per-client sandbox."""
    code = _extract_code(code)
    if not code.strip():
        return "[PythonInterpreter Error]: Empty code."
    b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    py_src = f"import base64; exec(base64.b64decode({repr(b64)}).decode('utf-8'))"
    command = f"{_q(python_exe)} -c {_q(py_src)}"
    try:
        result = await _session.shell(_cid(), command, timeout_ms=timeout_ms)
    except httpx.HTTPError as error:
        return f"[PythonInterpreter Error]: {_format_http_error(error)}"

    parts: list[str] = []
    stdout = result.stdout.rstrip("\n")
    stderr = result.stderr.rstrip("\n")
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if result.timed_out:
        parts.append("[PythonInterpreter Error] TimeoutError: Execution timed out.")
    output = "\n".join(parts)
    return output if output.strip() else "Finished execution."


@mcp.tool()
async def think(thought: str) -> str:
    """Record a thought for the current client."""
    return f"Thought recorded: {thought}"


@mcp.tool()
async def complete_task(summary: str) -> str:
    """Signal task completion for the current client."""
    return f"Task completed: {summary}"


def create_starlette_app(
    mcp_server: Server,
    *,
    debug: bool = False,
) -> Starlette:
    """Create the SSE app for the Coder MCP server."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        if _mcp_auth_token:
            provided = request.headers.get("authorization", "")
            expected = f"Bearer {_mcp_auth_token}"
            if not hmac.compare_digest(provided.encode(), expected.encode()):
                return JSONResponse({"error": "unauthorized"}, status_code=401)

        cid = request.query_params.get("client_id") or uuid.uuid4().hex[:12]
        image = request.query_params.get("image", "")
        token = _client_id.set(cid)
        try:
            await _session.ensure_session(cid, image=image)
        except httpx.HTTPError as error:
            _client_id.reset(token)
            LOGGER.warning("Failed to create sandbox for client %s: %s", cid, error)
            return JSONResponse(
                {"error": _format_http_error(error)},
                status_code=502,
            )

        _session.attach(cid)
        LOGGER.info("Client %s connected (image=%s)", cid, image or "<default>")
        try:
            async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
            ) as (read_stream, write_stream):
                try:
                    await mcp_server.run(
                        read_stream,
                        write_stream,
                        mcp_server.create_initialization_options(),
                    )
                finally:
                    _session.detach(cid)
                    LOGGER.info("Client %s disconnected", cid)
        finally:
            _client_id.reset(token)

        return Response()

    async def handle_health(_: Request):
        return JSONResponse(
            {
                "status": "ok",
                "sessions": _session.active_sessions,
            }
        )

    async def on_startup():
        _session.start_reaper()

    async def on_shutdown():
        await _session.shutdown()

    return Starlette(
        debug=debug,
        routes=[
            Route("/health", endpoint=handle_health, methods=["GET"]),
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        on_startup=[on_startup],
        on_shutdown=[on_shutdown],
    )


def _install_uvloop() -> bool:
    try:
        import uvloop  # type: ignore[import-untyped]

        uvloop.install()
        LOGGER.info("Using uvloop event loop")
        return True
    except ImportError:
        LOGGER.info("uvloop not available, using default asyncio event loop")
        return False


def main() -> None:
    """CLI entry point for the Coder MCP server."""
    global _session, _mcp_auth_token

    _install_uvloop()

    parser = argparse.ArgumentParser(description="Run the Apiary-backed Coder MCP server")
    parser.add_argument("--host", default="0.0.0.0", help="SSE bind host")
    parser.add_argument("--port", type=int, default=8083, help="SSE bind port")
    parser.add_argument(
        "--apiary-url",
        default=os.getenv("APIARY_URL", "http://127.0.0.1:8080"),
        help="Apiary daemon URL",
    )
    parser.add_argument(
        "--apiary-token",
        default=os.getenv("APIARY_API_TOKEN"),
        help="Apiary daemon bearer token",
    )
    parser.add_argument(
        "--mcp-token",
        default=os.getenv("MCP_AUTH_TOKEN"),
        help="Require this bearer token on the MCP SSE endpoint",
    )
    parser.add_argument(
        "--default-image",
        required=True,
        help="Fallback Docker image for sandbox sessions when an SSE client "
        "omits the `image` query parameter (must already be registered with the daemon)",
    )
    parser.add_argument(
        "--working-dir",
        default=os.getenv("APIARY_WORKING_DIR", "/workspace"),
        help="Default sandbox working directory",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=300.0,
        help="Seconds before an unconnected sandbox is reaped (default 300)",
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default="sse",
        help="MCP transport (default: sse)",
    )
    parser.add_argument(
        "--backlog",
        type=int,
        default=2048,
        help="TCP listen backlog (default 2048)",
    )
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=500,
        help="Max concurrent connections (default 500)",
    )
    args = parser.parse_args()

    _session = ApiarySessionMux(
        image=args.default_image,
        apiary_url=args.apiary_url,
        apiary_token=args.apiary_token,
        working_dir=args.working_dir,
        idle_timeout=args.idle_timeout,
        on_client_destroy=lambda cid: _plan_states.pop(cid, None),
    )
    _mcp_auth_token = args.mcp_token

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    app = create_starlette_app(mcp._mcp_server, debug=True)
    LOGGER.info("Starting Coder MCP server on %s:%s", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        backlog=args.backlog,
        limit_concurrency=args.limit_concurrency,
        timeout_keep_alive=30,
        log_level="info",
    )


if __name__ == "__main__":
    main()
