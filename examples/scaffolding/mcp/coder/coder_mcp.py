"""Apiary-backed MCP server for Coder tools.

This server preserves the Coder agent's MCP tool surface while moving all file
and shell execution into Apiary sandboxes. Each MCP client gets its own
persistent sandbox session identified by ``client_id`` on the SSE endpoint.
"""

import argparse
import asyncio
import contextvars
import hmac
import logging
import os
import shlex
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import uvicorn
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

mcp = FastMCP("coder_tools")

_client_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_client_id",
    default="stdio",
)

_REAPER_INTERVAL = 60


@dataclass
class PlanState:
    current_plan: list[dict[str, str]] = field(default_factory=list)
    explanation: Optional[str] = None


_plan_states: dict[str, PlanState] = {}


class SessionManager:
    """Manage per-client Apiary sandbox sessions."""

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        working_dir: str = "/workspace",
        idle_timeout: float = 1800.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._working_dir = working_dir
        self._idle_timeout = idle_timeout

        self._sessions: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._refcounts: dict[str, int] = {}
        self._detached_at: dict[str, float] = {}
        self._client_layers: dict[str, list[str]] = {}

        self._client: Optional[httpx.AsyncClient] = None
        self._reaper_task: Optional[asyncio.Task] = None

    @property
    def working_dir(self) -> str:
        return self._working_dir

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(timeout=300.0),
            )
        return self._client

    def _lock_for(self, cid: str) -> asyncio.Lock:
        if cid not in self._locks:
            self._locks[cid] = asyncio.Lock()
        return self._locks[cid]

    def set_client_layers(self, cid: str, layers: list[str]) -> None:
        """Register OverlayFS lower-dir layers for *cid*.

        Called when an SSE connection provides ``base_image`` query
        parameters.  The layers are passed to Apiary when the session
        for this client is created.
        """
        if layers:
            self._client_layers[cid] = layers

    async def _create_apiary_session(
        self,
        base_image: Optional[list[str]] = None,
    ) -> str:
        client = await self._get_client()
        payload: dict[str, Any] = {"working_dir": self._working_dir}
        if base_image:
            payload["base_image"] = base_image
        response = await client.post("/api/v1/sessions", json=payload)
        response.raise_for_status()
        return response.json()["session_id"]

    async def _destroy_apiary_session(self, session_id: str) -> None:
        try:
            client = await self._get_client()
            await client.delete(f"/api/v1/sessions/{session_id}")
        except Exception:
            LOGGER.warning("Failed to destroy Apiary session %s", session_id, exc_info=True)

    async def _ensure_session(self, cid: str) -> str:
        lock = self._lock_for(cid)
        async with lock:
            if cid in self._sessions:
                return self._sessions[cid]
            layers = self._client_layers.get(cid)
            session_id = await self._create_apiary_session(base_image=layers)
            self._sessions[cid] = session_id
            LOGGER.info("Created Apiary session %s for client %s", session_id, cid)
            return session_id

    async def _destroy_client(self, cid: str) -> None:
        session_id = self._sessions.pop(cid, None)
        self._locks.pop(cid, None)
        self._refcounts.pop(cid, None)
        self._detached_at.pop(cid, None)
        self._client_layers.pop(cid, None)
        _plan_states.pop(cid, None)
        if session_id:
            await self._destroy_apiary_session(session_id)
            LOGGER.info("Destroyed Apiary session %s for client %s", session_id, cid)

    def attach(self, cid: str) -> None:
        self._refcounts[cid] = self._refcounts.get(cid, 0) + 1
        self._detached_at.pop(cid, None)

    def detach(self, cid: str) -> None:
        count = self._refcounts.get(cid, 1) - 1
        if count <= 0:
            self._refcounts.pop(cid, None)
            self._detached_at[cid] = time.monotonic()
        else:
            self._refcounts[cid] = count

    def start_reaper(self) -> None:
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reap_loop())

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(_REAPER_INTERVAL)
            now = time.monotonic()
            for cid in list(self._detached_at):
                if cid in self._refcounts:
                    self._detached_at.pop(cid, None)
                    continue
                if cid not in self._sessions:
                    self._detached_at.pop(cid, None)
                    continue
                if now - self._detached_at[cid] >= self._idle_timeout:
                    LOGGER.info("Reaping idle client %s", cid)
                    await self._destroy_client(cid)

    async def execute(
        self,
        command: str,
        *,
        timeout_ms: Optional[int] = None,
        working_dir: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        cid = _client_id.get()
        wrapped = f"bash -c {shlex.quote(command)}"
        session_id = await self._ensure_session(cid)
        client = await self._get_client()

        payload: dict[str, Any] = {
            "command": wrapped,
            "session_id": session_id,
        }
        if timeout_ms is not None:
            payload["timeout_ms"] = timeout_ms
        if working_dir is not None:
            payload["working_dir"] = working_dir
        if env:
            payload["env"] = env

        response = await client.post("/api/v1/tasks", json=payload)
        if response.status_code == 404:
            LOGGER.warning("Lost session %s for client %s; recreating", session_id, cid)
            lock = self._lock_for(cid)
            async with lock:
                self._sessions.pop(cid, None)
            session_id = await self._ensure_session(cid)
            payload["session_id"] = session_id
            response = await client.post("/api/v1/tasks", json=payload)

        response.raise_for_status()
        return response.json()

    async def _file_request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to an Apiary file API endpoint; injects session_id and retries on 404."""
        cid = _client_id.get()
        session_id = await self._ensure_session(cid)
        client = await self._get_client()
        full_payload = {**payload, "session_id": session_id}
        response = await client.post(path, json=full_payload)
        if response.status_code == 404:
            LOGGER.warning("Lost session %s for client %s; recreating", session_id, cid)
            lock = self._lock_for(cid)
            async with lock:
                self._sessions.pop(cid, None)
            session_id = await self._ensure_session(cid)
            full_payload["session_id"] = session_id
            response = await client.post(path, json=full_payload)
        response.raise_for_status()
        return response.json()

    async def read_file(
        self,
        path: str,
        offset: int = 1,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """Read file via Apiary /api/v1/files/read."""
        payload: dict[str, Any] = {"path": path, "offset": offset}
        if limit is not None:
            payload["limit"] = limit
        return await self._file_request("/api/v1/files/read", payload)

    async def list_dir(
        self,
        path: str,
        offset: int = 1,
        limit: Optional[int] = None,
        depth: int = 1,
    ) -> dict[str, Any]:
        """List directory via Apiary /api/v1/files/list."""
        payload: dict[str, Any] = {"path": path, "offset": offset, "depth": depth}
        if limit is not None:
            payload["limit"] = limit
        return await self._file_request("/api/v1/files/list", payload)

    async def grep_files(
        self,
        pattern: str,
        include: Optional[str] = None,
        path: Optional[str] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Grep files via Apiary /api/v1/files/grep."""
        payload: dict[str, Any] = {"pattern": pattern, "limit": limit}
        if include is not None:
            payload["include"] = include
        if path is not None:
            payload["path"] = path
        return await self._file_request("/api/v1/files/grep", payload)

    async def apply_patch(self, patch: str) -> dict[str, Any]:
        """Apply patch via Apiary /api/v1/files/patch."""
        return await self._file_request("/api/v1/files/patch", {"patch": patch})

    async def shutdown(self) -> None:
        if self._reaper_task:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
        for cid in list(self._sessions):
            await self._destroy_client(cid)
        if self._client and not self._client.is_closed:
            await self._client.aclose()


_session = SessionManager(
    os.getenv("APIARY_URL", "http://127.0.0.1:8080"),
    os.getenv("APIARY_API_TOKEN"),
    os.getenv("APIARY_WORKING_DIR", "/workspace"),
)
_mcp_auth_token: Optional[str] = os.getenv("MCP_AUTH_TOKEN")


def _current_plan_state() -> PlanState:
    cid = _client_id.get()
    if cid not in _plan_states:
        _plan_states[cid] = PlanState()
    return _plan_states[cid]


def _format_shell_result(result: dict[str, Any]) -> str:
    stdout = (result.get("stdout") or "").rstrip("\n")
    stderr = (result.get("stderr") or "").rstrip("\n")
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(stderr)
    if result.get("timed_out"):
        parts.append("[timed out]")
    body = "\n".join(parts) if parts else "(no output)"
    return f"{body}\n\n[Exit code: {result.get('exit_code', -1)}]"


def _format_read_result(data: dict[str, Any]) -> str:
    """Format Apiary read_file JSON into the numbered-line text the agent expects."""
    if "error" in data:
        return f"Error: {data['error']}"
    lines = data.get("lines", [])
    total = data.get("total_lines", 0)
    returned = data.get("lines_returned", 0)
    out = "\n".join(f"{ln['line_no']:6d}|{ln['content']}" for ln in lines)
    return f"{out}\n\n[Total lines: {total}, Lines returned: {returned}]"


def _format_list_result(data: dict[str, Any], offset: int = 1) -> str:
    """Format Apiary list_dir JSON into the entry list text the agent expects."""
    if "error" in data:
        return f"Error: {data['error']}"
    entries = data.get("entries", [])
    total = data.get("total_entries", 0)
    returned = data.get("entries_returned", 0)
    out = "\n".join(
        f"{offset + i:6d}. [{e['entry_type']:5s}] {e['name']}" for i, e in enumerate(entries)
    )
    return f"{out}\n\n[Total entries: {total}, Entries returned: {returned}]"


def _format_grep_result(data: dict[str, Any]) -> str:
    """Format Apiary grep_files JSON into the grep-style text the agent expects."""
    if "error" in data:
        return f"Error: {data['error']}"
    files_list = data.get("files", [])
    total = data.get("total_files", 0)
    lines: list[str] = []
    for f in files_list:
        path = f.get("path", "")
        matches = f.get("matches", [])
        lines.append(f"\n{path}:")
        for m in matches[:10]:
            content = m.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"  {m.get('line_no', 0)}: {content}")
        if len(matches) > 10:
            lines.append(f"  ... ({len(matches) - 10} more matches)")
    body = "\n".join(lines).strip()
    return f"{body}\n\n[Files matched: {total}]"


def _format_patch_result(data: dict[str, Any]) -> str:
    """Format Apiary apply_patch JSON into the result text the agent expects."""
    if "error" in data:
        return f"Error: {data['error']}"
    results = data.get("results", [])
    return "; ".join(results) if results else "No changes applied"


@mcp.tool()
async def read_file(
    file_path: str,
    offset: int = 1,
    limit: Optional[int] = None,
    mode: str = "slice",
) -> str:
    """Read a file with 1-indexed line numbers from the Apiary sandbox."""
    _ = mode
    try:
        data = await _session.read_file(file_path, offset=offset, limit=limit)
        return _format_read_result(data)
    except httpx.HTTPStatusError as e:
        body = e.response.text
        try:
            err = e.response.json()
            body = err.get("error", body)
        except Exception:
            pass
        return f"Error: {body}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def list_dir(
    dir_path: str,
    offset: int = 1,
    limit: Optional[int] = None,
    depth: int = 1,
) -> str:
    """List directory contents from the Apiary sandbox."""
    try:
        data = await _session.list_dir(dir_path, offset=offset, limit=limit, depth=depth)
        return _format_list_result(data, offset=offset)
    except httpx.HTTPStatusError as e:
        body = e.response.text
        try:
            err = e.response.json()
            body = err.get("error", body)
        except Exception:
            pass
        return f"Error: {body}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def grep_files(
    pattern: str,
    include: Optional[str] = None,
    path: Optional[str] = None,
    limit: int = 100,
) -> str:
    """Search files for a regex pattern inside the Apiary sandbox."""
    try:
        data = await _session.grep_files(pattern, include=include, path=path, limit=limit)
        return _format_grep_result(data)
    except httpx.HTTPStatusError as e:
        body = e.response.text
        try:
            err = e.response.json()
            body = err.get("error", body)
        except Exception:
            pass
        return f"Error: {body}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def apply_patch(patch: str) -> str:
    """Apply a structured patch inside the Apiary sandbox."""
    try:
        data = await _session.apply_patch(patch)
        return _format_patch_result(data)
    except httpx.HTTPStatusError as e:
        body = e.response.text
        try:
            err = e.response.json()
            body = err.get("error", body)
        except Exception:
            pass
        return f"Error: {body}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def update_plan(
    plan: list[dict[str, str]],
    explanation: Optional[str] = None,
) -> str:
    """Update the per-client plan state."""
    if not plan:
        return "Error: plan is required and cannot be empty"

    valid_statuses = {"pending", "in_progress", "completed"}
    in_progress_count = 0
    validated_plan: list[dict[str, str]] = []

    for index, item in enumerate(plan, start=1):
        if not isinstance(item, dict):
            return f"Error: Plan item {index} must be an object"

        step = item.get("step")
        status = item.get("status")

        if not step:
            return f"Error: Plan item {index} missing 'step' field"
        if not status:
            return f"Error: Plan item {index} missing 'status' field"
        if status not in valid_statuses:
            return (
                f"Error: Plan item {index} has invalid status '{status}'. "
                f"Must be one of: {', '.join(sorted(valid_statuses))}"
            )
        if status == "in_progress":
            in_progress_count += 1

        validated_plan.append({"step": step, "status": status})

    if in_progress_count > 1:
        return (
            f"Error: At most one step can be in_progress at a time, but found {in_progress_count}"
        )

    state = _current_plan_state()
    state.current_plan = validated_plan
    state.explanation = explanation

    output_lines: list[str] = []
    if explanation:
        output_lines.append(f"Explanation: {explanation}")
        output_lines.append("")

    output_lines.append("Plan:")
    for index, item in enumerate(validated_plan, start=1):
        output_lines.append(f"  {index}. [{item['status']}] {item['step']}")

    total = len(validated_plan)
    completed = sum(1 for item in validated_plan if item["status"] == "completed")
    progress = round(completed / total * 100, 1) if total else 0.0
    output_lines.append("")
    output_lines.append(f"Progress: {completed}/{total} ({progress}%)")
    return "\n".join(output_lines)


@mcp.tool()
async def shell(
    command: list[str],
    workdir: Optional[str] = None,
    timeout_ms: Optional[int] = None,
) -> str:
    """Execute a command array inside the Apiary sandbox."""
    if not command:
        return "Error: Command array cannot be empty"
    result = await _session.execute(
        shlex.join(command),
        timeout_ms=timeout_ms,
        working_dir=workdir,
    )
    return _format_shell_result(result)


@mcp.tool()
async def shell_command(
    command: str,
    workdir: Optional[str] = None,
    timeout_ms: Optional[int] = None,
) -> str:
    """Execute a shell command string inside the Apiary sandbox."""
    if not command:
        return "Error: Command cannot be empty"
    result = await _session.execute(
        command,
        timeout_ms=timeout_ms,
        working_dir=workdir,
    )
    return _format_shell_result(result)


@mcp.tool()
async def think(thought: str) -> str:
    """Record a thought for the current client."""
    return f"Thought recorded: {thought}"


@mcp.tool()
async def complete_task(summary: str) -> str:
    """Signal task completion for the current client."""
    return f"Task completed: {summary}"


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create the SSE app for the Coder MCP server."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        if _mcp_auth_token:
            provided = request.headers.get("authorization", "")
            expected = f"Bearer {_mcp_auth_token}"
            if not hmac.compare_digest(provided.encode(), expected.encode()):
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(request.scope, request.receive, request._send)
                return

        cid = request.query_params.get("client_id") or uuid.uuid4().hex[:12]
        layers = request.query_params.getlist("base_image")
        _client_id.set(cid)
        if layers:
            _session.set_client_layers(cid, layers)
        _session.attach(cid)
        LOGGER.info("Client %s connected (layers=%d)", cid, len(layers))

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

    async def on_startup() -> None:
        _session.start_reaper()

    async def on_shutdown() -> None:
        await _session.shutdown()

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        on_startup=[on_startup],
        on_shutdown=[on_shutdown],
    )


if __name__ == "__main__":
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
    args = parser.parse_args()

    _session = SessionManager(
        args.apiary_url,
        args.apiary_token,
        args.working_dir,
        idle_timeout=args.idle_timeout,
    )
    _mcp_auth_token = args.mcp_token

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        app = create_starlette_app(mcp._mcp_server, debug=True)
        LOGGER.info("Starting Coder MCP server on %s:%s", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
