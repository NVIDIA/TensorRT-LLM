from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme

from .config import HumanRequest
from .types import (AgentTextEvent, CompactBoundaryEvent, RateLimitWarningEvent,
                    ServerToolCallEvent, SessionInitEvent, ThinkingEvent,
                    ToolCallEvent, UsageInfo)

_THEME = Theme({
    "layer.planner": "bold cyan",
    "layer.generator": "bold green",
    "layer.evaluator": "bold yellow",
    "layer.default": "bold magenta",
    "status.started": "bold blue",
    "status.completed": "bold green",
    "status.failed": "bold red",
    "status.warning": "bold yellow",
    "info": "dim",
    "target": "bold white",
    "subagent.badge": "italic bright_black",
    "subagent.label": "italic bright_white",
    "thinking.body": "italic bright_black",
    "todo.completed": "green",
    "todo.in_progress": "bold yellow",
    "todo.pending": "dim",
})

_THINKING_MAX_CHARS = 2000

# Indent applied to subagent panels so a glance at the log makes the
# "main agent vs. spawned subagent" hierarchy obvious.
_SUBAGENT_INDENT = 4

_AGENT_STYLES = {
    "planner": ("cyan", "layer.planner"),
    "generator": ("green", "layer.generator"),
    "evaluator": ("yellow", "layer.evaluator"),
}

_DEFAULT_STYLE = ("magenta", "layer.default")

console = Console(theme=_THEME, highlight=False)


def _ensure_blocking_streams() -> None:
    """Restore blocking mode on stdout/stderr.

    The ``claude`` CLI (a Node.js process spawned by the Claude Code SDK)
    inherits and shares the parent's stdout/stderr file descriptors, and
    Node sets them to non-blocking when stdout is a pipe (e.g. when the
    workflow is piped through ``tee``). After the subprocess exits the
    FD stays non-blocking in the parent, so subsequent ``rich`` writes
    fail with ``BlockingIOError: [Errno 11]``. Flipping the flag back
    here is cheap and safe to call repeatedly.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            fd = stream.fileno()
        except (OSError, ValueError, AttributeError):
            continue
        try:
            os.set_blocking(fd, True)
        except OSError:
            pass


def _console_write(method, *args, **kwargs) -> None:
    """Invoke a ``rich.Console`` method, recovering from non-blocking stdout.

    On ``BlockingIOError`` we restore blocking mode on the underlying FD
    and retry once — the retry then blocks normally until the pipe
    consumer drains, instead of raising.
    """
    try:
        method(*args, **kwargs)
    except BlockingIOError:
        _ensure_blocking_streams()
        method(*args, **kwargs)


# Make sure stdout/stderr start in blocking mode so the very first
# write isn't racing a subprocess that already flipped the flag.
_ensure_blocking_streams()


def _layer_style(layer_name: str) -> tuple[str, str]:
    return _AGENT_STYLES.get(layer_name.lower(), _DEFAULT_STYLE)


def _layer_title(layer_name: str, suffix: str = "") -> str:
    _, text_style = _layer_style(layer_name)
    label = layer_name.upper()
    title = f"[{text_style}]{label}[/{text_style}]"
    if suffix:
        title += f" [info]{suffix}[/info]"
    return title


def _print_panel(renderable: object, extra: Console | None = None) -> None:
    _console_write(console.print, renderable)
    if extra is not None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        extra.print(f"[info]{stamp}[/info]")
        extra.print(renderable)


def _subagent_title_suffix(suffix: str, agent_label: str | None) -> str:
    if not agent_label:
        return suffix
    badge = (f"[subagent.badge]↳[/subagent.badge] "
             f"[subagent.label]{agent_label}[/subagent.label]")
    if suffix:
        return f"{badge} [info]·[/info] {suffix}"
    return badge


def _subagent_border(border: str) -> str:
    """Visually distinguish subagent activity by dimming the border."""
    return f"dim {border}"


def _emit_panel(layer_name: str,
                title_suffix: str,
                body: object,
                agent_label: str | None,
                extra: Console | None = None) -> None:
    border, _ = _layer_style(layer_name)
    is_subagent = agent_label is not None
    title = _layer_title(layer_name,
                         _subagent_title_suffix(title_suffix, agent_label))
    panel = Panel(
        body,
        title=title,
        title_align="left",
        border_style=_subagent_border(border) if is_subagent else border,
        padding=(0, 1),
    )
    if is_subagent:
        _print_panel(Padding(panel, (0, 0, 0, _SUBAGENT_INDENT)), extra)
    else:
        _print_panel(panel, extra)


def print_rule(text: str, extra: Console | None = None) -> None:
    _console_write(console.rule, text)
    if extra is not None:
        extra.rule(text)


def print_message(text: str, extra: Console | None = None) -> None:
    _console_write(console.print, text)
    if extra is not None:
        extra.print(text)


def print_user_prompt(layer_name: str,
                      content: str,
                      extra: Console | None = None) -> None:
    border, _ = _layer_style(layer_name)
    text = content.strip()
    body = Markdown(text) if text else "[info](empty)[/info]"
    _print_panel(
        Panel(
            body,
            title=_layer_title(layer_name, "prompt"),
            title_align="left",
            border_style=border,
            padding=(0, 1),
        ), extra)


def print_agent_started(layer_name: str,
                        backend: str,
                        model: str,
                        extra: Console | None = None,
                        version: str | None = None,
                        reasoning_effort: str | None = None) -> None:
    border, _ = _layer_style(layer_name)
    lines = [
        "[status.started]▸ started[/status.started]",
        f"[info]backend[/info] [target]{backend}[/target]",
    ]
    if version:
        lines.append(f"[info]version[/info] [target]{version}[/target]")
    lines.append(f"[info]model[/info] [target]{model}[/target]")
    if reasoning_effort:
        lines.append(f"[info]reasoning effort[/info] "
                     f"[target]{reasoning_effort}[/target]")
    body = "\n".join(lines)
    _print_panel(
        Panel(
            body,
            title=_layer_title(layer_name),
            title_align="left",
            border_style=border,
            padding=(0, 1),
        ), extra)


def _format_tokens(value: int | None) -> str | None:
    if value is None:
        return None
    return f"{value:,}"


def _format_usage_body(usage: UsageInfo) -> str:
    parts: list[str] = []
    tokens_parts: list[str] = []
    for label, value in (
        ("in", usage.input_tokens),
        ("out", usage.output_tokens),
        ("cache+", usage.cache_creation_tokens),
        ("cache~", usage.cache_read_tokens),
    ):
        formatted = _format_tokens(value)
        if formatted is not None:
            tokens_parts.append(f"{label} [target]{formatted}[/target]")

    total = _format_tokens(usage.total_tokens)
    if total is not None:
        tokens_parts.append(f"total [target]{total}[/target]")

    if tokens_parts:
        parts.append("[info]tokens[/info] " + " · ".join(tokens_parts))

    if (usage.context_tokens is not None
            or usage.context_percentage is not None):
        segments: list[str] = []
        if usage.context_percentage is not None:
            segments.append(f"[target]{usage.context_percentage:.1f}%[/target]")
        ctx_tokens = _format_tokens(usage.context_tokens)
        ctx_window = _format_tokens(usage.context_window)
        if ctx_tokens is not None and ctx_window is not None:
            segments.append(f"[target]{ctx_tokens}[/target]"
                            f"/[target]{ctx_window}[/target]")
        elif ctx_tokens is not None:
            segments.append(f"[target]{ctx_tokens}[/target]")
        if segments:
            parts.append("[info]context[/info] " + " ".join(segments))

    if usage.cost_usd is not None:
        parts.append(
            f"[info]cost[/info] [target]${usage.cost_usd:.4f}[/target]")
    if usage.num_turns is not None:
        parts.append(f"[info]turns[/info] [target]{usage.num_turns}[/target]")
    if usage.duration_ms is not None:
        seconds = usage.duration_ms / 1000
        parts.append(f"[info]duration[/info] [target]{seconds:.2f}s[/target]")

    return "\n".join(parts)


def print_agent_completed(layer_name: str,
                          extra: Console | None = None,
                          usage: UsageInfo | None = None) -> None:
    border, _ = _layer_style(layer_name)
    body_lines = ["[status.completed]✔ completed[/status.completed]"]
    if usage is not None:
        usage_body = _format_usage_body(usage)
        if usage_body:
            body_lines.append(usage_body)
    _print_panel(
        Panel(
            "\n".join(body_lines),
            title=_layer_title(layer_name),
            title_align="left",
            border_style=border,
            padding=(0, 1),
        ), extra)


def print_agent_failed(layer_name: str,
                       error: Exception,
                       extra: Console | None = None) -> None:
    _print_panel(
        Panel(
            f"[status.failed]✘ failed[/status.failed]: {error}",
            title=_layer_title(layer_name),
            title_align="left",
            border_style="status.failed",
            padding=(0, 1),
        ), extra)


def print_tool_call(layer_name: str,
                    event: ToolCallEvent,
                    extra: Console | None = None) -> None:
    body = _tool_input_body(event.name, event.input)
    _emit_panel(layer_name, f"tool · {event.name}", body, event.agent_label,
                extra)


def print_server_tool_call(layer_name: str,
                           event: ServerToolCallEvent,
                           extra: Console | None = None) -> None:
    body = _tool_input_body(event.name, event.input)
    _emit_panel(layer_name, f"server tool · {event.name}", body,
                event.agent_label, extra)


def print_thinking(layer_name: str,
                   event: ThinkingEvent,
                   extra: Console | None = None) -> None:
    text = event.text.strip()
    truncated = text[:_THINKING_MAX_CHARS]
    if len(text) > _THINKING_MAX_CHARS:
        truncated += "\n[info]…[/info]"
    body = (f"[thinking.body]{truncated}[/thinking.body]"
            if truncated else "[info](empty)[/info]")
    _emit_panel(layer_name, "thinking", body, event.agent_label, extra)


def print_rate_limit(layer_name: str,
                     event: RateLimitWarningEvent,
                     extra: Console | None = None) -> None:
    parts: list[str] = [
        f"[status.warning]⚠ rate limit[/status.warning] "
        f"[target]{event.status}[/target]"
    ]
    if event.rate_limit_type:
        parts.append(f"[info]type[/info] [target]{event.rate_limit_type}"
                     f"[/target]")
    if event.utilization is not None:
        parts.append(f"[info]utilization[/info] [target]{event.utilization:.1%}"
                     f"[/target]")
    if event.resets_at is not None:
        parts.append(f"[info]resets_at[/info] [target]{event.resets_at}"
                     f"[/target]")
    body = "  ".join(parts)
    _print_panel(
        Panel(
            body,
            title=_layer_title(layer_name, "rate limit"),
            title_align="left",
            border_style="status.warning",
            padding=(0, 1),
        ), extra)


def print_compact_boundary(layer_name: str,
                           event: CompactBoundaryEvent,
                           extra: Console | None = None) -> None:
    parts: list[str] = ["[info]conversation auto-compacted[/info]"]
    if event.trigger:
        parts.append(f"[info]trigger[/info] [target]{event.trigger}[/target]")
    if event.pre_tokens is not None:
        parts.append(f"[info]pre_tokens[/info] [target]"
                     f"{event.pre_tokens:,}[/target]")
    body = "  ".join(parts)
    border, _ = _layer_style(layer_name)
    _print_panel(
        Panel(
            body,
            title=_layer_title(layer_name, "compact boundary"),
            title_align="left",
            border_style=border,
            padding=(0, 1),
        ), extra)


def _format_session_init_body(event: SessionInitEvent) -> str:
    sections: list[tuple[str, list[str]]] = [
        ("plugins", event.plugins),
        ("skills", event.skills),
        ("subagents", event.agents),
    ]
    lines: list[str] = []
    for label, items in sections:
        if not items:
            continue
        lines.append(f"[info]{label} ({len(items)})[/info]")
        for name in sorted(set(items)):
            lines.append(f"  [target]{name}[/target]")
    if not lines:
        return "[info](no skills, plugins, or subagents loaded)[/info]"
    return "\n".join(lines)


def print_session_init(layer_name: str,
                       event: SessionInitEvent,
                       extra: Console | None = None) -> None:
    border, _ = _layer_style(layer_name)
    _print_panel(
        Panel(
            _format_session_init_body(event),
            title=_layer_title(layer_name, "system"),
            title_align="left",
            border_style=border,
            padding=(0, 1),
        ), extra)


def print_layer_panel(layer_name: str,
                      title_suffix: str,
                      body: object,
                      extra: Console | None = None) -> None:
    """Print a layer-styled panel with an arbitrary body.

    ``body`` can be a string or any Rich renderable (``Syntax``,
    ``Markdown``, ...). Useful for surfacing application-level events
    (e.g. progress reads/writes) with the same styling as agent output.
    """
    border, _ = _layer_style(layer_name)
    _print_panel(
        Panel(
            body,
            title=_layer_title(layer_name, title_suffix),
            title_align="left",
            border_style=border,
            padding=(0, 1),
        ), extra)


def _build_options_table(request: HumanRequest) -> Table:
    table = Table(
        show_header=True,
        header_style="bold yellow",
        box=None,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("#", style="bold yellow", no_wrap=True, width=3)
    table.add_column("Option", style="bold white")
    table.add_column("Description", style="info", overflow="fold")
    for i, opt in enumerate(request.options, start=1):
        table.add_row(str(i), opt.label, opt.description or "")
    return table


def print_human_input_request(layer_name: str,
                              request: HumanRequest,
                              extra: Console | None = None) -> None:
    question = request.prompt.strip() or "(no prompt)"
    header = request.header.strip()
    title_suffix = f"ask_human · {header}" if header else "ask_human"
    if request.options:
        body = Group(
            Markdown(question),
            "",
            _build_options_table(request),
            "",
            ("[info]Reply with the option number, the exact label, or "
             "free-form text. Empty line = no answer.[/info]"),
        )
    else:
        body = Markdown(question)
    _print_panel(
        Panel(
            body,
            title=_layer_title(layer_name, title_suffix),
            title_align="left",
            border_style="status.warning",
            padding=(0, 1),
        ), extra)


def print_human_reply(layer_name: str,
                      request: HumanRequest,
                      reply: str,
                      extra: Console | None = None) -> None:
    text = reply.strip()
    body = Markdown(text) if text else "[info](no reply)[/info]"
    _print_panel(
        Panel(
            body,
            title=_layer_title(layer_name, "ask_human · reply"),
            title_align="left",
            border_style="status.completed",
            padding=(0, 1),
        ), extra)


def print_agent_text(layer_name: str,
                     event: AgentTextEvent,
                     extra: Console | None = None) -> None:
    text = event.text.strip()
    body = Markdown(text) if text else "[info](empty)[/info]"
    _emit_panel(layer_name, "message", body, event.agent_label, extra)


def _tool_input_body(name: str, tool_input: dict[str, object]) -> object:
    """Dispatch tool-call rendering on tool name.

    Routing by ``name`` (rather than by which keys happen to be present
    in ``tool_input``) avoids misrendering MCP/custom tools that share
    field names with built-ins — e.g. a DB-query tool with a ``command``
    field would otherwise look like a shell prompt. Each renderer still
    type-checks the fields it expects and returns ``None`` if the shape
    is off, so we fall back to the JSON dump on mismatch.
    """
    if not tool_input:
        return "[info]no arguments[/info]"
    renderer = _TOOL_RENDERERS.get(name)
    if renderer is not None:
        body = renderer(tool_input)
        if body is not None:
            return body
    pretty = json.dumps(tool_input, indent=2, sort_keys=True)
    return Syntax(pretty, "json", theme="ansi_dark", word_wrap=True)


def _render_bash_command(tool_input: dict[str, object]) -> object | None:
    """Render a Bash-style tool call as ``description`` + highlighted command.

    Claude's ``Bash`` tool ships ``command`` alongside a short
    ``description`` and occasional flags (``run_in_background``,
    ``timeout``). Dumping the raw JSON buries the actual shell line in
    quoting noise; instead we surface the description as a header, list
    any extra flags, and syntax-highlight the command itself.
    """
    command = tool_input.get("command")
    if not isinstance(command, str):
        return None
    syntax = Syntax(f"$ {command}", "bash", theme="ansi_dark", word_wrap=True)
    header_lines: list[str] = []
    description = tool_input.get("description")
    if isinstance(description, str) and description.strip():
        header_lines.append(f"[info]{description.strip()}[/info]")
    for key in sorted(tool_input):
        if key in ("command", "description"):
            continue
        header_lines.append(
            f"[info]{key}[/info] [target]{tool_input[key]}[/target]")
    if not header_lines:
        return syntax
    return Group(*header_lines, syntax)


# Pygments lexer aliases keyed on the file extension. We keep the table
# small and stick to lexer names that ship with Pygments — unknown
# extensions fall back to ``"text"`` so ``Syntax`` still renders cleanly.
_LEXER_BY_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".css": "css",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".h": "c",
    ".c": "c",
    ".sql": "sql",
    ".lua": "lua",
    ".tf": "hcl",
}

_LEXER_BY_FILENAME = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
}

# Cap the number of lines we syntax-highlight for a Write preview so a
# multi-thousand-line file doesn't blow the panel up. The original
# content is still passed to the model unchanged — this only affects
# the on-screen rendering.
_WRITE_PREVIEW_MAX_LINES = 10

# Per-side cap for the Edit unified-diff preview. We render up to this
# many lines from ``old_string`` and ``new_string`` each, so a very
# large replacement still keeps both halves visible side by side.
_EDIT_PREVIEW_MAX_LINES_PER_SIDE = 15

# Per-change cap for the Codex FileChange diff preview. A single Codex
# turn can apply several patches; capping each diff body keeps the
# panel readable when one of them is huge.
_FILE_CHANGE_PREVIEW_MAX_LINES = 30


def _lexer_from_path(file_path: str) -> str:
    p = Path(file_path)
    name_key = p.name.lower()
    if name_key in _LEXER_BY_FILENAME:
        return _LEXER_BY_FILENAME[name_key]
    return _LEXER_BY_EXT.get(p.suffix.lower(), "text")


def _render_read_call(tool_input: dict[str, object]) -> object | None:
    """Render a Read tool call as ``<path>`` + range metadata.

    Claude's ``Read`` tool carries a ``file_path`` plus optional
    ``offset``/``limit``/``pages`` modifiers. The default JSON dump
    drowns the path in braces and quoting; a couple of dim header lines
    make the path and range pop out at a glance.
    """
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    header_lines = [f"[target]{file_path}[/target]"]
    for key in sorted(tool_input):
        if key == "file_path":
            continue
        header_lines.append(
            f"[info]{key}[/info] [target]{tool_input[key]}[/target]")
    return "\n".join(header_lines)


def _render_write_call(tool_input: dict[str, object]) -> object | None:
    """Render a Write tool call as ``<path>`` + content preview.

    The default JSON dump puts the entire file content on a single
    quoted line — useless. We surface the path as a header and
    syntax-highlight the content using a lexer inferred from the file
    extension, truncating to ``_WRITE_PREVIEW_MAX_LINES`` so the panel
    stays readable.
    """
    file_path = tool_input.get("file_path")
    content = tool_input.get("content")
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    if not isinstance(content, str):
        return None
    header_lines = [f"[target]{file_path}[/target]"]
    for key in sorted(tool_input):
        if key in ("file_path", "content"):
            continue
        header_lines.append(
            f"[info]{key}[/info] [target]{tool_input[key]}[/target]")

    lines = content.splitlines()
    if len(lines) > _WRITE_PREVIEW_MAX_LINES:
        body_text = "\n".join(lines[:_WRITE_PREVIEW_MAX_LINES])
        omitted = len(lines) - _WRITE_PREVIEW_MAX_LINES
        footer = f"[info]… {omitted} more lines truncated[/info]"
    else:
        body_text = content
        footer = None

    body = Syntax(
        body_text,
        _lexer_from_path(file_path),
        theme="ansi_dark",
        word_wrap=True,
        line_numbers=True,
    )
    parts: list[object] = [*header_lines, body]
    if footer is not None:
        parts.append(footer)
    return Group(*parts)


def _count_unified_diff_stats(diff_text: str) -> tuple[int, int]:
    """Return ``(additions, deletions)`` for a unified-diff string.

    Header lines (``+++``/``---``) and hunk markers (``@@``) don't
    represent content changes, so they're excluded from the count.
    """
    additions = 0
    deletions = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def _render_edit_call(tool_input: dict[str, object]) -> object | None:
    """Render an Edit tool call as ``<path>`` + unified-diff preview.

    Claude's ``Edit`` tool replaces ``old_string`` with ``new_string`` in
    ``file_path``. The default JSON dump puts both strings on quoted
    single lines, hiding any structure; instead we surface the path as a
    header and render the change as a colored diff via Pygments' ``diff``
    lexer so the actual edit pops at a glance.
    """
    file_path = tool_input.get("file_path")
    old_string = tool_input.get("old_string")
    new_string = tool_input.get("new_string")
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        return None
    additions = len(new_string.splitlines())
    deletions = len(old_string.splitlines())
    header_lines = [
        f"[target]{escape(file_path)}[/target] "
        f"[green]+{additions}[/green] [red]-{deletions}[/red]"
    ]
    for key in sorted(tool_input):
        if key in ("file_path", "old_string", "new_string"):
            continue
        header_lines.append(
            f"[info]{key}[/info] [target]{tool_input[key]}[/target]")

    diff_lines: list[str] = []
    omitted = 0
    for prefix, text in (("-", old_string), ("+", new_string)):
        lines = text.splitlines()
        if len(lines) > _EDIT_PREVIEW_MAX_LINES_PER_SIDE:
            kept = lines[:_EDIT_PREVIEW_MAX_LINES_PER_SIDE]
            omitted += len(lines) - _EDIT_PREVIEW_MAX_LINES_PER_SIDE
        else:
            kept = lines
        for line in kept:
            diff_lines.append(f"{prefix}{line}")

    body_text = "\n".join(diff_lines) if diff_lines else "(no diff)"
    body = Syntax(body_text, "diff", theme="ansi_dark", word_wrap=True)
    parts: list[object] = [*header_lines, body]
    if omitted:
        parts.append(f"[info]… {omitted} more diff lines truncated[/info]")
    return Group(*parts)


_FILE_CHANGE_KIND_MARKERS = {
    "add": ("+", "green"),
    "delete": ("-", "red"),
    "update": ("~", "yellow"),
}


def _render_file_change_call(tool_input: dict[str, object]) -> object | None:
    """Render a Codex ``FileChange`` tool call as a list of patches.

    Each entry in ``changes`` carries ``path``, a ``kind`` tagged
    ``add``/``delete``/``update`` (with an optional ``move_path`` for
    renames), and a ``diff`` body. The default JSON dump puts each diff
    on a single quoted line, hiding the actual change; instead we
    surface a per-change header (``+ /path``, ``~ /old → /new``,
    ``- /path``) followed by a unified diff rendered through Pygments'
    ``diff`` lexer so additions and deletions pop at a glance.
    """
    changes = tool_input.get("changes")
    if not isinstance(changes, list):
        return None
    valid_changes = [c for c in changes if isinstance(c, dict)]
    if not valid_changes:
        return "[info](no changes)[/info]"

    parts: list[object] = []
    status = tool_input.get("status")
    if isinstance(status, str) and status:
        parts.append(f"[info]status[/info] [target]{escape(status)}[/target]")

    for change in valid_changes:
        if parts:
            parts.append("")
        parts.extend(_render_one_file_change(change))
    return Group(*parts)


def _render_one_file_change(change: dict[str, object]) -> list[object]:
    raw_path = change.get("path") or ""
    path = raw_path if isinstance(raw_path, str) else str(raw_path)
    kind = change.get("kind")
    kind_type = ""
    move_path: str | None = None
    if isinstance(kind, dict):
        raw_type = kind.get("type")
        kind_type = raw_type if isinstance(raw_type, str) else ""
        candidate = kind.get("move_path") or kind.get("movePath")
        if isinstance(candidate, str) and candidate:
            move_path = candidate

    marker, color = _FILE_CHANGE_KIND_MARKERS.get(kind_type, ("?", "info"))
    header = (f"[{color}]{marker}[/{color}] "
              f"[target]{escape(path)}[/target]")
    if move_path:
        header += f" [info]→[/info] [target]{escape(move_path)}[/target]"
    if kind_type:
        header += f" [info]({kind_type})[/info]"

    diff = change.get("diff")
    diff_text = diff if isinstance(diff, str) else ""
    if diff_text.strip():
        additions, deletions = _count_unified_diff_stats(diff_text)
        header += (f" [green]+{additions}[/green] "
                   f"[red]-{deletions}[/red]")

    parts: list[object] = [header]
    if not diff_text.strip():
        return parts
    diff_lines = diff_text.splitlines()
    if len(diff_lines) > _FILE_CHANGE_PREVIEW_MAX_LINES:
        body_text = "\n".join(diff_lines[:_FILE_CHANGE_PREVIEW_MAX_LINES])
        omitted = len(diff_lines) - _FILE_CHANGE_PREVIEW_MAX_LINES
        footer: str | None = (
            f"[info]… {omitted} more diff lines truncated[/info]")
    else:
        body_text = diff_text
        footer = None
    parts.append(Syntax(body_text, "diff", theme="ansi_dark", word_wrap=True))
    if footer is not None:
        parts.append(footer)
    return parts


_TODO_STATUS_MARKERS = {
    "completed": ("[todo.completed]☑[/todo.completed]", "todo.completed"),
    "in_progress":
    ("[todo.in_progress]▸[/todo.in_progress]", "todo.in_progress"),
    "pending": ("[todo.pending]☐[/todo.pending]", "todo.pending"),
}


def _render_todowrite_call(tool_input: dict[str, object]) -> object | None:
    """Render a TodoWrite tool call as a checkbox-style task list.

    Claude's ``TodoWrite`` tool ships ``todos`` as a list of
    ``{content, status, activeForm}`` dicts. The default JSON dump
    bloats a five-task list into a thirty-line block; instead, render
    each entry as a status-marked checkbox so progress is visible at a
    glance. ``in_progress`` items prefer ``activeForm`` ("Doing X") to
    surface what the agent is currently working on.
    """
    todos = tool_input.get("todos")
    if not isinstance(todos, list):
        return None
    if not todos:
        return "[info](no todos)[/info]"
    lines: list[str] = []
    for entry in todos:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        content = entry.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        active_form = entry.get("activeForm")
        marker, style = _TODO_STATUS_MARKERS.get(
            status if isinstance(status, str) else "",
            _TODO_STATUS_MARKERS["pending"],
        )
        if status == "in_progress" and isinstance(active_form,
                                                  str) and active_form.strip():
            display = active_form.strip()
        else:
            display = content
        lines.append(f"{marker} [{style}]{escape(display)}[/{style}]")
    return "\n".join(lines) if lines else "[info](no todos)[/info]"


_TOOL_RENDERERS = {
    "Bash": _render_bash_command,
    "Read": _render_read_call,
    "Write": _render_write_call,
    "Edit": _render_edit_call,
    "TodoWrite": _render_todowrite_call,
    "FileChange": _render_file_change_call,
}
