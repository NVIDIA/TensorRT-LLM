from __future__ import annotations

import io

from rich.console import Console

from agent_flow import console as console_module
from agent_flow.types import (AgentTextEvent, CompactBoundaryEvent,
                              RateLimitWarningEvent, ServerToolCallEvent,
                              SessionInitEvent, ThinkingEvent, ToolCallEvent)


def _capture_console(monkeypatch) -> Console:
    """Swap the module-level console for one that writes to a buffer."""
    buffer = io.StringIO()
    fake = Console(
        file=buffer,
        theme=console_module._THEME,
        highlight=False,
        force_terminal=False,
        width=120,
    )
    monkeypatch.setattr(console_module, "console", fake)
    fake._buffer_text = buffer  # type: ignore[attr-defined]
    return fake


def test_main_agent_tool_call_has_no_subagent_badge(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(name="Bash", input={"command": "ls"}, tool_use_id="t-1"),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · Bash" in output
    assert "↳" not in output


def test_bash_tool_call_extracts_description_and_command(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Bash",
            input={
                "command": "ls -la",
                "description": "List files in current directory",
            },
            tool_use_id="t-bash",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "List files in current directory" in output
    assert "ls -la" in output
    # The raw JSON braces / quoted keys should not leak through.
    assert '"command"' not in output
    assert '"description"' not in output


def test_bash_tool_call_surfaces_extra_flags(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Bash",
            input={
                "command": "pytest",
                "description": "Run tests",
                "run_in_background": True,
                "timeout": 60000,
            },
            tool_use_id="t-bash-flags",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "Run tests" in output
    assert "pytest" in output
    assert "run_in_background" in output
    assert "True" in output
    assert "timeout" in output
    assert "60000" in output


def test_bash_tool_call_without_description_renders_only_command(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(name="Bash", input={"command": "ls"}, tool_use_id="t-1"),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "ls" in output
    assert '"command"' not in output


def test_bash_tool_call_prefixes_command_with_shell_prompt(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(name="Bash",
                      input={"command": "ls -la"},
                      tool_use_id="t-prompt"),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "$ ls -la" in output


def test_read_tool_call_renders_path_without_json_braces(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Read",
            input={"file_path": "/tmp/foo/bar.py"},
            tool_use_id="t-read",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · Read" in output
    assert "/tmp/foo/bar.py" in output
    # No JSON noise around the path.
    assert '"file_path"' not in output
    assert "{" not in output


def test_read_tool_call_surfaces_offset_and_limit(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Read",
            input={
                "file_path": "/tmp/foo/bar.py",
                "offset": 50,
                "limit": 100,
            },
            tool_use_id="t-read-range",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "/tmp/foo/bar.py" in output
    assert "offset" in output
    assert "50" in output
    assert "limit" in output
    assert "100" in output


def test_write_tool_call_renders_path_and_content_without_json(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Write",
            input={
                "file_path": "/tmp/foo/bar.py",
                "content": "def hello():\n    return 42\n",
            },
            tool_use_id="t-write",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · Write" in output
    assert "/tmp/foo/bar.py" in output
    assert "def hello" in output
    assert "return 42" in output
    # JSON-shaped output would have a quoted ``"content"`` key.
    assert '"content"' not in output
    assert '"file_path"' not in output


def test_write_tool_call_truncates_very_long_content(monkeypatch):
    fake = _capture_console(monkeypatch)
    over_limit = console_module._WRITE_PREVIEW_MAX_LINES + 50
    long_content = "\n".join(f"line_{i}" for i in range(over_limit))
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Write",
            input={
                "file_path": "/tmp/foo/big.py",
                "content": long_content,
            },
            tool_use_id="t-write-long",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    # First line should appear; lines beyond the cap should be elided.
    assert "line_0" in output
    assert "line_5" in output
    assert "line_500" not in output
    assert "more lines truncated" in output


def test_edit_tool_call_renders_path_and_diff(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Edit",
            input={
                "file_path": "/tmp/foo/bar.py",
                "old_string": "def hello():\n    return 1\n",
                "new_string": "def hello():\n    return 2\n",
            },
            tool_use_id="t-edit",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · Edit" in output
    assert "/tmp/foo/bar.py" in output
    assert "return 1" in output
    assert "return 2" in output
    # No JSON noise around the strings.
    assert '"old_string"' not in output
    assert '"new_string"' not in output


def test_edit_tool_call_surfaces_replace_all_flag(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Edit",
            input={
                "file_path": "/tmp/foo/bar.py",
                "old_string": "old",
                "new_string": "new",
                "replace_all": True,
            },
            tool_use_id="t-edit-flag",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "replace_all" in output
    assert "True" in output


def test_edit_tool_call_shows_added_and_deleted_line_counts(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Edit",
            input={
                "file_path": "/tmp/foo/bar.py",
                "old_string": "line1\nline2\nline3\n",
                "new_string": "line1\nline2_changed\nline3\nline4\n",
            },
            tool_use_id="t-edit-stats",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    # old_string has 3 lines (deletions), new_string has 4 lines (additions).
    assert "+4" in output
    assert "-3" in output


def test_edit_tool_call_handles_empty_old_or_new_string(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Edit",
            input={
                "file_path": "/tmp/foo/new.py",
                "old_string": "",
                "new_string": "first line\nsecond line\n",
            },
            tool_use_id="t-edit-empty-old",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "+2" in output
    assert "-0" in output


def test_edit_tool_call_truncates_very_long_diffs(monkeypatch):
    fake = _capture_console(monkeypatch)
    over_limit = console_module._EDIT_PREVIEW_MAX_LINES_PER_SIDE + 50
    long_old = "\n".join(f"old_{i}" for i in range(over_limit))
    long_new = "\n".join(f"new_{i}" for i in range(over_limit))
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Edit",
            input={
                "file_path": "/tmp/foo/big.py",
                "old_string": long_old,
                "new_string": long_new,
            },
            tool_use_id="t-edit-long",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    # First lines from each side visible; later lines elided.
    assert "old_0" in output
    assert "new_0" in output
    assert f"old_{over_limit - 1}" not in output
    assert f"new_{over_limit - 1}" not in output
    assert "more diff lines truncated" in output


def test_todowrite_tool_call_renders_checkbox_list(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="TodoWrite",
            input={
                "todos": [
                    {
                        "content": "Wire up backend",
                        "status": "completed",
                        "activeForm": "Wiring up backend",
                    },
                    {
                        "content": "Render edit diffs",
                        "status": "in_progress",
                        "activeForm": "Rendering edit diffs",
                    },
                    {
                        "content": "Add tests",
                        "status": "pending",
                        "activeForm": "Adding tests",
                    },
                ],
            },
            tool_use_id="t-todo",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · TodoWrite" in output
    assert "Wire up backend" in output
    # in_progress prefers ``activeForm`` so the user sees "Rendering …".
    assert "Rendering edit diffs" in output
    assert "Add tests" in output
    # No JSON noise.
    assert '"todos"' not in output
    assert '"status"' not in output


def test_file_change_tool_call_renders_each_patch(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="FileChange",
            input={
                "status":
                "completed",
                "changes": [
                    {
                        "path": "/repo/added.py",
                        "kind": {
                            "type": "add"
                        },
                        "diff": "@@ -0,0 +1,2 @@\n+def hi():\n+    return 1\n",
                    },
                    {
                        "path": "/repo/old.py",
                        "kind": {
                            "type": "update",
                            "move_path": "/repo/new.py"
                        },
                        "diff": "@@ -1 +1 @@\n-old line\n+new line\n",
                    },
                    {
                        "path": "/repo/gone.py",
                        "kind": {
                            "type": "delete"
                        },
                        "diff": "@@ -1 +0,0 @@\n-removed line\n",
                    },
                ],
            },
            tool_use_id="t-fc",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · FileChange" in output
    assert "completed" in output
    # Each path should show up once, with its kind annotation.
    assert "/repo/added.py" in output
    assert "(add)" in output
    assert "/repo/old.py" in output
    assert "/repo/new.py" in output
    assert "(update)" in output
    assert "/repo/gone.py" in output
    assert "(delete)" in output
    # Diff bodies should leak through the diff lexer, not as JSON.
    assert "def hi" in output
    assert "new line" in output
    assert "removed line" in output
    assert '"changes"' not in output
    assert '"diff"' not in output


def test_file_change_tool_call_shows_added_and_deleted_line_counts(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="FileChange",
            input={
                "changes": [
                    {
                        "path": "/repo/added.py",
                        "kind": {
                            "type": "add"
                        },
                        "diff": "@@ -0,0 +1,2 @@\n+def hi():\n+    return 1\n",
                    },
                    {
                        "path":
                        "/repo/edit.py",
                        "kind": {
                            "type": "update"
                        },
                        "diff":
                        ("--- a/repo/edit.py\n+++ b/repo/edit.py\n"
                         "@@ -1,3 +1,2 @@\n-old1\n-old2\n-old3\n+new1\n"),
                    },
                ],
            },
            tool_use_id="t-fc-stats",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    # First change: 2 additions, 0 deletions.
    assert "+2" in output
    assert "-0" in output
    # Second change: 1 addition, 3 deletions. The unified-diff
    # ``+++``/``---`` headers must be excluded from the count.
    assert "+1" in output
    assert "-3" in output


def test_file_change_tool_call_truncates_long_diffs(monkeypatch):
    fake = _capture_console(monkeypatch)
    over_limit = console_module._FILE_CHANGE_PREVIEW_MAX_LINES + 50
    huge_diff = "\n".join(f"+line_{i}" for i in range(over_limit))
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="FileChange",
            input={
                "changes": [
                    {
                        "path": "/repo/big.py",
                        "kind": {
                            "type": "add"
                        },
                        "diff": huge_diff,
                    },
                ],
            },
            tool_use_id="t-fc-trunc",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "line_0" in output
    assert f"line_{over_limit - 1}" not in output
    assert "more diff lines truncated" in output


def test_file_change_tool_call_handles_empty_changes(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="FileChange",
            input={"changes": []},
            tool_use_id="t-fc-empty",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · FileChange" in output
    assert "no changes" in output


def test_todowrite_tool_call_handles_empty_list(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="TodoWrite",
            input={"todos": []},
            tool_use_id="t-todo-empty",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · TodoWrite" in output
    assert "no todos" in output


def test_unknown_tool_with_command_field_does_not_render_as_bash(monkeypatch):
    """An MCP/custom tool that happens to carry a ``command`` field
    should render as JSON, not as a shell prompt — name-based dispatch
    prevents the field-shape collision."""
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="QueryDB",
            input={"command": "SELECT * FROM users"},
            tool_use_id="t-mcp-cmd",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · QueryDB" in output
    # The query should appear, but NOT prefixed by a shell prompt — the
    # JSON dump preserves the ``"command"`` key, while the bash renderer
    # would have stripped it.
    assert "SELECT * FROM users" in output
    assert "$ SELECT" not in output
    assert '"command"' in output


def test_unknown_tool_with_file_path_does_not_render_as_write(monkeypatch):
    """A custom tool that ships ``file_path`` + ``content`` should fall
    through to the JSON dump rather than being misrendered as a Write."""
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="UploadDoc",
            input={
                "file_path": "/docs/spec.md",
                "content": "# Title\n\nBody.\n",
            },
            tool_use_id="t-mcp-write",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · UploadDoc" in output
    # JSON dump retains the field names; the Write renderer would have
    # surfaced the path as a header without the quoted key.
    assert '"file_path"' in output
    assert '"content"' in output


def test_known_tool_with_wrong_shape_falls_through_to_json(monkeypatch):
    """If a tool name matches the dispatch table but the payload shape
    doesn't, the renderer returns ``None`` and we fall back to JSON
    rather than crashing or producing garbled output."""
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Bash",
            input={"command": 42},  # wrong type — should not render as $ 42
            tool_use_id="t-bash-bad",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "tool · Bash" in output
    assert "$ 42" not in output
    assert '"command"' in output
    assert "42" in output


def test_subagent_tool_call_renders_label_and_indent(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_tool_call(
        "planner",
        ToolCallEvent(
            name="Bash",
            input={"command": "rg foo"},
            tool_use_id="t-2",
            parent_tool_use_id="task-42",
            agent_label="Explore",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    # Subagent badge + label live in the panel title.
    assert "↳" in output
    assert "Explore" in output
    assert "tool · Bash" in output
    # Each line of the rendered panel should be indented.
    panel_lines = [l for l in output.splitlines() if l.strip()]
    assert all(
        line.startswith(" " * console_module._SUBAGENT_INDENT)
        for line in panel_lines), output
    # The bottom border should be just the frame — no subtitle text.
    bottom = next(l for l in panel_lines if "╯" in l)
    assert "subagent" not in bottom


def test_session_init_panel_lists_skills_plugins_and_agents(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_session_init(
        "planner",
        SessionInitEvent(
            skills=["update-config", "loop", "update-config"],
            plugins=["code-review", "trtllm-agent-toolkit"],
            agents=["Explore", "Plan"],
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "system" in output
    assert "plugins (2)" in output
    # ``skills`` count reflects the raw list (3), but the body
    # de-duplicates aliases when listing.
    assert "skills (3)" in output
    assert output.count("update-config") == 1
    assert "loop" in output
    assert "subagents (2)" in output
    assert "Explore" in output


def test_session_init_panel_handles_empty_payload(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_session_init("planner", SessionInitEvent())
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "system" in output
    assert "no skills, plugins, or subagents loaded" in output


def test_subagent_text_event_renders_label(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_agent_text(
        "planner",
        AgentTextEvent(
            text="found it",
            parent_tool_use_id="task-42",
            agent_label="Explore",
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "↳" in output
    assert "Explore" in output
    assert "found it" in output


def test_server_tool_call_renders_distinct_label(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_server_tool_call(
        "planner",
        ServerToolCallEvent(name="web_search",
                            input={"query": "claude"},
                            tool_use_id="srv-1"),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    # Server-side tools are rendered with a different prefix so the UI
    # can distinguish them from MCP / local tool calls at a glance.
    assert "server tool · web_search" in output
    assert "claude" in output


def test_thinking_event_renders_dim_and_truncates(monkeypatch):
    fake = _capture_console(monkeypatch)
    long_thought = "y" * (console_module._THINKING_MAX_CHARS + 25)
    console_module.print_thinking(
        "planner",
        ThinkingEvent(text=long_thought),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "thinking" in output
    assert "…" in output


def test_rate_limit_warning_renders_status_and_metadata(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_rate_limit(
        "planner",
        RateLimitWarningEvent(
            status="allowed_warning",
            rate_limit_type="five_hour",
            resets_at=1700000000,
            utilization=0.9,
        ),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "rate limit" in output
    assert "allowed_warning" in output
    assert "five_hour" in output
    assert "90.0%" in output
    assert "1700000000" in output


def test_compact_boundary_renders_trigger_and_token_counts(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_compact_boundary(
        "planner",
        CompactBoundaryEvent(trigger="auto", pre_tokens=150000),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "compact boundary" in output
    assert "auto" in output
    assert "150,000" in output


def test_compact_boundary_renders_without_optional_fields(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_compact_boundary(
        "planner",
        CompactBoundaryEvent(),
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "compact boundary" in output
    assert "auto-compacted" in output


def test_started_panel_renders_version_segment_when_present(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_agent_started(
        "planner",
        "claude-code",
        "claude-opus-4-7[1m]",
        version="cli 2.1.123 · sdk 0.1.65",
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "started" in output
    assert "claude-code" in output
    assert "claude-opus-4-7[1m]" in output
    assert "version" in output
    assert "cli 2.1.123" in output
    assert "sdk 0.1.65" in output


def test_started_panel_omits_version_segment_when_blank(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_agent_started(
        "planner",
        "claude-code",
        "claude-opus-4-7[1m]",
        version="",
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    # No "version" label and no stray separators left over from the
    # missing segment — keeps the started panel clean for backends that
    # cannot resolve a version.
    assert "version" not in output
    assert "claude-code" in output


def test_started_panel_renders_reasoning_effort_when_present(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_agent_started(
        "planner",
        "claude-code",
        "claude-opus-4-7[1m]",
        reasoning_effort="max",
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "reasoning effort" in output
    assert "max" in output


def test_started_panel_omits_reasoning_effort_when_blank(monkeypatch):
    fake = _capture_console(monkeypatch)
    console_module.print_agent_started(
        "planner",
        "claude-code",
        "claude-opus-4-7[1m]",
        reasoning_effort="",
    )
    output = fake._buffer_text.getvalue()  # type: ignore[attr-defined]
    assert "reasoning effort" not in output
    assert "claude-opus-4-7[1m]" in output


def test_console_write_recovers_from_blocking_io_error(monkeypatch):
    """When ``rich.Console`` raises ``BlockingIOError`` (because a Node.js
    subprocess put the shared stdout FD into non-blocking mode), the
    wrapper should restore blocking mode and retry the write."""
    calls: list[str] = []
    restored: list[bool] = []

    def fake_method(payload):
        calls.append(payload)
        # Fail on first call, succeed on retry.
        if len(calls) == 1:
            raise BlockingIOError(11, "write would block")

    def fake_ensure():
        restored.append(True)

    monkeypatch.setattr(console_module, "_ensure_blocking_streams", fake_ensure)
    console_module._console_write(fake_method, "hello")
    assert calls == ["hello", "hello"]
    assert restored == [True]


def test_console_write_passes_through_when_no_error(monkeypatch):
    calls: list[str] = []
    restored: list[bool] = []

    def fake_method(payload):
        calls.append(payload)

    def fake_ensure():
        restored.append(True)

    monkeypatch.setattr(console_module, "_ensure_blocking_streams", fake_ensure)
    console_module._console_write(fake_method, "hello")
    assert calls == ["hello"]
    assert restored == []


def test_print_message_recovers_from_blocking_io_error(monkeypatch):
    """End-to-end: ``print_message`` should not crash when the underlying
    rich Console raises ``BlockingIOError`` once."""
    calls: list[object] = []

    class FlakyConsole:

        def print(self, payload):
            calls.append(payload)
            if len(calls) == 1:
                raise BlockingIOError(11, "write would block")

        def rule(self, text):  # pragma: no cover - not exercised here
            calls.append(("rule", text))

    monkeypatch.setattr(console_module, "console", FlakyConsole())
    # Avoid actually touching real stdout in the test.
    monkeypatch.setattr(console_module, "_ensure_blocking_streams",
                        lambda: None)
    console_module.print_message("hello")
    assert calls == ["hello", "hello"]


def test_ensure_blocking_streams_sets_flag(monkeypatch):
    """``_ensure_blocking_streams`` should call ``os.set_blocking(True)``
    on stdout/stderr file descriptors and tolerate failures."""
    set_calls: list[tuple[int, bool]] = []

    def fake_set_blocking(fd, blocking):
        set_calls.append((fd, blocking))

    monkeypatch.setattr(console_module.os, "set_blocking", fake_set_blocking)
    console_module._ensure_blocking_streams()
    # At least stdout (1) and stderr (2) should have been flipped to
    # blocking — order does not matter.
    blocked_fds = {fd for fd, blocking in set_calls if blocking}
    # ``sys.stdout.fileno()`` is normally 1, ``sys.stderr.fileno()`` is 2,
    # but under pytest the streams are captured. Either way, we should
    # see at least one call and all calls should request blocking=True.
    assert set_calls, "expected at least one os.set_blocking call"
    assert all(blocking for _, blocking in set_calls)
    assert blocked_fds
