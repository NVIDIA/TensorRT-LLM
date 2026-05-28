from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_flow.hooks import (_is_human_user_entry, _tool_uses_since_last_user,
                              called_required_tool_this_turn,
                              require_tool_call_stop_hook)


def _write_transcript(path: Path, entries: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n",
                    encoding="utf-8")
    return path


def _user(text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _user_tool_result(tool_use_id: str, result: str = "ok") -> dict:
    return {
        "type": "user",
        "message": {
            "role":
            "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
            }],
        },
    }


def _assistant_tool_use(name: str, tool_use_id: str = "tu_1") -> dict:
    return {
        "type": "assistant",
        "message": {
            "role":
            "assistant",
            "content": [{
                "type": "tool_use",
                "id": tool_use_id,
                "name": name,
                "input": {},
            }],
        },
    }


def _assistant_text(text: str) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": text
            }],
        },
    }


class TestHumanUserEntryDetection:

    def test_plain_string_user_is_human(self):
        assert _is_human_user_entry(_user("hello")) is True

    def test_tool_result_only_user_is_not_human(self):
        assert _is_human_user_entry(_user_tool_result("tu_1")) is False

    def test_mixed_content_is_human(self):
        entry = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": "please"
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1"
                    },
                ],
            },
        }
        assert _is_human_user_entry(entry) is True

    def test_assistant_is_not_user(self):
        assert _is_human_user_entry(_assistant_text("hi")) is False


class TestToolUsesSinceLastUser:

    def test_collects_tool_uses_after_last_human_message(self):
        entries = [
            _user("first turn"),
            _assistant_tool_use("Read"),
            _user("second turn"),
            _assistant_tool_use("mcp__agent-tools__append_planner_progress"),
            _assistant_text("done"),
        ]
        assert _tool_uses_since_last_user(entries) == [
            "mcp__agent-tools__append_planner_progress"
        ]

    def test_tool_result_user_is_not_a_boundary(self):
        entries = [
            _user("task"),
            _assistant_tool_use("Read", "tu_1"),
            _user_tool_result("tu_1"),
            _assistant_tool_use("mcp__agent-tools__append_planner_progress",
                                "tu_2"),
        ]
        # Boundary is the first (real) user message; both tool uses
        # count — including the one between the tool_result echo.
        assert _tool_uses_since_last_user(entries) == [
            "Read", "mcp__agent-tools__append_planner_progress"
        ]

    def test_empty_when_no_assistant_activity_this_turn(self):
        entries = [_user("hi")]
        assert _tool_uses_since_last_user(entries) == []


class TestCalledRequiredToolThisTurn:

    def test_matches_fully_qualified_name(self, tmp_path):
        transcript = _write_transcript(
            tmp_path / "t.jsonl",
            [
                _user("hi"),
                _assistant_tool_use(
                    "mcp__agent-tools__append_planner_progress"),
            ],
        )
        assert called_required_tool_this_turn(
            transcript, ["mcp__agent-tools__append_planner_progress"]) is True

    def test_matches_short_name(self, tmp_path):
        transcript = _write_transcript(
            tmp_path / "t.jsonl",
            [
                _user("hi"),
                _assistant_tool_use(
                    "mcp__agent-tools__append_planner_progress"),
            ],
        )
        assert called_required_tool_this_turn(
            transcript, ["append_planner_progress"]) is True

    def test_returns_false_when_tool_not_called(self, tmp_path):
        transcript = _write_transcript(tmp_path / "t.jsonl", [
            _user("hi"),
            _assistant_tool_use("Read"),
            _assistant_text("done"),
        ])
        assert called_required_tool_this_turn(
            transcript, ["append_planner_progress"]) is False

    def test_previous_turn_calls_dont_count(self, tmp_path):
        transcript = _write_transcript(
            tmp_path / "t.jsonl",
            [
                _user("turn 1"),
                _assistant_tool_use("mcp__agent-tools__append_planner_progress",
                                    "tu_1"),
                _user("turn 2"),
                _assistant_text("done without calling tool"),
            ],
        )
        assert called_required_tool_this_turn(
            transcript, ["append_planner_progress"]) is False

    def test_missing_transcript_returns_false(self, tmp_path):
        missing = tmp_path / "missing.jsonl"
        assert called_required_tool_this_turn(
            missing, ["append_planner_progress"]) is False

    def test_malformed_lines_are_skipped(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(
            "not json\n" + json.dumps(_user("hi")) + "\n" + json.dumps(
                _assistant_tool_use(
                    "mcp__agent-tools__append_planner_progress")) + "\n",
            encoding="utf-8",
        )
        assert called_required_tool_this_turn(
            path, ["append_planner_progress"]) is True

    def test_empty_required_is_vacuously_true(self, tmp_path):
        transcript = _write_transcript(tmp_path / "t.jsonl", [_user("hi")])
        assert called_required_tool_this_turn(transcript, []) is True


class TestRequireToolCallStopHook:

    def test_builds_stop_hook_structure(self, tmp_path):
        config = require_tool_call_stop_hook(["append_planner_progress"])

        assert set(config.keys()) == {"Stop"}
        assert len(config["Stop"]) == 1
        matcher = config["Stop"][0]
        assert len(matcher.hooks) == 1
        assert callable(matcher.hooks[0])

    def test_rejects_empty_required(self):
        with pytest.raises(ValueError, match="must not be empty"):
            require_tool_call_stop_hook([])

    async def test_hook_allows_when_tool_was_called(self, tmp_path):
        transcript = _write_transcript(
            tmp_path / "t.jsonl",
            [
                _user("hi"),
                _assistant_tool_use(
                    "mcp__agent-tools__append_planner_progress"),
            ],
        )
        config = require_tool_call_stop_hook(["append_planner_progress"])
        hook = config["Stop"][0].hooks[0]

        result = await hook(
            {
                "hook_event_name": "Stop",
                "transcript_path": str(transcript),
                "stop_hook_active": False,
            },
            None,
            {},
        )
        assert result == {}

    async def test_hook_blocks_when_tool_missing(self, tmp_path):
        transcript = _write_transcript(
            tmp_path / "t.jsonl",
            [_user("hi"), _assistant_text("no tool here")],
        )
        config = require_tool_call_stop_hook(["append_planner_progress"])
        hook = config["Stop"][0].hooks[0]

        result = await hook(
            {
                "hook_event_name": "Stop",
                "transcript_path": str(transcript),
                "stop_hook_active": False,
            },
            None,
            {},
        )
        assert result["decision"] == "block"
        assert "append_planner_progress" in result["reason"]

    async def test_hook_gives_up_when_stop_hook_active(self, tmp_path):
        transcript = _write_transcript(
            tmp_path / "t.jsonl",
            [_user("hi"), _assistant_text("still missing")],
        )
        config = require_tool_call_stop_hook(["append_planner_progress"])
        hook = config["Stop"][0].hooks[0]

        # Second stop attempt after we already blocked once — give up rather
        # than loop forever.
        result = await hook(
            {
                "hook_event_name": "Stop",
                "transcript_path": str(transcript),
                "stop_hook_active": True,
            },
            None,
            {},
        )
        assert result == {}

    async def test_hook_uses_custom_reason(self, tmp_path):
        transcript = _write_transcript(
            tmp_path / "t.jsonl",
            [_user("hi"), _assistant_text("nothing")],
        )
        config = require_tool_call_stop_hook(
            ["append_planner_progress"],
            reason="please log progress!",
        )
        hook = config["Stop"][0].hooks[0]

        result = await hook(
            {
                "hook_event_name": "Stop",
                "transcript_path": str(transcript),
                "stop_hook_active": False,
            },
            None,
            {},
        )
        assert result == {
            "decision": "block",
            "reason": "please log progress!",
        }
