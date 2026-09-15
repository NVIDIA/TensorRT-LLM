# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import sys

from agent_flow.workflow_tool import ToolCompletion, run_stop_hook


def test_tool_completion_tracks_only_the_current_turn(tmp_path):
    state_path = tmp_path / "completion.json"
    completion = ToolCompletion({"append_progress"}, state_path)

    completion.mark("append_progress")
    assert completion.missing() == set()
    assert json.loads(state_path.read_text())["completed"] == ["append_progress"]

    completion.reset()
    assert completion.missing() == {"append_progress"}


def test_codex_stop_hook_blocks_once_until_required_tool_finishes(tmp_path, monkeypatch, capsys):
    state_path = tmp_path / "completion.json"
    completion = ToolCompletion({"append_progress"}, state_path)

    monkeypatch.setattr(sys, "stdin", io.StringIO('{"stop_hook_active": false}'))
    run_stop_hook(state_path)
    assert json.loads(capsys.readouterr().out)["decision"] == "block"

    completion.mark("append_progress")
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"stop_hook_active": false}'))
    run_stop_hook(state_path)
    assert json.loads(capsys.readouterr().out) == {}
