# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``modeling-v2-onboard`` command line."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_flow import CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL, AgentLayer
from agent_flow.workflows.modeling_v2_onboard.cli import _parse_args, main, make_agent_factory


@pytest.mark.parametrize(
    ("backend", "model"),
    [("claude-code", CLAUDE_CODE_DEFAULT_MODEL), ("codex", CODEX_DEFAULT_MODEL)],
)
def test_every_agent_uses_the_selected_backend(tmp_path, backend, model):
    agent = make_agent_factory(backend, None)("planner", "sys", tmp_path, "persistent")
    assert isinstance(agent, AgentLayer)
    config = agent.config
    assert (config.backend.kind, config.backend.model) == (backend, model)
    assert config.backend.cwd == tmp_path
    assert config.session.mode == "persistent"
    assert config.system_prompt == "sys"


def test_explicit_model_wins(tmp_path):
    agent = make_agent_factory("codex", "my-model")("worker[a]", "sys", tmp_path, "persistent")
    assert agent.config.backend.model == "my-model"


def test_defaults():
    args = _parse_args(["--task", "t.yaml", "--workspace", "ws"])
    assert (args.backend, args.model, args.max_parallel, args.clean) == (
        "claude-code",
        None,
        8,
        False,
    )
    assert args.task == Path("t.yaml")


def test_max_parallel_must_be_positive():
    with pytest.raises(SystemExit):
        _parse_args(["--task", "t", "--workspace", "w", "--max-parallel", "0"])


def test_an_invalid_task_exits_with_an_error(tmp_path, capsys):
    task = tmp_path / "task.yaml"
    task.write_text("checkpoint_path: /nonexistent\n")
    with pytest.raises(SystemExit) as info:
        main(["--task", str(task), "--workspace", str(tmp_path / "ws")])
    assert info.value.code == 2
    assert "reference_code_path" in capsys.readouterr().err


def test_a_run_keeps_its_agents_off_claude_code_auto_memory(tmp_path, monkeypatch):
    # Set through monkeypatch so the test leaves the variable as it found it.
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "0")
    task = tmp_path / "task.yaml"
    task.write_text("checkpoint_path: /nonexistent\n")
    with pytest.raises(SystemExit):
        main(["--task", str(task), "--workspace", str(tmp_path / "ws")])
    assert os.environ["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"


def test_feedback_leaves_a_message_without_a_task(tmp_path, capsys):
    main(["--workspace", str(tmp_path / "ws"), "--feedback", "drop fla-fix-test"])
    (message,) = (tmp_path / "ws" / "inbox").glob("*.md")
    assert message.read_text() == "drop fla-fix-test\n"
    assert str(message) in capsys.readouterr().out

    note = tmp_path / "note.md"
    note.write_text("second\n")
    main(["--workspace", str(tmp_path / "ws"), "--feedback-file", str(note)])
    assert len(list((tmp_path / "ws" / "inbox").glob("*.md"))) == 2


def test_running_needs_a_task_and_feedback_excludes_clean(tmp_path):
    with pytest.raises(SystemExit):
        _parse_args(["--workspace", "ws"])
    with pytest.raises(SystemExit):
        _parse_args(["--workspace", "ws", "--feedback", "x", "--clean"])
    with pytest.raises(SystemExit):
        main(["--workspace", str(tmp_path), "--feedback", "   "])
