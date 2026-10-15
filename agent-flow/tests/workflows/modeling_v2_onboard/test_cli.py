# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``modeling-v2-onboard`` command line."""

from __future__ import annotations

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
