# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`trtllm-modeling-v2 gate <name>`: the orchestrator's own gates, run standalone."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult


def _task(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
        "arch": "sm_100",
        "parallel": "unified",
        "members": ["tp2", "tep2"],
        "features": ["mtp", "disagg"],
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "r.yaml",
            "model_key": "M",
            "disagg": {"test": DISAGG_TEST, "env": {"LLM_MODELS_ROOT": "/models"}},
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


DISAGG_TEST = (
    "tests/integration/defs/accuracy/test_disaggregated_serving.py::TestM::test_auto_dtype"
)


def _records(*texts):
    return "".join(
        f"{workload_io.RECORD_MARKER} {json.dumps({'index': n, 'text': t})}\n"
        for n, t in enumerate(texts)
    )


@pytest.fixture
def shell(monkeypatch):
    """Replace the executor-backed runner with a recorder answering `output`."""
    state = {"output": _records("Paris", "the process"), "seen": [], "junit": None}

    def fake_runner_for(executor, **kwargs):
        def run(command, env=None, timeout=0):
            state["seen"].append(command)
            report = re.search(r"--junitxml=([^\s']+)", command)
            if report and state["junit"] is not None:
                Path(report.group(1)).write_text(state["junit"], encoding="utf-8")
            output = state["output"]
            return CommandResult(
                returncode=0, stdout=output.pop(0) if isinstance(output, list) else output
            )

        return run

    monkeypatch.setattr(gate_cli, "shell_runner_for", fake_runner_for)
    return state


def test_a_relative_variant_is_judged_against_its_units_score_measured_first(
    tmp_path, shell, capsys
):
    path = _task(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["variants"] = {"switched": {"env": {"MY_SWITCH": "1"}, "gates": ["accuracy"]}}
    path.write_text(yaml.safe_dump(data))
    shell["output"] = [f"lm-eval gsm8k average accuracy: {score}\n" for score in (92, 85)]
    argv = ["accuracy", "--task", str(path), "--unit", "switched", "--workdir", str(tmp_path / "w")]
    assert gate_cli.main([*argv, "--member", "tep2"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("[FAIL] accuracy[gsm8k]@tep2: measured 85.0000 vs floor 89.8872")
    assert "the unit's own score 92.0000" in out and len(shell["seen"]) == 2
    assert "MY_SWITCH=1" not in shell["seen"][0] and "MY_SWITCH=1" in shell["seen"][1]
