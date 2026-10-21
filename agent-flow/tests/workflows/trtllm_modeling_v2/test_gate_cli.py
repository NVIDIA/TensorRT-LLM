# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`trtllm-modeling-v2 gate <name>`: the orchestrator's own gates, run standalone."""

from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult


def _task(tmp_path, **extra):
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
        "features": ["mtp", "kv_quant"],
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml", "model_key": "M"},
        **extra,
    }
    data["accuracy"]["witness"] = {
        "kv_quant": {"contrast_env": {"KV": "bf16"}, "expected_bytes_per_token": 890}
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _records(*texts):
    return "".join(
        f"{workload_io.RECORD_MARKER} {json.dumps({'index': n, 'text': t})}\n"
        for n, t in enumerate(texts)
    )


@pytest.fixture
def shell(monkeypatch):
    """Replace the executor-backed runner with a recorder answering `output`."""
    state = {"output": _records("Paris", "the process"), "seen": []}

    def fake_runner_for(executor, **kwargs):
        def run(command, env=None, timeout=0):
            state["seen"].append(command)
            return CommandResult(returncode=0, stdout=state["output"])

        return run

    monkeypatch.setattr(gate_cli, "shell_runner_for", fake_runner_for)
    return state


def test_a_boot_gate_runs_standalone_at_the_cheapest_member(tmp_path, shell, capsys):
    code = gate_cli.main(["boot", "--task", str(_task(tmp_path)), "--workdir", str(tmp_path / "w")])
    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("[PASS] boot@tep2")
    assert "--tp_size=2 --moe_ep_size=2" in shell["seen"][0]
    assert "CUDA_VISIBLE_DEVICES=0,1" in shell["seen"][0]


def test_the_cli_is_installed_where_an_agent_can_call_it_by_path(tmp_path):
    script = gate_cli.install_cli(tmp_path)
    assert script == tmp_path / "bin" / "trtllm-modeling-v2"
    out = subprocess.run(
        [str(script), "gate", "--help"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0 and "--unit" in out.stdout
    assert gate_cli.invocation("t.yaml", "w", "core", str(script)).startswith(f"{script} gate <")
