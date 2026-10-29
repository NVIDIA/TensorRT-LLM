# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`trtllm-modeling-v2 gate <name>`: the orchestrator's own gates, run standalone."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import disagg, gate_cli
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


def test_a_boot_gate_runs_standalone_at_the_cheapest_member(tmp_path, shell, capsys):
    code = gate_cli.main(["boot", "--task", str(_task(tmp_path)), "--workdir", str(tmp_path / "w")])
    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("[PASS] boot@tep2")
    config = json.loads((tmp_path / "w/core/boot.core.tep2.json").read_text())["llm"]
    assert config["tensor_parallel_size"] == config["moe_expert_parallel_size"] == 2
    assert "CUDA_VISIBLE_DEVICES=0,1" in shell["seen"][0]


def _junit(*cases):
    return "<testsuite>" + "".join(cases) + "</testsuite>"


_PASSED = '<testcase classname="accuracy.TestM" name="test_auto_dtype"/>'


def test_the_disagg_gate_runs_the_trees_test_from_its_pytest_root(tmp_path, shell, capsys):
    """Both engines' devices, the spec's environment, CI's root, and the JUnit verdict."""
    task = _task(tmp_path)
    root = tmp_path / "repo/tests/integration/defs"
    root.mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\n")
    shell["junit"] = _junit(_PASSED)
    argv = ["disagg", "--task", str(task), "--unit", "disagg", "--workdir", str(tmp_path / "w")]
    assert gate_cli.main(argv) == 0
    assert capsys.readouterr().out.startswith("[PASS] disagg@tep2")
    [command] = shell["seen"]
    assert command.startswith(f"cd {root} && ")
    assert "accuracy/test_disaggregated_serving.py::TestM::test_auto_dtype" in command
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in command and "LLM_MODELS_ROOT=/models" in command


def test_the_disagg_verdict_is_a_pass_only_when_every_selected_test_passed(tmp_path):
    report = tmp_path / "r.xml"
    skipped = '<testcase name="t2"><skipped message="needs 8 GPUs"/></testcase>'
    failed = '<testcase name="t"><failure message="accuracy 41.0 below 57.4"/></testcase>'
    for cases, detail in (
        ((_PASSED,), "1 test(s) passed"),
        ((_PASSED, skipped), "skipped"),
        ((failed,), "failure"),
        ((), "selected no test"),
    ):
        report.write_text(_junit(*cases))
        passed, said = disagg.judge_report(report)
        assert passed == (detail == "1 test(s) passed") and detail in said, detail
    assert disagg.judge_report(tmp_path / "missing.xml")[0] is False


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
