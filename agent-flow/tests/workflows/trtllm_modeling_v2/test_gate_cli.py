# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`trtllm-modeling-v2 gate <name>`: the orchestrator's own gates, run standalone."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli, gates, witness
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


def test_under_a_scheduler_the_job_keeps_the_devices_it_is_handed(tmp_path, shell, monkeypatch):
    monkeypatch.setattr(gate_cli, "select_command_executor", lambda *a, **k: None)
    (tmp_path / "s").mkdir()
    spool = {"kind": "session_spool", "python": "/usr/bin/python3", "spool": str(tmp_path / "s")}
    task = _task(tmp_path, environment=spool)
    argv = ["boot", "--task", str(task), "--workdir", str(tmp_path / "w"), "--member", "tp2"]
    assert gate_cli.main(argv) == 0
    assert "CUDA_VISIBLE_DEVICES" not in shell["seen"][-1]
    assert gate_cli.main([*argv, "--devices", "2,3"]) == 0
    assert "CUDA_VISIBLE_DEVICES=2,3" in shell["seen"][-1]


def test_a_tree_is_measured_from_inside_itself(tmp_path, shell):
    """A worker resolves `tensorrt_llm` from its cwd ahead of PYTHONPATH."""
    tree = tmp_path / "pre-fix"
    tree.mkdir()
    gate_cli.main(
        [
            "boot",
            "--task",
            str(_task(tmp_path)),
            "--tree",
            str(tree),
            "--member",
            "tp2",
            "--workdir",
            str(tmp_path / "w"),
        ]
    )
    command = shell["seen"][0]
    assert command.startswith(f"cd {tree.resolve()} && ")
    assert f"PYTHONPATH={tree.resolve()}" in command
    assert f"{tree.resolve()}/examples/llm-api/quickstart_advanced.py" in command


def test_on_a_git_checkout_a_snapshot_of_a_commit_is_measured(tmp_path, shell, capsys):
    """As the orchestrator's checks are: never the checkout the run's integrations move."""
    from .test_workflow import _git

    task = _task(tmp_path)
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir()
    (repo / "tensorrt_llm" / "__init__.py").write_text("")
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    first = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "commit", "-q", "--allow-empty", "-m", "second")
    argv = ["boot", "--task", str(task), "--member", "tp2", "--workdir", str(tmp_path / "w")]
    assert gate_cli.main(argv) == 0
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert shell["seen"][-1].startswith(f"cd {tmp_path / 'snapshots' / head} && ")
    assert f"[tree {head[:7]}]" in capsys.readouterr().out
    assert gate_cli.main([*argv, "--commit", first]) == 0
    assert shell["seen"][-1].startswith(f"cd {tmp_path / 'snapshots' / first} && ")
    with pytest.raises(SystemExit):
        gate_cli.main([*argv, "--commit", first, "--tree", str(repo)])


def test_a_failing_gate_exits_one_and_prints_its_evidence(tmp_path, shell, capsys):
    shell["output"] = "nothing structured\n"
    code = gate_cli.main(["boot", "--task", str(_task(tmp_path)), "--workdir", str(tmp_path)])
    assert code == 1
    assert "[FAIL] boot" in capsys.readouterr().out


def test_json_output_is_the_whole_verdict(tmp_path, shell, capsys):
    gate_cli.main(["boot", "--task", str(_task(tmp_path)), "--json", "--workdir", str(tmp_path)])
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["name"] == "boot" and verdict["passed"] is True and verdict["member"] == "tep2"


def test_the_witness_is_the_workflows_own_dispatch(tmp_path, shell, monkeypatch):
    seen = {}

    def fake(environment, **kwargs):
        seen.update(kwargs)
        return gates.GateResult(name="kv_quant", passed=True, detail="ok")

    monkeypatch.setattr(witness, "run_declared_witness", fake)
    code = gate_cli.main(
        [
            "witness",
            "--task",
            str(_task(tmp_path)),
            "--unit",
            "kv_quant",
            "--member",
            "tp2",
            "--workdir",
            str(tmp_path / "w"),
        ]
    )
    assert code == 0
    assert seen["unit"] == "kv_quant" and seen["member"] == "tp2"
    assert seen["features"] == ["mtp", "kv_quant"]
    assert seen["workdir"] == (tmp_path / "w").resolve() / "witness" / "kv_quant"


@pytest.mark.parametrize(("unit", "gpus"), [("disagg", "all"), ("kv_quant", "2")])
def test_an_equivalence_witness_asks_for_both_legs_devices(
    tmp_path, shell, monkeypatch, unit, gpus
):
    asked = []

    def executor(environment, **kwargs):
        asked.append(kwargs["gpus"])

    monkeypatch.setattr(gate_cli, "select_command_executor", executor)
    monkeypatch.setattr(
        witness, "run_declared_witness", lambda *a, **k: gates.GateResult(unit, True, "ok")
    )
    task = _task(tmp_path, features=["mtp", "kv_quant", "disagg"])
    argv = ["witness", "--task", str(task), "--unit", unit, "--member", "tp2"]
    assert gate_cli.main([*argv, "--workdir", str(tmp_path / "w")]) == 0
    # Leg B of the equivalence witness is two engines of the member's width.
    assert asked == [gpus]


def test_a_unit_without_a_witness_is_a_usage_error(tmp_path, shell, capsys):
    code = gate_cli.main(["witness", "--task", str(_task(tmp_path)), "--unit", "mtp"])
    assert code == 2 and "has no witness" in capsys.readouterr().err


def test_an_invalid_task_is_a_usage_error(tmp_path, shell, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("arch: sm_1\n", encoding="utf-8")
    assert gate_cli.main(["boot", "--task", str(bad)]) == 2


def test_the_run_cli_dispatches_the_subcommand(tmp_path, shell):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["gate", "boot", "--task", str(_task(tmp_path)), "--workdir", str(tmp_path)])
    assert exit_info.value.code == 0


def test_the_acceptance_gate_keeps_its_results_where_told(tmp_path, shell):
    rows = [
        {"index": n, "prompt_sha": f"p{n}", "tokens": 300, "iterations": 100} for n in range(32)
    ]
    shell["output"] = "".join(f"{workload_io.RECORD_MARKER} {json.dumps(r)}\n" for r in rows)
    results = tmp_path / "shared.jsonl"
    code = gate_cli.main(
        [
            "acceptance",
            "--task",
            str(_task(tmp_path)),
            "--unit",
            "mtp",
            "--results",
            str(results),
            "--workdir",
            str(tmp_path / "w"),
        ]
    )
    assert code == 0
    assert gates.GateRecords(results).all("acceptance")


def test_the_assembler_is_pointed_at_the_entry_point(tmp_path):
    from .test_workflow import _workflow

    wf = _workflow(tmp_path)
    wf.task_data = {"trtllm_repo_path": "/repo", "features": []}
    wf._preflight = type("P", (), {"target_rel": "t"})()
    prompt = wf._assembler_prompt("core")
    assert "trtllm-modeling-v2 gate <boot|" in prompt and "--unit core" in prompt
    assert str(Path(wf.workspace) / "gates") in prompt


def test_the_cli_is_installed_where_an_agent_can_call_it_by_path(tmp_path):
    import subprocess

    from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli

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


def test_the_workspace_cli_runs_the_orchestrators_package_from_any_directory(tmp_path):
    """A shell inside another agent-flow checkout still runs the package the orchestrator runs."""
    import subprocess

    from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli

    script = gate_cli.install_cli(tmp_path / "ws")
    other = tmp_path / "other-checkout"
    (other / "agent_flow").mkdir(parents=True)
    (other / "agent_flow" / "__init__.py").write_text("raise SystemExit('the wrong agent_flow')\n")
    out = subprocess.run(
        [str(script), "gate", "--help"],
        capture_output=True,
        text=True,
        cwd=other,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    assert "wrong agent_flow" not in out.stderr
