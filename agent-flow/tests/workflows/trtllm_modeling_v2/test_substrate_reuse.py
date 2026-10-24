# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A resumed run stands on a recent substrate pass in the same environment instead of re-proving it."""

from __future__ import annotations

import json
import time

import yaml

from agent_flow.logger import get_logger
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run.stages import preflight as stage
from agent_flow.workflows.trtllm_modeling_v2.run.stages.preflight import (
    SUBSTRATE_FRESH_S,
    SUBSTRATE_RECORD,
    substrate_inputs,
    substrate_reuse,
)

from .test_workflow import _workflow, passing_shell


def _substrate(command: str) -> bool:
    """The substrate gate's boot; the geometry probe boots the stock path too, and is not one."""
    return (
        "TRTLLM_MODELING_V2=off" in command
        and "quickstart" in command
        and "geometry_probe" not in command
    )


class Shell:
    """The passing gates, counting the substrate boots; ``fail`` fails them instead."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.boots = 0

    def __call__(self, command, env=None, timeout=0):
        if _substrate(command):
            self.boots += 1
            if self.fail:
                return CommandResult(returncode=1, stdout="MPI_ERR_SPAWN: could not spawn")
        return passing_shell()(command, env, timeout)


def _preflight(tmp_path, task_file, shell: Shell, **kwargs) -> tuple[bool, object]:
    workflow = _workflow(tmp_path, shell_runner=shell, skip_substrate_gate=False, **kwargs)
    log = get_logger().console
    state = workflow._init_state(task_file, log)
    return workflow._run_preflight(state, log), workflow


def _journal(tmp_path) -> str:
    return (tmp_path / "ws" / "journal.md").read_text(encoding="utf-8")


def _record(tmp_path) -> dict:
    return json.loads((tmp_path / "ws" / SUBSTRATE_RECORD).read_text(encoding="utf-8"))


def _edit_task(task_file, edit) -> None:
    spec = yaml.safe_load(task_file.read_text(encoding="utf-8"))
    edit(spec)
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")


def test_a_new_run_proves_the_substrate_and_records_the_pass(tmp_path, task_file):
    shell = Shell()
    passed, _ = _preflight(tmp_path, task_file, shell)
    assert passed and shell.boots == 1
    record = _record(tmp_path)
    assert set(record["inputs"]) == {
        "environment",
        "spec_environment",
        "engine",
        "options",
        "checkout",
    }
    assert record["notes"] and record["notes"][0].startswith("substrate gate: stock path generated")
    assert "**substrate gate passed** — run because this is a new run" in _journal(tmp_path)


def test_a_resumed_run_in_the_same_environment_stands_on_the_recorded_pass(tmp_path, task_file):
    _preflight(tmp_path, task_file, Shell())
    recorded = _record(tmp_path)

    shell = Shell()
    passed, workflow = _preflight(tmp_path, task_file, shell)
    assert passed and workflow.resume and shell.boots == 0
    journal = _journal(tmp_path)
    assert "**substrate gate not run** — passed 0.0 h ago in this environment" in journal
    # Its notes still reach the host facts, which every preflight rewrites.
    facts = workflow.host_facts_path.read_text(encoding="utf-8")
    assert recorded["notes"][0] in facts
    # The pass is not re-dated: the window runs from when it was proven.
    assert _record(tmp_path) == recorded


def test_a_resumed_run_proves_it_again_once_the_environment_changed(tmp_path, task_file):
    _preflight(tmp_path, task_file, Shell())
    _edit_task(task_file, lambda spec: spec["environment"]["env"].update({"NCCL_DEBUG": "INFO"}))

    shell = Shell()
    passed, _ = _preflight(tmp_path, task_file, shell)
    assert passed and shell.boots == 1
    journal = _journal(tmp_path)
    assert "run because the environment, spec_environment differ from the last pass's" in journal

    _edit_task(
        task_file, lambda spec: spec.update({"runtime": {"extra_flags": ["--max_seq_len=4096"]}})
    )
    shell = Shell()
    _preflight(tmp_path, task_file, shell)
    assert shell.boots == 1 and "run because the engine differs from" in _journal(tmp_path)


def test_a_resumed_run_proves_it_again_once_the_pass_is_old(tmp_path, task_file):
    _preflight(tmp_path, task_file, Shell())
    path = tmp_path / "ws" / SUBSTRATE_RECORD
    record = _record(tmp_path)
    path.write_text(json.dumps({**record, "at": time.time() - SUBSTRATE_FRESH_S - 60}))

    shell = Shell()
    _preflight(tmp_path, task_file, shell)
    assert shell.boots == 1
    assert "h old, past the 12 h window" in _journal(tmp_path)
    # Re-proven, so the next start stands on it.
    shell = Shell()
    _preflight(tmp_path, task_file, shell)
    assert shell.boots == 0


def test_a_clean_run_proves_it_whatever_is_recorded(tmp_path, task_file):
    _preflight(tmp_path, task_file, Shell())
    shell = Shell()
    passed, workflow = _preflight(tmp_path, task_file, shell, clean=True)
    assert passed and not workflow.resume and shell.boots == 1


def test_a_failed_proof_removes_the_record(tmp_path, task_file):
    _preflight(tmp_path, task_file, Shell())
    _edit_task(task_file, lambda spec: spec["environment"].update({"python": "/opt/other/python"}))

    passed, _ = _preflight(tmp_path, task_file, Shell(fail=True))
    assert not passed
    assert not (tmp_path / "ws" / SUBSTRATE_RECORD).exists()
    assert "substrate gate failed" in _journal(tmp_path)

    _edit_task(
        task_file, lambda spec: spec["environment"].update({"python": "/opt/venv/bin/python"})
    )
    shell = Shell()
    _preflight(tmp_path, task_file, shell)
    assert shell.boots == 1 and "run because no pass is recorded" in _journal(tmp_path)


# ------------------------------------------------------------------ the rule itself


def _environment(**overrides) -> RunEnvironment:
    fields = {"kind": "local", "python": "/py", "env": {"TMPDIR": "/scratch/t"}, "devices": (0,)}
    return RunEnvironment(**{**fields, **overrides})


def _task(tmp_path, **overrides) -> dict:
    return {
        "checkpoint_path": "/ckpt",
        "parallel": "tp1",
        "trtllm_repo_path": str(tmp_path / "checkout"),
        "environment": {"kind": "local", "python": "/py"},
        "runtime": {"cuda_graph": True},
        **overrides,
    }


def test_what_the_pass_stands_for_leaves_the_checkouts_content_out(tmp_path):
    (tmp_path / "checkout").mkdir()
    options = tmp_path / "core.llm_api.yaml"
    options.write_text("max_num_tokens: 8192\n")
    task = _task(tmp_path, accuracy={"llm_api_options": str(options)})
    inputs = substrate_inputs(task, _environment())

    (tmp_path / "checkout" / "model.py").write_text("changed = True\n")
    assert substrate_inputs(task, _environment()) == inputs
    # A unit's own options file is not what the stock boot reads; the default is.
    task["accuracy"]["llm_api_options"] = {"default": str(options), "mtp": "/elsewhere.yaml"}
    assert substrate_inputs(task, _environment()) == inputs

    changed = {
        "options": lambda: options.write_text("max_num_tokens: 4096\n"),
        "environment": lambda: None,
        "checkout": lambda: task.update({"trtllm_repo_path": "/another/checkout"}),
        "engine": lambda: task.update({"checkpoint_path": "/another/ckpt"}),
        "spec_environment": lambda: task["environment"].update({"substrate_checkpoint": "/proxy"}),
    }
    for part, edit in changed.items():
        edit()
        environment = _environment(devices=(1,)) if part == "environment" else _environment()
        now = substrate_inputs(task, environment)
        assert {name for name in now if now[name] != inputs[name]} == {part}, part
        inputs = substrate_inputs(task, _environment())


def test_a_pass_stands_only_for_its_inputs_and_inside_its_window():
    inputs = {"environment": "a", "engine": "b"}
    now = 1_000_000.0
    record = {"at": now - 3600, "inputs": dict(inputs), "notes": []}
    assert substrate_reuse(record, inputs, now=now) == (
        True,
        "passed 1.0 h ago in this environment",
    )
    assert substrate_reuse(None, inputs, now=now) == (False, "no pass is recorded")
    assert substrate_reuse({**record, "inputs": {"environment": "a"}}, inputs, now=now) == (
        False,
        "the engine differs from the last pass's",
    )
    stale, why = substrate_reuse(record, inputs, now=now + SUBSTRATE_FRESH_S)
    assert not stale and why == "the last pass is 13.0 h old, past the 12 h window"
    # A pass dated after now is no proof of anything.
    assert not substrate_reuse(record, inputs, now=now - 7200)[0]
    assert substrate_reuse({**record, "at": "yesterday"}, inputs, now=now) == (
        False,
        "the last pass has no time",
    )
    assert stage.SUBSTRATE_FRESH_S == 12 * 3600
