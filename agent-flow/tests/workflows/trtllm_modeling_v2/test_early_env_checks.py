# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment preconditions checked before sessions are submitted or agents dispatched.

A too-long ``TMPDIR`` fails at spec load and at ``session render``; a
multimodal checkpoint's processor imports are probed with the run's
interpreter; an unregistered ``large`` marker becomes a host fact; and every
gate gets the run's own Triton cache unless the task names one.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import cli as session_cli
from agent_flow.workflows.trtllm_modeling_v2.run import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.preflight import PreflightError
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

_SCRATCH = "/scratch/mv2flow-run-tmp"
_TOO_LONG = "/scratch/" + "d" * preflight.MAX_TMPDIR_LEN


def _spec(tmp_path, env) -> dict:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir(exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "environment": {"kind": "local", "python": "/usr/bin/python3", "env": env},
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
        },
    }


def _write(tmp_path, data) -> str:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------ TMPDIR, early


def test_the_spec_loader_rejects_a_tmpdir_too_long_for_a_socket(tmp_path):
    with pytest.raises(TaskSchemaError, match="unix socket path is capped"):
        load_and_validate_task_yaml(_write(tmp_path, _spec(tmp_path, {"TMPDIR": _TOO_LONG})))


def test_the_spec_loader_rejects_a_tmpdir_under_slash_tmp(tmp_path):
    with pytest.raises(TaskSchemaError, match="points into /tmp"):
        load_and_validate_task_yaml(_write(tmp_path, _spec(tmp_path, {"TMPDIR": "/tmp/x"})))


def test_the_spec_loader_leaves_an_unset_tmpdir_to_preflight(tmp_path):
    load_and_validate_task_yaml(_write(tmp_path, _spec(tmp_path, {"TMPDIR": _SCRATCH})))
    load_and_validate_task_yaml(_write(tmp_path, _spec(tmp_path, {})))


def _session_task(tmp_path, env, session_env=None) -> dict:
    session = {"name": "s", "spool": str(tmp_path / "spool-s"), "role": "gates"}
    if session_env:
        session["env"] = session_env
    return {
        "trtllm_repo_path": str(tmp_path / "repo"),
        "environment": {
            "kind": "session_spool",
            "python": "/opt/venv/bin/python",
            "spool": str(tmp_path / "spool-s"),
            "env": env,
            "container_image": "/images/trtllm.sqsh",
            "partition": "batch",
            "account": "acct",
            "sessions": [session],
        },
    }


@pytest.mark.parametrize("action", ["render", "submission"])
def test_a_session_is_not_rendered_or_submitted_with_a_long_tmpdir(tmp_path, capsys, action):
    path = _write(tmp_path, _session_task(tmp_path, {"TMPDIR": _TOO_LONG}))
    argv = [action, "--task", path, "--name", "s"] + (["--stdout"] if action == "render" else [])
    assert session_cli.main(argv) == 2
    assert "unix socket path is capped" in capsys.readouterr().err


def test_a_session_env_tmpdir_is_checked_too(tmp_path, capsys):
    task = _session_task(tmp_path, {"TMPDIR": _SCRATCH}, session_env={"TMPDIR": "/tmp/s"})
    path = _write(tmp_path, task)
    assert session_cli.main(["render", "--task", path, "--name", "s", "--stdout"]) == 2
    assert "points into /tmp" in capsys.readouterr().err


def test_a_session_with_a_short_tmpdir_renders(tmp_path, capsys):
    path = _write(tmp_path, _session_task(tmp_path, {"TMPDIR": _SCRATCH}))
    assert session_cli.main(["render", "--task", path, "--name", "s", "--stdout"]) == 0
    assert "#!/bin/bash" in capsys.readouterr().out


# --------------------------------------------------- multimodal processor probe


def _env() -> RunEnvironment:
    return RunEnvironment(kind="local", python="/opt/venv/bin/python", env={})


def _runner(verdict=None, **result):
    calls: list[str] = []

    def run(command, env, timeout):
        calls.append(command)
        stdout = "" if verdict is None else preflight.PROCESSOR_PROBE_MARKER + json.dumps(verdict)
        return CommandResult(returncode=result.get("returncode", 0), stdout=stdout)

    run.calls = calls
    return run


_MULTIMODAL = {"checkpoint_path": "/ckpt", "features": ["multimodal"]}


def test_no_processor_probe_without_a_multimodal_unit():
    run = _runner({"ok": True})
    assert (
        preflight.check_processor_dependencies(
            {"checkpoint_path": "/ckpt"}, _env(), shell_runner=run
        )
        == []
    )
    assert run.calls == []


def test_a_missing_processor_module_is_an_infrastructure_error():
    run = _runner({"import_error": True, "module": "some_vision_lib", "error": "No module"})
    with pytest.raises(PreflightError, match="infrastructure.*`some_vision_lib` is missing"):
        preflight.check_processor_dependencies(_MULTIMODAL, _env(), shell_runner=run)
    assert "/opt/venv/bin/python" in run.calls[0]
    assert "local_files_only=True" in run.calls[0]
    assert "/ckpt" in run.calls[0]


def test_a_processor_that_loads_is_a_note():
    notes = preflight.check_processor_dependencies(
        _MULTIMODAL, _env(), shell_runner=_runner({"ok": True})
    )
    assert notes and "loads" in notes[0]


@pytest.mark.parametrize(
    "run",
    [_runner({"error": "ValueError: unrecognized processor"}), _runner(None, returncode=1)],
)
def test_other_processor_failures_are_notes_not_stops(run):
    notes = preflight.check_processor_dependencies(_MULTIMODAL, _env(), shell_runner=run)
    assert len(notes) == 1


def test_the_probe_source_reports_a_missing_module_by_name(tmp_path):
    """The probe as the run's interpreter would execute it, with a fake processor loader."""
    fake = tmp_path / "transformers"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        "class AutoProcessor:\n"
        "    @staticmethod\n"
        "    def from_pretrained(*args, **kwargs):\n"
        "        import some_vision_lib_that_is_absent\n",
        encoding="utf-8",
    )
    out = subprocess.run(
        [sys.executable, "-c", preflight.PROCESSOR_PROBE_SOURCE, "/ckpt"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tmp_path)},
        check=True,
    ).stdout
    line = next(x for x in out.splitlines() if x.startswith(preflight.PROCESSOR_PROBE_MARKER))
    verdict = json.loads(line[len(preflight.PROCESSOR_PROBE_MARKER) :])
    assert verdict["import_error"] and verdict["module"] == "some_vision_lib_that_is_absent"


# ------------------------------------------------------------- `large` marker


def _ini(tmp_path, markers: str):
    path = tmp_path / preflight.UNITTEST_PYTEST_INI
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[pytest]\naddopts = -q\nmarkers =\n{markers}", encoding="utf-8")


def test_an_unregistered_large_marker_becomes_a_note(tmp_path):
    _ini(tmp_path, "    cpu_only: no GPU\n    gpu2: two GPUs\n")
    notes = preflight.check_large_marker(tmp_path)
    assert len(notes) == 1 and "`large`" in notes[0] and "planner" in notes[0]


def test_a_registered_large_marker_says_nothing(tmp_path):
    _ini(tmp_path, "    cpu_only: no GPU\n    large: bound-size cell\n")
    assert preflight.check_large_marker(tmp_path) == []


def test_no_pytest_ini_says_nothing(tmp_path):
    assert preflight.check_large_marker(tmp_path) == []


# --------------------------------------------------------------- Triton cache


def test_the_run_gets_its_own_triton_cache(tmp_path):
    task = {"environment": {"env": {"TMPDIR": _SCRATCH}}}
    gates.default_triton_cache(task, tmp_path)
    assert task["environment"]["env"][gates.TRITON_CACHE_DIR_ENV] == str(
        tmp_path.resolve() / gates.RUN_TRITON_CACHE_DIRNAME
    )


def test_a_declared_triton_cache_is_kept(tmp_path):
    task = {"environment": {"env": {gates.TRITON_CACHE_DIR_ENV: "/declared/cache"}}}
    gates.default_triton_cache(task, tmp_path)
    assert task["environment"]["env"][gates.TRITON_CACHE_DIR_ENV] == "/declared/cache"
