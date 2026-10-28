# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sessions declared in task.yaml: the rendered entry."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import SCHEDULER_PATH, sessions


def _task(tmp_path):
    return {
        "trtllm_repo_path": str(tmp_path / "repo"),
        "environment": {
            "kind": "session_spool",
            "python": "/opt/venv/bin/python",
            "spool": str(tmp_path / "spool-a"),
            "env": {"PATH": "/opt/venv/bin"},
            "sessions": [{"name": "a", "spool": str(tmp_path / "spool-a"), "role": "gates"}],
        },
    }


def _run_entry(tmp_path, name):
    task = _task(tmp_path)
    entry = sessions.install_session_entry(task, name)
    stub = tmp_path / "stub-python"
    capture = tmp_path / f"captured-{name}.json"
    stub.write_text(
        "#!/bin/bash\n"
        f'python3 -c \'import json, os, sys; json.dump({{"argv": sys.argv[1:], "env": dict(os.environ)}}, '
        f'open("{capture}", "w"))\' "$@"\n'
    )
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update({"MV2FLOW_SCHEDULER_PYTHON": str(stub), "SLURM_NODEID": "0"})
    done = subprocess.run(
        ["bash", str(entry)], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    return entry, json.loads(capture.read_text())


def test_the_rendered_entry_becomes_the_scheduler(tmp_path):
    entry, seen = _run_entry(tmp_path, "a")
    assert entry == tmp_path / "spool-a" / "session_entry.sh" and os.access(entry, os.X_OK)
    argv = seen["argv"]
    assert argv[:3] == [str(SCHEDULER_PATH), "watch", str(tmp_path / "spool-a")]
    assert argv[argv.index("--session") + 1] == "a"
    assert seen["env"]["MV2FLOW_SPOOL"] == str(tmp_path / "spool-a")
    assert seen["env"]["PATH"].startswith("/opt/venv/bin:")
    assert (tmp_path / "spool-a" / "session_entry.log").is_file()


def test_session_cli_prepares_a_missing_spool_without_submitting(tmp_path, monkeypatch, capsys):
    from agent_flow.workflows.trtllm_modeling_v2 import cli

    def no_submit(*args, **kwargs):
        raise AssertionError("session commands must not submit jobs")

    monkeypatch.setattr(subprocess, "run", no_submit)
    monkeypatch.setattr(subprocess, "Popen", no_submit)
    task = _task(tmp_path)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task), encoding="utf-8")
    spool = tmp_path / "spool-a"
    session = cli.SUBCOMMANDS["session"]
    options = ["--task", str(path), "--name", "a"]
    assert session(["render", *options, "--stdout"]) == 0
    assert "exec" in capsys.readouterr().out
    assert not spool.exists()
    assert session(["submission", *options, "--after", "123"]) == 0
    call = capsys.readouterr().out
    assert "mcp__slurm-broker__slurm_submit(" in call and "afterany:123" in call
    assert not spool.exists()

    assert session(["allocate", *options]) == 0
    plan = capsys.readouterr().out
    assert "Submit this through the broker" in plan
    assert "No dependency:" in plan and "session ack" in plan
    assert (spool / "session_entry.sh").is_file()
    assert not (spool / "handover").exists()
    assert session(["ack", *options, "--job", "456", "--after", "123"]) == 0
    ack = json.loads(Path(capsys.readouterr().out.strip()).read_text())
    assert {k: ack[k] for k in ("job_id", "succeeds", "session", "by")} == {
        "job_id": "456",
        "succeeds": "123",
        "session": "a",
        "by": "driver",
    }


def test_session_status_and_tune_host_follow_the_latest_watcher(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.tune.measure import spool_host

    task = _task(tmp_path)
    spool = tmp_path / "spool-a"
    (spool / "watchers").mkdir(parents=True)
    (spool / "status.json").write_text(
        json.dumps({"watcher": "old", "host": "old-node", "alive_epoch": 100, "slurm_job": "1"})
    )
    (spool / "watchers" / "successor.json").write_text(
        json.dumps({"watcher": "new", "host": "new-node", "alive_epoch": 190, "slurm_job": "2"})
    )
    status = sessions.session_status(task, now=200)[0]
    assert status["slurm_job"] == "2" and status["heartbeat_age_s"] == 10
    assert spool_host(str(spool)) == "new-node"
