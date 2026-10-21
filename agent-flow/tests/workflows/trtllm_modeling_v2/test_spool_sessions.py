# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sessions declared in task.yaml: the rendered entry and the successor request."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import SCHEDULER_PATH, sessions
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    Scheduler,
    SchedulerConfig,
)


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
    assert argv[argv.index("--entry") + 1] == str(entry)
    assert argv[argv.index("--session") + 1] == "a"
    assert seen["env"]["MV2FLOW_SPOOL"] == str(tmp_path / "spool-a")
    assert seen["env"]["PATH"].startswith("/opt/venv/bin:")
    assert (tmp_path / "spool-a" / "session_entry.log").is_file()


def _sched(spool, *, ends_in) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["SLURM_JOB_ID"] = "1001"
    config = SchedulerConfig(
        devices=("0", "1"),
        session="a",
        ends_at=time.time() + ends_in,
        handover_lead_s=3600.0,
        poll_s=0.02,
        leftover_grace_s=0,
    )
    return Scheduler(spool, config, environ=env, watcher_id="w1")


def test_requests_are_listed_until_acknowledged(tmp_path):
    task = _task(tmp_path)
    _sched(Path(task["environment"]["sessions"][0]["spool"]), ends_in=60).tick()
    (pending,) = sessions.pending_requests(task)
    assert pending["session"] == "a"
    sessions.acknowledge(pending["spool"], job_id="7777", by="test")
    assert sessions.pending_requests(task) == []
