# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator's loop tells a busy orchestrator from one that has exited."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra import files
from agent_flow.workflows.trtllm_modeling_v2.infra.files import write_json_atomic
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import status
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor.heartbeat import (
    ORCHESTRATOR_EXIT,
    ORCHESTRATOR_HEARTBEAT,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor.status import load_config


def test_monitor_cli_tells_a_busy_orchestrator_from_one_that_has_exited(
    tmp_path, monkeypatch, capsys
):
    from agent_flow.workflows.trtllm_modeling_v2 import cli

    monitor = cli.SUBCOMMANDS["monitor"]
    monkeypatch.delenv("MV2FLOW_WORKSPACE", raising=False)
    assert monitor(["status"]) == 2
    assert "pass --workspace or set MV2FLOW_WORKSPACE" in capsys.readouterr().err
    assert monitor(["--workspace", str(tmp_path / "missing"), "status"]) == 1
    assert "not a trtllm_modeling_v2 workspace" in capsys.readouterr().err

    (tmp_path / "task.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("MV2FLOW_WORKSPACE", str(tmp_path))
    assert monitor(["status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data == {
        "workspace": str(tmp_path.resolve()),
        "orchestrator": {"state": "unknown", "detail": "no orchestrator heartbeat"},
        "correctness": None,
        "sessions": [],
    }
    dispatch = tmp_path / "dispatch"
    dispatch.mkdir()
    write_json_atomic(dispatch / ORCHESTRATOR_HEARTBEAT, {"pid": 1, "host": "h", "at": time.time()})
    assert status.orchestrator_state(load_config(tmp_path))[0] == "running"
    assert monitor(["wait", "--timeout", "0", "--no-sessions"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "TIMEOUT  nothing new"
    # A closed run changes nothing else, so the waiter must notice the exit.
    write_json_atomic(dispatch / ORCHESTRATOR_EXIT, {"pid": 1, "host": "h", "at": time.time() + 1})
    assert monitor(["--workspace", str(tmp_path), "wait", "--timeout", "0"]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("ORCHESTRATOR_EXITED  pid 1")


def test_atomic_records_keep_complete_values_and_preserve_the_old_record_on_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "record.json"
    barrier = threading.Barrier(2)
    replace = files.os.replace

    def simultaneous(source, target):
        barrier.wait(timeout=5)
        replace(source, target)

    monkeypatch.setattr(files.os, "replace", simultaneous)
    records = [{"writer": n, "body": str(n) * 10000} for n in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda record: write_json_atomic(path, record), records))
    kept = files.read_json(path)
    assert kept in records
    assert list(tmp_path.iterdir()) == [path]

    def fail_replace(source, target):
        raise OSError("storage refused replacement")

    monkeypatch.setattr(files.os, "replace", fail_replace)
    with pytest.raises(OSError, match="storage refused"):
        write_json_atomic(path, {"writer": "failed"})
    assert files.read_json(path) == kept
    assert list(tmp_path.iterdir()) == [path]

    monkeypatch.setattr(files.os, "replace", replace)
    ordinary = tmp_path / "ordinary"
    ordinary.write_text("same umask")
    files.write_text_atomic(path, "shared")
    assert path.stat().st_mode & 0o777 == ordinary.stat().st_mode & 0o777
    files.write_text_atomic(path, "private options", mode=0o600)
    assert path.stat().st_mode & 0o077 == 0
