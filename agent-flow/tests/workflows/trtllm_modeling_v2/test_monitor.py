# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator's loop tells a busy orchestrator from one that has exited."""

from __future__ import annotations

import json
import time

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
