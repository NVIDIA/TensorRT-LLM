# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A successor acknowledged before the scheduler asks for one still answers the request.

A driver that submits a session's successor early records it with
``session ack`` before ``handover/request.json`` exists. The request written
later for that same allocation is answered by it; a request of another
session, or the successor's own request near its end, is not.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.driver import core
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import cli as substrate_cli
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import sessions, watchdog
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
    ack_answers,
    ack_follows,
    write_json_atomic,
)


def _task(tmp_path):
    return {
        "trtllm_repo_path": str(tmp_path / "repo"),
        "environment": {
            "kind": "session_spool",
            "python": "/opt/venv/bin/python",
            "spool": str(tmp_path / "spool-a"),
            "sessions": [
                {"name": "a", "spool": str(tmp_path / "spool-a"), "role": "gates"},
                {"name": "b", "spool": str(tmp_path / "spool-b"), "role": "work"},
            ],
        },
    }


def _spool(task, name):
    return Path(sessions.session_by_name(task, name).spool)


def _sched(spool, *, job, session="a", ends_in=7200.0, watcher_id=None) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["SLURM_JOB_ID"] = job
    config = SchedulerConfig(
        devices=("0",),
        session=session,
        ends_at=time.time() + ends_in,
        handover_lead_s=3600.0,
        poll_s=0.02,
        leftover_grace_s=0,
    )
    return Scheduler(spool, config, environ=env, watcher_id=watcher_id or f"w{job}")


def _near_its_end(sched: Scheduler) -> None:
    """The allocation reaches its hand-over lead: the scheduler writes its request."""
    sched.config.ends_at = time.time() + 600
    sched.tick()


def _ack(spool):
    """The one ack on ``spool``: ``handover/acks/<allocation it follows>.json``."""
    (path,) = (Path(spool) / "handover" / "acks").glob("*.json")
    return json.loads(path.read_text())


def test_an_ack_written_before_the_request_answers_it(tmp_path):
    task = _task(tmp_path)
    spool = _spool(task, "a")
    sched = _sched(spool, job="1001")
    sched.tick()
    assert sessions.pending_requests(task) == []
    sessions.acknowledge(spool, job_id="2002", by="driver", session="a")
    assert _ack(spool)["succeeds"] == "1001" and _ack(spool)["request_id"] is None
    _near_its_end(sched)
    assert (spool / "handover" / "request.json").is_file()
    assert sessions.pending_requests(task) == []


def test_the_successors_own_hand_over_is_not_answered_by_it(tmp_path):
    task = _task(tmp_path)
    spool = _spool(task, "a")
    old = _sched(spool, job="1001")
    old.tick()
    sessions.acknowledge(spool, job_id="2002", by="driver", session="a")
    _near_its_end(old)
    assert sessions.pending_requests(task) == []
    # The successor takes over the spool and, near its own end, asks for the next one.
    (SpoolPaths(spool).watchers / "w1001.json").unlink()
    _near_its_end(_sched(spool, job="2002"))
    (pending,) = sessions.pending_requests(task)
    assert pending["slurm_job"] == "2002" and pending["session"] == "a"


def test_an_early_ack_answers_no_other_sessions_request(tmp_path):
    task = _task(tmp_path)
    _sched(_spool(task, "a"), job="1001").tick()
    sessions.acknowledge(_spool(task, "a"), job_id="2002", by="driver", session="a")
    _near_its_end(_sched(_spool(task, "b"), job="1001", session="b"))
    (pending,) = sessions.pending_requests(task)
    assert pending["session"] == "b"


def test_an_ack_naming_another_session_or_allocation_answers_nothing():
    request = {"request_id": "a-1001-5", "session": "a", "slurm_job": "1001"}
    early = {"request_id": None, "job_id": "2002", "succeeds": "1001", "session": "a"}
    assert ack_answers(early, request)
    assert not ack_answers({**early, "session": "b"}, request)
    assert not ack_answers({**early, "succeeds": "999"}, request)
    # The ack the successor's own job was submitted under does not follow it.
    assert not ack_follows({**early, "succeeds": "2002"}, "2002", "a")
    # Outside Slurm every allocation is `unknown`: only the request's id can answer it.
    assert not ack_answers({**early, "succeeds": "unknown"}, {**request, "slurm_job": "unknown"})
    assert ack_answers({"request_id": "a-1001-5", "job_id": "2002"}, request)


def test_an_earlier_hand_overs_request_on_disk_is_not_the_one_acked(tmp_path):
    task = _task(tmp_path)
    spool = _spool(task, "a")
    handover = spool / "handover"
    handover.mkdir(parents=True)
    write_json_atomic(
        handover / "request.json", {"request_id": "a-999-1", "session": "a", "slurm_job": "999"}
    )
    write_json_atomic(handover / "ack.json", {"request_id": "a-999-1", "job_id": "1001"})
    sched = _sched(spool, job="1001")
    sched.tick()
    sessions.acknowledge(spool, job_id="2002", by="driver", session="a")
    assert _ack(spool)["request_id"] is None and _ack(spool)["succeeds"] == "1001"
    _near_its_end(sched)
    assert sessions.pending_requests(task) == []


def test_an_ack_after_the_request_names_it_as_before(tmp_path):
    task = _task(tmp_path)
    spool = _spool(task, "a")
    _near_its_end(_sched(spool, job="1001"))
    (pending,) = sessions.pending_requests(task)
    sessions.acknowledge(spool, job_id="2002", by="driver")
    ack = _ack(spool)
    assert ack["request_id"] == pending["request_id"]
    assert ack["succeeds"] == "1001" and ack["session"] == "a"
    assert sessions.pending_requests(task) == []


def test_the_cli_records_the_session_it_acks(tmp_path):
    task = _task(tmp_path)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task), encoding="utf-8")
    _sched(_spool(task, "b"), job="1001", session="b").tick()
    argv = ["ack", "--task", str(path), "--name", "b", "--job", "2002"]
    assert substrate_cli.main(argv) == 0
    ack = _ack(_spool(task, "b"))
    assert ack["session"] == "b" and ack["succeeds"] == "1001" and ack["job_id"] == "2002"


# ------------------------------------------------------------------ the watchdog


def _watcher(spool, **fields):
    paths = SpoolPaths(spool)
    paths.ensure()
    (paths.root / "handover").mkdir()
    record = {"watcher": "w", "slurm_job": "9", "session": "a", "alive_epoch": time.time()}
    write_json_atomic(paths.watchers / "w.json", {**record, **fields})


def test_the_watchdog_counts_an_ack_written_before_the_request(tmp_path):
    _watcher(tmp_path / "a", ends_at_epoch=time.time() + 600)
    assert [a.kind for a in watchdog.check([tmp_path / "a"])] == ["ENDING"]
    write_json_atomic(
        tmp_path / "a" / "handover" / "ack.json",
        {"request_id": None, "job_id": "12", "succeeds": "9", "session": "a"},
    )
    assert watchdog.check([tmp_path / "a"]) == []


def test_the_watchdog_ignores_an_earlier_hand_overs_request_and_ack(tmp_path):
    _watcher(tmp_path / "a", ends_at_epoch=time.time() + 600)
    write_json_atomic(
        tmp_path / "a" / "handover" / "request.json", {"request_id": "r8", "slurm_job": "8"}
    )
    write_json_atomic(
        tmp_path / "a" / "handover" / "ack.json",
        {"request_id": "r8", "job_id": "9", "succeeds": "8", "session": "a"},
    )
    assert [a.kind for a in watchdog.check([tmp_path / "a"])] == ["ENDING"]


# -------------------------------------------------------------------- the driver


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "repo").mkdir()
    (ws / "task.yaml").write_text(yaml.safe_dump(_task(tmp_path)), encoding="utf-8")
    return ws


def test_the_driver_does_not_ask_again_for_a_successor_acked_early(workspace, tmp_path):
    task = _task(tmp_path)
    spool = _spool(task, "a")
    sched = _sched(spool, job="1001")
    sched.tick()
    sessions.acknowledge(spool, job_id="2002", by="driver", session="a")
    _near_its_end(sched)
    cfg = core.load_config(workspace)
    assert core.poll_once(cfg) is None
    # The successor's own request, near its end, is raised as usual.
    (SpoolPaths(spool).watchers / "w1001.json").unlink()
    _near_its_end(_sched(spool, job="2002"))
    event = core.poll_once(cfg)
    assert event is not None and event.kind == "SESSION_REQUEST" and "2002" in event.detail
