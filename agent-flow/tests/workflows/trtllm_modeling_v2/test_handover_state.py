# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-over state keyed by the allocation it is about, so none of it goes stale.

The first long run of this flow left a predecessor's ``request.json`` in place
after its successor came up; the next early ``session ack`` replaced the one
``ack.json`` that answered it, and the driver asked for a successor to an
allocation that had ended a day before. ``session allocate`` built its
dependency from that request and would have started a duplicate at once.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.driver import core
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import cli as substrate_cli
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import sessions, watchdog
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
    stale_request,
    write_json_atomic,
)

HOUR = 3600.0


def _task(tmp_path):
    return {
        "trtllm_repo_path": str(tmp_path / "repo"),
        "environment": {
            "kind": "session_spool",
            "python": "/opt/venv/bin/python",
            "spool": str(tmp_path / "gates-3"),
            "container_image": "/images/trtllm.sqsh",
            "sessions": [
                {
                    "name": "g3",
                    "spool": str(tmp_path / "gates-3"),
                    "role": "gates",
                    "time_limit": "24:00:00",
                },
            ],
        },
    }


def _spool(tmp_path) -> Path:
    return tmp_path / "gates-3"


def _sched(spool, *, job, watcher=None, ends_in=20 * HOUR) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["SLURM_JOB_ID"] = job
    config = SchedulerConfig(
        devices=("0",),
        session="g3",
        ends_at=time.time() + ends_in,
        handover_lead_s=HOUR,
        poll_s=0.02,
        leftover_grace_s=0,
    )
    return Scheduler(spool, config, environ=env, watcher_id=watcher or f"w{job}")


def _requesting(spool, *, job, watcher=None) -> Scheduler:
    """A watcher of ``job`` that has reached its hand-over lead and written its request."""
    sched = _sched(spool, job=job, watcher=watcher, ends_in=600)
    sched.tick()
    assert SpoolPaths(spool).handover_request.is_file()
    return sched


def _ended(sched: Scheduler) -> None:
    """The allocation is gone: its watcher stops heart-beating (SIGTERM leaves its record)."""
    path = sched.paths.watchers / f"{sched.watcher_id}.json"
    record = json.loads(path.read_text())
    write_json_atomic(path, {**record, "alive_epoch": time.time() - 2 * HOUR})


def _archived(spool) -> dict[str, dict]:
    """``handover/archive/``, by the name each file had before its time stamp."""
    return {
        p.name.split(".", 1)[0]: json.loads(p.read_text())
        for p in sorted((Path(spool) / "handover" / "archive").glob("*.json"))
    }


def _acks(spool) -> dict[str, dict]:
    return {
        p.stem: json.loads(p.read_text())
        for p in sorted((Path(spool) / "handover" / "acks").glob("*.json"))
    }


# ------------------------------------------------------------------- the ack


def test_a_later_ack_for_the_next_allocation_leaves_the_answering_one_in_place(tmp_path):
    task, spool = _task(tmp_path), _spool(tmp_path)
    _requesting(spool, job="1000027")
    sessions.acknowledge(spool, job_id="1000247", by="driver", session="g3")
    assert sessions.pending_requests(task) == []
    # The next allocation's successor, submitted early: before, this replaced the ack above.
    sessions.acknowledge(spool, job_id="1000421", by="driver", session="g3", after="1000247")
    acks = _acks(spool)
    assert acks["1000027"]["job_id"] == "1000247" and acks["1000247"]["job_id"] == "1000421"
    assert sessions.pending_requests(task) == []


def test_an_ack_of_another_job_for_the_same_allocation_is_archived_not_lost(tmp_path):
    spool = _spool(tmp_path)
    _requesting(spool, job="1000027")
    sessions.acknowledge(spool, job_id="1000247", by="driver")
    sessions.acknowledge(spool, job_id="1000300", by="driver")
    assert _acks(spool)["1000027"]["job_id"] == "1000300"
    (archived,) = _archived(spool).values()
    assert (
        archived["job_id"] == "1000247"
        and "replaced by the ack of job 1000300" in (archived["archived"]["why"])
    )


def test_the_cli_ack_records_the_allocation_it_was_told_to_follow(tmp_path):
    task, spool = _task(tmp_path), _spool(tmp_path)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task), encoding="utf-8")
    _sched(spool, job="1000247").tick()
    argv = ["ack", "--task", str(path), "--name", "g3", "--job", "1000421", "--after", "1000247"]
    assert substrate_cli.main(argv) == 0
    assert _acks(spool)["1000247"]["succeeds"] == "1000247"


def test_the_watchdog_counts_a_keyed_ack_written_before_the_request(tmp_path):
    spool = _spool(tmp_path)
    _sched(spool, job="1000247", ends_in=1200).tick()
    sessions.acknowledge(spool, job_id="1000421", by="driver", after="1000247")
    sessions.acknowledge(spool, job_id="1099999", by="driver", after="1000421")
    assert watchdog.check([spool]) == []


# ------------------------------------------------- the successor sets it aside


def test_the_successor_sets_aside_its_predecessors_request_and_acks_when_it_starts(tmp_path):
    task, spool = _task(tmp_path), _spool(tmp_path)
    old = _requesting(spool, job="1000027")
    request_id = json.loads(SpoolPaths(spool).handover_request.read_text())["request_id"]
    sessions.acknowledge(spool, job_id="1000247", by="driver", session="g3")
    _ended(old)
    successor = _sched(spool, job="1000247")
    # An ack for the successor's own successor, written before it came up, is its own.
    sessions.acknowledge(spool, job_id="1000421", by="driver", session="g3", after="1000247")
    successor.tick()

    assert not SpoolPaths(spool).handover_request.exists()
    assert list(_acks(spool)) == ["1000247"]
    archived = _archived(spool)
    assert archived["request-1000027"]["request_id"] == request_id
    assert (
        "successor w1000247 (allocation 1000247) started"
        in (archived["request-1000027"]["archived"]["why"])
    )
    assert archived["ack-1000027"]["job_id"] == "1000247"
    assert "set aside the hand-over it succeeds" in SpoolPaths(spool).status.read_text()
    assert sessions.pending_requests(task) == []


def test_a_requeued_allocation_keeps_its_ack_and_drops_its_earlier_request(tmp_path):
    """Slurm requeues an allocation under the same job id with a fresh time limit."""
    task, spool = _task(tmp_path), _spool(tmp_path)
    first = _requesting(spool, job="1000024", watcher="first-start")
    sessions.acknowledge(spool, job_id="1000400", by="driver", session="g3")
    _ended(first)
    again = _sched(spool, job="1000024", watcher="second-start")
    again.tick()
    assert not SpoolPaths(spool).handover_request.exists()
    assert list(_acks(spool)) == ["1000024"], "the queued successor still follows 1000024"
    # Near its own end it asks again, and the ack it kept answers that request.
    again.config.ends_at = time.time() + 600
    again.tick()
    assert json.loads(SpoolPaths(spool).handover_request.read_text())["watcher"] == "second-start"
    assert sessions.pending_requests(task) == []


def test_a_watcher_does_not_set_aside_the_request_of_an_allocation_that_outlasts_it(tmp_path):
    spool = _spool(tmp_path)
    _requesting(spool, job="2000")
    _sched(spool, job="2001", ends_in=60).tick()
    assert json.loads(SpoolPaths(spool).handover_request.read_text())["slurm_job"] == "2000"


def test_no_request_is_written_once_a_successor_is_up(tmp_path):
    spool = _spool(tmp_path)
    _sched(spool, job="1000247").tick()
    old = _sched(spool, job="1000027", ends_in=600)
    old.tick()
    assert old.handover == {"successor": "w1000247"}
    assert not SpoolPaths(spool).handover_request.exists()


def test_a_request_written_before_the_successor_came_up_is_set_aside_by_its_writer(tmp_path):
    spool = _spool(tmp_path)
    old = _requesting(spool, job="1000027")
    write_json_atomic(
        SpoolPaths(spool).watchers / "late.json",
        {"watcher": "late", "slurm_job": "1000247", "alive_epoch": time.time(),
         "ends_at_epoch": time.time() + 20 * HOUR},
    )  # fmt: skip
    old.tick()
    assert old.handover["successor"] == "late"
    assert not SpoolPaths(spool).handover_request.exists()
    assert "its successor late is up" in _archived(spool)["request-1000027"]["archived"]["why"]


def test_a_watcher_writes_its_request_again_when_it_left_the_disk(tmp_path):
    spool = _spool(tmp_path)
    sched = _requesting(spool, job="1000027")
    first = json.loads(SpoolPaths(spool).handover_request.read_text())
    SpoolPaths(spool).handover_request.unlink()
    sched.tick()
    assert json.loads(SpoolPaths(spool).handover_request.read_text()) == first
    assert "written again" in SpoolPaths(spool).status.read_text()


# ---------------------------------------------- the driver ignores a stale one


def _stale_request_on_disk(spool, *, job="1000027", writer="w1000027") -> dict:
    """A request its watcher left when the allocation ended (that watcher's record is old)."""
    request = {
        "request_id": f"g3-{job}-1792258324",
        "session": "g3",
        "slurm_job": job,
        "watcher": writer,
        "ends_at_epoch": time.time() - 20 * HOUR,
    }
    paths = SpoolPaths(spool)
    paths.ensure()
    paths.handover.mkdir(exist_ok=True)
    write_json_atomic(paths.handover_request, request)
    write_json_atomic(
        paths.watchers / f"{writer}.json",
        {"watcher": writer, "slurm_job": job, "alive_epoch": time.time() - 20 * HOUR},
    )
    return request


def test_a_request_stands_only_while_its_writer_heart_beats():
    live = [{"watcher": "b", "slurm_job": "1000247", "ends_at_epoch": 2e9}]
    request = {"slurm_job": "1000027", "watcher": "a", "ends_at_epoch": 1e9}
    assert "allocation 1000027 has no live watcher: it ended" in stale_request(request, live)
    requeued = {**request, "slurm_job": "1000247"}
    assert "runs on under a new watcher (requeued or restarted)" in stale_request(requeued, live)
    mine = {"slurm_job": "1000247", "watcher": "b", "ends_at_epoch": 2e9}
    assert stale_request(mine, live) == ""
    successor = [*live, {"watcher": "c", "slurm_job": "1000421", "ends_at_epoch": 3e9}]
    assert "its successor c (allocation 1000421) is up" in stale_request(mine, successor)


def test_wait_archives_a_request_no_live_watcher_stands_behind_and_says_so(tmp_path):
    task, spool = _task(tmp_path), _spool(tmp_path)
    _sched(spool, job="1000247").tick()
    request = _stale_request_on_disk(spool)
    pending, ignored = sessions.review_requests(task)
    assert pending == [] and "ignored request" in ignored[0]
    assert SpoolPaths(spool).handover_request.exists(), "a listing moves nothing"

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (tmp_path / "repo").mkdir()
    (workspace / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    event = core.wait(core.load_config(workspace), timeout_s=0.05, poll_s=0.01)
    assert event.kind == "TIMEOUT"
    (note,) = event.notes
    assert note.startswith(f"session g3: archived request {request['request_id']}")
    assert "allocation 1000027 has no live watcher: it ended" in note
    assert event.render().splitlines()[1] == f"  note: {note}"
    assert not SpoolPaths(spool).handover_request.exists()
    assert _archived(spool)["request-1000027"]["archived"]["by"].startswith("driver ")
    assert core.wait(core.load_config(workspace), timeout_s=0.05, poll_s=0.01).notes == ()


def test_the_cli_lists_an_ignored_request(tmp_path, capsys):
    task, spool = _task(tmp_path), _spool(tmp_path)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task), encoding="utf-8")
    _sched(spool, job="1000247").tick()
    _stale_request_on_disk(spool)
    assert substrate_cli.main(["requests", "--task", str(path)]) == 0
    out = capsys.readouterr().out
    assert "(no successor requests)" in out and "ignored request g3-1000027" in out


# ---------------------------------------------------------- session allocate


def test_allocate_follows_the_live_watcher_not_the_request_on_disk(tmp_path):
    task, spool = _task(tmp_path), _spool(tmp_path)
    _sched(spool, job="1000247").tick()
    _stale_request_on_disk(spool)
    plan = sessions.plan_successor(task, "g3")
    assert plan.after == "1000247"
    assert 'dependency="afterany:1000247"' in plan.call and "1000027" not in plan.call
    assert "It starts when allocation 1000247 ends" in plan.reason
    assert any("names allocation 1000027" in note for note in plan.notes)
    assert sessions.allocate(task, "g3") == plan.call


def test_allocate_with_no_live_watcher_has_no_dependency_and_says_why(tmp_path):
    task, spool = _task(tmp_path), _spool(tmp_path)
    _stale_request_on_disk(spool)
    plan = sessions.plan_successor(task, "g3")
    assert plan.after is None and "dependency" not in plan.call
    assert plan.reason.startswith(f"No dependency: no watcher on {spool} has heart-beat")


def test_allocate_follows_the_allocation_that_ends_last(tmp_path):
    """Two allocations overlap on one spool: the next successor follows the later one."""
    task, spool = _task(tmp_path), _spool(tmp_path)
    _sched(spool, job="1000247").tick()
    _sched(spool, job="1000027", ends_in=600).tick()
    assert sessions.plan_successor(task, "g3").after == "1000247"


def test_allocate_says_when_a_successor_is_already_acknowledged(tmp_path):
    task, spool = _task(tmp_path), _spool(tmp_path)
    _sched(spool, job="1000247").tick()
    sessions.acknowledge(spool, job_id="1000421", by="driver", session="g3")
    (note,) = sessions.plan_successor(task, "g3").notes
    assert "Allocation 1000247 already has a successor acknowledged: job 1000421" in note


def test_the_cli_allocate_prints_the_call_why_and_the_ack_to_run(tmp_path, capsys):
    task, spool = _task(tmp_path), _spool(tmp_path)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task), encoding="utf-8")
    _sched(spool, job="1000247").tick()
    assert substrate_cli.main(["allocate", "--task", str(path), "--name", "g3"]) == 0
    out = capsys.readouterr().out
    assert f"session ack --task {path} --name g3 --job <id> --after 1000247" in out
    assert 'dependency="afterany:1000247"' in out
    assert "It starts when allocation 1000247 ends" in out
    assert (spool / "session_entry.sh").is_file()


def test_outside_slurm_an_ack_answers_the_request_by_its_id(tmp_path):
    task, spool = _task(tmp_path), _spool(tmp_path)
    _requesting(spool, job="unknown", watcher="local")
    (pending,) = sessions.pending_requests(task)
    path = sessions.acknowledge(spool, job_id="777", by="driver", session="g3")
    assert path.name == "job-777.json"
    assert json.loads(path.read_text())["request_id"] == pending["request_id"]
    assert sessions.pending_requests(task) == []
