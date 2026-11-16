# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The controller's own deadline: inside a job it stops in time and says how to continue.

The flow submits no job for its controller. One that finds itself inside a job stops starting
turns before that job's end, says the command that continues the run, and exits "to be
continued" (75): what a site's script needs to submit the same command again.
"""

from __future__ import annotations

import threading
import time

import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import controller
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import slurm as slurm_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.files import write_json_atomic
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import status
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor.heartbeat import ORCHESTRATOR_EXIT
from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator as orchestrator_mod

from ._helpers import Script, _install, _state, _workflow


def test_what_continues_a_run_resumes_it_and_carries_nothing_of_this_job(tmp_path):
    # The same command with `--clean` would start the run over.
    workspace = tmp_path / "ws"
    argv = ["--task", "task.yaml", "--workspace", str(workspace), "--clean"]
    said: list[str] = []
    environ = {"SLURM_JOB_ID": "5", "SLURM_JOB_END_TIME": str(int(time.time()) + 0)}
    limit = controller.watch_limit(
        workspace, argv, stop=lambda r: True, leave=lambda c: None,
        environ=environ, say=said.append,
    )  # fmt: skip
    # Said at the start, to whoever reads the job's output: when it stops, and with what status.
    assert said[0].startswith("→ this controller runs inside job 5, which ends at ")
    assert said[0].endswith(
        "it stops starting turns 10 minutes before that, says the command that continues, "
        "and exits 75"
    )
    deadline = time.time() + 5
    while not limit.said and time.time() < deadline:
        time.sleep(0.02)
    limit.finish()
    assert limit.said.startswith("this controller runs inside job 5, which ends at ")
    assert ": stopped for the limit; continue with: " in limit.said
    assert limit.said.endswith(f"--task task.yaml --workspace {workspace}")
    assert "--clean" not in limit.said and said[-1] == f"→ {limit.said}"
    # The line is left where the monitor reads it.
    assert controller.last_handover(workspace)["line"] == limit.said
    # A caller that is not the flow's own CLI names the command that continues it.
    limit = controller.watch_limit(
        workspace, [], stop=lambda r: True, leave=lambda c: None,
        environ=environ, say=said.append, command=["driver", "resume"],
    )  # fmt: skip
    deadline = time.time() + 5
    while not limit.said and time.time() < deadline:
        time.sleep(0.02)
    assert limit.finish() and limit.said.endswith("continue with: driver resume")
    # Inside a job, that job's own variables are not exported into what the flow submits.
    inside = {
        "SLURM_JOB_ID": "5", "SLURM_MEM_PER_NODE": "32768", "SLURM_JOB_END_TIME": "1",
        "SLURM_NTASKS": "1", "SLURM_CONF": "/etc/slurm.conf", "SBATCH_TIMELIMIT": "1", "HOME": "/h",
    }  # fmt: skip
    assert slurm_mod.command_environment(inside) == {
        "SLURM_CONF": "/etc/slurm.conf",
        "HOME": "/h",
        "SLURM_TIME_FORMAT": "%s",
    }
    outside = {"SLURM_NTASKS": "4", "HOME": "/h"}
    assert slurm_mod.command_environment(outside)["SLURM_NTASKS"] == "4"


def test_the_limit_hands_over_stops_the_run_and_leaves_in_time():
    events: list[str] = []
    answers = iter([False, True, True, True, True, True, True, True, True, True])
    left = threading.Event()

    def stop(reason: str) -> bool:
        events.append("stop")
        assert reason.endswith("stopped for the limit")
        return next(answers)

    def leave(code: int) -> None:
        events.append(f"leave {code}")
        left.set()

    limit = controller.Limit(
        end=time.time() + 0.8, stop=stop, hand_over=lambda: events.append("over") or "line",
        leave=leave, before_s=0.6, exit_before_s=0.2, poll_s=0.05,
    ).start()  # fmt: skip
    assert left.wait(5), "a run that does not return is left before the job is killed"
    # Nothing more starts once the limit is near.
    assert events[0] == "stop" and events[1] == "over" and events[-1] == "leave 75"
    assert events.count("stop") >= 2, "a run that had not opened yet is asked again"
    assert limit.reached and limit.said == "line"
    # A run that returns once it is stopped: no forced exit, and the process does not go
    # before the hand-over under way has said what continues the run.
    events.clear()

    def slow_hand_over() -> str:
        time.sleep(0.5)
        events.append("over")
        return "line"

    limit = controller.Limit(
        end=time.time() + 5.3, stop=lambda reason: True, hand_over=slow_hand_over,
        leave=leave, before_s=5.0, exit_before_s=0.2, poll_s=0.05,
    ).start()  # fmt: skip
    deadline = time.time() + 5
    while not limit.reached and time.time() < deadline:
        time.sleep(0.02)
    assert limit.finish() is True and events == ["over"] and limit.said == "line"
    # A run that ends long before its job does: nothing is handed over.
    events.clear()
    limit = controller.Limit(
        end=time.time() + 3600, stop=stop, hand_over=lambda: events.append("over") or "",
        leave=leave, poll_s=0.05,
    ).start()  # fmt: skip
    assert limit.finish() is False and events == []


def test_a_run_that_can_start_nothing_more_before_its_jobs_end_is_continued_at_once(tmp_path):
    """The same stop, line and status as at the limit, hours before it."""
    stops: list[str] = []
    said: list[str] = []
    left: list[int] = []
    assert controller.continue_now(tmp_path, "nothing fits") is False, "no job's end is watched"
    limit = controller.watch_limit(
        tmp_path,
        ["--task", "t.yaml", "--workspace", str(tmp_path)],
        stop=lambda reason: stops.append(reason) or True,
        leave=left.append,
        environ={"SLURM_JOB_ID": "7", "SLURM_JOB_END_TIME": str(int(time.time()) + 7200)},
        say=said.append,
    )
    try:
        time.sleep(0.2)
        assert not limit.reached and stops == []
        assert controller.continue_now(tmp_path, "job test-final does not fit what is left")
        deadline = time.time() + 10
        while not limit.said and time.time() < deadline:
            time.sleep(0.05)
        assert stops == ["job test-final does not fit what is left: stopped for the limit"]
        assert (
            "job test-final does not fit what is left; stopped for the limit; continue with: "
            in (limit.said)
        )
        assert controller.last_handover(tmp_path)["line"] == limit.said
    finally:
        assert limit.finish() is True and left == [], "the command line then exits 75 itself"


def test_a_controller_knows_its_jobs_end_and_the_run_takes_a_stop_from_outside(tmp_path):
    import subprocess

    assert controller.job_end({}) is None
    assert controller.job_end({"SLURM_JOB_END_TIME": "1800000000"}) == 1800000000.0

    def queue(argv, **kwargs):
        assert argv[:4] == ["squeue", "-h", "-j", "5"]
        return subprocess.CompletedProcess(argv, 0, "1800000123\n", "")

    assert controller.job_end({"SLURM_JOB_ID": "5"}, runner=queue) == 1800000123.0
    assert controller.job_end({"SLURM_JOB_ID": "5"}, runner=lambda *a, **k: 1 / 0) is None
    # Outside a job there is no limit to watch.
    assert (
        controller.watch_limit(tmp_path, [], stop=lambda r: True, leave=lambda c: None, environ={})
        is None
    )
    # The stop the limit asks for is the run's own: turns in flight are left, tasks keep their attempt.
    assert orchestrator_mod.stop_run(tmp_path, "too early") is False
    with orchestrator_mod.stoppable(tmp_path):
        assert orchestrator_mod.stop_run(tmp_path, "stopped for the limit") is True
        assert orchestrator_mod._STOPS[tmp_path.resolve()].result() == "stopped for the limit"
    assert orchestrator_mod.stop_run(tmp_path, "too late") is False


def test_a_turn_cut_for_the_limit_runs_again_as_the_attempt_it_was(
    tmp_path, task_file, monkeypatch
):
    workspace = (tmp_path / "ws").resolve()
    reason = f"the controller's job ends at T: {controller.LIMIT_STOP}"
    cut: list[int] = []

    def assembler(turn, prompt):
        if not cut:
            # The limit arrives while this turn runs: the run stops, and the turn is cut
            # before it answered.
            cut.append(turn)
            assert orchestrator_mod.stop_run(workspace, reason)
            return None
        return {"decision": "ready", "summary": "assembled"}

    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None, assembler=assembler))
    with workflow:
        workflow.run(task_file)
    state = _state(tmp_path)
    assert cut and not state.done and state.verdict == ""
    journal = (workspace / "journal.md").read_text()
    assert f"stopped: {reason}" in journal and "run closed" not in journal
    # What continues the run starts the cut turn again: no attempt was spent on the cut.
    second = _workflow(tmp_path)
    again = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert _state(tmp_path).verdict == "delivered"
    assert "assembler" in [role for role, _ in again]
    graph = yaml.safe_load((workspace / "graph.yaml").read_text())
    assert {int(task.get("attempts", 1) or 1) for task in graph["tasks"]} == {1}


def test_the_monitor_says_what_continues_a_run_that_stopped_for_its_limit(tmp_path):
    (tmp_path / "task.yaml").write_text("{}\n")
    dispatch = tmp_path / "dispatch"
    dispatch.mkdir()
    (tmp_path / controller.DIRNAME).mkdir()
    now = time.time()
    line = "this controller runs inside job 9, which ends at T: stopped for the limit; continue with: C"
    write_json_atomic(
        tmp_path / controller.DIRNAME / controller.HANDOVER, {"at": now - 30, "line": line}
    )
    write_json_atomic(dispatch / ORCHESTRATOR_EXIT, {"pid": 1, "host": "h", "at": now})
    event = status.poll_once(status.load_config(tmp_path), now=now + 5)
    assert event.kind == "ORCHESTRATOR_EXITED" and event.detail.endswith(line)
    # An exit long after a hand-over is another exit.
    write_json_atomic(dispatch / ORCHESTRATOR_EXIT, {"pid": 1, "host": "h", "at": now + 7200})
    assert line not in status.poll_once(status.load_config(tmp_path), now=now + 7300).detail
