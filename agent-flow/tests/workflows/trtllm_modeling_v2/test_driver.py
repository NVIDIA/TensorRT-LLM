# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The external transport's driver, packaged with no run's paths written into it.

These tests pin both sides: the orchestrator side (per-role timeouts,
heartbeat, job prefixes, adoption across a restart) and the driver side
(wait, claim, prompt, status, instructions).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import socket
import threading
import time
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli as cli_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.driver import core
from agent_flow.workflows.trtllm_modeling_v2.infra.driver import reload as reload_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import TurnRequest, TurnTimeout
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import SPOOLCTL_PATH
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import write_json_atomic
from agent_flow.workflows.trtllm_modeling_v2.infra.transport import (
    DEFAULT_ROLE_TURN_TIMEOUTS,
    ORCHESTRATOR_EXIT,
    ORCHESTRATOR_HEARTBEAT,
    ExternalAgentTransport,
)


def _turn(transport, **request):
    """One turn through the transport; its text."""
    return transport.run(TurnRequest(**request)).text


def _answer(transport, request_id, result):
    """What a driver does: the result, then ``.done`` last."""
    (transport.root / f"{request_id}.result.md").write_text(result, encoding="utf-8")
    (transport.root / f"{request_id}.done").write_text("1\n", encoding="utf-8")


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    task = {
        "trtllm_repo_path": str(tmp_path / "repo"),
        "source_embargo": ["tensorrt_llm/_torch/models/modeling_secret.py"],
        "environment": {
            "kind": "session_spool",
            "python": "/venv/bin/python",
            "spool": str(tmp_path / "spool-a"),
            "env": {"TMPDIR": "/short/tmp"},
            "sessions": [
                {"name": "a", "spool": str(tmp_path / "spool-a"), "role": "gates"},
                {"name": "b", "spool": str(tmp_path / "spool-b"), "role": "work"},
            ],
        },
    }
    (tmp_path / "repo").mkdir()
    (ws / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    return ws


def _ask(
    transport, *, role="smith", what="onboard x", decision_file=None, backend=None, model=None
):
    """Send one turn from a thread; return the thread and a box for its outcome."""
    box: dict = {}

    def run():
        try:
            box["outcome"] = transport.run(
                TurnRequest(
                    role=role,
                    what=what,
                    prompt="do it",
                    system_prompt="you are a smith",
                    decision_file=decision_file,
                    backend=backend,
                    model=model,
                )
            )
        except TurnTimeout as exc:
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def _wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def _request_ids(ws) -> list[str]:
    return sorted(p.name[: -len(".request.json")] for p in (ws / "dispatch").glob("*.request.json"))


# ------------------------------------------------------- the orchestrator side


def test_each_role_has_its_own_turn_timeout(workspace):
    """A core assembly runs well past 3 h, so the assembler gets 16 h."""
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    assert transport.timeout_for("assembler") == 16 * 3600
    assert transport.timeout_for("smith") == DEFAULT_ROLE_TURN_TIMEOUTS["smith"]
    assert transport.timeout_for("plan_reviewer") < transport.timeout_for("assembler")


def test_an_explicit_timeout_still_binds_every_role(workspace):
    transport = ExternalAgentTransport(workspace, timeout_s=600, heartbeat_s=None)
    assert {transport.timeout_for(r) for r in DEFAULT_ROLE_TURN_TIMEOUTS} == {600}


def test_a_turn_times_out_on_its_roles_bound(workspace):
    transport = ExternalAgentTransport(
        workspace, poll_s=0.01, role_timeouts={"reviewer": 0.2}, heartbeat_s=None
    )
    with pytest.raises(TurnTimeout, match="reviewer turn timeout"):
        _turn(transport, role="reviewer", what="review", prompt="p")


def test_a_request_names_its_job_prefix_timeout_and_agent(workspace):
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, _ = _ask(transport, backend="codex", model="gpt-x")
    _wait_for(lambda: _request_ids(workspace))
    (request_id,) = _request_ids(workspace)
    data = json.loads((workspace / "dispatch" / f"{request_id}.request.json").read_text())
    assert data["job_prefix"] == f"r{request_id.split('-')[0]}"
    assert data["timeout_s"] == DEFAULT_ROLE_TURN_TIMEOUTS["smith"]
    assert data["agent"] == {"backend": "codex", "model": "gpt-x", "reasoning_effort": None}
    _answer(transport, request_id, "ok")
    thread.join(timeout=10)


def test_the_orchestrator_heart_beats_and_says_when_it_exits(workspace):
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=0.05)
    beat = workspace / "dispatch" / ORCHESTRATOR_HEARTBEAT
    _wait_for(beat.exists)
    assert json.loads(beat.read_text())["pid"] == os.getpid()
    transport._mark_exit({"pid": os.getpid(), "host": "h", "started": 0})
    assert (workspace / "dispatch" / ORCHESTRATOR_EXIT).exists()
    transport.close()


def test_the_heartbeat_survives_a_full_filesystem(workspace, monkeypatch):
    """A shared project quota can stay full for minutes.

    A beat thread that died on EDQUOT would make the driver read a live
    orchestrator as exited.
    """
    import errno

    from agent_flow.workflows.trtllm_modeling_v2.infra.transport import ExternalAgentTransport

    real, fails = ExternalAgentTransport._write_atomic, {"n": 2}

    def full(path, text):
        if path.name == ORCHESTRATOR_HEARTBEAT and fails["n"]:
            fails["n"] -= 1
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real(path, text)

    monkeypatch.setattr(ExternalAgentTransport, "_write_atomic", staticmethod(full))
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=0.02)
    beat = workspace / "dispatch" / ORCHESTRATOR_HEARTBEAT
    _wait_for(beat.exists)
    assert fails["n"] == 0
    transport.close()


def test_a_relaunch_adopts_an_orphaned_turn_whose_decision_path_moved(workspace):
    """A relaunch numbers the same turn differently; it waits on the unanswered request."""
    dispatch = workspace / "dispatch"
    dispatch.mkdir()
    old_decision = workspace / "decisions" / "assembler-1.yaml"
    new_decision = workspace / "decisions" / "assembler-2.yaml"
    write_json_atomic(
        dispatch / "046-assembler.request.json",
        {
            "id": "046-assembler",
            "role": "assembler",
            "what": "dspark",
            "decision_file": str(old_decision),
            "requested_at": time.time() - 600,
        },
    )
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, box = _ask(transport, role="assembler", what="dspark", decision_file=str(new_decision))
    time.sleep(0.3)
    assert _request_ids(workspace) == ["046-assembler"], "no second request for the same turn"
    old_decision.parent.mkdir(parents=True)
    old_decision.write_text("decision: UNIT_READY\n")
    _answer(transport, "046-assembler", "assembled")
    thread.join(timeout=10)
    assert box["outcome"].request_id == "046-assembler"
    assert new_decision.read_text() == "decision: UNIT_READY\n"


def test_a_turn_this_process_asked_is_never_adopted_across_decision_paths(workspace):
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    first, _ = _ask(transport, role="assembler", what="core", decision_file="/d/assembler-1.yaml")
    _wait_for(lambda: len(_request_ids(workspace)) == 1)
    second, _ = _ask(transport, role="assembler", what="core", decision_file="/d/assembler-2.yaml")
    _wait_for(lambda: len(_request_ids(workspace)) == 2)
    for request_id in _request_ids(workspace):
        _answer(transport, request_id, "ok")
    first.join(timeout=10)
    second.join(timeout=10)


def test_the_driver_runs_as_many_turns_as_the_orchestrator_runs_tasks():
    """Capped, because many more concurrent turns on one account stall the API stream."""
    from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import DEFAULT_MAX_PARALLEL_TASKS

    assert core.DEFAULT_MAX_INFLIGHT == DEFAULT_MAX_PARALLEL_TASKS == 8


# ---------------------------------------------------------------- the driver


def _orchestrate(workspace, *turns, **kwargs):
    transport = ExternalAgentTransport(
        workspace, poll_s=0.01, heartbeat_s=kwargs.pop("heartbeat_s", None)
    )
    threads = [_ask(transport, **turn)[0] for turn in turns]
    _wait_for(lambda: len(_request_ids(workspace)) == len(turns))
    return transport, threads


def test_claims_are_exclusive(workspace):
    transport, threads = _orchestrate(workspace, {"what": "a"})
    cfg = core.load_config(workspace)
    (request_id,) = _request_ids(workspace)
    core.claim(cfg, request_id, by="driver-1")
    with pytest.raises(core.DriverError, match="already claimed by driver-1"):
        core.claim(cfg, request_id, by="driver-2")
    assert core.pending(cfg) == [] and len(core.in_flight(cfg)) == 1
    core.release(cfg, request_id)
    assert [r["id"] for r in core.pending(cfg)] == [request_id]
    _answer(transport, request_id, "ok")
    with pytest.raises(core.DriverError, match="already answered"):
        core.claim(cfg, request_id)
    for thread in threads:
        thread.join(timeout=10)


def test_wait_returns_a_new_turn(workspace):
    transport, threads = _orchestrate(workspace, {"what": "onboard rmsnorm"})
    event = core.wait(core.load_config(workspace), timeout_s=5, poll_s=0.01, watch_sessions=False)
    assert event.kind == "NEW" and event.request_id == _request_ids(workspace)[0]
    assert "smith: onboard rmsnorm" in event.detail
    _answer(transport, event.request_id, "ok")
    threads[0].join(timeout=10)


def test_wait_respects_the_in_flight_cap(workspace):
    transport, threads = _orchestrate(workspace, {"what": "a"}, {"what": "b"})
    cfg = core.load_config(workspace, max_inflight=1)
    core.claim(cfg, _request_ids(workspace)[0])
    event = core.wait(cfg, timeout_s=0.2, poll_s=0.01, watch_sessions=False)
    assert event.kind == "TIMEOUT" and "in-flight cap, 1/1" in event.detail
    for request_id in _request_ids(workspace):
        _answer(transport, request_id, "ok")
    for thread in threads:
        thread.join(timeout=10)


def _request_of(workspace, role, what):
    for request_id in _request_ids(workspace):
        data = json.loads((workspace / "dispatch" / f"{request_id}.request.json").read_text())
        if data["role"] == role and data["what"] == what:
            return request_id
    raise AssertionError(f"no {role} request for {what}")


def test_the_planners_and_the_critics_turns_never_wait_behind_workers(workspace):
    """A planner turn is handed out even with every worker slot taken."""
    transport, threads = _orchestrate(
        workspace,
        {"what": "a"},
        {"what": "b"},
        {"role": "planner", "what": "planner"},
        {"role": "plan_reviewer", "what": "critic"},
        {"role": "plan_reviewer", "what": "plan-review"},
    )
    cfg = core.load_config(workspace, max_inflight=1)
    core.claim(cfg, _request_of(workspace, "smith", "a"))
    first = core.wait(cfg, timeout_s=5, poll_s=0.01, watch_sessions=False)
    assert first.kind == "NEW" and first.request_id == _request_of(workspace, "planner", "planner")
    core.claim(cfg, first.request_id)
    second = core.wait(cfg, timeout_s=5, poll_s=0.01, watch_sessions=False)
    assert second.request_id == _request_of(workspace, "plan_reviewer", "critic")
    core.claim(cfg, second.request_id)
    # A paper review of the survey is a worker's turn: it waits at the cap with b.
    third = core.wait(cfg, timeout_s=0.2, poll_s=0.01, watch_sessions=False)
    assert third.kind == "TIMEOUT" and "in-flight cap, 1/1" in third.detail
    data = core.status(cfg)
    assert (data["in_flight"], data["planner_in_flight"]) == (1, 2)
    assert "in flight: 1/1, and 2 planner or critic turn(s) beside the cap" in core.format_status(
        data
    )
    for request_id in _request_ids(workspace):
        _answer(transport, request_id, "ok")
    for thread in threads:
        thread.join(timeout=10)


def test_a_planner_turn_goes_ahead_of_older_worker_turns(workspace):
    transport, threads = _orchestrate(workspace, {"what": "a"})
    planner_thread, _ = _ask(transport, role="planner", what="planner")
    _wait_for(lambda: len(_request_ids(workspace)) == 2)
    cfg = core.load_config(workspace)
    event = core.wait(cfg, timeout_s=5, poll_s=0.01, watch_sessions=False)
    assert event.request_id == _request_of(workspace, "planner", "planner")
    assert core.planner_turn({"role": "plan_reviewer", "what": "critic"})
    assert not core.planner_turn({"role": "plan_reviewer", "what": "plan-review"})
    for request_id in _request_ids(workspace):
        _answer(transport, request_id, "ok")
    for thread in [*threads, planner_thread]:
        thread.join(timeout=10)


def test_wait_returns_when_the_orchestrator_exits(workspace):
    """A closed run writes no further request, so the waiter must notice the exit."""
    dispatch = workspace / "dispatch"
    dispatch.mkdir()
    cfg = core.load_config(workspace)
    assert core.orchestrator_state(cfg)[0] == "unknown"
    write_json_atomic(dispatch / ORCHESTRATOR_HEARTBEAT, {"pid": 1, "host": "h", "at": time.time()})
    assert core.orchestrator_state(cfg)[0] == "running"
    assert core.wait(cfg, timeout_s=0.1, poll_s=0.01, watch_sessions=False).kind == "TIMEOUT"
    write_json_atomic(dispatch / ORCHESTRATOR_EXIT, {"pid": 1, "host": "h", "at": time.time() + 1})
    event = core.wait(cfg, timeout_s=5, poll_s=0.01, watch_sessions=False)
    assert event.kind == "ORCHESTRATOR_EXITED" and "pid 1" in event.detail


def test_a_stale_orchestrator_heartbeat_reads_as_an_exit(workspace):
    (workspace / "dispatch").mkdir()
    write_json_atomic(
        workspace / "dispatch" / ORCHESTRATOR_HEARTBEAT,
        {"pid": 9, "host": "h", "at": time.time() - 3600},
    )
    state, detail = core.orchestrator_state(core.load_config(workspace))
    assert state == "exited" and "stopped heart-beating" in detail


def test_an_orchestrator_spool_job_receipt_reads_as_an_exit(workspace, tmp_path):
    spool = tmp_path / "spool-orch"
    (spool / "done").mkdir(parents=True)
    (spool / "done" / "free-mv2flow-run3.rc").write_text("0\n")
    cfg = core.load_config(workspace, orchestrator_job=f"{spool}:free-mv2flow-run3")
    state, detail = core.orchestrator_state(cfg)
    assert state == "exited" and "free-mv2flow-run3 finished rc=0" in detail


def test_wait_surfaces_a_sessions_successor_request(workspace, tmp_path):
    spool = tmp_path / "spool-a"
    (spool / "handover").mkdir(parents=True)
    (spool / "watchers").mkdir()
    ends = time.time() + 600
    write_json_atomic(
        spool / "watchers" / "w77.json",
        {"watcher": "w77", "slurm_job": "77", "alive_epoch": time.time(), "ends_at_epoch": ends},
    )
    write_json_atomic(
        spool / "handover" / "request.json",
        {"request_id": "a-77-2", "slurm_job": "77", "watcher": "w77", "ends_at_epoch": ends},
    )
    event = core.wait(core.load_config(workspace), timeout_s=5, poll_s=0.01)
    assert event.kind == "SESSION_REQUEST" and "session a" in event.detail
    assert "session allocate" in event.detail


def test_wait_surfaces_a_dead_session(workspace, tmp_path):
    spool = tmp_path / "spool-b"
    (spool / "watchers").mkdir(parents=True)
    write_json_atomic(
        spool / "watchers" / "w.json", {"watcher": "w", "alive_epoch": time.time() - 900}
    )
    event = core.wait(core.load_config(workspace), timeout_s=5, poll_s=0.01)
    assert event.kind == "SESSION_STALE" and str(spool) in event.detail


def test_the_prompt_resolves_everything_a_subagent_needs(workspace):
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, _ = _ask(
        transport,
        what="onboard topk",
        decision_file=str(workspace / "decisions" / "smith-3.yaml"),
        backend="claude-code",
        model="claude-opus-5-5",
    )
    _wait_for(lambda: _request_ids(workspace))
    (request_id,) = _request_ids(workspace)
    text = core.render_prompt(core.load_config(workspace), request_id)
    assert "$" not in re.sub(r"\$[A-Z_]+|\$\(", "", text), "an unrendered template field"
    for fragment in (
        f"export MV2FLOW_JOB_PREFIX=r{request_id.split('-')[0]}",
        str(workspace / "dispatch" / f"{request_id}.system.md"),
        str(workspace / "dispatch" / f"{request_id}.done"),
        str(workspace / "decisions" / "smith-3.yaml"),
        "model `claude-opus-5-5`",
        "spoolctl.py cancel",
        "6.0 h",
    ):
        assert fragment in text, fragment
    _answer(transport, request_id, "ok")
    thread.join(timeout=10)


def test_the_driver_can_hand_a_turn_back_in_order(workspace):
    transport, threads = _orchestrate(workspace, {"what": "x"})
    (request_id,) = _request_ids(workspace)
    source = workspace / "result.md"
    source.write_text("blocked: no GPU")
    core.answer(core.load_config(workspace), request_id, source.read_text())
    threads[0].join(timeout=10)
    assert (workspace / "dispatch" / f"{request_id}.result.md").read_text() == "blocked: no GPU"


def test_the_instructions_are_rendered_from_the_task(workspace, tmp_path):
    (workspace / "TURN_INSTRUCTIONS.local.md").write_text("Read 03-plan/ before starting.\n")
    text = core.render_instructions(core.load_config(workspace))
    for fragment in (
        f"Working root: `{tmp_path / 'repo'}`",
        f"trtllm-modeling-v2 driver --workspace {workspace} prompt",
        f"| `b` | catalog tests, microbenchmarks | `{tmp_path / 'spool-b'}` |",
        f"python3 {SPOOLCTL_PATH} submit",
        "modeling_secret.py",
        "/short/tmp",
        "safe-kill rule",
        "Read 03-plan/ before starting.",
    ):
        assert fragment in text, fragment


def test_the_packaged_templates_name_no_machine_specific_path():
    for path in (core.TURN_INSTRUCTIONS_TEMPLATE, core.TURN_PROMPT_TEMPLATE):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("/scratch/", "/lustre/", "/home/", "deepseek", "DeepSeek"):
            assert forbidden not in text


def test_the_cli_drives_a_workspace(workspace, capsys):
    transport, threads = _orchestrate(workspace, {"what": "x"})
    (request_id,) = _request_ids(workspace)
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(
            ["driver", "--workspace", str(workspace), "prompt", request_id, "--claim", "--by", "me"]
        )
    assert exit_info.value.code == 0
    assert "MV2FLOW_JOB_PREFIX" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["driver", "--workspace", str(workspace), "status"])
    out = capsys.readouterr().out
    assert exit_info.value.code == 0 and f"{request_id}" in out and "CLAIMED" in out
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["driver", "--workspace", str(workspace), "instructions"])
    assert exit_info.value.code == 0 and (workspace / "TURN_INSTRUCTIONS.md").is_file()
    _answer(transport, request_id, "ok")
    threads[0].join(timeout=10)


def test_the_cli_refuses_a_directory_that_is_not_a_workspace(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["driver", "--workspace", str(tmp_path / "nowhere"), "status"])
    assert exit_info.value.code == 1
    assert "not a trtllm_modeling_v2 workspace" in capsys.readouterr().err


def test_the_wait_loop_documents_itself():
    from agent_flow.workflows.trtllm_modeling_v2.infra.driver import cli as driver_cli

    for event in ("NEW", "ORCHESTRATOR_EXITED", "SESSION_REQUEST", "TIMEOUT", "--sessions-only"):
        assert event in driver_cli.__doc__
    assert Path(driver_cli.__file__).with_name("TURN_INSTRUCTIONS.md").is_file()


# ------------------------------------------- decision files are named, never overwritten


def test_the_prompt_names_the_decision_file_exactly_as_the_request_does(workspace):
    """Decisions are `<unit>-<role>-<turn>.yaml` and the driver must write that exact file."""
    named = workspace / "decisions" / "core-assembler-2.yaml"
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, _ = _ask(transport, role="assembler", what="core", decision_file=str(named))
    _wait_for(lambda: _request_ids(workspace))
    (request_id,) = _request_ids(workspace)
    text = core.render_prompt(core.load_config(workspace), request_id)
    assert f"`{named}`" in text
    assert text.count("decisions/") == 1, "exactly one decision file is named"
    _answer(transport, request_id, "ok")
    thread.join(timeout=10)


def test_a_relative_decision_file_resolves_against_the_orchestrators_directory(
    workspace, tmp_path, monkeypatch
):
    """The driver runs elsewhere; a relative path means what it meant to the orchestrator."""
    orchestrator_cwd = tmp_path / "orchestrator-cwd"
    orchestrator_cwd.mkdir()
    monkeypatch.chdir(orchestrator_cwd)
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, _ = _ask(
        transport, role="reviewer", what="core", decision_file="ws/decisions/core-reviewer-1.yaml"
    )
    _wait_for(lambda: _request_ids(workspace))
    (request_id,) = _request_ids(workspace)
    monkeypatch.chdir(tmp_path)  # the driver is somewhere else
    text = core.render_prompt(core.load_config(workspace), request_id)
    assert f"`{orchestrator_cwd / 'ws' / 'decisions' / 'core-reviewer-1.yaml'}`" in text
    _answer(transport, request_id, "ok")
    thread.join(timeout=10)


def test_answering_a_turn_never_touches_a_decision_file(workspace):
    named = workspace / "decisions" / "core-smith-3.yaml"
    named.parent.mkdir()
    named.write_text("decision: LANDED\n")
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, _ = _ask(transport, decision_file=str(named))
    _wait_for(lambda: _request_ids(workspace))
    (request_id,) = _request_ids(workspace)
    core.answer(core.load_config(workspace), request_id, "done")
    thread.join(timeout=10)
    assert named.read_text() == "decision: LANDED\n"


def test_adoption_never_overwrites_a_decision_that_already_exists(workspace):
    dispatch = workspace / "dispatch"
    dispatch.mkdir()
    theirs = workspace / "decisions" / "core-assembler-1.yaml"
    ours = workspace / "decisions" / "core-assembler-1.2.yaml"
    theirs.parent.mkdir()
    theirs.write_text("decision: UNIT_READY\n")
    ours.write_text("decision: BLOCKED\n")
    write_json_atomic(
        dispatch / "010-assembler.request.json",
        {
            "id": "010-assembler",
            "role": "assembler",
            "what": "core",
            "decision_file": str(theirs),
            "requested_at": time.time() - 60,
        },
    )
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, box = _ask(transport, role="assembler", what="core", decision_file=str(ours))
    time.sleep(0.2)
    _answer(transport, "010-assembler", "ok")
    thread.join(timeout=10)
    assert box["outcome"].request_id == "010-assembler"
    assert ours.read_text() == "decision: BLOCKED\n", (
        "an existing decision file is never overwritten"
    )
    assert theirs.read_text() == "decision: UNIT_READY\n"


def test_only_the_cli_starts_the_heartbeat_thread(workspace):
    """A thread per transport would make every test process multi-threaded before it forks."""
    import inspect

    ExternalAgentTransport(workspace, poll_s=0.01)
    assert not (workspace / "dispatch" / ORCHESTRATOR_HEARTBEAT).exists()
    assert "heartbeat_s=ORCHESTRATOR_HEARTBEAT_S" in inspect.getsource(cli_mod.main)


# ------------------------------------------------- a run whose turns run in process


def _ledger(workspace, *, host=None, stopped=None, started_at=None, held=()):
    """``inflight.json`` as an orchestrator keeps it, under any transport."""
    write_json_atomic(
        workspace / reload_mod.INFLIGHT_FILENAME,
        {"pid": 4242, "host": host or socket.gethostname(),
         "started_at": time.time() - 60 if started_at is None else started_at,
         "held": list(held), "stopped": stopped},
    )  # fmt: skip


@contextlib.contextmanager
def _orchestrator_lock(workspace):
    """The workspace lock, taken exactly as the orchestrator's CLI takes it."""
    handle = cli_mod._claim_workspace(workspace)
    try:
        yield
    finally:
        cli_mod._HELD_LOCKS.remove(handle)
        handle.close()


def _dispatched_turn(workspace):
    """A turn left in ``dispatch/`` by an earlier, external-transport process."""
    (workspace / "dispatch").mkdir(exist_ok=True)
    write_json_atomic(
        workspace / "dispatch" / "001-smith.request.json",
        {"id": "001-smith", "role": "smith", "what": "onboard x", "requested_at": time.time()},
    )


def test_an_in_process_orchestrator_holding_the_workspace_lock_is_running(workspace):
    _ledger(workspace, host="another-host")
    with _orchestrator_lock(workspace):
        state, detail = core.orchestrator_state(core.load_config(workspace))
    assert state == "running" and "holds the workspace lock" in detail


def test_an_in_process_orchestrator_that_let_the_lock_go_on_this_host_has_exited(workspace):
    _ledger(workspace)
    with _orchestrator_lock(workspace):
        pass
    state, detail = core.orchestrator_state(core.load_config(workspace))
    assert state == "exited" and "no longer holds the workspace lock" in detail


def test_a_free_lock_seen_from_another_host_proves_nothing(workspace):
    _ledger(workspace, host="another-host")
    with _orchestrator_lock(workspace):
        pass
    state, detail = core.orchestrator_state(core.load_config(workspace))
    assert state == "unknown" and "a driver on another-host also sees it stop" in detail


def test_an_in_process_run_that_closed_has_exited_from_any_host(workspace):
    _ledger(workspace, host="another-host", stopped={"how": "closed:delivered", "at": time.time()})
    state, detail = core.orchestrator_state(core.load_config(workspace))
    assert state == "exited" and "closed the run (delivered)" in detail


def test_a_reexec_for_a_reload_is_not_an_exit(workspace):
    _ledger(workspace, stopped={"how": "reload", "at": time.time()})
    with _orchestrator_lock(workspace):
        pass
    state, detail = core.orchestrator_state(core.load_config(workspace))
    assert state == "running" and "re-executing" in detail


def test_a_heartbeat_an_earlier_process_left_does_not_speak_for_the_current_one(workspace):
    (workspace / "dispatch").mkdir()
    now = time.time()
    beat = {"pid": 7, "host": "h", "started": now - 7200, "at": now - 3600}
    write_json_atomic(workspace / "dispatch" / ORCHESTRATOR_HEARTBEAT, beat)
    _ledger(workspace, started_at=now - 60)
    with _orchestrator_lock(workspace):
        assert core.orchestrator_state(core.load_config(workspace))[0] == "running"
    # The heartbeat of the ledger's own process (an external run) still decides.
    write_json_atomic(workspace / "dispatch" / ORCHESTRATOR_HEARTBEAT, {**beat, "started": now})
    state, detail = core.orchestrator_state(core.load_config(workspace))
    assert state == "exited" and "stopped heart-beating" in detail


def test_sessions_only_never_hands_out_a_turn(workspace):
    _dispatched_turn(workspace)
    cfg = core.load_config(workspace)
    assert core.wait(cfg, timeout_s=5, poll_s=0.01, watch_sessions=False).kind == "NEW"
    event = core.wait(cfg, timeout_s=0.05, poll_s=0.01, watch_sessions=False, sessions_only=True)
    assert event.kind == "TIMEOUT" and event.detail == "nothing new"


def test_sessions_only_reports_the_orchestrators_exit_and_the_sessions(workspace, tmp_path):
    _ledger(workspace, stopped={"how": "closed:delivered", "at": time.time()})
    event = core.wait(core.load_config(workspace), timeout_s=5, poll_s=0.01, sessions_only=True)
    assert event.kind == "ORCHESTRATOR_EXITED" and "turn(s) unanswered" not in event.detail
    _ledger(workspace)
    spool = tmp_path / "spool-b"
    (spool / "watchers").mkdir(parents=True)
    write_json_atomic(
        spool / "watchers" / "w.json", {"watcher": "w", "alive_epoch": time.time() - 900}
    )
    with _orchestrator_lock(workspace):
        event = core.wait(core.load_config(workspace), timeout_s=5, poll_s=0.01, sessions_only=True)
    assert event.kind == "SESSION_STALE" and str(spool) in event.detail


def test_the_cli_sessions_only_loop_shows_the_turns_the_orchestrator_holds(workspace, capsys):
    _dispatched_turn(workspace)
    since = time.time() - 300
    _ledger(workspace, held=[{"kind": "turn", "what": "smith onboard rmsnorm", "since": since}])
    argv = ["driver", "--workspace", str(workspace), "wait", "--sessions-only", "--no-sessions"]
    with _orchestrator_lock(workspace), pytest.raises(SystemExit) as exit_info:
        cli_mod.main([*argv, "--timeout", "0.05", "--poll", "0.01"])
    out = capsys.readouterr().out
    assert exit_info.value.code == 0 and out.startswith("TIMEOUT  nothing new")
    assert "orchestrator: running" in out
    assert "orchestrator holds: turn smith onboard rmsnorm 5m" in out
    assert "001-smith" not in out and "in flight:" not in out


def test_status_names_where_correctness_was_delivered(workspace):
    from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod

    cfg = core.load_config(workspace)
    assert core.status(cfg)["correctness"] is None
    assert "correctness:" not in core.format_status(core.status(cfg))

    contract_mod.record_milestone(
        workspace,
        contract_mod.Milestone(sha="abcdef0123456789", marker="m", ref="mv2/x-correct", checks=3),
    )
    text = core.format_status(core.status(cfg))
    assert "correctness: Correctness delivered at abcdef0123 (`mv2/x-correct`)" in text
