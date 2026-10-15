# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The external transport's driver, productized (F041, F030, F017, F010).

The DSV4.1 run drove its turns with a run-local `driver.py`, a
`wait_for_request.sh` and a hand-written `TURN_INSTRUCTIONS.md`, all with the
run's paths written into them. These tests pin the packaged replacement: the
orchestrator side (per-role timeouts, heartbeat, job prefixes, adoption across
a restart) and the driver side (wait, claim, prompt, status, instructions).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

import pytest
import yaml

from agent_flow.runtime.turns import ORCHESTRATOR_EXIT, ORCHESTRATOR_HEARTBEAT, TurnRequest
from agent_flow.workflows.staircase_onboard import cli as cli_mod
from agent_flow.workflows.staircase_onboard.driver import core
from agent_flow.workflows.staircase_onboard.substrate import SCRIPTS_DIR
from agent_flow.workflows.staircase_onboard.substrate.scheduler import write_json_atomic
from agent_flow.workflows.staircase_onboard.transport import (
    DEFAULT_ROLE_TURN_TIMEOUTS,
    EXTERNAL_MAX_PARALLEL_SMITHS,
    ExternalAgentTransport,
    ExternalTransportTimeout,
)


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
        except ExternalTransportTimeout as exc:
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
    """F017: one 3 h bound killed a core assembly; the assembler now gets 16 h."""
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
    with pytest.raises(ExternalTransportTimeout, match="reviewer turn timeout"):
        transport(role="reviewer", what="review", prompt="p")


def test_a_request_names_its_job_prefix_timeout_and_agent(workspace):
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, _ = _ask(transport, backend="codex", model="gpt-x")
    _wait_for(lambda: _request_ids(workspace))
    (request_id,) = _request_ids(workspace)
    data = json.loads((workspace / "dispatch" / f"{request_id}.request.json").read_text())
    assert data["job_prefix"] == f"r{request_id.split('-')[0]}"
    assert data["timeout_s"] == DEFAULT_ROLE_TURN_TIMEOUTS["smith"]
    assert data["agent"] == {"backend": "codex", "model": "gpt-x"}
    transport.answer(request_id, "ok")
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
    """A shared project quota filled for minutes.

    The beat thread died on EDQUOT and the driver read a live orchestrator as exited.
    """
    import errno

    from agent_flow.runtime.turns import file_dispatch

    real, fails = file_dispatch.FileDispatchTurnExecutor._write_atomic, {"n": 2}

    def full(path, text):
        if path.name == ORCHESTRATOR_HEARTBEAT and fails["n"]:
            fails["n"] -= 1
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real(path, text)

    monkeypatch.setattr(file_dispatch.FileDispatchTurnExecutor, "_write_atomic", staticmethod(full))
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=0.02)
    beat = workspace / "dispatch" / ORCHESTRATOR_HEARTBEAT
    _wait_for(beat.exists)
    assert fails["n"] == 0
    transport.close()


def test_a_relaunch_adopts_an_orphaned_turn_whose_decision_path_moved(workspace):
    """F030: run6 wrote turn 2 of 6 for the turn run5 had asked and nobody answered."""
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
    transport.answer("046-assembler", "assembled")
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
        transport.answer(request_id, "ok")
    first.join(timeout=10)
    second.join(timeout=10)


def test_the_external_transport_caps_concurrency_at_eight():
    """F010: twelve concurrent turns lost seven to API stream stalls."""
    assert EXTERNAL_MAX_PARALLEL_SMITHS == 8
    assert core.DEFAULT_MAX_INFLIGHT == 8


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
    transport.answer(request_id, "ok")
    with pytest.raises(core.DriverError, match="already answered"):
        core.claim(cfg, request_id)
    for thread in threads:
        thread.join(timeout=10)


def test_wait_returns_a_new_turn(workspace):
    transport, threads = _orchestrate(workspace, {"what": "onboard rmsnorm"})
    event = core.wait(core.load_config(workspace), timeout_s=5, poll_s=0.01, watch_sessions=False)
    assert event.kind == "NEW" and event.request_id == _request_ids(workspace)[0]
    assert "smith: onboard rmsnorm" in event.detail
    transport.answer(event.request_id, "ok")
    threads[0].join(timeout=10)


def test_wait_respects_the_in_flight_cap(workspace):
    transport, threads = _orchestrate(workspace, {"what": "a"}, {"what": "b"})
    cfg = core.load_config(workspace, max_inflight=1)
    core.claim(cfg, _request_ids(workspace)[0])
    event = core.wait(cfg, timeout_s=0.2, poll_s=0.01, watch_sessions=False)
    assert event.kind == "TIMEOUT" and "in-flight cap, 1/1" in event.detail
    for request_id in _request_ids(workspace):
        transport.answer(request_id, "ok")
    for thread in threads:
        thread.join(timeout=10)


def test_wait_returns_when_the_orchestrator_exits(workspace):
    """F041: the waiter sat 2 h 10 min past a run that had already closed."""
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
    (spool / "done" / "free-staircase-run3.rc").write_text("0\n")
    cfg = core.load_config(workspace, orchestrator_job=f"{spool}:free-staircase-run3")
    state, detail = core.orchestrator_state(cfg)
    assert state == "exited" and "free-staircase-run3 finished rc=0" in detail


def test_wait_surfaces_a_sessions_successor_request(workspace, tmp_path):
    spool = tmp_path / "spool-a"
    (spool / "handover").mkdir(parents=True)
    write_json_atomic(
        spool / "handover" / "request.json", {"request_id": "a-1-2", "slurm_job": "77"}
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
        f"export STAIRCASE_JOB_PREFIX=r{request_id.split('-')[0]}",
        str(workspace / "dispatch" / f"{request_id}.system.md"),
        str(workspace / "dispatch" / f"{request_id}.done"),
        str(workspace / "decisions" / "smith-3.yaml"),
        "model `claude-opus-5-5`",
        "modeling_secret.py",
        "spoolctl.sh cancel",
        "6.0 h",
    ):
        assert fragment in text, fragment
    transport.answer(request_id, "ok")
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
        f"staircase-onboard driver --workspace {workspace} prompt",
        f"| `b` | catalog tests, microbenchmarks | `{tmp_path / 'spool-b'}` |",
        str(SCRIPTS_DIR / "spool_submit.sh"),
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
    assert "STAIRCASE_JOB_PREFIX" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["driver", "--workspace", str(workspace), "status"])
    out = capsys.readouterr().out
    assert exit_info.value.code == 0 and f"{request_id}" in out and "CLAIMED" in out
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["driver", "--workspace", str(workspace), "instructions"])
    assert exit_info.value.code == 0 and (workspace / "TURN_INSTRUCTIONS.md").is_file()
    transport.answer(request_id, "ok")
    threads[0].join(timeout=10)


def test_the_cli_refuses_a_directory_that_is_not_a_workspace(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["driver", "--workspace", str(tmp_path / "nowhere"), "status"])
    assert exit_info.value.code == 1
    assert "not a staircase workspace" in capsys.readouterr().err


def test_the_wait_loop_documents_itself():
    from agent_flow.workflows.staircase_onboard.driver import cli as driver_cli

    for event in ("NEW", "ORCHESTRATOR_EXITED", "SESSION_REQUEST", "TIMEOUT"):
        assert event in driver_cli.__doc__
    assert Path(driver_cli.__file__).with_name("TURN_INSTRUCTIONS.md").is_file()


# ------------------------------------------- decision files are named, never overwritten


def test_the_prompt_names_the_decision_file_exactly_as_the_request_does(workspace):
    """F033: decisions are `<unit>-<role>-<turn>.yaml` and the driver must write that exact file."""
    named = workspace / "decisions" / "core-assembler-2.yaml"
    transport = ExternalAgentTransport(workspace, poll_s=0.01, heartbeat_s=None)
    thread, _ = _ask(transport, role="assembler", what="core", decision_file=str(named))
    _wait_for(lambda: _request_ids(workspace))
    (request_id,) = _request_ids(workspace)
    text = core.render_prompt(core.load_config(workspace), request_id)
    assert f"`{named}`" in text
    assert text.count("decisions/") == 1, "exactly one decision file is named"
    transport.answer(request_id, "ok")
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
    transport.answer(request_id, "ok")
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
    transport.answer("010-assembler", "ok")
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
