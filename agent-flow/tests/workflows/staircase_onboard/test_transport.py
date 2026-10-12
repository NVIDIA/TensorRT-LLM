# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The external agent transport: the orchestrator's turn, delivered as a file.

What these protect is the hand-off contract, because both sides are written
independently and a mismatch presents as a hang rather than an error. The
ordering rule (`.done` last) and the system prompt travelling with the turn
are the two that would be silently wrong.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from agent_flow.workflows.staircase_onboard.transport import (
    DISPATCH_DIRNAME,
    ExternalAgentTransport,
    ExternalTransportTimeout,
)


def _answer_after(transport: ExternalAgentTransport, request_id: str, result: str, delay=0.05):
    """Play the driver from another thread."""

    def run():
        time.sleep(delay)
        transport.answer(request_id, result)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_a_turn_round_trips(tmp_path):
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "001-smith", "LANDED, receipt written")
    assert transport(role="smith", what="onboard foo", prompt="do it") == (
        "LANDED, receipt written"
    )


def test_the_system_prompt_travels_with_the_turn(tmp_path):
    """A driver given only the user prompt runs a DIFFERENT agent.

    No role, no invariants, no citation discipline, no embargo -- and the
    answer still looks plausible, which is what makes the omission dangerous
    rather than obvious.
    """
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "001-assembler", "ok")
    transport(role="assembler", what="core", prompt="p", system_prompt="YOU ARE THE ASSEMBLER")

    root = tmp_path / DISPATCH_DIRNAME
    # `startswith`, not equality: the transport appends its own note about the
    # one instruction it cannot honour. See `_transport_note`.
    assert (
        (root / "001-assembler.system.md")
        .read_text(encoding="utf-8")
        .startswith("YOU ARE THE ASSEMBLER")
    )
    request = json.loads((root / "001-assembler.request.json").read_text(encoding="utf-8"))
    assert request["system_prompt_file"].endswith("001-assembler.system.md")


def test_the_request_names_the_decision_file_the_orchestrator_will_read(tmp_path):
    """The driver has to know where the typed decision goes.

    Otherwise the turn completes and the orchestrator finds nothing to branch on.
    """
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "001-reviewer", "ok")
    transport(role="reviewer", what="core", prompt="p", decision_file="/ws/decisions/r-1.yaml")
    request = json.loads(
        (tmp_path / DISPATCH_DIRNAME / "001-reviewer.request.json").read_text(encoding="utf-8")
    )
    assert request["decision_file"] == "/ws/decisions/r-1.yaml"


def test_done_is_written_after_the_result(tmp_path):
    """A reader that sees `.done` must be able to trust the result beside it.

    Same rule as the compute session's `.rc`, and for the same reason: the
    two are read by a different process than the one that writes them.
    """
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    root = tmp_path / DISPATCH_DIRNAME
    transport.answer("007-smith", "the answer")
    assert (root / "007-smith.done").stat().st_mtime_ns >= (
        root / "007-smith.result.md"
    ).stat().st_mtime_ns


def test_pending_lists_only_unanswered_requests_in_order(tmp_path):
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    for n, role in ((1, "smith"), (2, "smith"), (3, "assembler")):
        (tmp_path / DISPATCH_DIRNAME / f"{n:03d}-{role}.request.json").write_text(
            json.dumps({"id": f"{n:03d}-{role}"}), encoding="utf-8"
        )
    transport.answer("002-smith", "done")
    assert [p.name for p in transport.pending()] == [
        "001-smith.request.json",
        "003-assembler.request.json",
    ]


def test_ids_continue_across_a_resume(tmp_path):
    """Renumbering from 1 would overwrite the previous attempt's request files.

    Those files are the evidence for what the run asked for.
    """
    root = tmp_path / DISPATCH_DIRNAME
    root.mkdir(parents=True)
    (root / "004-smith.request.json").write_text("{}", encoding="utf-8")
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "005-smith", "ok")
    transport(role="smith", what="x", prompt="p")
    assert (root / "005-smith.result.md").is_file()


def test_concurrent_turns_get_distinct_ids(tmp_path):
    """Ten smiths run in one batch; two sharing an id would cross their answers."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    seen: list[str] = []
    lock = threading.Lock()

    def one():
        with lock:
            n = len(seen) + 1
        request_id = f"{n:03d}-smith"
        _answer_after(transport, request_id, request_id, delay=0.2)
        with lock:
            seen.append(request_id)

    # Drive the transport itself concurrently and check the ids it minted.
    results: list[str] = []
    threads = []
    for _ in range(5):
        one()
    for _ in range(5):
        t = threading.Thread(
            target=lambda: results.append(transport(role="smith", what="x", prompt="p"))
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=10)
    assert sorted(results) == sorted(seen)


def test_a_stale_driver_heartbeat_fails_fast_instead_of_waiting_hours(tmp_path):
    """The three-hour turn timeout bounds a slow turn, not a dead driver."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01, timeout_s=60)
    beat = tmp_path / DISPATCH_DIRNAME / "driver.heartbeat"
    beat.write_text("x", encoding="utf-8")
    import os

    old = time.time() - 10_000
    os.utime(beat, (old, old))
    with pytest.raises(ExternalTransportTimeout, match="heartbeat is stale"):
        transport(role="smith", what="x", prompt="p")


def test_no_heartbeat_at_all_is_not_treated_as_a_dead_driver(tmp_path):
    """A driver that does not bother writing one must not be penalized."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01, timeout_s=60)
    _answer_after(transport, "001-smith", "ok")
    assert transport(role="smith", what="x", prompt="p") == "ok"


def test_the_timeout_names_the_prompt_file_so_the_turn_can_be_run_by_hand(tmp_path):
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01, timeout_s=0.05)
    with pytest.raises(ExternalTransportTimeout, match="001-smith.prompt.md"):
        transport(role="smith", what="x", prompt="p")


def test_an_answer_with_no_result_file_is_empty_not_an_error(tmp_path):
    """A driver may legitimately have nothing to say.

    The decision file is what the orchestrator branches on.
    """
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    root = tmp_path / DISPATCH_DIRNAME

    def run():
        time.sleep(0.05)
        (root / "001-smith.done").write_text("1\n", encoding="utf-8")

    threading.Thread(target=run, daemon=True).start()
    assert transport(role="smith", what="x", prompt="p") == ""


def test_files_are_renamed_into_place_never_written_in_place(tmp_path):
    """A driver polling the directory must never see a half-written request."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "001-smith", "ok")
    transport(role="smith", what="x", prompt="p" * 10000)
    leftovers = list((tmp_path / DISPATCH_DIRNAME).glob("*.tmp"))
    assert not leftovers, f"staging files left behind: {leftovers}"


def test_the_workflow_accepts_a_transport_and_uses_it_for_every_role(tmp_path):
    """Reachability, in the same spirit as test_gate_wiring.py."""
    import inspect

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    assert "agent_transport" in inspect.signature(StaircaseOnboardWorkflow.__init__).parameters
    for method in ("_dispatch", "_run_one_smith"):
        source = inspect.getsource(getattr(StaircaseOnboardWorkflow, method))
        assert "_run_turn" in source, f"{method} bypasses the transport"
    turn = inspect.getsource(StaircaseOnboardWorkflow._run_turn)
    assert "system_prompt=" in turn


# ---------------------------------------- persistence, for the roles that need it


def test_prior_turns_restore_what_a_persistent_role_would_have_remembered(tmp_path):
    """plan_drafter, plan_reviewer and assembler are persistent by design.

    The transport starts a fresh session per turn, so their prompts' claim
    ("You are in the same session as your first draft") is false under it. A
    drafter that believes it will revise a file it has never read. The
    artifacts are on disk either way; naming them is what makes them usable.
    """
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "001-plan_drafter", "draft one done")
    transport(role="plan_drafter", what="round 1", prompt="draft it")

    _answer_after(transport, "002-plan_drafter", "draft two done")
    transport(role="plan_drafter", what="round 2", prompt="redraft it")

    request = json.loads(
        (tmp_path / DISPATCH_DIRNAME / "002-plan_drafter.request.json").read_text(encoding="utf-8")
    )
    assert [t["id"] for t in request["prior_turns"]] == ["001-plan_drafter"]
    assert request["prior_turns"][0]["prompt_file"].endswith("001-plan_drafter.prompt.md")


def test_prior_turns_are_scoped_to_one_role(tmp_path):
    """A smith must not be handed the assembler's history."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "001-assembler", "ok")
    transport(role="assembler", what="core", prompt="p")
    _answer_after(transport, "002-smith", "ok")
    transport(role="smith", what="x", prompt="p")
    assert [t["id"] for t in transport.prior_turns("smith")] == ["002-smith"]
    assert [t["id"] for t in transport.prior_turns("assembler")] == ["001-assembler"]


def test_an_unanswered_turn_is_not_prior_history(tmp_path):
    """Concurrent smiths would otherwise hand each other half-finished turns."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    (tmp_path / DISPATCH_DIRNAME / "001-smith.request.json").write_text("{}", encoding="utf-8")
    assert transport.prior_turns("smith") == []


# --------------------------------------------------- one orchestrator per workspace


def test_a_second_orchestrator_on_one_workspace_is_refused(tmp_path):
    """Two on one workspace is not a race anybody notices.

    They share the checkpoint file, the dispatch directory and the request-id
    counter, so they mint the SAME id, overwrite each other's request JSON,
    and both unblock on one answer. Measured: a stale process a `pkill` had
    failed to kill raced a fresh one for three seconds, and the survivor's
    request was the older code's — missing the system-prompt pointer the
    driver needed, which is a turn run with no role.
    """
    import subprocess
    import sys
    import textwrap

    workspace = tmp_path / "ws"
    script = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
        from agent_flow.workflows.staircase_onboard.cli import _claim_workspace
        from pathlib import Path
        _claim_workspace(Path({str(workspace)!r}))
        print("HELD", flush=True)
        time.sleep(30)
        """
    )
    holder = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "HELD"
        second = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
        )
        assert second.returncode == 3
        assert "already holds" in second.stderr
    finally:
        holder.kill()


def test_the_lock_is_released_when_the_process_exits(tmp_path):
    """A crash must not leave a workspace unusable; flock is held by the fd."""
    import subprocess
    import sys
    import textwrap

    workspace = tmp_path / "ws"
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
        from agent_flow.workflows.staircase_onboard.cli import _claim_workspace
        from pathlib import Path
        _claim_workspace(Path({str(workspace)!r}))
        """
    )
    for _ in range(2):
        done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr


# -------------------------------- what this transport cannot honour, said once


def test_the_system_prompt_says_the_progress_tool_is_unavailable(tmp_path):
    """Every role prompt ends by telling the agent to call `append_<role>_progress`.

    The tool is real -- the in-process transport hands it to the session it
    spawns -- but this transport has no session. The shared decision-file
    block that follows says to write a file instead, so the prompt arrives
    carrying two different endings and only the second one works here.

    Three auditor turns in a row noticed and picked correctly. The next role
    might not: calling a missing tool looks from the inside like a turn that
    ended, so the run would block for its full timeout on a turn that believed
    it was finished.
    """
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "001-auditor", "ok")
    transport(role="auditor", what="w", prompt="p", system_prompt="ROLE PROMPT")

    system = (tmp_path / DISPATCH_DIRNAME / "001-auditor.system.md").read_text(encoding="utf-8")
    assert system.startswith("ROLE PROMPT")
    assert "append_auditor_progress" in system
    assert "not available here" in system
    assert "`<id>.done`" in system


def test_the_note_names_the_role_that_was_dispatched(tmp_path):
    """A generic note would send a smith looking for the auditor's tool."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "001-smith", "ok")
    transport(role="smith", what="w", prompt="p", system_prompt="ROLE")

    system = (tmp_path / DISPATCH_DIRNAME / "001-smith.system.md").read_text(encoding="utf-8")
    assert "append_smith_progress" in system
    assert "append_auditor_progress" not in system


def test_the_role_prompt_is_never_edited_only_appended_to(tmp_path):
    """The instruction is correct for the transport it was written for.

    Rewriting the role prompts to suit this one would make them wrong in
    process, so the difference stays where it belongs.
    """
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "001-reviewer", "ok")
    original = "ROLE\n\n## Ending your turn\n\nCall `append_reviewer_progress`.\n"
    transport(role="reviewer", what="w", prompt="p", system_prompt=original)

    system = (tmp_path / DISPATCH_DIRNAME / "001-reviewer.system.md").read_text(encoding="utf-8")
    assert system.startswith(original)
