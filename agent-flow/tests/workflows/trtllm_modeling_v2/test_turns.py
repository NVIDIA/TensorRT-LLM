# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract both turn transports answer.

The point of the abstraction is that an orchestrator cannot tell them apart,
so most of this file is parametrized over both. What differs — a mailbox has
request ids and artifacts, a direct call has a session — is asserted
separately and deliberately, because those are the differences a caller is
allowed to see.
"""

from __future__ import annotations

import threading
import time

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.execution.turns import (
    InProcessTurnExecutor,
    TurnExecutionError,
    TurnOutcome,
    TurnRequest,
    TurnTimeout,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.transport import ExternalAgentTransport


def _answer(transport, request_id, result):
    """What a driver does: the result, then ``.done`` last."""
    (transport.root / f"{request_id}.result.md").write_text(result, encoding="utf-8")
    (transport.root / f"{request_id}.done").write_text("1\n", encoding="utf-8")


class RecordingLayer:
    """Stands in for an ``AgentLayer``: called with a prompt, returns text."""

    def __init__(self, reply: str = "answered", name: str = "layer") -> None:
        self.reply = reply
        self.layer_name = name
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def _answer_after(executor: ExternalAgentTransport, request_id: str, result: str, delay=0.02):
    """Play the driver from another thread."""

    def run():
        time.sleep(delay)
        _answer(executor, request_id, result)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


@pytest.fixture(params=["inprocess", "file-dispatch"])
def executor_pair(request, tmp_path):
    """``(executor, arrange_answer, prompt_seen)`` for each transport.

    ``arrange_answer(role, text)`` makes that transport return ``text`` for
    the next turn, however that transport gets its answers.
    ``prompt_seen()`` returns what the far side was actually handed.
    """
    if request.param == "inprocess":
        layer = RecordingLayer()

        def arrange(role: str, text: str) -> None:
            layer.reply = text

        yield InProcessTurnExecutor(lambda req: layer), arrange, lambda: layer.prompts[-1]
    else:
        executor = ExternalAgentTransport(tmp_path, poll_s=0.01)
        counter = {"n": 0}

        def arrange(role: str, text: str) -> None:
            counter["n"] += 1
            _answer_after(executor, f"{counter['n']:03d}-{role}", text)

        def prompt_seen() -> str:
            latest = sorted(executor.root.glob("*.prompt.md"))[-1]
            return latest.read_text(encoding="utf-8")

        yield executor, arrange, prompt_seen


# ------------------------------------------------- the part that must not differ


def test_a_turn_returns_the_role_the_text_and_the_transport(executor_pair):
    executor, arrange, _ = executor_pair
    arrange("smith", "landed foo")
    outcome = executor.run(TurnRequest(role="smith", what="onboard foo", prompt="do it"))

    assert isinstance(outcome, TurnOutcome)
    assert outcome.role == "smith"
    assert outcome.text == "landed foo"
    assert outcome.transport == executor.name


def test_every_transport_accepts_bound_even_without_a_session(executor_pair):
    """Callers must not have to ask which transport they hold.

    A concurrent batch owns its own sessions, so it calls ``bound``
    unconditionally. A transport with no session concept has to tolerate
    that rather than make the caller branch — which is the branch this
    interface exists to delete.
    """
    executor, arrange, _ = executor_pair
    arrange("smith", "ok")
    bound = executor.bound(RecordingLayer("ok"))
    assert bound.run(TurnRequest(role="smith", what="x", prompt="p")).text == "ok"


def test_the_prompt_reaches_the_other_side_unedited(executor_pair):
    executor, arrange, prompt_seen = executor_pair
    arrange("assembler", "done")
    prompt = "line one\n\nline two with `backticks` and — an em dash"
    executor.run(TurnRequest(role="assembler", what="core", prompt=prompt))
    assert prompt_seen() == prompt


# ------------------------------------------------------------- in-process only


def test_in_process_calls_the_layer_the_provider_returns():
    layer = RecordingLayer("from the layer")
    seen: list[str] = []

    def provider(request: TurnRequest):
        seen.append(request.role)
        return layer

    outcome = InProcessTurnExecutor(provider).run(
        TurnRequest(role="reviewer", what="core", prompt="review it")
    )
    assert seen == ["reviewer"]
    assert outcome.text == "from the layer"
    assert layer.prompts == ["review it"]


def test_bound_pins_the_session_the_caller_already_owns():
    """Ten concurrent dispatches are ten agents, not ten turns of one.

    The provider resolves a role's shared layer; a batch that built its own
    must be able to say so, or every dispatch in it would land on the same
    session.
    """
    shared = RecordingLayer("shared")
    mine = RecordingLayer("mine")
    executor = InProcessTurnExecutor(lambda req: shared)

    assert executor.run(TurnRequest(role="smith", what="a", prompt="p")).text == "shared"
    assert executor.bound(mine).run(TurnRequest(role="smith", what="b", prompt="p")).text == "mine"
    # Binding returns a view; the original is untouched.
    assert executor.run(TurnRequest(role="smith", what="c", prompt="p")).text == "shared"


def test_a_provider_with_no_layer_is_a_named_failure_not_an_attribute_error():
    """A role the workflow never configured must say so, not crash on None."""
    executor = InProcessTurnExecutor(lambda req: None)
    with pytest.raises(TurnExecutionError, match="no layer available for role 'ghost'"):
        executor.run(TurnRequest(role="ghost", what="x", prompt="p"))


def test_the_system_prompt_is_not_resent_in_process():
    """In process the session already carries it; resending would double it."""
    layer = RecordingLayer("ok")
    InProcessTurnExecutor(lambda req: layer).run(
        TurnRequest(role="smith", what="x", prompt="task only", system_prompt="ROLE PROMPT")
    )
    assert layer.prompts == ["task only"]


def test_an_interrupt_during_a_turn_reaches_the_caller():
    """Ctrl-C must not become an empty answer.

    The orchestrator catches it around the whole run to checkpoint and say
    how to resume. Swallowing it here would produce a turn that "ended" with
    no decision, which every role's caller reads as a role that declined to
    report.
    """

    def interrupted(prompt: str) -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        InProcessTurnExecutor(lambda req: interrupted).run(
            TurnRequest(role="smith", what="x", prompt="p")
        )


def test_an_exception_from_a_layer_is_not_converted_into_an_outcome():
    """A batch scores a raised dispatch differently from one that decided nothing."""
    boom = RuntimeError("backend returned 401")

    def failing(prompt: str) -> str:
        raise boom

    with pytest.raises(RuntimeError, match="401"):
        InProcessTurnExecutor(lambda req: failing).run(
            TurnRequest(role="plan_reviewer", what="x", prompt="p")
        )


# ---------------------------------------------------------- file-dispatch only


def test_the_system_prompt_travels_with_a_dispatched_turn(tmp_path):
    """A driver handed only the task prompt is running a different agent."""
    executor = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(executor, "001-smith", "ok")
    executor.run(TurnRequest(role="smith", what="x", prompt="task", system_prompt="ROLE PROMPT"))
    system = (executor.root / "001-smith.system.md").read_text(encoding="utf-8")
    assert system == "ROLE PROMPT"


def test_a_turn_nobody_answers_times_out_with_the_prompt_path(tmp_path):
    """A structured failure naming where to run the turn by hand."""
    executor = ExternalAgentTransport(tmp_path, poll_s=0.01, timeout_s=0.05)
    with pytest.raises(TurnTimeout, match="001-smith.prompt.md"):
        executor.run(TurnRequest(role="smith", what="x", prompt="p"))


def test_a_stale_heartbeat_fails_before_the_full_timeout(tmp_path):
    """The turn timeout bounds a slow turn; the heartbeat bounds a dead driver."""
    import os

    executor = ExternalAgentTransport(tmp_path, poll_s=0.01, timeout_s=600)
    beat = executor.root / "driver.heartbeat"
    beat.write_text("x", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(beat, (old, old))

    started = time.monotonic()
    with pytest.raises(TurnTimeout, match="heartbeat is stale"):
        executor.run(TurnRequest(role="smith", what="x", prompt="p"))
    assert time.monotonic() - started < 10


def test_bound_is_a_no_op_for_a_transport_with_no_session(tmp_path):
    executor = ExternalAgentTransport(tmp_path, poll_s=0.01)
    assert executor.bound(object()) is executor


def test_ids_continue_across_a_resume_so_evidence_is_never_overwritten(tmp_path):
    """The request files are the record of what the run asked for."""
    first = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(first, "001-smith", "ok")
    first.run(TurnRequest(role="smith", what="x", prompt="p"))

    resumed = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(resumed, "002-smith", "ok")
    outcome = resumed.run(TurnRequest(role="smith", what="y", prompt="p"))
    assert outcome.request_id == "002-smith"


def test_a_resume_waits_on_an_unanswered_twin_instead_of_asking_again(tmp_path):
    """A dead process's in-flight turn is the same question, possibly being answered.

    A compute session can die with dozens of smith turns in flight; re-asking
    each would duplicate every agent's work and hand two agents the same
    decision path.
    """
    first = ExternalAgentTransport(tmp_path, poll_s=0.01, timeout_s=0.05)
    with pytest.raises(TurnTimeout):
        first.run(
            TurnRequest(role="smith", what="onboard a", prompt="p", decision_file="d/smith-1.yaml")
        )

    resumed = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(resumed, "001-smith", "answered by the first driver")
    outcome = resumed.run(
        TurnRequest(role="smith", what="onboard a", prompt="p2", decision_file="d/smith-1.yaml")
    )
    assert outcome.request_id == "001-smith"
    assert outcome.text == "answered by the first driver"
    assert not (resumed.root / "002-smith.request.json").exists()


def test_an_answered_or_different_request_is_never_adopted(tmp_path):
    """Only an unanswered request for the same turn and decision path is a twin."""
    first = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(first, "001-smith", "done once")
    first.run(TurnRequest(role="smith", what="onboard a", prompt="p", decision_file="d/1"))

    second = ExternalAgentTransport(tmp_path, poll_s=0.01, timeout_s=0.05)
    with pytest.raises(TurnTimeout):
        second.run(TurnRequest(role="smith", what="onboard b", prompt="p", decision_file="d/2"))

    third = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(third, "003-smith", "fresh")
    outcome = third.run(
        TurnRequest(role="smith", what="onboard a", prompt="p", decision_file="d/1")
    )
    assert outcome.request_id == "003-smith"


def _wait_for(path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.005)


def test_a_concurrent_caller_asking_alike_is_never_handed_a_siblings_request(tmp_path):
    """Two in-flight turns of one process are two questions, even when they read alike.

    The ordering is forced rather than left to the scheduler: the second
    caller looks only after the first caller's request file is on disk. That
    is the interleaving in which a twin check that ignored in-flight requests
    would adopt the sibling's request, so both callers would wait on one
    answer and one prompt would never be delivered.
    """
    # Bounded, and daemon threads: a regression must fail here, not hang the run.
    executor = ExternalAgentTransport(tmp_path, poll_s=0.005, timeout_s=10)
    alike = dict(role="smith", what="onboard a", decision_file=None)
    outcomes: dict[str, TurnOutcome] = {}

    def ask(key: str) -> threading.Thread:
        prompt = f"{key} prompt"
        thread = threading.Thread(
            target=lambda: outcomes.__setitem__(
                key, executor.run(TurnRequest(prompt=prompt, **alike))
            ),
            daemon=True,
        )
        thread.start()
        return thread

    first = ask("first")
    _wait_for(executor.root / "001-smith.request.json")
    second = ask("second")
    _wait_for(executor.root / "002-smith.request.json")

    _answer(executor, "001-smith", "answer one")
    _answer(executor, "002-smith", "answer two")
    first.join(timeout=5)
    second.join(timeout=5)

    assert outcomes["first"].request_id == "001-smith"
    assert outcomes["second"].request_id == "002-smith"
    assert outcomes["first"].text == "answer one"
    assert outcomes["second"].text == "answer two"
    assert (executor.root / "002-smith.prompt.md").read_text(encoding="utf-8") == "second prompt"


def test_a_retry_in_the_same_process_waits_on_its_own_unanswered_request(tmp_path):
    """Once its caller has given up, a request is an orphan like a dead process's.

    Re-asking would duplicate work a driver may still be doing, exactly as on
    a resume.
    """
    executor = ExternalAgentTransport(tmp_path, poll_s=0.005, timeout_s=0.05)
    request = TurnRequest(role="smith", what="onboard a", prompt="p", decision_file="d/1")
    with pytest.raises(TurnTimeout):
        executor.run(request)

    def answer_once_adopted():
        # Answering before the retry has looked would make 001 an answered
        # request, which is rightly never adopted; wait until the retry holds it.
        deadline = time.monotonic() + 5
        while "001-smith" not in executor._in_flight and time.monotonic() < deadline:
            time.sleep(0.005)
        _answer(executor, "001-smith", "late answer")

    executor.timeout_s = 5.0
    threading.Thread(target=answer_once_adopted, daemon=True).start()
    outcome = executor.run(request)
    assert outcome.request_id == "001-smith"
    assert outcome.text == "late answer"
    assert not (executor.root / "002-smith.request.json").exists()


def test_the_request_file_names_the_backend_the_role_is_configured_for(tmp_path):
    """A driver can only honour a role's model if the request says which one it is."""
    import json

    executor = ExternalAgentTransport(tmp_path, poll_s=0.005)
    _answer_after(executor, "001-plan_reviewer", "ok")
    executor.run(
        TurnRequest(role="plan_reviewer", what="r1", prompt="p", backend="codex", model="m")
    )
    request = json.loads(
        (executor.root / "001-plan_reviewer.request.json").read_text(encoding="utf-8")
    )
    assert request["agent"] == {"backend": "codex", "model": "m", "reasoning_effort": None}
