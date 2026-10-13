# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staircase's use of the file-dispatch turn transport.

The protocol — request/system/prompt/result/done, written atomically, `.done`
last, a monotonic id per turn, a heartbeat that fails a dead driver fast —
is not Staircase's. It solves "this host cannot authenticate an agent CLI",
which is a property of the host and not of onboarding a checkpoint, so it
lives in :mod:`agent_flow.runtime.turns.file_dispatch` and any workflow can
use it.

What IS Staircase's is one paragraph: every role prompt in this workflow ends
by telling the agent to call ``append_<role>_progress``, and that tool cannot
exist here. So this module supplies that note and nothing else.

The names below are kept because the CLI, the workflow and the tests all
reach for them, and because the thing they name has not changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from agent_flow.runtime.turns import (
    DEFAULT_TURN_TIMEOUT_S,
    DISPATCH_DIRNAME,
    HEARTBEAT_STALE_S,
    FileDispatchTurnExecutor,
    TurnTimeout,
)

from .embargo import embargo_transport_note

#: Per-role turn timeouts, in seconds. One bound for every role was wrong in
#: both directions: a core assembly for a new architecture (modeling, weights,
#: routing, golden and long-context proofs) ran past the old 3 h default and
#: raised out of the orchestrator (F017), while a stuck plan review would wait
#: just as long as an assembly.
DEFAULT_ROLE_TURN_TIMEOUTS: dict[str, float] = {
    "plan_drafter": 4 * 3600.0,
    "plan_reviewer": 3 * 3600.0,
    "smith": 6 * 3600.0,
    "assembler": 16 * 3600.0,
    "reviewer": 4 * 3600.0,
    "auditor": 6 * 3600.0,
    "runner": 3 * 3600.0,
    # One candidate: implement, test on GPU, micro-bench -- hours of spool jobs.
    "tuner": 8 * 3600.0,
}

#: How often the orchestrator tells the driver it is alive.
ORCHESTRATOR_HEARTBEAT_S = 30.0

#: Concurrent smith dispatches when turns go through the external transport.
#: The binding limit there is concurrent agent streams on one account: twelve
#: concurrent turns lost seven to "Agent stalled: no progress for 600s" within
#: twelve minutes (F010).
EXTERNAL_MAX_PARALLEL_SMITHS = 8

#: The historical name, kept because it is what the workflow, the CLI and
#: their tests raise and catch. A `TurnTimeout` from any transport is the
#: same event, so this is an alias rather than a parallel hierarchy.
ExternalTransportTimeout = TurnTimeout


def staircase_transport_note(role: str) -> str:
    """Say which of the role prompt's instructions this transport cannot honour.

    Every role prompt ends by telling the agent to call
    ``append_<role>_progress``. That tool is real — the in-process transport
    hands it to the session it spawns — but **this** transport does not,
    because there is no session to hand it to. The shared
    `decision_file_block` that follows says to write the decision file
    instead, so the prompt arrives carrying two different endings and only
    the second one works here.

    Three auditor turns in a row noticed the contradiction and picked
    correctly, and all three reported it. That is a judgement none of them
    should have had to make at the end of a two-hour turn, and the next role
    to face it might pick the other one: calling a missing tool looks from
    the inside like a turn that ended, so the run would block for its full
    timeout on a turn that believed it was finished.

    Correcting the role prompts would be the wrong repair — the instruction
    is right for the transport they were written for. The difference belongs
    to the transport, so the transport states it.
    """
    return (
        "\n\n---\n\n"
        "## This turn runs on the external transport\n\n"
        f"Your role prompt above tells you to call `append_{role}_progress`. "
        "**That tool is not available here** — it belongs to the in-process "
        "transport, which spawns a session to hand it to. This transport has no "
        "session: it writes your turn to disk and blocks on your answer.\n\n"
        "So ignore that instruction and follow the decision-file instruction "
        "that comes after it. Writing the decision file with `Write`, then "
        "`<id>.result.md`, then `<id>.done`, is the **only** way a verdict "
        "reaches the orchestrator from here. Everything else in your role "
        "prompt applies unchanged.\n"
    )


class ExternalAgentTransport(FileDispatchTurnExecutor):
    """The file-dispatch transport, carrying Staircase's role-prompt note.

    Call it exactly where an ``AgentLayer`` would be called::

        result = transport(role="smith", what="onboard foo", prompt=text)

    or, from the orchestrator's transport-independent path, through
    ``run(TurnRequest(...))``. Both are the same turn.
    """

    def __init__(
        self,
        workspace: Path | str,
        *,
        poll_s: float = 5.0,
        timeout_s: float | None = None,
        role_timeouts: Mapping[str, float] | None = None,
        heartbeat_s: float | None = None,
        source_embargo: Sequence[str] = (),
        repo_root: Path | str | None = None,
    ) -> None:
        # A driver-run agent is outside this process's tool hooks, so the
        # embargo they enforce in process is restated here as an instruction.
        embargo_note = embargo_transport_note(tuple(source_embargo), repo_root=repo_root)
        # An explicit `timeout_s` is one bound for every role, as it always
        # was; without one each role gets its own default.
        timeouts = dict(DEFAULT_ROLE_TURN_TIMEOUTS) if timeout_s is None else {}
        timeouts.update(role_timeouts or {})
        super().__init__(
            workspace,
            poll_s=poll_s,
            timeout_s=DEFAULT_TURN_TIMEOUT_S if timeout_s is None else timeout_s,
            transport_note=lambda role: staircase_transport_note(role) + embargo_note,
            role_timeouts=timeouts,
            orchestrator_heartbeat_s=heartbeat_s,
            adopt_across_decisions=True,
        )


__all__ = [
    "DEFAULT_ROLE_TURN_TIMEOUTS",
    "DEFAULT_TURN_TIMEOUT_S",
    "EXTERNAL_MAX_PARALLEL_SMITHS",
    "ORCHESTRATOR_HEARTBEAT_S",
    "DISPATCH_DIRNAME",
    "ExternalAgentTransport",
    "ExternalTransportTimeout",
    "HEARTBEAT_STALE_S",
    "staircase_transport_note",
]
