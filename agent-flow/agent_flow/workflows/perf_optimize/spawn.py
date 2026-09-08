"""Starting a sibling campaign, and the only place in this package that may.

:mod:`.gitops` opens by stating that it holds the package's only
``subprocess`` call, so that nothing can reach a shell without passing one
reviewed door. That invariant is why a campaign cannot quietly run a
benchmark of its own instead of going through ``ibc-bench``, and it is worth
keeping exactly as strict as it is.

This module is the second door, and it is narrow on purpose:

- It starts **one kind of process**: ``perf-optimize`` itself. Not a
  benchmark, not a harness command, not a shell. The argv is built from a
  :class:`.disagg_sol.CampaignLaunch`, which was already validated — its
  checkout is unshared, its workspace is its own — so by the time anything
  gets here there is no decision left to make.
- It never interprets the child's output. A campaign reports through its
  workspace, the same way it does when a person starts it; reading a pipe
  would make this module a second, worse channel for results.

**Why a process rather than a call.** The two campaigns could be run in
this process by importing the workflow twice, and that would need no new
door at all. It would also give up the property that made the hand-started
pair safe: separate processes cannot corrupt each other's state, one dying
does not take the other with it, and each is independently attachable and
killable. Those were not incidental in the run this module is modelled on —
the two campaigns landed on nine and one cluster nodes respectively, ran for
three and a half hours, and neither could have observed the other if it had
tried. A shared interpreter would have made that a claim rather than a fact.

**What is deliberately not here.** No retry, no backoff, no supervision
loop. A campaign that dies leaves a workspace that says how far it got and
resumes from it; a supervisor that restarted it automatically would be
guessing that the failure was transient, and the failures actually observed
in this stack — a corrupted wheel download, a hung fill gate — are not.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from agent_flow.workflows.perf_optimize.disagg_sol import CampaignLaunch, DisaggSolError


def materialize(launch: CampaignLaunch) -> Path:
    """Write the campaign's ``task.yaml`` and return where it landed.

    Separate from :func:`start` so the spec a campaign will run under is on
    disk, and reviewable, before any process exists — and so a dry run is
    the same code path minus one call.
    """
    launch.workspace.mkdir(parents=True, exist_ok=True)
    launch.task_path.write_text(
        yaml.safe_dump(launch.spec, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return launch.task_path


def start(launch: CampaignLaunch, *, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start one campaign and return without waiting for it.

    Not waiting is the point: the two halves are independent under this
    module's scope, so serializing them would add hours of wall clock and buy
    nothing. The caller holds the handles and decides when to join.
    """
    if not launch.task_path.is_file():
        raise DisaggSolError(
            f"{launch.task_path} does not exist — call materialize() before start(), "
            f"so that what a campaign runs under is on disk before it runs."
        )
    return subprocess.Popen(  # noqa: S603 - argv is built, never a shell string
        launch.argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )


def start_all(
    launches: Iterable[CampaignLaunch], *, env: dict[str, str] | None = None
) -> list[tuple[CampaignLaunch, subprocess.Popen]]:
    """Materialize every campaign, then start every campaign.

    In two passes rather than one, so a spec that cannot be written stops the
    run before any process exists. Half-started is the one state a supervisor
    must not produce: the campaign that did start would claim a checkout the
    other was going to need, and the failure would surface as a repo-claim
    refusal minutes later, pointing at the wrong cause.
    """
    listed = list(launches)
    for launch in listed:
        materialize(launch)
    return [(launch, start(launch, env=env)) for launch in listed]


def wait_all(started: Sequence[tuple[CampaignLaunch, subprocess.Popen]]) -> dict[str, int]:
    """Join every campaign, returning each track's exit status.

    Every one is waited on even after one fails: the other is still running
    on a cluster, and abandoning it would leave an allocation and a claimed
    checkout behind with nothing tracking either.
    """
    return {launch.track: process.wait() for launch, process in started}
