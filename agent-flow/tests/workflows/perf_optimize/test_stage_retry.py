"""Giving a stage that wrote nothing another turn.

`_require_stage_outputs` raises before the checkpoint advances, and its
docstring already names the recovery: "simply re-running the workflow retries
the same stage". That sentence was addressed to a person. Unattended campaigns
have no person, so a role that ended its turn early destroyed every completed
round behind it -- in the run this was written for, six hours of work and a
+5.67% cumulative gain, twenty-one minutes before the Slurm job it was waiting
for completed successfully.

These tests are about the loop, not the workflow: a real one is hours of GPU
time, so the subject here is a stand-in that raises on cue. What is being
pinned is which failures are retried, how many times, and against what the
budget is counted.
"""

from __future__ import annotations

import pytest

from agent_flow.workflows.perf_optimize.cli import _run_with_stage_retries
from agent_flow.workflows.perf_optimize.workflow import StageOutputsMissing


class FakeWorkflow:
    """Replays a scripted sequence of outcomes, one per ``run`` call.

    ``positions`` is what the checkpoint reports after each failure, which is
    the thing the budget is keyed on -- so a test can say "failed twice at the
    same stage" or "failed once here, once there" and mean it.
    """

    def __init__(self, outcomes, positions=None):
        self.outcomes = list(outcomes)
        self.positions = list(positions or [])
        self.calls = 0

    def run(self, task):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if outcome is not None:
            raise outcome

    def checkpoint_position(self):
        return self.positions.pop(0) if self.positions else ("analyzer", 0, 0, 0)


def _missing(stage="analyzer", name="profile_findings.md"):
    return StageOutputsMissing(f"{stage} left {name}", stage=stage, missing=(name,))


def test_a_stage_that_wrote_nothing_is_given_another_turn():
    """The whole point: one early turn no longer ends the campaign."""
    wf = FakeWorkflow([_missing(), None])

    _run_with_stage_retries(wf, "task.yaml", budget=1)

    assert wf.calls == 2


def test_the_budget_is_spent_and_then_the_failure_surfaces():
    """Retrying is not the same as retrying forever.

    A stage that cannot succeed -- a role wedged on a cluster that is down --
    must still end the run, and with its own error rather than a timeout.
    """
    wf = FakeWorkflow([_missing(), _missing()])

    with pytest.raises(StageOutputsMissing):
        _run_with_stage_retries(wf, "task.yaml", budget=1)

    assert wf.calls == 2


def test_the_budget_is_per_position_not_per_run():
    """Two unrelated hiccups in a long campaign are not "the same failure".

    A campaign is dozens of stages over many rounds. Counted globally, a budget
    of one would let a round-2 stumble consume the allowance that a round-7
    stumble needs, and the second failure would end a run that was fine. The
    count therefore resets whenever the checkpoint moves.
    """
    wf = FakeWorkflow(
        [_missing(), _missing(), None],
        positions=[("analyzer", 1, 0, 0), ("qa", 6, 0, 0)],
    )

    _run_with_stage_retries(wf, "task.yaml", budget=1)

    assert wf.calls == 3


def test_a_stage_stuck_in_one_place_still_stops():
    """The flip side of per-position: it must not become unbounded.

    Same position twice with a budget of one is the stuck case, and it has to
    stop there even though the campaign as a whole has retries left elsewhere.
    """
    wf = FakeWorkflow(
        [_missing(), _missing()],
        positions=[("analyzer", 3, 0, 0), ("analyzer", 3, 0, 0)],
    )

    with pytest.raises(StageOutputsMissing):
        _run_with_stage_retries(wf, "task.yaml", budget=1)


def test_only_a_missing_deliverable_is_retried():
    """A deliverable that EXISTS but is invalid replays identically.

    Roadmap schema failures are the real instance: the file is on disk and a
    re-run reads the same bytes, so retrying cannot change the outcome -- it
    would only bury the error under repeated attempts. Two campaigns died that
    way (`baseline.value` None, `expected_gain_pct` 0.0), and both should fail
    on the first report, not the third.
    """
    wf = FakeWorkflow([RuntimeError("roadmap.yaml failed schema validation")])

    with pytest.raises(RuntimeError, match="schema validation"):
        _run_with_stage_retries(wf, "task.yaml", budget=3)

    assert wf.calls == 1


def test_zero_budget_restores_the_previous_behaviour():
    """An operator who wants the old fail-fast contract can still have it."""
    wf = FakeWorkflow([_missing()])

    with pytest.raises(StageOutputsMissing):
        _run_with_stage_retries(wf, "task.yaml", budget=0)

    assert wf.calls == 1
