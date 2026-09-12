"""Tests for the staircase-bringup prompt bundle.

These lock the two decisions that are easy to undo by accident: QA's rubric is
a *replacement* for the base prompt rather than an extension of it, and
importing the package must not spawn backend sessions.
"""

from __future__ import annotations

import subprocess
import sys

from agent_flow.workflows.agent_team.prompts import DEFAULT_PROMPTS, PromptBundle
from agent_flow.workflows.staircase_bringup import STAIRCASE_PROMPTS, build_staircase_prompts
from agent_flow.workflows.staircase_bringup.prompts.qa import EVALUATION_CRITERIA

# Base-prompt dimensions staircase deliberately drops. Performance is the
# load-bearing one: perf is recorded and never gated here, and PlanDrafter may
# not return DONE below ``min_score``, so an un-scoreable dimension dragging
# the weighted average down deadlocks the workflow rather than merely adding
# noise.
_DROPPED_BASE_DIMENSIONS = {"Performance", "Code Quality", "Technical Sophistication"}


def _flat(text: str) -> str:
    """Collapse whitespace so a prose assertion survives the prompt's line wrapping.

    Prompt bodies are hard-wrapped, so a phrase that reads as one sentence is
    often split across a newline. Asserting on the raw text makes these checks
    fail for formatting reasons rather than content ones.
    """
    return " ".join(text.split())


def test_evaluation_criteria_shape():
    """Five dimensions, weights summing to 10, none of the dropped base ones."""
    names = [c["name"] for c in EVALUATION_CRITERIA]
    assert len(names) == len(set(names)) == 5
    assert sum(c["weight"] for c in EVALUATION_CRITERIA) == 10.0
    assert set(names).isdisjoint(_DROPPED_BASE_DIMENSIONS)
    assert names[0] == "Gate Outcome", "the release criterion must lead the rubric"


def test_every_criterion_reaches_the_qa_prompt():
    """Each dimension's name and weight must render into the prompt text."""
    for c in EVALUATION_CRITERIA:
        assert f"**{c['name']}** (weight {c['weight']})" in STAIRCASE_PROMPTS.qa


def test_qa_prompt_is_a_replacement_not_an_extension():
    """QA must not be the agent_team base with staircase text appended."""
    assert not STAIRCASE_PROMPTS.qa.startswith(DEFAULT_PROMPTS.qa[:200])
    # The base rubric's prose must be gone, not merely outweighed.
    assert "Technical Sophistication" not in STAIRCASE_PROMPTS.qa


def test_qa_prompt_keeps_the_orchestrator_contract():
    """Tool-facing contracts are shared with the base and must survive the swap."""
    for token in (
        "append_qa_progress",
        "read_human_feedback",
        "`APPROVE`",
        "`REJECT`",
        "weighted_score",
        "acceptance-criteria.md",
    ):
        assert token in STAIRCASE_PROMPTS.qa, token
    assert "read_latest_progress" in STAIRCASE_PROMPTS.qa, "the no-such-tool rule must be stated"


def test_gate_overrides_the_weighted_average():
    """The accuracy gate is the release criterion, not one term in an average."""
    assert "your `decision` is REJECT" in _flat(STAIRCASE_PROMPTS.qa)
    assert "There is no Performance dimension" in _flat(STAIRCASE_PROMPTS.qa)


def test_other_roles_extend_rather_than_replace():
    """The four extended roles must keep the agent_team base as a strict prefix.

    ``with_extensions`` appends, so the base prompt surviving verbatim at the
    front is what distinguishes an extension from an accidental replacement.
    QA is the deliberate exception, covered above.
    """
    for role in ("plan_drafter", "plan_reviewer", "coder", "reviewer"):
        base = getattr(DEFAULT_PROMPTS, role)
        built = getattr(STAIRCASE_PROMPTS, role)
        assert built.startswith(base.rstrip()), role
        assert len(built) > len(base), role


def test_each_role_carries_the_blocks_its_matrix_row_claims():
    """Spot-check the block-to-role matrix: a dropped import is otherwise silent."""
    expected = {
        "plan_drafter": [
            "Staircase bring-up frame",
            "closed-vocabulary rule",
            "required mechanisms",
            "Catalog entry spec",
            "Target products",
            "gate ladder",
            "reference ladder",
        ],
        "plan_reviewer": [
            "Staircase bring-up frame",
            "closed-vocabulary rule",
            "required mechanisms",
            "Validation standard",
        ],
        "coder": [
            "Staircase bring-up frame",
            "Receipts",
            "Lint",
            "Signal versus noise",
            "Done / TODO",
        ],
        "reviewer": ["Receipts", "Lint", "Signal versus noise", "Done / TODO"],
    }
    for role, fragments in expected.items():
        prompt = getattr(STAIRCASE_PROMPTS, role)
        for fragment in fragments:
            assert fragment in prompt, f"{role} is missing {fragment!r}"


def test_planner_side_gets_the_outcome_bound_escape_hatch():
    """Without it, the first closed-vocabulary criterion the planner writes is REJECTed.

    Criteria are normally outcome-bound, but staircase's central criterion
    names a mechanism by construction. Both the drafter and its reviewer need
    the carve-out, or they deadlock on format rather than content.
    """
    for role in ("plan_drafter", "plan_reviewer"):
        prompt = _flat(getattr(STAIRCASE_PROMPTS, role))
        assert "Project-level required mechanisms" in prompt, role
        assert "leaked prescription" in prompt, role
    # The reviewer additionally needs the explicit do-not-reject instruction,
    # since it is the side that would otherwise fire.
    assert "do not REJECT a criterion for naming them" in _flat(STAIRCASE_PROMPTS.plan_reviewer)


def test_anchor_is_read_only_on_both_planning_sides():
    """The release criterion must not be movable by the agents that could move it."""
    assert "not yours to move" in _flat(STAIRCASE_PROMPTS.plan_drafter)
    assert "lowers the accuracy anchor" in _flat(STAIRCASE_PROMPTS.plan_reviewer)
    assert "never the bar" in _flat(STAIRCASE_PROMPTS.coder)


def test_replan_flag_defaults_to_enabled():
    """Stage/Goal control flow carries staircase's per-Goal atomicity; default it on."""
    assert build_staircase_prompts() == build_staircase_prompts(replan_on_qa=True)
    assert isinstance(STAIRCASE_PROMPTS, PromptBundle)


def test_slurm_blocks_are_task_scoped():
    """Slurm guidance reaches every role only when the task asked for it."""
    plain = build_staircase_prompts()
    slurm = build_staircase_prompts(include_slurm_environment=True)
    for role in ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"):
        assert len(getattr(slurm, role)) > len(getattr(plain, role)), role
    # Coder/Reviewer/QA additionally get the verified-command cache.
    assert "test_command.md" in slurm.coder
    assert "test_command.md" not in slurm.plan_drafter


def test_importing_the_package_spawns_no_backend_session():
    """Importing this package must not pull in another workflow's prompts.

    ``modeling_bringup.prompts._common`` calls its skill probe at module
    scope, so importing it starts one backend session per configured backend
    — measured at 5.07s against 1.16s for the ``agent_team`` prompts alone.
    The staircase Slurm blocks are local copies specifically so that cost, and
    the coupling behind it, stay out of this package. This guards the
    regression where someone replaces the copies with an import to
    de-duplicate them.

    Asserted in a clean interpreter, since this test session has already
    imported everything.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import agent_flow.workflows.staircase_bringup as s; "
            "print(any(m.startswith('agent_flow.workflows.modeling_bringup') "
            "for m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "False", (
        "importing staircase_bringup pulled in modeling_bringup — the Slurm "
        "borrow regressed to a module-scope import and now probes at import time"
    )
