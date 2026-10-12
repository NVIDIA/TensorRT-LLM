# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the six role prompts and the bundle that scopes them to a run."""

from __future__ import annotations

import dataclasses

import pytest

from agent_flow.workflows.staircase_onboard import prompts as prompts_mod
from agent_flow.workflows.staircase_onboard.environment import resolve_environment
from agent_flow.workflows.staircase_onboard.prompts import (
    DEFAULT_PROMPTS,
    PromptBundle,
    build_staircase_onboard_prompts,
)

ROLES = tuple(f.name for f in dataclasses.fields(DEFAULT_PROMPTS))

# The one role that reports no typed decision, because it makes none: it
# relays a submission to a scheduler the orchestrator cannot reach, and the
# verdict comes from files the job wrote. Every rule below that is about
# reporting a verdict, or about reading the in-tree layout, is therefore not
# about this role.
RUNNER = "runner"
DECIDING_ROLES = tuple(r for r in ROLES if r != RUNNER)

# Things the port retired. A prompt naming one of them is telling an agent to
# use a product that no longer exists, which is worse than saying nothing.
RETIRED = (
    "uv run",
    "ty check",
    "TARGET.md",
    "smoke.py",
    "llm_args.yaml",
    "model_dir",
    "1.3.0rc25",
    ".allow-trtllm",
    "isolation hook",
    "bench/accuracy.py",
    "bench/perf.py",
)


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/opt/venv/bin/python", "env": {}}},
        devices=[0, 1],
        world_size=2,
    )


def test_the_bundle_has_exactly_the_roles_the_workflow_dispatches():
    assert ROLES == (
        "plan_drafter",
        "plan_reviewer",
        "smith",
        "assembler",
        "reviewer",
        "auditor",
        "runner",
    )


def test_every_prompt_is_substantial():
    for role in ROLES:
        assert len(getattr(DEFAULT_PROMPTS, role)) > 2000, role


def test_every_deciding_prompt_names_its_own_progress_tool():
    """The typed decision is the only channel a verdict reaches the orchestrator."""
    for role in DECIDING_ROLES:
        assert f"append_{role}_progress" in getattr(DEFAULT_PROMPTS, role), role


def test_the_runner_is_told_that_nothing_reads_its_answer():
    """It has no progress tool because it has no verdict.

    Stating that in the prompt is what keeps it from inventing one: a role
    that believes it is reporting a result will report a result.
    """
    prompt = DEFAULT_PROMPTS.runner
    assert "append_runner_progress" not in prompt
    assert "nothing reads your answer" in prompt
    assert "trigger, not a witness" in prompt


def test_no_prompt_names_a_retired_product():
    for role in ROLES:
        text = getattr(DEFAULT_PROMPTS, role)
        for retired in RETIRED:
            assert retired not in text, f"{role} still names {retired!r}"


def test_every_prompt_that_reads_the_tree_carries_its_layout():
    """The runner is excluded because it never opens the repository.

    It is handed a script path and a submission call. Telling it where the
    catalog lives would be telling it about work that is not its own, which
    is how a relay starts having opinions.
    """
    for role in DECIDING_ROLES:
        assert "_experimental/modeling_v2" in getattr(DEFAULT_PROMPTS, role), role
    assert "_experimental/modeling_v2" not in DEFAULT_PROMPTS.runner


def test_every_role_has_a_red_flags_table():
    """Each row has an incident behind it; they are the expensive part."""
    for role in ROLES:
        assert "## Red flags" in getattr(DEFAULT_PROMPTS, role), role
        assert "| Impulse | Reality |" in getattr(DEFAULT_PROMPTS, role), role


def test_the_test_naming_convention_reaches_the_roles_that_write_tests():
    for role in ("smith", "assembler", "reviewer", "auditor"):
        assert "test_modeling_v2_" in getattr(DEFAULT_PROMPTS, role), role


def test_the_modeling_v2_switch_reaches_the_roles_that_run_the_engine():
    for role in ("plan_drafter", "assembler"):
        assert "TRTLLM_MODELING_V2" in getattr(DEFAULT_PROMPTS, role), role
        assert "require" in getattr(DEFAULT_PROMPTS, role), role


def test_the_anchor_is_never_invented_reaches_the_planner():
    assert "never invented" in DEFAULT_PROMPTS.plan_drafter
    assert "never invented" in DEFAULT_PROMPTS.plan_reviewer.lower() or (
        "invented" in DEFAULT_PROMPTS.plan_reviewer
    )


def test_certification_is_three_state_in_the_smith_prompt():
    assert "**unknown** — a value outside every cell, or an arch not in `ARCHS`" in (
        DEFAULT_PROMPTS.smith
    )
    assert "Tolerance inflation manufactures green" in DEFAULT_PROMPTS.smith


def test_the_smith_prompt_teaches_the_19977_entry_form():
    """An entry is an OpWrapper class with an executable contract, in one of two homes."""
    smith = DEFAULT_PROMPTS.smith
    for member in ("`raw_call`", "`reference`", "`is_valid`", "`compare`", "`CELLS`", "`ARCHS`"):
        assert member in smith
    assert "there is no `.md` beside it" in smith
    assert "/ops/<category>/<name>.py" in smith
    assert "models/<family>/<ckpt>__<arch>__<parallel>/<category>/test_modeling_v2_<name>.py" in (
        smith
    )
    assert "catalog/torch/" not in smith.replace("there is no `catalog/torch/`", "")


def test_the_runtime_dump_incident_reaches_the_roles_that_can_catch_it():
    """34 of 35 batch sizes outside the contract — the failure the auditor exists for."""
    for role in ("assembler", "reviewer", "auditor"):
        assert "34 of the 35" in getattr(DEFAULT_PROMPTS, role), role


def test_the_source_boundary_survives_the_hook_removal():
    """The hook is gone; the rule it defended is prose in every writing role."""
    for role in ("plan_drafter", "smith", "assembler"):
        text = getattr(DEFAULT_PROMPTS, role)
        assert "never to copy a structure" in text, role
    assert "ordinary read" in DEFAULT_PROMPTS.smith


def test_the_reviewer_is_told_not_to_re_run_the_gates():
    assert "Do not re-run the gates" in DEFAULT_PROMPTS.reviewer
    assert "already exists" in DEFAULT_PROMPTS.reviewer


def test_the_plan_phase_is_told_it_does_not_block_the_build():
    assert "not** a failure" in DEFAULT_PROMPTS.plan_drafter
    assert "the build is not blocked on you" in DEFAULT_PROMPTS.plan_reviewer.lower() or (
        "not blocked on you" in DEFAULT_PROMPTS.plan_reviewer
    )


def test_the_assembler_is_told_not_to_reformat():
    assert "lints once" in DEFAULT_PROMPTS.assembler


# ------------------------------------------------------------------- bundling


def test_empty_extensions_leave_the_bundle_unchanged():
    extended = DEFAULT_PROMPTS.with_extensions(smith="", assembler="   \n ")
    for role in ROLES:
        assert getattr(extended, role) == getattr(DEFAULT_PROMPTS, role)


def test_a_non_empty_extension_is_appended_after_one_blank_line():
    extended = DEFAULT_PROMPTS.with_extensions(smith="## Extra\nSomething.\n")
    assert extended.smith.startswith(DEFAULT_PROMPTS.smith.rstrip())
    assert extended.smith.endswith("\n\n## Extra\nSomething.\n")
    assert "\n\n\n" not in extended.smith
    assert extended.auditor == DEFAULT_PROMPTS.auditor


def test_the_bundle_is_frozen():
    with pytest.raises(Exception):
        DEFAULT_PROMPTS.smith = "mutated"  # type: ignore[misc]


def test_replace_swaps_one_prompt_outright():
    swapped = DEFAULT_PROMPTS.replace(auditor="tiny")
    assert swapped.auditor == "tiny"
    assert swapped.smith == DEFAULT_PROMPTS.smith


def test_the_environment_block_reaches_every_role():
    """A role that was not told the rule is a role that invents a launcher command."""
    bundle = build_staircase_onboard_prompts(environment=_env())
    for role in ROLES:
        text = getattr(bundle, role)
        assert "CUDA_VISIBLE_DEVICES=0,1" in text, role
        assert "symptom to report" in text, role


def test_without_an_environment_the_bundle_is_the_default():
    assert build_staircase_onboard_prompts() == DEFAULT_PROMPTS


def test_full_reviewer_mode_adds_its_extension_only_to_the_reviewer():
    bundle = build_staircase_onboard_prompts(reviewer_mode="full")
    assert "full` reviewer mode" in bundle.reviewer
    assert bundle.smith == DEFAULT_PROMPTS.smith


def test_light_reviewer_mode_needs_no_extension():
    assert build_staircase_onboard_prompts(reviewer_mode="light") == DEFAULT_PROMPTS


def test_full_mode_still_refuses_to_re_run_the_accuracy_gate():
    """An hour of wall-clock to reproduce a number inside its own noise is delay."""
    bundle = build_staircase_onboard_prompts(reviewer_mode="full")
    assert "not re-run the accuracy gate" in bundle.reviewer


def test_default_prompts_match_the_module_constants():
    assert DEFAULT_PROMPTS.plan_drafter == prompts_mod.PLAN_DRAFTER_SYSTEM_PROMPT
    assert DEFAULT_PROMPTS.smith == prompts_mod.SMITH_SYSTEM_PROMPT
    assert DEFAULT_PROMPTS.auditor == prompts_mod.AUDITOR_SYSTEM_PROMPT


def test_bundle_equality_is_by_value():
    assert PromptBundle(*(getattr(DEFAULT_PROMPTS, r) for r in ROLES)) == DEFAULT_PROMPTS


def test_every_role_is_told_how_to_write_down_what_it_read():
    """The rule has to reach the role, not just the contract-writing skill.

    A smith writes contracts, but an assembler writes target comments and a
    reviewer writes findings — all of it read back later against an engine
    that has moved. Whoever records a fact needs the rule.
    """
    prompts = build_staircase_onboard_prompts()
    for role in (
        prompts.plan_drafter,
        prompts.plan_reviewer,
        prompts.smith,
        prompts.assembler,
        prompts.reviewer,
        prompts.auditor,
    ):
        assert "Describe the behaviour and name the symbol — never the line" in role
        assert "quoted diagnostic" in role


def test_the_planner_and_its_reviewer_search_flashinfer_before_writing_a_kernel():
    """FlashInfer kernels no trtllm op wraps are tier 3, not tier 4-5 and not a ceiling."""
    drafter, reviewer = DEFAULT_PROMPTS.plan_drafter, DEFAULT_PROMPTS.plan_reviewer
    assert "check the catalog, the op inventory and\nFlashInfer's API" in drafter
    assert "tier 3 whose `candidate` is\n`flashinfer.<module>.<name>`" in drafter
    assert "tier: 1 | 2 | 3 | 4 | 5" in drafter
    assert "the survey's FlashInfer API" in reviewer
    assert "A writing row that never looked at FlashInfer is a REJECT" in reviewer
