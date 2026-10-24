# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the role prompts and the bundle that scopes them to a run."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.agents import prompts as prompts_mod
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import (
    DEFAULT_PROMPTS,
    PromptBundle,
    build_trtllm_modeling_v2_prompts,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment

ROLES = tuple(f.name for f in dataclasses.fields(DEFAULT_PROMPTS))

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
    # The pre-merge modeling_v2 draft: an entry's op call is `__call__`, there
    # is no bind layer, the catalog is its entry files, and a target's
    # weights.py and modeling.py carry their own classes.
    "raw_call",
    "bind_const",
    "bind_layered",
    "index.yaml",
    "index.d/",
    "ModelWeights",
    "phase_of",
    "_target.py",
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
        "tuner",
        "planner",
        "improver",
    )


def test_every_prompt_is_substantial():
    for role in ROLES:
        assert len(getattr(DEFAULT_PROMPTS, role)) > 2000, role


def test_every_prompt_tells_its_role_to_write_a_decision_file():
    """The typed decision is the only channel a verdict reaches the orchestrator."""
    for role in ROLES:
        prompt = getattr(DEFAULT_PROMPTS, role)
        assert "decision file" in prompt, role
        assert "append_" not in prompt.replace("append_paged", ""), role


def test_no_prompt_names_a_retired_product():
    for role in ROLES:
        text = getattr(DEFAULT_PROMPTS, role)
        for retired in RETIRED:
            assert retired not in text, f"{role} still names {retired!r}"


def test_every_prompt_that_reads_the_tree_carries_its_layout():
    # The improver reads the run's records, not the tree.
    for role in set(ROLES) - {"improver"}:
        assert "_experimental/modeling_v2" in getattr(DEFAULT_PROMPTS, role), role


def test_every_role_has_a_red_flags_table():
    """Each row has an incident behind it; the planner has none yet."""
    for role in set(ROLES) - {"planner"}:
        assert "## Red flags" in getattr(DEFAULT_PROMPTS, role), role
        assert "| Impulse | Reality |" in getattr(DEFAULT_PROMPTS, role), role


def test_the_test_naming_convention_reaches_the_roles_that_write_tests():
    for role in ("smith", "assembler", "reviewer"):
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
    for member in ("`__call__`", "`reference`", "`is_valid`", "`CELLS`", "`ARCHS`"):
        assert member in smith
    for gone in ("raw_call", "bind_const", "`compare`", "index.yaml", "index.d"):
        assert gone not in smith, gone
    assert "one `OpWrapper` subclass per file" in smith
    assert "/ops/<category>/<name>.py" in smith
    assert "models/<family>/<ckpt>__<arch>__<parallel>/<category>/test_modeling_v2_<name>.py" in (
        smith
    )
    assert "catalog/torch/" not in smith.replace("there is no `catalog/torch/`", "")


def test_the_runtime_dump_incident_reaches_the_roles_that_can_catch_it():
    """34 of 35 batch sizes outside the contract — the reviewer reads for it."""
    for role in ("reviewer",):
        assert "34 of the 35" in getattr(DEFAULT_PROMPTS, role), role


def test_the_source_boundary_survives_the_hook_removal():
    """The hook is gone; the rule it defended is prose in every writing role."""
    for role in ("plan_drafter", "smith", "assembler"):
        text = getattr(DEFAULT_PROMPTS, role)
        assert "never to copy a structure" in text, role
    assert "ordinary read" in DEFAULT_PROMPTS.smith


def test_the_reviewer_runs_no_gate_and_knows_accuracy_comes_after_it():
    """A unit review is dispatched after boot and before the accuracy gate, and runs nothing."""
    assert "A unit review runs nothing" in DEFAULT_PROMPTS.reviewer
    assert "accuracy gate runs after your verdict" in DEFAULT_PROMPTS.reviewer
    # A regression review runs what its own dispatch says, not the unit review's rule.
    assert "## A regression review" in DEFAULT_PROMPTS.reviewer
    assert "run what it says and nothing else of the suite" in DEFAULT_PROMPTS.reviewer


def test_the_plan_phase_is_told_it_does_not_block_the_build():
    assert "not** a failure" in DEFAULT_PROMPTS.plan_drafter
    assert "The build does not wait for you" in DEFAULT_PROMPTS.plan_reviewer
    assert "cap" not in DEFAULT_PROMPTS.plan_reviewer.split("## What to check")[0]


def test_the_assembler_is_told_not_to_reformat():
    assert "lints the branch once" in DEFAULT_PROMPTS.assembler


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
    assert extended.reviewer == DEFAULT_PROMPTS.reviewer


def test_the_bundle_is_frozen():
    with pytest.raises(Exception):
        DEFAULT_PROMPTS.smith = "mutated"  # type: ignore[misc]


def test_replace_swaps_one_prompt_outright():
    swapped = DEFAULT_PROMPTS.replace(reviewer="tiny")
    assert swapped.reviewer == "tiny"
    assert swapped.smith == DEFAULT_PROMPTS.smith


def test_the_environment_block_reaches_every_role():
    """A role that was not told the rule is a role that invents a launcher command."""
    bundle = build_trtllm_modeling_v2_prompts(environment=_env())
    for role in ROLES:
        text = getattr(bundle, role)
        assert "CUDA_VISIBLE_DEVICES=0,1" in text, role
        assert "symptom to report" in text, role


def test_without_an_environment_the_bundle_is_the_default():
    assert build_trtllm_modeling_v2_prompts() == DEFAULT_PROMPTS


def test_default_prompts_are_the_module_constants_plus_their_decision_block():
    for role, constant in (
        ("plan_drafter", prompts_mod.PLAN_DRAFTER_SYSTEM_PROMPT),
        ("smith", prompts_mod.SMITH_SYSTEM_PROMPT),
        ("reviewer", prompts_mod.REVIEWER_SYSTEM_PROMPT),
    ):
        prompt = getattr(DEFAULT_PROMPTS, role)
        assert prompt.startswith(constant.rstrip()), role
        assert "## The decision file" in prompt[len(constant.rstrip()) :], role
    assert DEFAULT_PROMPTS.tuner == prompts_mod.TUNER_SYSTEM_PROMPT


def test_bundle_equality_is_by_value():
    assert PromptBundle(*(getattr(DEFAULT_PROMPTS, r) for r in ROLES)) == DEFAULT_PROMPTS


def test_every_role_is_told_how_to_write_down_what_it_read():
    """The rule has to reach the role, not just the contract-writing skill.

    A smith writes contracts, but an assembler writes target comments and a
    reviewer writes findings — all of it read back later against an engine
    that has moved. Whoever records a fact needs the rule.
    """
    prompts = build_trtllm_modeling_v2_prompts()
    for role in (
        prompts.plan_drafter,
        prompts.plan_reviewer,
        prompts.smith,
        prompts.assembler,
        prompts.reviewer,
    ):
        assert "Describe the behaviour and name the symbol — never the line" in role
        assert "quoted diagnostic" in role


def test_the_planner_and_its_reviewer_search_flashinfer_before_writing_a_kernel():
    """FlashInfer kernels no trtllm op wraps are tier 3, not tier 4-5 and not a ceiling."""
    drafter, reviewer = DEFAULT_PROMPTS.plan_drafter, DEFAULT_PROMPTS.plan_reviewer
    assert "check the catalog, the op inventory and FlashInfer's API" in drafter
    assert "tier 3 whose `candidate` is `flashinfer.<module>.<name>`" in drafter
    assert "tier: 1 | 2 | 3 | 4 | 5" in drafter
    assert "the survey's FlashInfer API" in reviewer
    assert "A writing row that never looked at FlashInfer is a REJECT" in reviewer


def test_the_tasks_notes_reach_every_role_verbatim():
    bundle = build_trtllm_modeling_v2_prompts(notes="Follow the gpt_oss naming.")
    for role in ROLES:
        text = getattr(bundle, role)
        assert "## Notes from the task" in text, role
        assert "Follow the gpt_oss naming." in text, role
    assert build_trtllm_modeling_v2_prompts(notes="  ") == DEFAULT_PROMPTS


def test_the_smith_hands_back_a_shared_code_defect_at_once():
    smith = " ".join(DEFAULT_PROMPTS.smith.split())
    assert "When the defect is in shared code outside the entry" in smith
    assert "end the turn with `FAILED_RECEIPT` as soon as you have measured that" in smith
    assert "Leave the remaining cells to the re-certification that follows the fix" in smith


def test_the_improver_reads_only_the_history_head_reaches():
    improver = " ".join(DEFAULT_PROMPTS.improver.split())
    assert "read only what its HEAD reaches" in improver
    assert "Never walk its other refs (`git log --all`, `--branches`" in improver


def test_the_core_skill_leaves_cross_model_regression_to_the_review():
    skill = (
        Path(prompts_mod.__file__).parent.parent / "skills" / "trtllm-modeling-v2-core" / "SKILL.md"
    ).read_text(encoding="utf-8")
    text = " ".join(skill.split())
    assert "Run the tests you added or changed and those of the files you touch" in text
    assert "do not run `tests affected` or `tests compare` in your own turn" in text
