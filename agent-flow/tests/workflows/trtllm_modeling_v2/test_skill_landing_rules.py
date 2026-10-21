# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the kernel skills tell a smith agrees with what the orchestrator does.

A FAILED_RECEIPT's guards and tests are integrated when they touch nothing
but the entry (`run/stages/smith.py`); the skills said "land it" while the
orchestrator integrated only finished work, so a smith's refusals never
reached the target. The test budget and the `large` marker are rules every
entry's test follows, so they are read where the smith reads.
"""

from __future__ import annotations

from pathlib import Path

import agent_flow.workflows.trtllm_modeling_v2 as pkg

SKILLS = Path(pkg.__file__).parent / "agents" / "skills"


def _read(*parts: str) -> str:
    return " ".join(SKILLS.joinpath(*parts).read_text(encoding="utf-8").split())


def test_the_kernel_skills_say_which_failed_receipts_are_integrated():
    verify = _read("trtllm-modeling-v2-kernel-verify", "SKILL.md")
    onboard = _read("trtllm-modeling-v2-kernel-onboard", "SKILL.md")
    spec = _read("trtllm-modeling-v2-kernel-onboard", "references", "entry-spec.md")
    assert (
        "a failed receipt whose commits change only the entry's own files is integrated" in verify
    )
    assert "a `FAILED_RECEIPT` whose commits are the entry's files only is integrated" in onboard
    assert "A failed receipt whose commits change only the entry's own files" in spec
    assert "stays on the task's branch, and the task's next attempt starts from it" in spec


def test_the_entry_spec_states_the_cell_budget_and_where_the_marker_lives():
    spec = _read("trtllm-modeling-v2-kernel-onboard", "references", "entry-spec.md")
    assert "**Budget: a new cell fits 10 s and 40 GiB of device memory.**" in spec
    assert "`pytest.param(cell, marks=pytest.mark.large)`" in spec
    assert '-m "not large"' in spec
    assert "registered in `tests/unittest/pytest.ini` under `markers`" in spec
    assert "**One launch for many shapes.**" in spec
    assert "**Iterate on what you change, certify once.**" in spec


def test_the_kernel_skills_iterate_on_cells_and_certify_once():
    verify = _read("trtllm-modeling-v2-kernel-verify", "SKILL.md")
    onboard = _read("trtllm-modeling-v2-kernel-onboard", "SKILL.md")
    assert "**Iterate on the cells you add or change** (`pytest -k`)" in verify
    assert "**Run the entry's whole test file and the claims gates once, at the end**" in verify
    assert "Iterate on the cells you are working on (`-k`)" in onboard
    assert "run the whole test file and the claims gates once" in onboard


def test_the_dispatch_sets_the_certification_scope_and_a_defect_outside_is_handed_back():
    verify = _read("trtllm-modeling-v2-kernel-verify", "SKILL.md")
    onboard = _read("trtllm-modeling-v2-kernel-onboard", "SKILL.md")
    for skill in (verify, onboard):
        assert "certify exactly those cells, plus one" in skill
        assert "goes in the `note` as uncertified" in skill
        assert (
            "`engine_defect` block" in skill
            and "(`affects`, `trigger`, `detect`, `proposal`" in (skill)
        )
    assert "by bucket boundary" in verify
