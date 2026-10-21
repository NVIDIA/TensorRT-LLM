# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``source_embargo`` entries: what they may name, and how the role prompts show them."""

from __future__ import annotations

import pytest

REL_FILE = "tensorrt_llm/_torch/models/modeling_famv4.py"


@pytest.fixture
def tree(tmp_path):
    repo = tmp_path / "repo"
    models = repo / "tensorrt_llm" / "_torch" / "models"
    models.mkdir(parents=True)
    (models / "modeling_famv4.py").write_text("SECRET = 'the existing implementation'\n")
    sibling = tmp_path / "vllm"
    (sibling / "vllm" / "model_executor").mkdir(parents=True)
    return repo, sibling


def test_package_relative_entries_name_the_workflows_own_files(tree, tmp_path):
    """`package:<path>` is relative to the workflow package, wherever it is installed."""
    from agent_flow.workflows.trtllm_modeling_v2.onboard.embargo import (
        PACKAGE_ROOT,
        display_entry,
        resolve_entry,
    )

    repo, _ = tree
    entry = "package:task.example.yaml"
    assert resolve_entry(entry) == PACKAGE_ROOT / "task.example.yaml"
    target = PACKAGE_ROOT / "task.example.yaml"
    # The prefix is kept for the prompt's own wording; outside the repository
    # the path is absolute, inside it repo-relative.
    assert display_entry(entry, repo) == f"package:{target}"
    assert (
        display_entry(entry, PACKAGE_ROOT.parent) == "package:trtllm_modeling_v2/task.example.yaml"
    )
    assert display_entry(REL_FILE, repo) == REL_FILE


def test_the_schema_accepts_package_entries_that_exist_and_only_those(tree):
    from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import _validate_source_embargo

    repo, _ = tree
    errors: list[str] = []
    _validate_source_embargo(["package:task.example.yaml", REL_FILE], errors, str(repo))
    assert errors == []
    _validate_source_embargo(["package:examples/no-such-model.md"], errors, str(repo))
    assert len(errors) == 1 and "workflow package" in errors[0]


@pytest.mark.parametrize(
    "entry,why",
    [
        ("package:../perf_analyze", "outside the workflow package"),
        ("package:examples/../../perf_analyze", "outside the workflow package"),
        ("package:../../workflows", "outside the workflow package"),
        ("package:", "whole workflow package"),
        ("package:.", "whole workflow package"),
    ],
)
def test_a_package_entry_may_not_escape_the_package(tree, entry, why):
    """Unchecked, `package:../perf_analyze` resolves to the sibling workflow and validates."""
    from agent_flow.workflows.trtllm_modeling_v2.onboard.embargo import (
        EmbargoEntryError,
        resolve_entry,
    )
    from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import _validate_source_embargo

    repo, _ = tree
    errors: list[str] = []
    _validate_source_embargo([entry], errors, str(repo))
    assert len(errors) == 1 and why in errors[0], errors
    with pytest.raises(EmbargoEntryError, match=why):
        resolve_entry(entry)


def test_the_role_prompt_calls_a_package_entry_reference_material(tree):
    """Not "this checkpoint already has an implementation in this repository"."""
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import source_embargo_block
    from agent_flow.workflows.trtllm_modeling_v2.onboard.embargo import (
        PACKAGE_ROOT,
        display_entries,
    )

    repo, _ = tree
    inside = {
        "trtllm_repo_path": str(PACKAGE_ROOT.parent),
        "source_embargo": ["package:task.example.yaml"],
    }
    block = source_embargo_block(display_entries(inside))
    assert "- `trtllm_modeling_v2/task.example.yaml` (workflow reference material)" in block
    assert "reference material about this model" in block
    assert "already has an implementation in this repository" not in block
    assert "Running the in-repository implementation" not in block

    mixed = {
        "trtllm_repo_path": str(repo),
        "source_embargo": [REL_FILE, "package:task.example.yaml"],
    }
    block = source_embargo_block(display_entries(mixed))
    assert "already has an implementation in this repository" in block
    assert "reference material about this model" in block
    assert f"- `{REL_FILE}`\n" in block
    assert f"- `{PACKAGE_ROOT / 'task.example.yaml'}` (workflow reference material)" in block
