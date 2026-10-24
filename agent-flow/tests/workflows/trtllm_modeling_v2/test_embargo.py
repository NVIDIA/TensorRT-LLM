# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the role prompts show ``source_embargo`` entries."""

from __future__ import annotations

import pytest

REL_FILE = "tensorrt_llm/_torch/models/modeling_famv4.py"


@pytest.fixture
def tree(tmp_path):
    repo = tmp_path / "repo"
    models = repo / "tensorrt_llm" / "_torch" / "models"
    models.mkdir(parents=True)
    (models / "modeling_famv4.py").write_text("SECRET = 'the existing implementation'\n")
    return repo


def test_the_role_prompt_calls_a_package_entry_reference_material(tree):
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import source_embargo_block
    from agent_flow.workflows.trtllm_modeling_v2.onboard.embargo import (
        PACKAGE_ROOT,
        display_entries,
    )

    repo = tree
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
