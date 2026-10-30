# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worked examples stay examples: valid spec fragments that no role is handed.

`examples/` holds one checkpoint's values for the generic features. The
skills, the registries and the example spec describe the features without
naming a model, because a run that onboards that model to measure the workflow
must derive those values itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

from .test_workflow import _workflow

PACKAGE = Path(task_schema.__file__).parents[1]
EXAMPLES = sorted((PACKAGE / "examples").glob("*.md"))
YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.S)

#: What a role is handed or can be pointed at by the run itself, and the
#: documents the package ships beside it.
ROLE_FACING = (
    *sorted((PACKAGE / "agents" / "skills").rglob("*.md")),
    PACKAGE / "onboard" / "runtime_index.yaml",
    PACKAGE / "onboard" / "kernel_sources.yaml",
    PACKAGE / "task.example.yaml",
    *sorted((PACKAGE / "agents" / "prompts").glob("*.py")),
    PACKAGE / "README.md",
    PACKAGE / "LESSONS.md",
    PACKAGE / "TROUBLESHOOTING.md",
    PACKAGE / "BUILD_AND_TEST.md",
)

#: Identifiers of the checkpoints the worked examples describe.
EXAMPLE_CHECKPOINT_MARKERS = ("V4.1", "DSV41", "dsv4.1", "deepseek_v41")


def test_there_is_a_worked_example():
    assert EXAMPLES


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_spec_fragment_in_an_example_uses_real_keys(example):
    """A fragment an operator copies must not fail validation on a misspelled key."""
    blocks = YAML_BLOCK.findall(example.read_text(encoding="utf-8"))
    assert blocks, f"{example.name} has no yaml fragment"
    for block in blocks:
        data = yaml.safe_load(block)
        assert isinstance(data, dict), block
        assert set(data) <= set(task_schema.TOP_LEVEL_KEYS), block
        accuracy = data.get("accuracy") or {}
        assert set(accuracy) <= set(task_schema.ACCURACY_KEYS), block
        for feature in data.get("features") or ():
            assert feature in task_schema.FEATURE_SKILLS, feature
        for feature in accuracy.get("witness") or {}:
            assert feature in task_schema.FEATURE_SKILLS, feature


def test_a_run_materializes_skills_and_never_examples(tmp_path):
    workflow = _workflow(tmp_path)
    destination = workflow._materialize_skills()
    copied = {path.name for path in destination.rglob("*")}
    assert copied
    assert not copied & {path.name for path in EXAMPLES}


@pytest.mark.parametrize("path", ROLE_FACING, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_role_facing_material_names_no_example_checkpoint(path):
    text = path.read_text(encoding="utf-8")
    found = [marker for marker in EXAMPLE_CHECKPOINT_MARKERS if marker in text]
    assert not found, f"{path.relative_to(PACKAGE)} names {found}; move it to examples/"
