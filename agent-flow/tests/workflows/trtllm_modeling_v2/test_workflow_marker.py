# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The target marker: a fingerprint of what the gates measure, over a real checkout."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from .test_workflow import _git

_FILES = {
    "tensorrt_llm/models/target.py": "T = 1\n",
    "tensorrt_llm/README.md": "the package\n",
    "tests/integration/defs/accuracy/references/gsm8k.yaml": "m: [{accuracy: 50}]\n",
    "examples/llm-api/quickstart_advanced.py": "print('q')\n",
}


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout"
    for name, text in _FILES.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _workflow(repo: Path) -> TrtllmModelingV2Workflow:
    workflow = TrtllmModelingV2Workflow(workspace=repo / "ws")
    workflow.task_data = {
        "trtllm_repo_path": str(repo),
        "accuracy": {"reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml"},
    }
    return workflow


def _commit(repo: Path, name: str, text: str) -> None:
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"change {name}")


@pytest.mark.parametrize(
    "name,seen",
    [
        ("tests/unittest/test_new.py", False),
        ("tensorrt_llm/README.md", False),
        ("examples/llm-api/quickstart_advanced.py", False),
        ("tensorrt_llm/models/target.py", True),
        ("tensorrt_llm/llmapi/llm.py", True),
    ],
)
def test_only_a_commit_a_gate_can_see_changes_the_marker(checkout, name, seen):
    workflow = _workflow(checkout)
    before = workflow._target_marker()
    assert before.startswith("tree:")
    _commit(checkout, name, "edited\n")
    assert (workflow._target_marker() != before) is seen
