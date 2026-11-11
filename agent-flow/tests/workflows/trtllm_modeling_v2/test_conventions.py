# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conventions scan, and the delivery checks the certification still runs."""

from __future__ import annotations

import subprocess

from agent_flow.workflows.trtllm_modeling_v2.checks import conventions
from agent_flow.workflows.trtllm_modeling_v2.infra.git import CommitPolicy
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import model_ops_tests_dir
from agent_flow.workflows.trtllm_modeling_v2.run import certify

from ._helpers import _ENTRY, TARGET_REL, _build_repo


def test_the_scan_reads_each_entry_its_test_and_the_targets_v1_imports(tmp_path):
    root = _build_repo(tmp_path)
    assert conventions.scan(root, TARGET_REL, "sm_103") == []
    target = root / TARGET_REL
    router = f"{TARGET_REL}/ops/moe/router.py"
    tests = model_ops_tests_dir(TARGET_REL)
    # Driven on another architecture than the run's.
    assert conventions.scan(root, TARGET_REL, "sm_100") == [
        f"{router}: Entry's ARCHS names sm_103, not sm_100"
    ]
    # A bare function is no entry; one with no cells certifies nothing.
    (root / router).write_text("def router(x):\n    return x\n")
    assert conventions.scan(root, TARGET_REL, "sm_103") == [
        f"{router}: no OpWrapper subclass: an entry is its contract "
        "(__call__, reference, CELLS, ARCHS)"
    ]
    (root / router).write_text(_ENTRY.replace("    CELLS", "    CASES"))
    assert conventions.scan(root, TARGET_REL, "sm_103") == [
        f"{router}: Entry declares no CELLS: nothing is certified"
    ]
    (root / router).write_text(_ENTRY)
    # Its test is there and reads the cells it claims; a family's shared entry is read too.
    test = root / tests / "moe/test_modeling_v2_router.py"
    test.write_text("def test_it():\n    assert True\n")
    assert conventions.scan(root, TARGET_REL, "sm_103") == [
        f"{router}: {tests}/moe/test_modeling_v2_router.py never reads the entry's CELLS: "
        "they were not driven"
    ]
    test.unlink()
    shared = target.parent / "ops/norm/shared.py"
    shared.parent.mkdir(parents=True)
    shared.write_text(_ENTRY)
    family = model_ops_tests_dir(str(target.parent.relative_to(root)))
    assert conventions.scan(root, TARGET_REL, "sm_103") == [
        f"{router}: no test at {tests}/moe/test_modeling_v2_router.py",
        f"{shared.relative_to(root)}: no test at {family}/norm/test_modeling_v2_shared.py",
    ]
    shared.unlink()
    (root / router).unlink()
    # What the target takes from Modeling V1 is said, by file and line.
    modeling = target / "modeling.py"
    modeling.write_text(
        "from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3\n" + modeling.read_text()
    )
    assert conventions.scan(root, TARGET_REL, "sm_103") == [
        f"{TARGET_REL}/modeling.py:1 imports Modeling V1 "
        "(tensorrt_llm._torch.models.modeling_qwen3)"
    ]
    # Kept once, with the task that first showed it.
    workspace = tmp_path / "ws"
    assert conventions.record(workspace, "core", ["a", "b"]) == ["a", "b"]
    assert conventions.record(workspace, "mtp", ["b", "c"]) == ["c"]
    assert conventions.load(workspace) == {"a": "core", "b": "core", "c": "mtp"}


def test_the_certification_still_holds_the_tree_to_git(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    assert certify.git_checks(tmp_path, CommitPolicy(), ("modeling.py",)) == []  # not a checkout
    git("init", "-q")
    (tmp_path / "seed").write_text("s\n", encoding="utf-8")
    git("add", "seed")
    git("-c", "user.email=t@e.st", "-c", "user.name=t", "commit", "-qm", "seed")
    (tmp_path / "modeling.py").write_text("# never committed\n", encoding="utf-8")
    [check] = certify.git_checks(tmp_path, CommitPolicy(), ("modeling.py",))
    assert not check.passed and check.name == "git:target-committed"
    assert "a clone would not have them" in check.detail and "modeling.py" in check.detail
