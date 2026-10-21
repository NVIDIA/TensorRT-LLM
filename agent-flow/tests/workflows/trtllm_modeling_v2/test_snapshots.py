# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract checks measure read-only snapshots of one commit, never the checkout that moves."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run import snapshots as snapshots_mod

from .test_workflow import _git, _workflow


def _commit(repo: Path, name: str, text: str) -> str:
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"change {name}")
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path) -> Path:
    """A checkout with a package, native code, a quickstart and a built library git ignores."""
    repo = tmp_path / "checkout"
    (repo / "tensorrt_llm" / "libs").mkdir(parents=True)
    (repo / "tensorrt_llm" / "libs" / "libth_common.so").write_bytes(b"\x7fELF")
    (repo / ".gitignore").write_text("*.so\n")
    _git(repo, "init", "-q", ".")
    for name in ("tensorrt_llm/__init__.py", "cpp/kernel.cu", "examples/llm-api/quickstart.py"):
        _commit(repo, name, "x = 1\n")
    return repo


def test_checks_of_one_commit_share_one_read_only_sparse_snapshot_until_none_needs_it(
    tmp_path, repo
):
    head = _git(repo, "rev-parse", "HEAD").strip()
    snapshots = snapshots_mod.Snapshots(repo, tmp_path / "ws" / "snapshots")
    with snapshots.lease(head) as tree, snapshots.lease(head) as again:
        assert again == tree and _git(tree, "rev-parse", "HEAD").strip() == head
        package = tree / "tensorrt_llm" / "__init__.py"
        assert package.is_file() and not (tree / "cpp").exists()
        assert not os.stat(package).st_mode & 0o222
        assert (tree / "tensorrt_llm" / "libs").is_symlink()
        # A file of the checkout the spec names by its path is the commit's; one it does not track is not.
        command = f"{repo}/.venv/bin/python {repo}/examples/llm-api/quickstart.py"
        spec = snapshots_mod.CheckTree(commit=head, tree=tree).spec(
            {"trtllm_repo_path": str(repo), "accuracy": {"leg_b": command}},
            snapshots.tracked(head),
        )
        assert spec["trtllm_repo_path"] == str(tree)
        assert spec["accuracy"]["leg_b"] == command.replace(f"{repo}/examples", f"{tree}/examples")
        _commit(repo, "tensorrt_llm/later.py", "y = 2\n")
        assert snapshots.prune() == []  # held by a running check
    assert not tree.exists()  # given back, and the checkout has moved past it
    later = _git(repo, "rev-parse", "HEAD").strip()
    with snapshots.lease(later):
        pass
    assert snapshots.tree(later).is_dir()  # the head's waits for the next check
    assert len(_git(repo, "worktree", "list").splitlines()) == 2  # the checkout and that one


@pytest.mark.parametrize("imported_from", ["snapshot", "checkout"])
def test_a_check_measures_one_commit_while_integrations_move_the_checkout(
    tmp_path, repo, monkeypatch, imported_from
):
    head = _git(repo, "rev-parse", "HEAD").strip()
    seen: list[str] = []

    def shell(command, env=None, timeout=0):
        seen.append(command)
        if len(seen) == 1:  # a task lands between the legs
            _commit(repo, "tensorrt_llm/landed.py", "z = 3\n")
        root = command.split("cd ", 1)[1].split(" ", 1)[0] if imported_from == "snapshot" else repo
        return CommandResult(
            returncode=0,
            stdout=f"[mv2flow-resolve] pid=7 rank=0 module=tensorrt_llm "
            f"origin={root}/tensorrt_llm/__init__.py\n",
        )

    def two_legs(environment, **kwargs):
        for leg in ("A", "B"):
            kwargs["runner"](f"leg {leg}", None, 10)
        return gates.GateResult(name="boot", member="tp1", passed=True, detail="legs agree")

    monkeypatch.setattr(gates, "run_boot_gate", two_legs)
    monkeypatch.setattr(gates, "resolve_boot_spec", lambda *args: None)
    workflow = _workflow(tmp_path, shell_runner=shell, git_runner=None)
    workflow.task_data = {"trtllm_repo_path": str(repo)}
    workflow.environment = RunEnvironment(kind="local", python="/py", world_size=1)
    workflow._snapshots = snapshots_mod.Snapshots(repo, workflow.workspace / "snapshots")
    marker = workflow._target_marker()
    check = contract_mod.Check("core", "boot", "tp1", contract_mod.TIER_INNER)

    result = workflow._run_contract_check(check, None)
    status = contract_mod.ContractStatus.load(workflow.workspace)
    verdict = status.record("core", result, marker="the target when the check was queued")

    snapshot = workflow._snapshots.tree(head)
    assert seen and all(command.startswith(f"cd {snapshot} && ") for command in seen)
    assert verdict.marker == result.marker == marker and verdict.tree_state == head[:7]
    assert not snapshot.exists()  # the checkout moved on and nothing holds it
    journal = workflow.journal_path.read_text()
    if imported_from == "snapshot":
        assert result.passed and verdict.imported == [head, head] and not verdict.void
        return
    # Each leg imported the checkout an integration moved: void, measured once more, never passed.
    assert len(seen) == 2 * (snapshots_mod.VOID_RERUNS + 1) and "void verdict" in journal
    assert not result.passed and not verdict.passed and str(repo) in verdict.void
    assert "[VOID] boot@tp1" in result.summary()
    rendered = contract_mod.evaluate(contract_mod.Contract(checks=(check,)), status).render()
    assert f"- VOID {check.key}: " in rendered
