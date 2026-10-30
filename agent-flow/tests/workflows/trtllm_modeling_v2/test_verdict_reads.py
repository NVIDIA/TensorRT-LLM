# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A verdict holds on a later commit while what its check read of the tree is unchanged.

A run lands work while its checks hold: kernel variants under the target's
``ops/``, a feature's files. Judged by the whole tree, every landing made
every verdict stale and the contract ran again; judged by what each check's
processes imported, a landing the check never read leaves it standing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import snapshots as snap
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

from .test_workflow import _git

# ---------------------------------------------------------------- the rank hook


def test_every_process_writes_the_trees_modules_it_imported(tmp_path):
    tree = tmp_path / "tree"
    (tree / "pkg").mkdir(parents=True)
    (tree / "pkg" / "__init__.py").write_text("")
    (tree / "pkg" / "used.py").write_text("X = 1\n")
    (tree / "pkg" / "unused.py").write_text("Y = 1\n")
    reads = tmp_path / "reads"
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": os.pathsep.join([str(gates.RANK_HOOK_DIR), str(tree)]),
        snap.READS_DIR_ENV: str(reads),
        snap.TREE_ENV: str(tree),
    }
    done = subprocess.run(
        [sys.executable, "-c", "import pkg.used"], env=env, capture_output=True, text=True,
        timeout=120, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    [written] = reads.glob("*.json")
    assert json.loads(written.read_text()) == ["pkg/__init__.py", "pkg/used.py"]


# ---------------------------------------------------------------- the check's tree


def _tree(tmp_path: Path) -> snap.CheckTree:
    tree = tmp_path / "snapshots" / ("a" * 40)
    tree.mkdir(parents=True)
    return snap.CheckTree(commit="a" * 40, tree=tree, marker="m1")


def _runner(files: list[str] | None, imported: bool = True):
    """A command that imported the package from the tree and wrote ``files`` as its reads."""

    def run(command, env=None, timeout=0):
        directory = Path(command.split(f"{snap.READS_DIR_ENV}=", 1)[1].split()[0].strip("'"))
        tree = command.split(f"{snap.TREE_ENV}=", 1)[1].split()[0].strip("'")
        if files is not None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "host-1.json").write_text(json.dumps(files))
        said = f"[mv2flow-resolve] module=tensorrt_llm origin={tree}/tensorrt_llm/__init__.py"
        return SimpleNamespace(stdout="", stderr=said if imported else "", returncode=0)

    return run


def test_a_check_names_what_its_commands_read_and_nothing_when_one_did_not_say(tmp_path):
    check = _tree(tmp_path)
    result = gates.GateResult(name="boot", passed=True, detail="", member="tp1")
    check.runner(_runner(["tensorrt_llm/a.py"]))("python run.py")
    check.runner(_runner(["tensorrt_llm/b.py"]))("python run.py")
    check.runner(_runner(None, imported=False))("nvidia-smi")  # imported nothing: not asked
    assert check.stamp(result).reads == ("tensorrt_llm/a.py", "tensorrt_llm/b.py")
    assert not list((check.tree.parent / snap.READS_DIRNAME).glob("*")), "cleaned up"
    check.runner(_runner(None))("python run.py")  # imported the package, said nothing
    assert check.stamp(result).reads == ()


# ---------------------------------------------------------------- the contract


def test_a_verdict_on_another_state_holds_only_where_its_reads_hold():
    contract = c.Contract(checks=(c.Check("core", "boot", "tp1", c.TIER_INNER),))
    verdict = c.Verdict("core", "boot", "tp1", True, marker="m1", reads="r", reads_marker="n1")
    assert c.evaluate(contract, [verdict], marker="m2").missing
    assert c.evaluate(contract, [verdict], marker="m2", holds=lambda v: True).holds
    assert c.evaluate(contract, [verdict], marker="m2", holds=lambda v: False).missing
    bare = c.Verdict("core", "boot", "tp1", True, marker="m1")
    assert c.evaluate(contract, [bare], marker="m2", holds=lambda v: True).missing


# ---------------------------------------------------------------- the workflow


_FILES = {
    "tensorrt_llm/__init__.py": "",
    "tensorrt_llm/models/core.py": "C = 1\n",
    "tensorrt_llm/models/ops/attn__r3.py": "A = 1\n",
    "tensorrt_llm/models/table.json": "{}\n",
}


def _workflow(tmp_path: Path) -> tuple[TrtllmModelingV2Workflow, Path]:
    repo = tmp_path / "checkout"
    for name, text in _FILES.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.task_data = {"trtllm_repo_path": str(repo), "accuracy": {}}
    return workflow, repo


def _land(repo: Path, name: str, text: str) -> None:
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"land {name}")


def test_a_verdict_holds_until_a_file_its_check_read_changes(tmp_path):
    workflow, repo = _workflow(tmp_path)
    measured = _git(repo, "rev-parse", "HEAD")
    result = SimpleNamespace(
        reads=("tensorrt_llm/__init__.py", "tensorrt_llm/models/core.py"),
        imported=(measured,),
        void="",
    )
    ident, marker = workflow._narrowed(result)
    assert ident and marker
    verdict = c.Verdict("core", "boot", "tp1", True, marker="m", reads=ident, reads_marker=marker)
    assert workflow._verdict_holds(verdict)

    _land(repo, "tensorrt_llm/models/ops/attn__r9.py", "B = 1\n")  # a variant core never imports
    _land(repo, "tensorrt_llm/models/ops/attn__r3.py", "A = 2\n")
    assert workflow._verdict_holds(verdict)

    _land(repo, "tensorrt_llm/models/table.json", '{"x": 1}\n')  # not Python: always read
    assert not workflow._verdict_holds(verdict)


def test_a_change_to_an_imported_file_makes_it_stale(tmp_path):
    workflow, repo = _workflow(tmp_path)
    measured = _git(repo, "rev-parse", "HEAD")
    result = SimpleNamespace(reads=("tensorrt_llm/models/core.py",), imported=(measured,), void="")
    ident, marker = workflow._narrowed(result)
    verdict = c.Verdict("core", "boot", "tp1", True, marker="m", reads=ident, reads_marker=marker)
    _land(repo, "tensorrt_llm/models/core.py", "C = 2\n")
    assert not workflow._verdict_holds(verdict)


def test_nothing_is_narrowed_without_one_commit_or_a_read_list(tmp_path):
    workflow, repo = _workflow(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    for result in (
        SimpleNamespace(reads=(), imported=(head,), void=""),
        SimpleNamespace(reads=("tensorrt_llm/models/core.py",), imported=(head, "b" * 40), void=""),
        SimpleNamespace(reads=("tensorrt_llm/models/core.py",), imported=(head,), void="two trees"),
    ):
        assert workflow._narrowed(result) == ("", "")
