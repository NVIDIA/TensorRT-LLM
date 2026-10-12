# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mechanical delivery checks.

Two kinds here. Most build a synthetic repo tree, so a check's failure mode
is exercised directly. The last one runs the whole verification against the
**real merged target in this checkout**, which is the only way to find out
that the path conventions are still the ones the repository actually uses.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import delivery

TARGET_REL = f"{delivery.MODELING_V2_PACKAGE}/models/gpt_oss/gpt_oss_120b__sm_103__tp1"

_MODELING = '''\
"""A synthetic target, just enough for the static checks."""

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.torch.embedding import embedding

REQUIRED_TRTLLM_OPS = (
    "flashinfer_rmsnorm",
    "cublas_mm",
)
'''


def _build_repo(tmp_path: Path, *, modeling: str = _MODELING) -> Path:
    root = tmp_path / "repo"
    target = root / TARGET_REL
    target.mkdir(parents=True)
    (target / "modeling.py").write_text(modeling, encoding="utf-8")
    (target / "weights.py").write_text("MANIFEST = ()\n", encoding="utf-8")

    catalog = root / delivery.CATALOG_DIR
    for category, name, contracted in (
        ("norm", "flashinfer_rmsnorm", True),
        ("gemm", "cublas_mm", True),
        ("torch", "embedding", False),
    ):
        (catalog / category).mkdir(parents=True, exist_ok=True)
        (catalog / category / f"{name}.py").write_text("def f(): ...\n", encoding="utf-8")
        if contracted:
            (catalog / category / f"{name}.md").write_text("# contract\n", encoding="utf-8")
        tests = root / delivery.CATALOG_TESTS_DIR / category
        tests.mkdir(parents=True, exist_ok=True)
        if contracted:
            (tests / f"test_modeling_v2_{name}.py").write_text(
                "def test(): ...\n", encoding="utf-8"
            )
    (catalog / "index.yaml").write_text(
        yaml.safe_dump(
            {
                "entries": [
                    {"path": "norm/flashinfer_rmsnorm.py"},
                    {"path": "gemm/cublas_mm.py"},
                    {"path": "torch/embedding.py"},
                ]
            }
        ),
        encoding="utf-8",
    )

    (root / delivery.CATALOG_TESTS_DIR / "test_modeling_v2_claims.py").write_text(
        "def test(): ...\n", encoding="utf-8"
    )
    routing = root / delivery.MODELING_V2_PACKAGE / "models" / "gpt_oss" / "routing.py"
    routing.write_text(
        'TARGET_MODULES = {"X": "models.gpt_oss.gpt_oss_120b__sm_103__tp1.modeling"}\n',
        encoding="utf-8",
    )

    refs = root / delivery.ACCURACY_REFERENCES_DIR
    refs.mkdir(parents=True)
    (refs / "gsm8k.yaml").write_text(
        yaml.safe_dump({"openai/gpt-oss-120b": [{"accuracy": 85.6}]}), encoding="utf-8"
    )
    tests_dir = root / delivery.ACCURACY_TESTS_DIR
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_modeling_v2_gpt_oss.py").write_text("def test(): ...\n", encoding="utf-8")
    db = root / delivery.TEST_DB_DIR
    db.mkdir(parents=True)
    (db / "l0_gb300.yml").write_text(
        "- condition: {}\n  tests:\n  - accuracy/test_modeling_v2_gpt_oss.py::T::t\n",
        encoding="utf-8",
    )
    return root


def _verify(root: Path):
    return delivery.verify_delivery(
        repo_root=root,
        target_rel=TARGET_REL,
        family="gpt_oss",
        reference_yaml=f"{delivery.ACCURACY_REFERENCES_DIR}/gsm8k.yaml",
        model_key="openai/gpt-oss-120b",
        accuracy_test="test_modeling_v2_gpt_oss.py",
    )


def test_a_complete_delivery_passes(tmp_path):
    assert _verify(_build_repo(tmp_path)).passed


def test_imports_are_read_from_the_ast_not_by_importing(tmp_path):
    root = _build_repo(tmp_path)
    entries = delivery.imported_catalog_entries(root / TARGET_REL / "modeling.py")
    assert entries == [
        ("gemm", "cublas_mm"),
        ("norm", "flashinfer_rmsnorm"),
        ("torch", "embedding"),
    ]


def test_required_ops_are_read_statically(tmp_path):
    root = _build_repo(tmp_path)
    assert delivery.declared_required_ops(root / TARGET_REL / "modeling.py") == [
        "flashinfer_rmsnorm",
        "cublas_mm",
    ]


def test_a_target_declaring_no_ops_is_flagged(tmp_path):
    root = _build_repo(tmp_path, modeling="X = 1\n")
    report = _verify(root)
    failure = next(c for c in report.failures if c.name == "target:REQUIRED_TRTLLM_OPS")
    assert "renamed op" in failure.detail


def test_a_missing_catalog_test_fails(tmp_path):
    root = _build_repo(tmp_path)
    (root / delivery.CATALOG_TESTS_DIR / "norm" / "test_modeling_v2_flashinfer_rmsnorm.py").unlink()
    report = _verify(root)
    assert not report.passed
    assert any(c.name == "test:norm/flashinfer_rmsnorm" for c in report.failures)


def test_mirrors_need_no_contract_or_test(tmp_path):
    """`catalog/torch/` mirrors carry no contract, test or receipts by design."""
    root = _build_repo(tmp_path)
    report = _verify(root)
    assert report.passed
    assert not any(c.name == "test:torch/embedding" for c in report.checks)


def test_an_unindexed_entry_fails(tmp_path):
    root = _build_repo(tmp_path)
    index = root / delivery.CATALOG_DIR / "index.yaml"
    index.write_text(yaml.safe_dump({"entries": [{"path": "gemm/cublas_mm.py"}]}), encoding="utf-8")
    report = _verify(root)
    assert any("index.yaml" in c.detail for c in report.failures)


def test_an_unrouted_target_is_caught(tmp_path):
    """A target nothing routes to is worse than missing: `auto` answers instead."""
    root = _build_repo(tmp_path)
    routing = root / delivery.MODELING_V2_PACKAGE / "models" / "gpt_oss" / "routing.py"
    routing.write_text("TARGET_MODULES = {}\n", encoding="utf-8")
    report = _verify(root)
    failure = next(c for c in report.failures if c.name == "routing:target")
    assert "built-in implementation would answer" in failure.detail


def test_a_missing_anchor_is_caught(tmp_path):
    root = _build_repo(tmp_path)
    refs = root / delivery.ACCURACY_REFERENCES_DIR / "gsm8k.yaml"
    refs.write_text(yaml.safe_dump({"someone/else": [{"accuracy": 1.0}]}), encoding="utf-8")
    report = _verify(root)
    assert any(c.name == "accuracy:reference" for c in report.failures)


def test_an_unregistered_test_is_caught(tmp_path):
    """A gate on no list passes exactly once."""
    root = _build_repo(tmp_path)
    (root / delivery.TEST_DB_DIR / "l0_gb300.yml").write_text("- condition: {}\n", encoding="utf-8")
    report = _verify(root)
    failure = next(c for c in report.failures if c.name == "tests:test-db")
    assert "CI would never run it" in failure.detail


def test_summary_for_auditor_includes_the_dump(tmp_path):
    report = _verify(_build_repo(tmp_path))
    rendered = delivery.summarize_for_auditor(report, dump_excerpt="batch=1 dtype=bf16")
    assert "already run" in rendered
    assert "batch=1 dtype=bf16" in rendered


# ------------------------------------- delivery must check git, not just disk


def _git(tmp_path, *args):
    subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)


def _git_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@e.st")
    _git(tmp_path, "config", "user.name", "t")
    return tmp_path


def test_outside_a_git_work_tree_no_check_is_made(tmp_path):
    """Absence of git is absence of a claim, not a failure.

    A fixture in a temporary directory is not a delivery, and a check that
    failed there would make every other test in this file red for a reason
    that has nothing to do with delivery.
    """
    assert delivery.check_tree_committed(tmp_path, ("anything",)) == []


def test_a_clean_tree_passes(tmp_path):
    _git_repo(tmp_path)
    (tmp_path / "t.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "t.py")
    _git(tmp_path, "commit", "-qm", "t")
    checks = delivery.check_tree_committed(tmp_path, ("t.py",))
    assert len(checks) == 1 and checks[0].passed


def test_an_uncommitted_target_fails_even_though_the_file_is_there(tmp_path):
    """The failure this exists for, and it was measured once.

    A run passed `routing:target` and reported an accuracy number while
    `routing.py`, `modeling.py` (984 changed lines) and `weights.py` were
    uncommitted — and the catalog commits certifying those lines were in
    history. It would have closed `delivered` and shipped nothing.
    """
    _git_repo(tmp_path)
    (tmp_path / "seed").write_text("s\n", encoding="utf-8")
    _git(tmp_path, "add", "seed")
    _git(tmp_path, "commit", "-qm", "seed")
    (tmp_path / "modeling.py").write_text("# never committed\n", encoding="utf-8")

    checks = delivery.check_tree_committed(tmp_path, ("modeling.py",))
    assert len(checks) == 1
    assert not checks[0].passed
    assert "a clone would not have them" in checks[0].detail
    assert "modeling.py" in checks[0].detail


def test_a_modified_but_uncommitted_target_also_fails(tmp_path):
    """Not just untracked — the real incident was tracked files with edits.

    It is also the F011 shape: a tree dirtied while gates were queued, where
    a pass attributes a number to a commit it was never measured on.
    """
    _git_repo(tmp_path)
    (tmp_path / "routing.py").write_text("v = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "routing.py")
    _git(tmp_path, "commit", "-qm", "r")
    (tmp_path / "routing.py").write_text("v = 2\n", encoding="utf-8")
    assert not delivery.check_tree_committed(tmp_path, ("routing.py",))[0].passed


def test_verify_delivery_includes_the_git_check():
    """Reachability, not behaviour: an uncalled check checks nothing."""
    assert "check_tree_committed" in inspect.getsource(delivery.verify_delivery)


# --------------------------------------------------------- against the real tree

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.skipif(
    not (_REPO_ROOT / delivery.MODELING_V2_PACKAGE).is_dir(),
    reason="not running inside a TensorRT-LLM checkout with modeling_v2 present",
)
def test_the_real_merged_target_passes_every_check():
    """The path conventions must match what the repository actually uses.

    `gpt_oss / sm_103 / tp1` is merged and gated, so every check here has a
    known-good answer. If this fails, a convention moved and the checks are
    describing a layout that no longer exists.
    """
    report = delivery.verify_delivery(
        repo_root=_REPO_ROOT,
        target_rel=TARGET_REL,
        family="gpt_oss",
        reference_yaml=f"{delivery.ACCURACY_REFERENCES_DIR}/gsm8k.yaml",
        model_key="openai/gpt-oss-120b",
        accuracy_test="test_modeling_v2_gpt_oss.py",
    )
    assert report.passed, report.render()
