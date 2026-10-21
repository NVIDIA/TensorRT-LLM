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

from agent_flow.workflows.trtllm_modeling_v2.onboard import delivery

TARGET_REL = f"{delivery.MODELING_V2_PACKAGE}/models/gpt_oss/gpt_oss_120b__sm_103__tp1"

_MODELING = '''\
"""A synthetic target, just enough for the static checks."""

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.silu_and_mul import (
    SiluAndMul,
)

from .ops.moe.router import Router


class SyntheticCore(DecoderModel):
    def forward(self, attn_metadata, *args, **kwargs):
        return self._select_target(attn_metadata).forward(attn_metadata, *args, **kwargs)


class PrefillTarget:
    def step_args(self, md):
        return dict(num_contexts=md.num_contexts, num_ctx_tokens=md.num_ctx_tokens)

    def forward(self, attn_metadata, *args, **kwargs): ...


class DecodeTarget:
    def step_args(self, md):
        return dict(num_contexts=0, num_ctx_tokens=0)

    def forward(self, attn_metadata, *args, **kwargs): ...
'''

_WEIGHTS = '''\
"""A synthetic weight table."""


class SyntheticWeights:
    WEIGHTS: tuple = ()


MODEL_WEIGHTS = SyntheticWeights()
'''


#: An entry as modeling_v2 states it: an OpWrapper whose class is its contract.
_ENTRY = """\
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class Entry(OpWrapper):
    ARCHS = frozenset({Arch.SM_103})
    CELLS = (Cell(why="decode", spec=dict(rows=1)), Cell(why="prefill", spec=dict(rows=2048)))

    def __call__(self, x):
        return x

    def reference(self, x):
        return x
"""

_ENTRY_TEST = """\
import pytest


@pytest.mark.parametrize("cell", op.CELLS)
def test_certified_cells(cell): ...
"""

#: (category, name) of every shared entry `_MODELING` imports.
_SHARED_ENTRIES = (
    ("norm", "flashinfer_rmsnorm"),
    ("gemm", "cublas_mm"),
    ("activation", "silu_and_mul"),
)


def _write_entry(wrapper: Path, test: Path, *, entry: str = _ENTRY) -> None:
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(entry, encoding="utf-8")
    test.parent.mkdir(parents=True, exist_ok=True)
    test.write_text(_ENTRY_TEST, encoding="utf-8")


def _build_repo(tmp_path: Path, *, modeling: str = _MODELING) -> Path:
    root = tmp_path / "repo"
    target = root / TARGET_REL
    target.mkdir(parents=True)
    (target / "modeling.py").write_text(modeling, encoding="utf-8")
    (target / "weights.py").write_text(_WEIGHTS, encoding="utf-8")

    catalog = root / delivery.CATALOG_DIR
    for category, name in _SHARED_ENTRIES:
        _write_entry(
            catalog / category / f"{name}.py",
            root / delivery.CATALOG_TESTS_DIR / category / f"test_modeling_v2_{name}.py",
        )
    # The one op only this target calls: in its own ops/, tested under its path.
    _write_entry(
        target / delivery.MODEL_OPS_DIRNAME / "moe" / "router.py",
        root / delivery.model_ops_tests_dir(TARGET_REL) / "moe" / "test_modeling_v2_router.py",
    )
    (root / delivery.CATALOG_TESTS_DIR / "test_modeling_v2_claims.py").write_text(
        "def test(): ...\n", encoding="utf-8"
    )
    routing = root / delivery.MODELING_V2_PACKAGE / "models" / "gpt_oss" / "routing.py"
    routing.write_text(
        "def route(*args):\n"
        "    from .gpt_oss_120b__sm_103__tp1.modeling import ModelingV2GptOss120bSm103Tp1\n"
        "    return ModelingV2GptOss120bSm103Tp1\n",
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
        ("activation", "silu_and_mul"),
        ("gemm", "cublas_mm"),
        ("norm", "flashinfer_rmsnorm"),
    ]
    assert delivery.imported_model_ops(root / TARGET_REL / "modeling.py") == [
        ("gpt_oss_120b__sm_103__tp1", "moe", "router")
    ]


def _replace_modeling(root: Path, old: str, new: str) -> None:
    modeling = root / TARGET_REL / "modeling.py"
    text = modeling.read_text(encoding="utf-8")
    assert old in text
    modeling.write_text(text.replace(old, new), encoding="utf-8")


def test_a_core_without_its_decode_target_is_flagged(tmp_path):
    """The core dispatches every step to one of two targets; one alone is half a model."""
    root = _build_repo(tmp_path)
    _replace_modeling(root, "class DecodeTarget:", "class Decode:")
    failure = next(c for c in _verify(root).failures if c.name == "target:prefill-decode")
    assert "no DecodeTarget" in failure.detail


def test_a_target_that_skips_step_args_is_flagged(tmp_path):
    """`step_args` is the probe the step-contract check calls; a target owes one."""
    root = _build_repo(tmp_path)
    _replace_modeling(
        root,
        "    def step_args(self, md):\n        return dict(num_contexts=0",
        "    def other(self, md):\n        return dict(num_contexts=0",
    )
    failure = next(c for c in _verify(root).failures if c.name == "target:prefill-decode")
    assert "DecodeTarget defines no step_args" in failure.detail


def test_a_decode_target_reading_the_phase_back_is_flagged(tmp_path):
    """Decode runs only when num_contexts == 0; reading it is a branch frozen at capture."""
    root = _build_repo(tmp_path)
    _replace_modeling(root, "dict(num_contexts=0,", "dict(num_contexts=md.num_contexts,")
    failure = next(c for c in _verify(root).failures if c.name == "target:decode-reads-no-phase")
    assert ".num_contexts" in failure.detail
    # Prefill holds the mixed batch and reads both counts; that is the design,
    # so only the one read inside DecodeTarget is named.
    assert failure.detail.count(".num_contexts") == 1
    assert ".num_ctx_tokens" not in failure.detail


def test_a_weights_file_without_a_table_is_flagged(tmp_path):
    root = _build_repo(tmp_path)
    (root / TARGET_REL / "weights.py").write_text("MODEL_WEIGHTS = object()\n", encoding="utf-8")
    failure = next(c for c in _verify(root).failures if c.name == "target:weight-table")
    assert "no class with a WEIGHTS table" in failure.detail


def test_a_table_the_loader_cannot_reach_is_flagged(tmp_path):
    """The loader reaches the table through MODEL_WEIGHTS; a table without it loads nothing."""
    root = _build_repo(tmp_path)
    weights = root / TARGET_REL / "weights.py"
    weights.write_text(_WEIGHTS.replace("MODEL_WEIGHTS = SyntheticWeights()", ""), encoding="utf-8")
    failure = next(c for c in _verify(root).failures if c.name == "target:weight-table")
    assert "MODEL_WEIGHTS" in failure.detail


def test_a_hand_written_manifest_beside_the_table_is_flagged(tmp_path):
    """The table generates the manifest; a second copy of every key is the drift it stops."""
    root = _build_repo(tmp_path)
    weights = root / TARGET_REL / "weights.py"
    weights.write_text(_WEIGHTS + "\n\ndef _manifest(core):\n    return {}\n", encoding="utf-8")
    failure = next(c for c in _verify(root).failures if c.name == "target:weight-table")
    assert "_manifest" in failure.detail


def test_the_target_check_no_longer_asks_for_a_required_ops_list(tmp_path):
    """The step contract covers engine-surface drift; a target with no REQUIRED_TRTLLM_OPS is complete."""
    root = _build_repo(tmp_path)
    assert "REQUIRED_TRTLLM_OPS" not in (root / TARGET_REL / "modeling.py").read_text()
    report = _verify(root)
    assert report.passed, report.render()
    assert not any("REQUIRED_TRTLLM_OPS" in c.name for c in report.checks)


def test_a_missing_catalog_test_fails(tmp_path):
    root = _build_repo(tmp_path)
    (root / delivery.CATALOG_TESTS_DIR / "norm" / "test_modeling_v2_flashinfer_rmsnorm.py").unlink()
    report = _verify(root)
    assert not report.passed
    assert any(c.name == "test:norm/flashinfer_rmsnorm" for c in report.failures)


def test_an_entry_is_its_executable_contract(tmp_path):
    """No `.md` contract and no exempt category: the OpWrapper class is the contract."""
    contract = delivery.entry_contract(
        _build_repo(tmp_path) / delivery.CATALOG_DIR / "norm" / "flashinfer_rmsnorm.py"
    )
    assert contract.certified
    assert contract.classes == ("Entry",)
    assert contract.archs == ("sm_103",)
    assert contract.cells == 2


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("def flashinfer_rmsnorm(x): ...\n", "no OpWrapper subclass"),
        (_ENTRY.replace("    ARCHS = frozenset({Arch.SM_103})\n", ""), "declares no ARCHS"),
        (_ENTRY.replace("CELLS = (", "UNUSED = ("), "declares no CELLS"),
        (_ENTRY.replace("def reference", "def ref"), "defines no reference"),
    ],
    ids=["bare-function", "no-archs", "no-cells", "no-reference"],
)
def test_an_entry_that_certifies_nothing_fails(tmp_path, entry, expected):
    root = _build_repo(tmp_path)
    (root / delivery.CATALOG_DIR / "norm" / "flashinfer_rmsnorm.py").write_text(
        entry, encoding="utf-8"
    )
    failure = next(
        c for c in _verify(root).failures if c.name == "catalog:norm/flashinfer_rmsnorm.py"
    )
    assert expected in failure.detail


def test_a_test_that_never_drives_the_cells_fails(tmp_path):
    root = _build_repo(tmp_path)
    test = root / delivery.CATALOG_TESTS_DIR / "gemm" / "test_modeling_v2_cublas_mm.py"
    test.write_text("def test(): ...\n", encoding="utf-8")
    failure = next(c for c in _verify(root).failures if c.name == "test:gemm/cublas_mm")
    assert ".CELLS" in failure.detail


def test_a_model_op_is_checked_at_the_target_paths(tmp_path):
    """An op only this target calls lives in its `ops/`, tested under its path."""
    root = _build_repo(tmp_path)
    report = _verify(root)
    names = {c.name for c in report.checks}
    assert {"ops:ops/moe/router.py", "test:ops/moe/router"} <= names
    test = root / delivery.model_ops_tests_dir(TARGET_REL) / "moe" / "test_modeling_v2_router.py"
    test.unlink()
    failure = next(c for c in _verify(root).failures if c.name == "test:ops/moe/router")
    assert "models/gpt_oss/gpt_oss_120b__sm_103__tp1/moe/test_modeling_v2_router.py" in (
        failure.detail
    )
    (root / TARGET_REL / "ops" / "moe" / "router.py").unlink()
    assert any(c.name == "ops:ops/moe/router.py" for c in _verify(root).failures)


def test_another_targets_op_is_refused(tmp_path):
    """Targets share the catalog and nothing else: a sibling's op gets promoted."""
    foreign = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v3."
        "r1_0528_nvfp4__sm_103__dep4.ops.moe.fused_moe import FusedMoe\n"
    )
    root = _build_repo(tmp_path, modeling=_MODELING + foreign)
    failure = next(c for c in _verify(root).failures if c.name == "ops:ops/moe/fused_moe.py")
    assert "promote the entry" in failure.detail


def test_the_catalog_needs_no_index(tmp_path):
    """The catalog is its entries: there is no index file to be listed in."""
    root = _build_repo(tmp_path)
    assert not (root / delivery.CATALOG_DIR / "index.yaml").exists()
    report = _verify(root)
    assert report.passed, report.render()
    assert not any("index" in c.detail for c in report.checks)


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
    """The failure this exists for.

    `routing:target` passes and the gates report accuracy while `routing.py`,
    `modeling.py` and `weights.py` are uncommitted, even with the catalog
    commits certifying those lines in history. Closing `delivered` on that
    ships nothing.
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
    """Not just untracked — tracked files with uncommitted edits fail too.

    It is also the shape of a tree dirtied while gates are queued, where a
    pass attributes a number to a commit it was never measured on.
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
@pytest.mark.parametrize(
    ("target_rel", "family", "model_key", "accuracy_test"),
    [
        (TARGET_REL, "gpt_oss", "openai/gpt-oss-120b", "test_modeling_v2_gpt_oss.py"),
        (
            f"{delivery.MODELING_V2_PACKAGE}/models/deepseek_v3/r1_0528_nvfp4__sm_103__dep4",
            "deepseek_v3",
            "deepseek-ai/DeepSeek-R1-0528",
            "test_modeling_v2_deepseek_v3.py",
        ),
    ],
    ids=["gpt_oss", "deepseek_v3"],
)
def test_the_real_merged_target_passes_every_check(target_rel, family, model_key, accuracy_test):
    """The path conventions must match what the repository actually uses.

    Both targets are merged and gated, so every check here has a known-good
    answer. If this fails, a convention moved and the checks are describing a
    layout the repository does not use. The `deepseek_v3` case is multi-rank,
    so it also holds the convention for the `comm` entries' op-matrix tests.
    """
    report = delivery.verify_delivery(
        repo_root=_REPO_ROOT,
        target_rel=target_rel,
        family=family,
        reference_yaml=f"{delivery.ACCURACY_REFERENCES_DIR}/gsm8k.yaml",
        model_key=model_key,
        accuracy_test=accuracy_test,
    )
    assert report.passed, report.render()


def test_a_multi_rank_entry_is_satisfied_by_its_op_matrix_test(tmp_path):
    """The collected file launches the ranks; the rank body beside it drives the cells."""
    tests = tmp_path / delivery.CATALOG_TESTS_DIR / "comm"
    tests.mkdir(parents=True)
    (tests / "test_modeling_v2_allgather_op_matrix.py").write_text("", encoding="utf-8")
    (tests / "_allgather_op_matrix.py").write_text("for cell in op.CELLS: ...\n", encoding="utf-8")
    [check] = delivery.check_catalog_tests(tmp_path, [("comm", "allgather")])
    assert check.passed, check.detail
    [missing] = delivery.check_catalog_tests(tmp_path, [("comm", "reducescatter")])
    assert not missing.passed
    assert missing.detail.endswith("comm/test_modeling_v2_reducescatter.py")


def test_the_accuracy_test_is_looked_up_in_the_trees_own_layout(tmp_path):
    assert delivery.accuracy_test_path(tmp_path, "newmodel") == "test_modeling_v2_newmodel.py"
    (tmp_path / delivery.ACCURACY_TESTS_DIR / "modeling_v2").mkdir(parents=True)
    assert delivery.accuracy_test_path(tmp_path, "newmodel") == "modeling_v2/test_newmodel.py"


def test_a_pending_anchor_fails_the_reference_check_whatever_the_file_holds(tmp_path):
    """The row is the user's to record from a written source, never the run's from its score."""
    reference = tmp_path / "gsm8k.yaml"
    reference.write_text(yaml.safe_dump({"x/y": [{"accuracy": 57.4}]}))
    [check] = delivery.check_accuracy_reference(tmp_path, str(reference), "x/y", pending=True)
    assert not check.passed and "anchor pending" in check.detail
    [check] = delivery.check_accuracy_reference(tmp_path, str(reference), "x/y")
    assert check.passed
