# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mechanical delivery checks."""

from __future__ import annotations

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


def test_an_entry_that_certifies_nothing_fails(tmp_path):
    root = _build_repo(tmp_path)
    (root / delivery.CATALOG_DIR / "norm" / "flashinfer_rmsnorm.py").write_text(
        _ENTRY.replace("CELLS = (", "UNUSED = ("), encoding="utf-8"
    )
    failure = next(
        c for c in _verify(root).failures if c.name == "catalog:norm/flashinfer_rmsnorm.py"
    )
    assert "declares no CELLS" in failure.detail


def test_a_test_that_never_drives_the_cells_fails(tmp_path):
    root = _build_repo(tmp_path)
    test = root / delivery.CATALOG_TESTS_DIR / "gemm" / "test_modeling_v2_cublas_mm.py"
    test.write_text("def test(): ...\n", encoding="utf-8")
    failure = next(c for c in _verify(root).failures if c.name == "test:gemm/cublas_mm")
    assert ".CELLS" in failure.detail


def test_an_unrouted_target_is_caught(tmp_path):
    root = _build_repo(tmp_path)
    routing = root / delivery.MODELING_V2_PACKAGE / "models" / "gpt_oss" / "routing.py"
    routing.write_text("TARGET_MODULES = {}\n", encoding="utf-8")
    report = _verify(root)
    failure = next(c for c in report.failures if c.name == "routing:target")
    assert "built-in implementation would answer" in failure.detail


def test_an_unregistered_test_is_caught(tmp_path):
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


def test_an_uncommitted_target_fails_even_though_the_file_is_there(tmp_path):
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

    The `deepseek_v3` case is multi-rank, so it also holds the convention for
    the `comm` entries' op-matrix tests.
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


def test_a_pending_anchor_fails_the_reference_check_whatever_the_file_holds(tmp_path):
    reference = tmp_path / "gsm8k.yaml"
    reference.write_text(yaml.safe_dump({"x/y": [{"accuracy": 57.4}]}))
    [check] = delivery.check_accuracy_reference(tmp_path, str(reference), "x/y", pending=True)
    assert not check.passed and "anchor pending" in check.detail
    [check] = delivery.check_accuracy_reference(tmp_path, str(reference), "x/y")
    assert check.passed
