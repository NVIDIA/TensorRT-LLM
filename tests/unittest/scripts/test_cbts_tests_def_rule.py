#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for CBTS test-definition scope recovery."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CBTS_ROOT = REPO_ROOT / "jenkins/scripts/cbts"
sys.path.insert(0, str(CBTS_ROOT))

from blocks import Stage, YAMLIndex  # noqa: E402
from rules.base import PRInputs  # noqa: E402
from rules.tests_def_rule import (  # noqa: E402
    ACCURACY_DIR,
    ACCURACY_REFS_PREFIX,
    _direct_test_importers,
    _py_class_scopes_from_deletions,
    _scope_start_line,
    _yaml_top_keys_from_deletions,
)
from rules.tests_def_rule import TestsDefRule as CbtsTestsDefRule  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _make_rule(repo_root: Path) -> CbtsTestsDefRule:
    return CbtsTestsDefRule(YAMLIndex(), {}, repo_root)


def _write_defs_file(tmp_path: Path, relative_path: str, content: str) -> None:
    path = tmp_path / "tests/integration/defs" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_perf_rule(tmp_path: Path) -> CbtsTestsDefRule:
    test_db_dir = tmp_path / "test-db"
    test_db_dir.mkdir()
    (test_db_dir / "l0_perf.yml").write_text(
        "l0_perf:\n- tests:\n  - perf/test_perf.py::test_perf[case]\n",
        encoding="utf-8",
    )
    (test_db_dir / "l0_b200.yml").write_text(
        "l0_b200:\n- tests:\n  - perf/test_perf_sanity.py::test_e2e[case]\n",
        encoding="utf-8",
    )
    (test_db_dir / "l0_other.yml").write_text(
        "l0_other:\n- tests:\n  - other/test_other.py::test_other\n",
        encoding="utf-8",
    )
    stages = {
        "H100_PCIe-PyTorch-Perf-1": Stage("H100_PCIe-PyTorch-Perf-1", "l0_perf", "x86_64", 1, 1),
        "DGX_B200-PyTorch-1": Stage("DGX_B200-PyTorch-1", "l0_b200", "x86_64", 1, 1),
    }
    return CbtsTestsDefRule(YAMLIndex.load(test_db_dir), stages, tmp_path)


@pytest.mark.parametrize(
    "import_statement",
    (
        "from .helper import VALUE",
        "from defs.perf.helper import VALUE",
        "import helper",
    ),
)
def test_direct_test_importers_resolve_static_imports(
    tmp_path: Path,
    import_statement: str,
) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(tmp_path, "perf/test_consumer.py", f"{import_statement}\n")
    _write_defs_file(tmp_path, "perf/test_other.py", "VALUE = 2\n")

    assert _direct_test_importers(tmp_path, "tests/integration/defs/perf/helper.py") == [
        "perf/test_consumer.py"
    ]


def test_direct_test_importers_fall_back_for_transitive_imports(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(tmp_path, "perf/bridge.py", "from .helper import VALUE\n")
    _write_defs_file(tmp_path, "perf/test_consumer.py", "from .bridge import VALUE\n")

    assert _direct_test_importers(tmp_path, "tests/integration/defs/perf/helper.py") is None


def test_direct_test_importers_fall_back_for_pytest_plugins(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "disaggregated/plugin.py", "VALUE = 1\n")
    _write_defs_file(
        tmp_path,
        "disaggregated/test_consumer.py",
        "pytest_plugins = ['plugin']\n",
    )

    assert (
        _direct_test_importers(tmp_path, "tests/integration/defs/disaggregated/plugin.py") is None
    )


def test_direct_test_importers_fall_back_for_conftest_import(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(tmp_path, "perf/conftest.py", "from .helper import VALUE\n")

    assert _direct_test_importers(tmp_path, "tests/integration/defs/perf/helper.py") is None


def test_direct_test_importers_fall_back_for_dynamic_import(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(
        tmp_path,
        "perf/test_consumer.py",
        "import importlib\nmodule_name = '.helper'\nimportlib.import_module(module_name)\n",
    )

    assert _direct_test_importers(tmp_path, "tests/integration/defs/perf/helper.py") is None


def test_pytorch_model_config_only_selects_test_perf_consumers(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/pytorch_model_config.py", "VALUE = 1\n")
    _write_defs_file(
        tmp_path,
        "perf/test_perf.py",
        "from .pytorch_model_config import VALUE\n",
    )
    _write_defs_file(tmp_path, "perf/test_perf_sanity.py", "VALUE = 2\n")
    _write_defs_file(
        tmp_path,
        "conftest.py",
        "from .perf.test_perf import generate_perf_tests\n",
    )

    result = _make_perf_rule(tmp_path).apply(
        PRInputs(
            changed_files=["tests/integration/defs/perf/pytorch_model_config.py"],
            diffs={},
        )
    )

    assert result is not None
    assert result.affected_stages == {"H100_PCIe-PyTorch-Perf-1"}
    assert set(result.block_filters) == {("l0_perf", 0)}


def test_conftest_import_keeps_directory_fallback(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(tmp_path, "perf/conftest.py", "from .helper import VALUE\n")

    result = _make_perf_rule(tmp_path).apply(
        PRInputs(
            changed_files=["tests/integration/defs/perf/helper.py"],
            diffs={},
        )
    )

    assert result is not None
    assert result.affected_stages == {
        "DGX_B200-PyTorch-1",
        "H100_PCIe-PyTorch-Perf-1",
    }
    assert set(result.block_filters) == {("l0_b200", 0), ("l0_perf", 0)}


def test_uncovered_import_consumers_keep_directory_fallback(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(
        tmp_path,
        "other/test_consumer.py",
        "from defs.perf.helper import VALUE\n",
    )

    result = _make_perf_rule(tmp_path).apply(
        PRInputs(
            changed_files=["tests/integration/defs/perf/helper.py"],
            diffs={},
        )
    )

    assert result is not None
    assert result.affected_stages == {
        "DGX_B200-PyTorch-1",
        "H100_PCIe-PyTorch-Perf-1",
    }
    assert set(result.block_filters) == {("l0_b200", 0), ("l0_perf", 0)}


def test_scope_start_line_includes_decorators() -> None:
    tree = ast.parse("@decorator\nclass TestExample:\n    pass\n")
    node = tree.body[0]
    assert isinstance(node, ast.ClassDef)
    assert _scope_start_line(node) == 1


def test_compute_anchors_recovers_deleted_scope_before_post_image(
    tmp_path: Path,
) -> None:
    git_path = "tests/integration/defs/test_example.py"
    yaml_path = "test_example.py"
    test_file = tmp_path / git_path
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "class TestB:\n    def test_b(self):\n        pass\n",
        encoding="utf-8",
    )
    diff = (
        "@@ -1,6 +1,3 @@\n"
        "-class TestA:\n"
        "-    def test_a(self):\n"
        "-        pass\n"
        " class TestB:\n"
        "     def test_b(self):\n"
        "         pass\n"
    )

    assert _make_rule(tmp_path)._compute_anchors(git_path, yaml_path, diff) == [
        "test_example.py::TestA"
    ]


def test_deleted_scope_recovery_resets_at_hunk_boundary() -> None:
    diff = "@@ -1,2 +1 @@\n class TestA:\n-    value = 1\n@@ -10 +9,0 @@\n-module_value = 2\n"

    assert _py_class_scopes_from_deletions(diff) is None


def test_deleted_decorator_without_visible_owner_falls_back() -> None:
    diff = "@@ -5 +5,0 @@\n-    @pytest.mark.parametrize('value', [1])\n"

    assert _py_class_scopes_from_deletions(diff) is None


def test_deleted_yaml_body_without_visible_key_falls_back() -> None:
    diff = "@@ -4 +4,0 @@\n-  - expected: 1\n"

    assert _yaml_top_keys_from_deletions(diff) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            'def test_model():\n    task = GSM8K("GPT-OSS/20B-MXFP4")\n',
            ["accuracy/references/gsm8k.yaml"],
        ),
        ("def test_other():\n    pass\n", []),
    ),
)
def test_deleted_accuracy_key_requires_absence_from_test_sources(
    tmp_path: Path,
    source: str,
    expected: list[str],
) -> None:
    git_path = f"{ACCURACY_REFS_PREFIX}gsm8k.yaml"
    yaml_path = "accuracy/references/gsm8k.yaml"
    reference = tmp_path / git_path
    reference.parent.mkdir(parents=True)
    reference.write_text("Other:\n  - expected: 2\n", encoding="utf-8")
    accuracy_test = tmp_path / ACCURACY_DIR / "test_models.py"
    accuracy_test.write_text(source, encoding="utf-8")
    diff = "@@ -1,4 +1,2 @@\n-GPT-OSS/20B-MXFP4:\n-  - expected: 1\n Other:\n   - expected: 2\n"

    assert _make_rule(tmp_path)._compute_anchors(git_path, yaml_path, diff) == expected
