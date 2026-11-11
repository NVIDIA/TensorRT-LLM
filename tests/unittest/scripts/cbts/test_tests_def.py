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
"""Tests for CBTS TestsDefRule: narrowing CI when files under tests/ change."""

from __future__ import annotations

import ast
import difflib
from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Stage, YAMLIndex
from cbts.rules._helpers import strip_noop_diff_lines
from cbts.rules.base import PRInputs
from cbts.rules.tests_def import (
    ACCURACY_DIR,
    ACCURACY_REFS_PREFIX,
    DEFS_GIT_PREFIX,
    _direct_test_importers,
    _py_class_scopes_from_deletions,
    _scope_start_line,
    _yaml_top_keys_from_deletions,
)
from cbts.rules.tests_def import TestsDefRule as CbtsTestsDefRule

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
    (test_db_dir / "l0_cpu.yml").write_text(
        "l0_cpu:\n- tests:\n  - unittest/others\n", encoding="utf-8"
    )
    (test_db_dir / "l0_b200.yml").write_text(
        "l0_b200:\n- tests:\n  - perf/test_perf_sanity.py::test_e2e[case]\n",
        encoding="utf-8",
    )
    (test_db_dir / "l0_gb200_multi_gpus_perf_sanity.yml").write_text(
        "l0_gb200_multi_gpus_perf_sanity:\n"
        "- tests:\n"
        "  - perf/test_perf_sanity.py::test_e2e[case]\n",
        encoding="utf-8",
    )
    (test_db_dir / "l0_other.yml").write_text(
        "l0_other:\n- tests:\n  - other/test_other.py::test_other\n",
        encoding="utf-8",
    )
    stages = {
        "CPU-Generic-x86-1": Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1),
        "DGX_B200-PyTorch-1": Stage("DGX_B200-PyTorch-1", "l0_b200", "x86_64", 1, 1),
        "GB200-4_GPUs-PyTorch-PerfSanity-1": Stage(
            "GB200-4_GPUs-PyTorch-PerfSanity-1",
            "l0_gb200_multi_gpus_perf_sanity",
            "x86_64",
            1,
            1,
        ),
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


def test_perf_sanity_definition_keeps_perfsanity_required(tmp_path: Path) -> None:
    result = _make_perf_rule(tmp_path).apply(
        PRInputs(
            changed_files=["tests/integration/defs/perf/test_perf_sanity.py"],
            diffs={},
        )
    )

    assert result is not None
    assert result.affected_stages == {
        "DGX_B200-PyTorch-1",
        "GB200-4_GPUs-PyTorch-PerfSanity-1",
    }
    assert result.perfsanity_relevant is True


@pytest.mark.parametrize(
    "helper", ["pytorch_model_config.py", "sampler_options_config.py", "new_config.py"]
)
def test_qa_only_helper_selects_cpu_harness(tmp_path: Path, helper: str) -> None:
    _write_defs_file(tmp_path, f"perf/{helper}", "VALUE = 1\n")
    _write_defs_file(tmp_path, "perf/test_perf.py", f"from .{Path(helper).stem} import VALUE\n")
    _write_defs_file(tmp_path, "perf/test_perf_sanity.py", "VALUE = 2\n")
    result = _make_perf_rule(tmp_path).apply(
        PRInputs(changed_files=[f"tests/integration/defs/perf/{helper}"], diffs={})
    )
    assert result is not None
    assert result.affected_stages == {"CPU-Generic-x86-1"}
    assert set(result.block_filters) == {("l0_cpu", 0)}
    assert result.perfsanity_relevant is False


@pytest.mark.parametrize("helper", ["_model_paths.py", "warmup_pass.py"])
def test_shared_perf_helper_keeps_cpu_and_gpu_coverage(tmp_path: Path, helper: str) -> None:
    _write_defs_file(tmp_path, f"perf/{helper}", "VALUE = 1\n")
    for consumer in ["test_perf.py", "test_perf_sanity.py", "utils.py"]:
        _write_defs_file(tmp_path, f"perf/{consumer}", f"from .{Path(helper).stem} import VALUE\n")
    result = _make_perf_rule(tmp_path).apply(
        PRInputs(changed_files=[f"tests/integration/defs/perf/{helper}"], diffs={})
    )
    assert result is not None
    assert result.affected_stages == {
        "CPU-Generic-x86-1",
        "DGX_B200-PyTorch-1",
        "GB200-4_GPUs-PyTorch-PerfSanity-1",
    }
    assert result.perfsanity_relevant is True


def test_qa_helper_without_cpu_coverage_falls_back(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/pytorch_model_config.py", "VALUE = 1\n")
    _write_defs_file(tmp_path, "perf/test_perf.py", "from .pytorch_model_config import VALUE\n")
    assert (
        _make_rule(tmp_path).apply(
            PRInputs(
                changed_files=["tests/integration/defs/perf/pytorch_model_config.py"], diffs={}
            )
        )
        is None
    )


@pytest.mark.parametrize("source", ["test_perf.py", "utils.py"])
def test_qa_perf_harness_changes_select_cpu_functional_test(tmp_path: Path, source: str) -> None:
    result = _make_perf_rule(tmp_path).apply(
        PRInputs(changed_files=[f"tests/integration/defs/perf/{source}"], diffs={})
    )
    assert result is not None
    assert "CPU-Generic-x86-1" in result.affected_stages
    assert ("l0_cpu", 0) in result.block_filters
    if source == "utils.py":
        assert "GB200-4_GPUs-PyTorch-PerfSanity-1" in result.affected_stages
        assert result.perfsanity_relevant is True
    else:
        assert result.affected_stages == {"CPU-Generic-x86-1"}


def test_missing_qa_perf_functional_coverage_falls_back(tmp_path: Path) -> None:
    rule = _make_rule(tmp_path)
    assert (
        rule.apply(PRInputs(changed_files=["tests/integration/defs/perf/test_perf.py"], diffs={}))
        is None
    )


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
        "GB200-4_GPUs-PyTorch-PerfSanity-1",
    }
    assert set(result.block_filters) == {
        ("l0_b200", 0),
        ("l0_gb200_multi_gpus_perf_sanity", 0),
    }


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
        "GB200-4_GPUs-PyTorch-PerfSanity-1",
    }
    assert set(result.block_filters) == {
        ("l0_b200", 0),
        ("l0_gb200_multi_gpus_perf_sanity", 0),
    }


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


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    (
        (
            "@pytest.mark.skip_less_device(4)\ndef test_eval():\n    pass\n",
            "@pytest.mark.skip_less_mpi_world_size(16)\ndef test_eval():\n    pass\n",
            ["test_example.py::test_eval"],
        ),
        (
            "class TestEval:\n    def test_eval(self):\n        assert old()\n",
            "class TestEval:\n    def test_eval(self):\n        assert new()\n",
            ["test_example.py::TestEval::test_eval"],
        ),
        (
            "def test_old():\n    pass\n\ndef test_neighbor():\n    pass\n",
            "def test_neighbor():\n    pass\n",
            ["test_example.py::test_old"],
        ),
        (
            "def test_old():\n    pass\n",
            "def test_new():\n    pass\n",
            ["test_example.py::test_new", "test_example.py::test_old"],
        ),
        (
            "@old\nclass TestEval:\n    def test_eval(self):\n        pass\n",
            "@new\nclass TestEval:\n    def test_eval(self):\n        pass\n",
            ["test_example.py::TestEval"],
        ),
        (
            "VALUE = 1\ndef test_eval():\n    pass\n",
            "VALUE = 2\ndef test_eval():\n    pass\n",
            ["test_example.py"],
        ),
        (
            "def helper():\n    return 1\ndef test_eval():\n    pass\n",
            "def test_eval():\n    pass\n",
            ["test_example.py"],
        ),
        (
            "# removed comment\n\n@old\ndef test_eval():\n    pass\n",
            "@new\ndef test_eval():\n    pass\n",
            ["test_example.py::test_eval"],
        ),
        (
            "def test_a():\n    first()\n    second()\n\ndef test_b():\n    old()\n",
            "def test_a():\n    second()\n\ndef test_b():\n    new()\n",
            ["test_example.py::test_a", "test_example.py::test_b"],
        ),
    ),
)
@pytest.mark.parametrize("context", (0, 3))
def test_verified_pre_image_scopes(
    tmp_path: Path, before: str, after: str, expected: list[str], context: int
) -> None:
    _write_defs_file(tmp_path, "test_example.py", after)
    raw_diff = "".join(
        difflib.unified_diff(before.splitlines(True), after.splitlines(True), n=context)
    )
    assert (
        _make_rule(tmp_path)._compute_anchors(
            f"{DEFS_GIT_PREFIX}test_example.py",
            "test_example.py",
            strip_noop_diff_lines(raw_diff),
            raw_diff,
        )
        == expected
    )


def test_decorator_replacement_filters_only_affected_test(tmp_path: Path) -> None:
    path = f"{DEFS_GIT_PREFIX}test_example.py"
    before = "@old\ndef test_eval():\n    pass\ndef test_other():\n    pass\n"
    after = before.replace("@old", "@new")
    _write_defs_file(tmp_path, "test_example.py", after)
    raw_diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))
    test_db = tmp_path / "test-db"
    test_db.mkdir()
    (test_db / "l0_target.yml").write_text(
        "l0_target:\n- tests:\n  - test_example.py::test_eval[tp16]\n"
        "  - test_example.py::test_other\n- tests:\n  - test_unrelated.py::test_other\n",
        encoding="utf-8",
    )
    rule = CbtsTestsDefRule(YAMLIndex.load(test_db), {}, tmp_path)
    result = rule.apply(
        PRInputs(
            changed_files=[path],
            diffs={path: strip_noop_diff_lines(raw_diff)},
            raw_diffs={path: raw_diff},
        )
    )
    assert result is not None
    assert result.scope == "testdefonly"
    assert set(result.block_filters) == {("l0_target", 0)}
    assert set(result.block_filters[("l0_target", 0)]) == {"test_example.py::test_eval"}


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
@pytest.mark.parametrize("test_path", ("test_models.py", "modeling_v2/test_models.py"))
def test_deleted_accuracy_key_requires_absence_from_test_sources(
    tmp_path: Path,
    source: str,
    expected: list[str],
    test_path: str,
) -> None:
    """Deleted reference sections remain relevant while a nested test still reads them."""
    git_path = f"{ACCURACY_REFS_PREFIX}gsm8k.yaml"
    yaml_path = "accuracy/references/gsm8k.yaml"
    reference = tmp_path / git_path
    reference.parent.mkdir(parents=True)
    reference.write_text("Other:\n  - expected: 2\n", encoding="utf-8")
    accuracy_test = tmp_path / ACCURACY_DIR / test_path
    accuracy_test.parent.mkdir(parents=True, exist_ok=True)
    accuracy_test.write_text(source, encoding="utf-8")
    diff = "@@ -1,4 +1,2 @@\n-GPT-OSS/20B-MXFP4:\n-  - expected: 1\n Other:\n   - expected: 2\n"

    assert _make_rule(tmp_path)._compute_anchors(git_path, yaml_path, diff) == expected


@pytest.mark.parametrize(
    ("source", "key", "expected"),
    (
        (
            'class TestTarget:\n    MODEL_NAME = "provider/model"\n',
            "provider/model",
            ["accuracy/modeling_v2/test_target.py::TestTarget"],
        ),
        (
            "class TestTarget:\n    def test_accuracy(self):\n        pass\n",
            "TestTarget::test_accuracy",
            ["accuracy/modeling_v2/test_target.py::TestTarget::test_accuracy"],
        ),
        (
            'class TestTarget:\n    KEY = "TestTarget::mtp3"\n'
            "    def test_accuracy(self):\n        pass\n",
            "TestTarget::mtp3",
            ["accuracy/modeling_v2/test_target.py::TestTarget"],
        ),
        (
            "class TestTarget(BaseTest):\n    pass\n",
            "TestTarget::test_inherited",
            ["accuracy/modeling_v2/test_target.py::TestTarget"],
        ),
        (
            'class TestTarget:\n    KEY = ("ModelingV2Target::" "mtp3")\n',
            "ModelingV2Target::mtp3",
            ["accuracy/modeling_v2/test_target.py::TestTarget"],
        ),
        (
            'class TestTarget:\n    KEY = "ModelingV2Target::mtp3"\n    KV_KEY = KEY + "_fp8_kv"\n',
            "ModelingV2Target::mtp3_fp8_kv",
            ["accuracy/modeling_v2/test_target.py::TestTarget"],
        ),
        (
            'KEY = "ModelingV2Target::mtp3"\nclass TestTarget:\n    pass\n',
            "ModelingV2Target::mtp3",
            ["accuracy/modeling_v2/test_target.py"],
        ),
        (
            'def read_reference():\n    return "ModelingV2Target::mtp3"\n',
            "ModelingV2Target::mtp3",
            ["accuracy/modeling_v2/test_target.py"],
        ),
        (
            'class TestTarget:\n    KEY = f"ModelingV2Target::{VARIANT}"\n',
            "ModelingV2Target::mtp3",
            ["accuracy/modeling_v2/test_target.py::TestTarget"],
        ),
        (
            'class TestTarget:\n    KEY = "ModelingV2Other::mtp3"\n',
            "ModelingV2Target::mtp3",
            ["accuracy/references/acceptance_length.yaml"],
        ),
    ),
)
def test_nested_accuracy_reference_anchors(
    tmp_path: Path, source: str, key: str, expected: list[str]
) -> None:
    """Resolve nested model, method and variant keys without inventing method anchors."""
    _write_defs_file(tmp_path, "accuracy/modeling_v2/test_target.py", source)
    _write_defs_file(tmp_path, "accuracy/test_other.py", "class TestOther:\n    pass\n")
    yaml_path = "accuracy/references/acceptance_length.yaml"
    _write_defs_file(tmp_path, yaml_path, f"{key}:\n  min_al: 2.5\n")
    diff = f"@@ -0,0 +1,2 @@\n+{key}:\n+  min_al: 2.5\n"

    assert (
        _make_rule(tmp_path)._compute_anchors(f"{DEFS_GIT_PREFIX}{yaml_path}", yaml_path, diff)
        == expected
    )


@pytest.mark.parametrize("namespace", ("ModelingV2Target", "TestTarget"))
def test_acceptance_reference_only_selects_consuming_classes(
    tmp_path: Path, namespace: str
) -> None:
    """A variant reference keeps listed test methods through their consuming class."""
    key = f"{namespace}::mtp3"
    yaml_path = "accuracy/references/acceptance_length.yaml"
    _write_defs_file(tmp_path, yaml_path, f"{key}:\n  min_al: 2.5\n")
    for module in ("modeling_v2", "other"):
        _write_defs_file(
            tmp_path,
            f"accuracy/{module}/test_target.py",
            f'class TestTarget:\n    KEY = "{key}"\n',
        )
    test_db = tmp_path / "test-db"
    test_db.mkdir()
    (test_db / "l0_target.yml").write_text(
        "l0_target:\n- tests:\n"
        "  - accuracy/modeling_v2/test_target.py::TestTarget::test_accuracy\n"
        "  - accuracy/modeling_v2/test_target.py::TestUnrelated::test_accuracy\n"
        "  - accuracy/other/test_target.py::TestTarget::test_accuracy\n"
        "- tests:\n  - accuracy/test_unrelated.py::TestUnrelated::test_accuracy\n",
        encoding="utf-8",
    )
    rule = CbtsTestsDefRule(YAMLIndex.load(test_db), {}, tmp_path)
    result = rule.apply(
        PRInputs(
            changed_files=[f"{DEFS_GIT_PREFIX}{yaml_path}"],
            diffs={f"{DEFS_GIT_PREFIX}{yaml_path}": f"@@ -0,0 +1,2 @@\n+{key}:\n+  min_al: 2.5\n"},
        )
    )

    assert result is not None
    assert result.scope == "testdefonly"
    assert set(result.block_filters) == {("l0_target", 0)}
    assert set(result.block_filters[("l0_target", 0)]) == {
        "accuracy/modeling_v2/test_target.py::TestTarget",
        "accuracy/other/test_target.py::TestTarget",
    }


def test_mixed_known_and_unknown_reference_keys_keep_fallback(tmp_path: Path) -> None:
    """One unresolved namespace prevents narrowing a mixed reference change."""
    _write_defs_file(
        tmp_path,
        "accuracy/modeling_v2/test_target.py",
        'class TestTarget:\n    KEY = "ModelingV2Target::mtp3"\n',
    )
    yaml_path = "accuracy/references/acceptance_length.yaml"
    content = "ModelingV2Target::mtp3:\n  min_al: 2.5\nUnknown::variant:\n  min_al: 1.5\n"
    _write_defs_file(tmp_path, yaml_path, content)
    diff = "@@ -0,0 +1,4 @@\n" + "".join(f"+{line}\n" for line in content.splitlines())

    assert _make_rule(tmp_path)._compute_anchors(
        f"{DEFS_GIT_PREFIX}{yaml_path}", yaml_path, diff
    ) == [yaml_path]
