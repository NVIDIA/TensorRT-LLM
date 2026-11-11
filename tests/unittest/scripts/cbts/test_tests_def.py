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
from cbts.blocks import Stage, YAMLIndex, parse_stages_from_groovy
from cbts.rules._helpers import strip_noop_diff_lines
from cbts.rules.base import PRInputs, RuleResult
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


@pytest.mark.parametrize(
    ("helper", "select_family"),
    [
        pytest.param("_fp4_moe_test_utils.py", True, id="root-helper"),
        pytest.param("moe/_fp4_moe_test_utils.py", True, id="shared-helper"),
        pytest.param("moe/_other_test_utils.py", False, id="local-helper"),
    ],
)
def test_shared_fp4_helper_keeps_modeling_v2_family_consumers(
    tmp_path: Path, helper: str, select_family: bool
) -> None:
    modeling_v2 = "unittest/_torch/modeling_v2"
    family_consumers = {
        f"{modeling_v2}/models/qwen4_exp/ops/moe/{name}"
        for name in (
            "test_modeling_v2_fp4_block_scale_moe_renorm_routed.py",
            "test_modeling_v2_fp4_block_scale_moe_renorm_permuted.py",
            "test_modeling_v2_fp4_block_scale_moe_renorm_routed_windows.py",
            "test_modeling_v2_fp4_moe_renorm_cute_dsl.py",
        )
    }
    shared_test = f"{modeling_v2}/moe/ops/test_modeling_v2_fp4_block_scale_moe_runner.py"
    unrelated_test = "unittest/_torch/moe/test_moe_module.py"
    test_db_dir = tmp_path / "test-db"
    test_db_dir.mkdir()
    entries = sorted(family_consumers | {shared_test, unrelated_test})
    (test_db_dir / "l0_b200.yml").write_text(
        "l0_b200:\n- tests:\n" + "".join(f"  - {entry}\n" for entry in entries),
        encoding="utf-8",
    )
    (test_db_dir / "l0_other.yml").write_text(
        "l0_other:\n- tests:\n  - unittest/others/test_other.py\n", encoding="utf-8"
    )
    stage = Stage("DGX_B200-PyTorch-1", "l0_b200", "x86_64", 1, 1)
    rule = CbtsTestsDefRule(YAMLIndex.load(test_db_dir), {stage.name: stage}, tmp_path)

    result = rule.apply(PRInputs(changed_files=[f"tests/{modeling_v2}/{helper}"], diffs={}))

    assert result is not None
    assert result.scope == "testdefonly"
    assert result.affected_stages == {stage.name}
    expected = {shared_test} | (family_consumers if select_family else set())
    prefix = modeling_v2 if select_family else f"{modeling_v2}/moe"
    assert result.block_filters == {("l0_b200", 0): {prefix: expected}}


@pytest.mark.parametrize(
    ("tp_size", "helper", "select_sh_take"),
    [
        pytest.param(2, "_m3_moe_operands.py", True, id="tp2-operands"),
        pytest.param(2, "_local_test_utils.py", False, id="tp2-local-helper"),
        pytest.param(4, "_m3_moe_operands.py", False, id="tp4-operands"),
        pytest.param(8, "_m3_moe_operands.py", False, id="tp8-operands"),
    ],
)
def test_m3_tp2_operands_keep_sibling_sh_take_consumer(
    tmp_path: Path, tp_size: int, helper: str, select_sh_take: bool
) -> None:
    targets = "unittest/_torch/modeling_v2/models/minimax_m3"
    ops = f"{targets}/minimax_m3_nvfp4__sm_107__tp{tp_size}/ops"
    sh_take_dir = f"{targets}/minimax_m3_nvfp4__sm_107__tp2/ops/m3_moe"
    moe_tests = {f"{ops}/moe/test_modeling_v2_m3_moe_engine.py"}
    if tp_size == 2:
        moe_tests.add(f"{ops}/moe/test_modeling_v2_m3_moe_epoch_wrap.py")
    comm_test = f"{ops}/comm/test_modeling_v2_m3_mc_allreduce_norm_op_matrix.py"
    entries = sorted(moe_tests | {sh_take_dir, comm_test})
    test_db_dir = tmp_path / "test-db"
    test_db_dir.mkdir()
    (test_db_dir / "l0_gr100.yml").write_text(
        "l0_gr100:\n- condition:\n    terms:\n      stage: post_merge\n"
        "  tests:\n" + "".join(f"  - {entry}\n" for entry in entries),
        encoding="utf-8",
    )
    (test_db_dir / "l0_other.yml").write_text(
        "l0_other:\n- tests:\n  - unittest/others/test_other.py\n", encoding="utf-8"
    )
    stages = {
        name: Stage(name, "l0_gr100", "x86_64", 1, 1, {"stage": stage})
        for name, stage in (
            ("GR100-PyTorch-Post-Merge-1", "post_merge"),
            ("GR100-PyTorch-1", "pre_merge"),
        )
    }
    rule = CbtsTestsDefRule(YAMLIndex.load(test_db_dir), stages, tmp_path)

    result = rule.apply(PRInputs(changed_files=[f"tests/{ops}/moe/{helper}"], diffs={}))

    assert result is not None
    assert result.scope == "testdefonly"
    assert result.affected_stages == {"GR100-PyTorch-Post-Merge-1"}
    expected = moe_tests | ({sh_take_dir} if select_sh_take else set())
    assert _selected_entries(result) == {("l0_gr100", 0): expected}


@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize(
    ("helper", "select_moved_consumer"),
    [("_rank_job.py", True), ("_local_test_utils.py", False)],
)
def test_m3_rank_helper_keeps_moved_collective_consumer(
    tmp_path: Path, tp_size: int, helper: str, select_moved_consumer: bool
) -> None:
    target = f"unittest/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp{tp_size}"
    comm_dir = f"{target}/comm"
    moved_comm_dir = f"{target}/ops/comm"
    unrelated_test = f"{target}/ops/moe/test_modeling_v2_m3_moe_engine.py"
    test_db_dir = tmp_path / "test-db"
    test_db_dir.mkdir()
    (test_db_dir / "l0_gb300_multi_gpus.yml").write_text(
        "l0_gb300_multi_gpus:\n- condition:\n"
        "    ranges:\n      system_gpu_count:\n        gte: 4\n        lte: 4\n"
        "    terms:\n      stage: post_merge\n  tests:\n"
        f"  - {comm_dir}\n  - {moved_comm_dir}\n  - {unrelated_test}\n",
        encoding="utf-8",
    )
    (test_db_dir / "l0_other.yml").write_text(
        "l0_other:\n- tests:\n  - unittest/others/test_other.py\n", encoding="utf-8"
    )
    expected_stages = {f"GB300-4_GPUs-PyTorch-Post-Merge-{i}" for i in range(1, 4)}
    stages = {
        name: Stage(
            name,
            "l0_gb300_multi_gpus",
            "aarch64",
            i,
            3,
            {"stage": "post_merge", "system_gpu_count": "4"},
        )
        for i, name in enumerate(sorted(expected_stages), start=1)
    }
    for name, stage, gpu_count in (
        ("GB300-4_GPUs-PyTorch-1", "pre_merge", "4"),
        ("GB300-8_GPUs-PyTorch-Post-Merge-1", "post_merge", "8"),
    ):
        stages[name] = Stage(
            name,
            "l0_gb300_multi_gpus",
            "aarch64",
            1,
            1,
            {"stage": stage, "system_gpu_count": gpu_count},
        )
    rule = CbtsTestsDefRule(YAMLIndex.load(test_db_dir), stages, tmp_path)

    result = rule.apply(PRInputs(changed_files=[f"tests/{comm_dir}/{helper}"], diffs={}))

    assert result is not None
    assert result.scope == "testdefonly"
    assert result.affected_stages == expected_stages
    expected = {comm_dir: {comm_dir}}
    if select_moved_consumer:
        expected[moved_comm_dir] = {moved_comm_dir}
    assert result.block_filters == {("l0_gb300_multi_gpus", 0): expected}


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


_COHORT_MODELS = "unittest/_torch/modeling_v2/models"
_COHORT_M3 = f"{_COHORT_MODELS}/minimax_m3"
_COHORT_K3 = f"{_COHORT_MODELS}/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1"
_COHORT_GLM = f"{_COHORT_MODELS}/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency"
_COHORT_GLM_ROUTER = f"{_COHORT_GLM}/ops/gemm/test_modeling_v2_router_gemm.py"
_COHORT_M3_MOE_FILES = {
    2: (
        "ops/moe/test_modeling_v2_m3_moe_engine.py",
        "ops/moe/test_modeling_v2_m3_moe_epoch_wrap.py",
        "ops/m3_moe/test_modeling_v2_m3_moe_flat_gate.py",
        "ops/m3_moe/test_modeling_v2_m3_moe_sh_take.py",
        "ops/moe/test_modeling_v2_m3_moe_triton_finalize.py",
        "ops/moe/test_modeling_v2_m3_fp32_router_rows.py",
        "ops/moe/test_modeling_v2_m3_moe_finalize_rows.py",
    ),
    4: (
        "ops/moe/test_modeling_v2_m3_moe_engine.py",
        "ops/moe/test_modeling_v2_m3_moe16_engine.py",
        "ops/m3_moe/test_modeling_v2_m3_moe_flat_gate_tp4.py",
    ),
    8: ("ops/moe/test_modeling_v2_m3_moe_engine.py",),
}


def _selected_entries(result: RuleResult) -> dict[tuple[str, int], set[str]]:
    return {
        key: {entry for entries in prefixes.values() for entry in entries}
        for key, prefixes in result.block_filters.items()
    }


def _make_cohort_rule(
    tmp_path: Path,
    entries_by_list: dict[str, set[str]],
    post_merge_lists: frozenset[str] = frozenset(),
) -> CbtsTestsDefRule:
    test_db = tmp_path / "test-db"
    test_db.mkdir()
    stages: dict[str, Stage] = {}
    for stem, entries in {
        **entries_by_list,
        "l0_other": {"unittest/others/test_other.py"},
    }.items():
        condition = (
            "- condition:\n    terms:\n      stage: post_merge\n  tests:\n"
            if stem in post_merge_lists
            else "- tests:\n"
        )
        (test_db / f"{stem}.yml").write_text(
            f"{stem}:\n{condition}" + "".join(f"  - {entry}\n" for entry in sorted(entries)),
            encoding="utf-8",
        )
        for mode in ("pre_merge", "post_merge"):
            name = f"{stem}-{mode}"
            stages[name] = Stage(name, stem, "x86_64", 1, 1, {"stage": mode})
    return CbtsTestsDefRule(YAMLIndex.load(test_db), stages, tmp_path)


def _m3_target(tp_size: int, sm: int = 107) -> str:
    return f"{_COHORT_M3}/minimax_m3_nvfp4__sm_{sm}__tp{tp_size}"


def _m3_router_test(tp_size: int) -> str:
    return f"{_COHORT_M3}/minimax_m3_nvfp4__sm_107__tp{tp_size}/ops/moe/test_modeling_v2_m3_router_tc.py"


def _m3_moe_cohort(tp_size: int) -> set[str]:
    return {f"{_m3_target(tp_size)}/{name}" for name in _COHORT_M3_MOE_FILES[tp_size]} | {
        _m3_router_test(tp_size)
    }


def test_modeling_v2_cohort_k3_helper_keeps_target_checks(tmp_path: Path) -> None:
    entries = {
        f"{_COHORT_K3}/ops/moe/test_modeling_v2_k3_moe_m1.py",
        f"{_COHORT_K3}/test_tp16_moetp16ep1_construction.py",
        f"{_COHORT_K3}/kernels/k3_moe_engines/test_k3_moe_engines_source_checks.py",
    }
    foreign = f"{_COHORT_MODELS}/kimi_k3_vl/other_target/test_other.py"
    rule = _make_cohort_rule(tmp_path, {"l0_b300": entries | {foreign}})
    result = rule.apply(
        PRInputs(changed_files=[f"tests/{_COHORT_K3}/ops/moe/_k3_moe_engines.py"], diffs={})
    )
    assert result is not None
    assert result.scope == "testdefonly"
    assert _selected_entries(result) == {("l0_b300", 0): entries}
    assert result.affected_stages == {"l0_b300-pre_merge", "l0_b300-post_merge"}


@pytest.mark.parametrize(
    "relative",
    [
        pytest.param("ops/gemm/test_modeling_v2_router_gemm.py", id="router"),
        *(
            pytest.param(f"gemm/{name}", id=f"sibling-{name}")
            for name in (
                "test_modeling_v2_dense_mlp_small_m.py",
                "test_modeling_v2_draft_hidden_half.py",
                "test_modeling_v2_fused_a_gemm.py",
                "test_modeling_v2_indexer_q_tail.py",
                "test_modeling_v2_low_m_gemm.py",
                "test_modeling_v2_o_proj_publish.py",
                "test_modeling_v2_q_b_fp8_rope.py",
            )
        ),
    ],
)
def test_modeling_v2_cohort_glm_keeps_stage_specific_gemm_scope(
    tmp_path: Path, relative: str
) -> None:
    gemm = f"{_COHORT_GLM}/gemm"
    moved_gemm = f"{_COHORT_GLM}/ops/gemm"
    broad_target = f"{_COHORT_GLM} TIMEOUT (30)"
    rule = _make_cohort_rule(
        tmp_path,
        {
            "l0_b300": {broad_target},
            "l0_gr100": {gemm, moved_gemm, f"{_COHORT_GLM}/ops/moe"},
        },
        frozenset({"l0_gr100"}),
    )
    result = rule.apply(PRInputs(changed_files=[f"tests/{_COHORT_GLM}/{relative}"], diffs={}))
    assert result is not None
    assert _selected_entries(result) == {
        ("l0_b300", 0): {broad_target},
        ("l0_gr100", 0): {gemm, moved_gemm},
    }
    assert result.affected_stages == {
        "l0_b300-pre_merge",
        "l0_b300-post_merge",
        "l0_gr100-post_merge",
    }


@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize(
    "relative",
    [
        "comm/_rank_job.py",
        pytest.param("comm/_m3_allreduce_op_matrix.py", id="sibling-allreduce-helper"),
        pytest.param("comm/test_modeling_v2_m3_allreduce.py", id="sibling-allreduce-test"),
        "ops/comm/_m3_mc_allreduce_norm_op_matrix.py",
        "ops/comm/test_modeling_v2_m3_mc_allreduce_norm_op_matrix.py",
        "ops/comm/test_modeling_v2_m3_mc_finalize_allreduce_norm.py",
    ],
)
def test_modeling_v2_cohort_sm103_collectives_keep_both_directories(
    tmp_path: Path, tp_size: int, relative: str
) -> None:
    target = _m3_target(tp_size, sm=103)
    expected = {f"{target}/comm", f"{target}/ops/comm"}
    other_target = _m3_target(8 if tp_size == 4 else 4, sm=103)
    rule = _make_cohort_rule(
        tmp_path,
        {
            "l0_gb300_multi_gpus": expected
            | {f"{target}/ops/moe", f"{other_target}/comm", f"{other_target}/ops/comm"}
        },
        frozenset({"l0_gb300_multi_gpus"}),
    )
    result = rule.apply(PRInputs(changed_files=[f"tests/{target}/{relative}"], diffs={}))
    assert result is not None
    assert _selected_entries(result) == {("l0_gb300_multi_gpus", 0): expected}
    assert result.affected_stages == {"l0_gb300_multi_gpus-post_merge"}


@pytest.mark.parametrize(
    ("tp_size", "changed_path"),
    [
        pytest.param(tp_size, path, id=f"tp{tp_size}-{Path(path).name}")
        for tp_size in (2, 4, 8)
        for path in sorted(
            _m3_moe_cohort(tp_size)
            | {
                f"{_m3_target(tp_size)}/ops/moe/_m3_moe_operands.py",
                f"{_COHORT_M3}/minimax_m3_nvfp4__sm_107__tp{tp_size}/ops/moe/_m3_router.py",
            }
        )
    ],
)
def test_modeling_v2_cohort_sm107_moe_keeps_siblings_without_comm(
    tmp_path: Path, tp_size: int, changed_path: str
) -> None:
    target = _m3_target(tp_size)
    entries = set().union(*(_m3_moe_cohort(tp) for tp in (2, 4, 8))) | {
        f"{target}/ops/moe/test_modeling_v2_m3_moe_push_op_matrix.py",
        f"{target}/ops/comm/test_modeling_v2_m3_mc_allreduce_norm_op_matrix.py",
        f"{target}/norm/test_norm.py",
    }
    rule = _make_cohort_rule(tmp_path, {"l0_gr100": entries}, frozenset({"l0_gr100"}))
    result = rule.apply(PRInputs(changed_files=[f"tests/{changed_path}"], diffs={}))
    assert result is not None
    assert _selected_entries(result) == {("l0_gr100", 0): _m3_moe_cohort(tp_size)}
    assert result.affected_stages == {"l0_gr100-post_merge"}


@pytest.mark.parametrize(
    ("tp_size", "relative"),
    [
        pytest.param(
            tp_size,
            relative,
            id=(
                "sibling-"
                if relative
                in {
                    "comm/_m3_allreduce_op_matrix.py",
                    "comm/_m3_capture_in_boundary_op_matrix.py",
                    "comm/_m3_draft_ar_norm_op_matrix.py",
                    "comm/_sampled_accept_tp8.py",
                }
                else ""
            )
            + f"tp{tp_size}-{Path(relative).name}",
        )
        for tp_size in (2, 4, 8)
        for relative in (
            "comm/_rank_job.py",
            "comm/_m3_allreduce_op_matrix.py",
            "ops/comm/_m3_mc_allreduce_norm_op_matrix.py",
            *(("ops/moe/_m3_moe_push_op_matrix.py",) if tp_size in (2, 8) else ()),
            *(
                ("comm/_m3_capture_in_boundary_op_matrix.py", "comm/_m3_draft_ar_norm_op_matrix.py")
                if tp_size == 4
                else ()
            ),
            *(("comm/_sampled_accept_tp8.py",) if tp_size == 8 else ()),
        )
    ],
)
def test_modeling_v2_cohort_sm107_comm_keeps_target_and_own_router(
    tmp_path: Path, tp_size: int, relative: str
) -> None:
    all_entries: set[str] = set()
    expected: set[str] = set()
    for tp in (2, 4, 8):
        target = _m3_target(tp)
        entries = _m3_moe_cohort(tp) | {f"{target}/norm", f"{target}/spec"}
        all_entries.update(entries)
        if tp == tp_size:
            expected = entries
    rule = _make_cohort_rule(tmp_path, {"l0_gr100": all_entries}, frozenset({"l0_gr100"}))
    result = rule.apply(
        PRInputs(changed_files=[f"tests/{_m3_target(tp_size)}/{relative}"], diffs={})
    )
    assert result is not None
    assert _selected_entries(result) == {("l0_gr100", 0): expected}
    assert result.affected_stages == {"l0_gr100-post_merge"}


@pytest.mark.parametrize(
    ("change", "scoped"),
    [
        ("no-hunk", False),
        ("module-add", False),
        ("module-delete", False),
        ("file-delete", False),
        ("body-add", True),
        ("body-delete", True),
    ],
)
def test_modeling_v2_cohort_preserves_function_diff_narrowing(
    tmp_path: Path, change: str, scoped: bool
) -> None:
    source = "VALUE = 1\n\ndef test_selected() -> None:\n    assert True\n"
    before = source
    after = source
    if change == "module-add":
        after = "OTHER = 2\n" + source
    elif change == "module-delete":
        after = source.replace("VALUE = 1\n", "")
    elif change == "file-delete":
        after = ""
    elif change == "body-add":
        after = source + "    assert 1 == 1\n"
    elif change == "body-delete":
        before = source + "    assert 1 == 1\n"
    diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), n=3))
    git_path = f"tests/{_COHORT_GLM_ROUTER}"
    test_file = tmp_path / git_path
    test_file.parent.mkdir(parents=True)
    if change != "file-delete":
        test_file.write_text(after, encoding="utf-8")
    selected = f"{_COHORT_GLM_ROUTER}::test_selected"
    untouched = f"{_COHORT_GLM_ROUTER}::test_untouched"
    sibling = f"{_COHORT_GLM}/gemm/test_dense_gemm.py"
    unrelated = f"{_COHORT_GLM}/ops/moe/test_experts.py"
    rule = _make_cohort_rule(
        tmp_path,
        {"l0_gr100": {selected, untouched, sibling, unrelated}},
        frozenset({"l0_gr100"}),
    )
    assert rule._compute_anchors(git_path, _COHORT_GLM_ROUTER, diff) == [
        selected if scoped else _COHORT_GLM_ROUTER
    ]
    result = rule.apply(PRInputs(changed_files=[git_path], diffs={git_path: diff}))
    assert result is not None
    expected = {selected} if scoped else {selected, untouched, sibling}
    assert _selected_entries(result) == {("l0_gr100", 0): expected}
    assert result.affected_stages == {"l0_gr100-post_merge"}


@pytest.fixture(scope="module")
def modeling_v2_cohort_real_rule() -> CbtsTestsDefRule:
    repo_root = Path(__file__).resolve().parents[4]
    return CbtsTestsDefRule(
        YAMLIndex.load(repo_root / "tests/integration/test_lists/test-db"),
        parse_stages_from_groovy(repo_root / "jenkins/L0_Test.groovy", include_post_merge=True),
        repo_root,
    )


def test_modeling_v2_cohort_real_glm_gemm_excludes_rubin_moe(
    modeling_v2_cohort_real_rule: CbtsTestsDefRule,
) -> None:
    result = modeling_v2_cohort_real_rule.apply(
        PRInputs(changed_files=[f"tests/{_COHORT_GLM_ROUTER}"], diffs={})
    )
    assert result is not None
    entries = _selected_entries(result)
    assert entries[("l0_b300", 0)] == {f"{_COHORT_GLM} TIMEOUT (30)"}
    assert entries[("l0_gr100", 0)] == {
        f"{_COHORT_GLM}/gemm",
        f"{_COHORT_GLM}/ops/gemm",
    }
    assert result.affected_stages == {
        "B300-PyTorch-1",
        "B300-PyTorch-2",
        "GR100-PyTorch-Post-Merge-1",
    }


def test_modeling_v2_cohort_real_k3_helper_keeps_root_checks(
    modeling_v2_cohort_real_rule: CbtsTestsDefRule,
) -> None:
    result = modeling_v2_cohort_real_rule.apply(
        PRInputs(changed_files=[f"tests/{_COHORT_K3}/ops/moe/_k3_moe_engines.py"], diffs={})
    )
    assert result is not None
    entries = _selected_entries(result)
    assert {
        f"{_COHORT_K3}/ops/moe/test_modeling_v2_k3_moe_m1.py",
        f"{_COHORT_K3}/test_tp16_moetp16ep1_construction.py",
    } <= entries[("l0_b300", 0)]
    assert any(stem == "l0_cpu" for stem, _ in entries)
    assert any(stem == "l0_gb300" for stem, _ in entries)
    assert all(
        entry.startswith(_COHORT_K3 + "/") for selected in entries.values() for entry in selected
    )


def test_modeling_v2_cohort_real_tp2_moe_keeps_only_moe_group(
    modeling_v2_cohort_real_rule: CbtsTestsDefRule,
) -> None:
    target = _m3_target(2)
    result = modeling_v2_cohort_real_rule.apply(
        PRInputs(changed_files=[f"tests/{target}/ops/moe/_m3_moe_operands.py"], diffs={})
    )
    assert result is not None
    expected = {
        path for path in _m3_moe_cohort(2) if not path.startswith(f"{target}/ops/m3_moe/")
    } | {f"{target}/ops/m3_moe"}
    assert _selected_entries(result) == {("l0_gr100", 0): expected}
    assert result.affected_stages == {"GR100-PyTorch-Post-Merge-1"}
