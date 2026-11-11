# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script changes retain their test consumers and compose with test-list edits."""

from pathlib import Path

import pytest
import yaml

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex
from cbts.command.main import Selector, build_rules
from cbts.rules.base import PRInputs
from cbts.rules.docs import DOCS_STAGE
from cbts.rules.out_of_scope import OutOfScopeRule
from cbts.rules.test_infrastructure import TestInfrastructureRule as CbtsTestInfrastructureRule

pytestmark = pytest.mark.cpu_only


def _index(entries: list[str]) -> YAMLIndex:
    index = YAMLIndex()
    index.blocks = [Block("l0_cpu", 0, {}, entries)]
    return index


def test_checker_keeps_script_suite_and_indirect_consumers(tmp_path: Path) -> None:
    entries = [
        "unittest/scripts/test_check_test_list.py",
        "unittest/scripts/test_split_qa_by_gpu.py",
        "unittest/scripts/test_other_script.py -k smoke",
        "unittest/others",
    ]
    index = _index(entries)
    stage = Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1)
    stages = {stage.name: stage}
    result = Selector(stages).run(
        PRInputs(changed_files=["scripts/check_test_list.py"], diffs={}),
        build_rules(index, stages, tmp_path),
    )
    assert result.scope == "testinfraonly"
    assert result.affected_stages == {stage.name}
    assert set(result.block_filters[("l0_cpu", 0)]) == set(entries[:2]) | {
        "unittest/scripts/test_other_script.py"
    }
    assert result.sanity_required is False
    assert result.perfsanity_required is False


@pytest.mark.parametrize("entries", [[], ["unittest/others"], ["unittest/scripts"]])
def test_missing_test_mapping_or_serving_stage_falls_back(entries: list[str]) -> None:
    result = CbtsTestInfrastructureRule(_index(entries), {}).apply(
        PRInputs(changed_files=["scripts/check_test_list.py"], diffs={})
    )
    assert result is not None
    assert result.scope is None


def test_unmapped_script_remains_unhandled(tmp_path: Path) -> None:
    index = _index(["unittest/scripts"])
    result = Selector({}).run(
        PRInputs(changed_files=["scripts/other_script.py"], diffs={}),
        build_rules(index, {}, tmp_path),
    )
    assert result.scope is None


def test_checker_composes_with_changed_tests_and_keeps_gpu_residual(tmp_path: Path) -> None:
    (tmp_path / "l0_cpu.yml").write_text(
        yaml.safe_dump({"l0_cpu": [{"tests": ["unittest/scripts", "unittest/others"]}]}),
        encoding="utf-8",
    )
    index = YAMLIndex.load(tmp_path)
    index.blocks.extend(
        Block("l0_b200", number, {}, [f"accuracy/test_model_{number}.py"]) for number in range(2)
    )
    stage = Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1)
    stages = {stage.name: stage}
    paths = ["scripts/check_test_list.py", "tests/unittest/others/test_other.py"]
    selector = Selector(stages)
    result = selector.run(
        PRInputs(changed_files=paths, diffs={}), build_rules(index, stages, tmp_path)
    )
    assert result.scope == "testsonly"
    assert set(result.block_filters[("l0_cpu", 0)]) == {"unittest/scripts", "unittest/others"}

    result = selector.run(
        PRInputs(changed_files=paths + ["cpp/tensorrt_llm/thop/reducescatterOp.cpp"], diffs={}),
        build_rules(index, stages, tmp_path),
    )
    assert result.scope is None


@pytest.mark.parametrize(
    ("source", "expected_entry", "docs_required"),
    (
        ("scripts/split_qa_by_gpu.py", "unittest/scripts", False),
        ("scripts/legacy_utils.py", "unittest/scripts", False),
        ("scripts/templates/precommit-legacy-files.j2", "unittest/scripts", False),
        ("scripts/templates/pyproject-format-exclude.j2", "unittest/scripts", False),
        ("scripts/templates/ruff-legacy.toml.j2", "unittest/scripts", False),
        ("scripts/test_to_stage_mapping.py", "unittest/tools", False),
        ("scripts/disagg_transfer_diagnostics.py", "unittest/tools", False),
        ("scripts/generate_llm_args_golden_manifest.py", "unittest/usage", False),
        ("tensorrt_llm/usage/llm_args_golden_manifest.json", "unittest/usage", True),
    ),
)
def test_reviewed_infrastructure_selects_registered_consumers(
    tmp_path: Path, source: str, expected_entry: str, docs_required: bool
) -> None:
    index = _index(["unittest/scripts", "unittest/tools", "unittest/usage", "unittest/others"])
    stage = Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1)
    stages = {stage.name: stage}
    pr = PRInputs([source], {})
    assert OutOfScopeRule(index, stages, tmp_path).apply(pr) is None

    result = Selector(stages).run(pr, build_rules(index, stages, tmp_path))

    assert result.scope == "testinfraonly"
    assert result.affected_stages == {stage.name} | ({DOCS_STAGE} if docs_required else set())
    assert set(result.block_filters[("l0_cpu", 0)]) == {expected_entry}
    assert not result.sanity_required
    assert not result.perfsanity_required


def test_each_consumer_requires_a_serving_stage() -> None:
    index = _index(["unittest/scripts"])
    index.blocks.append(Block("l0_tools", 0, {}, ["unittest/tools"]))
    stage = Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1)
    rule = CbtsTestInfrastructureRule(index, {stage.name: stage})
    result = rule.apply(
        PRInputs(["scripts/split_qa_by_gpu.py", "scripts/test_to_stage_mapping.py"], {})
    )
    assert result is not None
    assert result.scope is None


def test_cbts_migration_preserves_cpu_policy_and_tools_follow_all_stages() -> None:
    index = _index(["unittest/scripts"])
    index.blocks.append(Block("l0_a10", 0, {}, ["unittest/scripts"]))
    cpu = Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1)
    gpu = Stage("A10-PyTorch-1", "l0_a10", "x86_64", 1, 1)
    rule = CbtsTestInfrastructureRule(index, {stage.name: stage for stage in (cpu, gpu)})
    cbts_path = "jenkins/scripts/cbts/cbts/coverage/selection/selector.py"

    result = rule.apply(PRInputs([cbts_path], {}))
    assert result is not None
    assert result.scope == "testsonly"
    assert result.affected_stages == {cpu.name}
    assert set(result.block_filters) == {("l0_cpu", 0)}

    result = rule.apply(PRInputs([cbts_path, "scripts/split_qa_by_gpu.py"], {}))
    assert result is not None
    assert result.affected_stages == {cpu.name, gpu.name}


def test_manifest_generator_and_artifact_union_consumers(tmp_path: Path) -> None:
    index = _index(["unittest/scripts", "unittest/tools", "unittest/usage"])
    stage = Stage("CPU-Generic-x86-1", "l0_cpu", "x86_64", 1, 1)
    stages = {stage.name: stage}
    paths = [
        "scripts/generate_llm_args_golden_manifest.py",
        "tensorrt_llm/usage/llm_args_golden_manifest.json",
        "scripts/split_qa_by_gpu.py",
    ]
    selector = Selector(stages)
    result = selector.run(PRInputs(paths, {}), build_rules(index, stages, tmp_path))
    assert result.scope == "testinfraonly"
    assert result.affected_stages == {stage.name, DOCS_STAGE}
    assert set(result.block_filters[("l0_cpu", 0)]) == {"unittest/scripts", "unittest/usage"}

    result = selector.run(
        PRInputs(paths + ["tensorrt_llm/llmapi/llm_args.py"], {}),
        build_rules(index, stages, tmp_path),
    )
    assert result.scope is None
    assert "tensorrt_llm/llmapi/llm_args.py" not in selector.handled


@pytest.mark.parametrize(
    "source",
    (
        "scripts/build_wheel.py",
        "scripts/generate_openengine_protos.py",
        "scripts/bolt/manifest.py",
        "scripts/templates/new-lint-config.j2",
        "tensorrt_llm/usage/llmapi_config.py",
        "tensorrt_llm/usage/other_manifest.json",
        "jenkins/scripts/slurm_install.sh",
        "jenkins/scripts/test_rerun.py",
    ),
)
def test_runtime_and_unreviewed_paths_are_not_claimed(source: str) -> None:
    rule = CbtsTestInfrastructureRule(_index(["unittest/scripts", "unittest/usage"]), {})
    assert rule.apply(PRInputs([source], {})) is None
