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
"""ModelingV2Rule selects the tests of the family a change lands in."""

from __future__ import annotations

from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, YAMLIndex
from cbts.rules.base import PRInputs
from cbts.rules.modeling_v2 import ModelingV2Rule

pytestmark = pytest.mark.cpu_only

_SRC = "tensorrt_llm/_torch/_experimental/modeling_v2"
_UNIT = "unittest/_torch/modeling_v2"
_GATE_A = "accuracy/modeling_v2/test_fam.py::TestModelingV2A"
_GATE_B = "accuracy/modeling_v2/test_fam.py::TestModelingV2B"

_ENTRIES = (
    f"{_UNIT}/test_modeling_v2_routing.py",
    f"{_UNIT}/norm/test_modeling_v2_rmsnorm.py",
    f"{_UNIT}/gemm/test_modeling_v2_mm.py",
    f"{_UNIT}/models/fam",
    f"{_UNIT}/models/other",
    f"{_UNIT}/comm",
    f"{_GATE_A}::test_gsm8k[tp1]",
    f"{_GATE_B}::test_gsm8k[tp4]",
    "accuracy/modeling_v2/test_other.py::TestModelingV2C::test_gsm8k",
)

_FAMILY = {f"{_UNIT}/models/fam", f"{_GATE_A}::test_gsm8k[tp1]", f"{_GATE_B}::test_gsm8k[tp4]"}


def _rule() -> ModelingV2Rule:
    index = YAMLIndex()
    index.blocks = [Block("l0_b300", 0, {}, list(_ENTRIES))]
    return ModelingV2Rule(index, {})


@pytest.mark.parametrize(
    ("changed", "selected"),
    (
        pytest.param("models/fam/a/ops/norm/fused.py", _FAMILY, id="target-file-runs-its-family"),
        pytest.param("models/fam/b_sm103/ops/gemm.py", _FAMILY, id="any-directory-runs-its-family"),
        pytest.param("models/fam/routing.py", _FAMILY, id="family-file-runs-its-family"),
        pytest.param("catalog/norm/rmsnorm.py", set(_ENTRIES), id="catalog-runs-everything"),
        pytest.param("models/__init__.py", set(_ENTRIES), id="models-init-runs-everything"),
    ),
)
def test_change_selects_its_family(changed: str, selected: set[str]) -> None:
    result = _rule().apply(PRInputs(changed_files=[f"{_SRC}/{changed}"], diffs={}))
    kept = {entry for waives in result.block_filters[("l0_b300", 0)].values() for entry in waives}
    assert kept == selected | {f"{_UNIT}/test_modeling_v2_routing.py"}


@pytest.mark.parametrize(
    ("changed", "selected"),
    (
        pytest.param(
            "models/fam/a/ops/norm/fused.py",
            {f"{_UNIT}/models/fam/test_shared.py", f"{_UNIT}/models/fam/a/test_a.py"},
            id="target-change-keeps-its-family-test-files",
        ),
        pytest.param(
            "models/fam/routing.py",
            {f"{_UNIT}/models/fam/test_shared.py", f"{_UNIT}/models/fam/a/test_a.py"},
            id="family-change-keeps-the-family-test-files",
        ),
        pytest.param(
            "models/other/c/modeling.py",
            {f"{_UNIT}/models/other/test_o.py"},
            id="other-family-change-leaves-the-family-test-files",
        ),
    ),
)
def test_family_test_files_run_for_any_change_in_the_family(
    changed: str, selected: set[str]
) -> None:
    rule = _rule()
    rule.yaml_index.blocks = [
        Block(
            "l0_cpu",
            0,
            {},
            [
                f"{_UNIT}/test_modeling_v2_routing.py",
                f"{_UNIT}/models/fam/test_shared.py",
                f"{_UNIT}/models/fam/a/test_a.py",
                f"{_UNIT}/models/other/test_o.py",
            ],
        )
    ]
    result = rule.apply(PRInputs(changed_files=[f"{_SRC}/{changed}"], diffs={}))
    kept = {entry for waives in result.block_filters[("l0_cpu", 0)].values() for entry in waives}
    assert kept == selected | {f"{_UNIT}/test_modeling_v2_routing.py"}


@pytest.fixture(scope="module")
def repository_cpu_rule() -> ModelingV2Rule:
    repo = Path(__file__).resolve().parents[4]
    index = YAMLIndex.load(repo / "tests/integration/test_lists/test-db")
    index.blocks = [block for block in index.blocks if block.yaml_stem == "l0_cpu"]
    return ModelingV2Rule(index, {})


@pytest.mark.parametrize(
    "family", ["deepseek_v3", "gpt_oss", "glm_moe_dsa", "kimi_k3_vl", "qwen4_exp"]
)
def test_repository_cpu_selection_keeps_model_tests_in_their_family(
    repository_cpu_rule: ModelingV2Rule, family: str
) -> None:
    result = repository_cpu_rule.apply(
        PRInputs(changed_files=[f"{_SRC}/models/{family}/routing.py"], diffs={})
    )
    kept = {
        entry
        for blocks in result.block_filters.values()
        for entries in blocks.values()
        for entry in entries
    }
    expected_family = f"{_UNIT}/models/{family}/"
    family_tests = {entry for entry in kept if entry.startswith(f"{_UNIT}/models/")}
    assert family_tests
    assert all(entry.startswith(expected_family) for entry in family_tests), family_tests
    assert f"{_UNIT}/test_modeling_v2_routing.py" in kept
    expected_tests = {
        entry
        for block in repository_cpu_rule.yaml_index.blocks
        for entry in block.tests
        if entry.startswith(expected_family)
    }
    assert expected_tests <= kept
