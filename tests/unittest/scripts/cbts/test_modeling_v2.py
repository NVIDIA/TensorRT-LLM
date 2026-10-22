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
"""ModelingV2Rule selects the tests of the node a change lands in."""

from __future__ import annotations

from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, YAMLIndex
from cbts.rules.base import PRInputs
from cbts.rules.modeling_v2 import ModelingV2Rule

_SRC = "tensorrt_llm/_torch/_experimental/modeling_v2"
_UNIT = "unittest/_torch/modeling_v2"
_GATE_A = "accuracy/modeling_v2/test_fam.py::TestModelingV2A"
_GATE_B = "accuracy/modeling_v2/test_fam.py::TestModelingV2B"

_ENTRIES = (
    f"{_UNIT}/test_modeling_v2_claims.py",
    f"{_UNIT}/norm/test_modeling_v2_rmsnorm.py",
    f"{_UNIT}/gemm/test_modeling_v2_mm.py",
    f"{_UNIT}/models/fam",
    f"{_UNIT}/models/other",
    f"{_UNIT}/comm",
    f"{_GATE_A}::test_gsm8k[tp1]",
    f"{_GATE_B}::test_gsm8k[tp4]",
    "accuracy/modeling_v2/test_other.py::TestModelingV2C::test_gsm8k",
)


def _rule(repo: Path) -> ModelingV2Rule:
    """A tree with family `fam` (targets a, b, and an ops-only dir) and family `other`."""
    files = {
        f"{_SRC}/models/fam/routing.py": (
            "from .a.modeling import ModelingV2A\nfrom .b.modeling import ModelingV2B\n"
        ),
        f"{_SRC}/models/fam/a/modeling.py": "",
        f"{_SRC}/models/fam/a/ops/norm/fused.py": "",
        f"{_SRC}/models/fam/a/cells.py": (
            "from ....catalog._op import Cell\n"
            "from ....catalog.norm.rmsnorm import Rmsnorm\n"
            "CELLS = {Rmsnorm: (Cell(why='x'),)}\n"
        ),
        f"{_SRC}/models/fam/b/modeling.py": "",
        f"{_SRC}/models/fam/b_sm103/ops/gemm.py": "",
        f"{_SRC}/models/other/routing.py": "from .c.modeling import ModelingV2C\n",
        f"{_SRC}/models/other/c/modeling.py": "",
        f"tests/{_UNIT}/norm/test_modeling_v2_rmsnorm.py": "",
    }
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    index = YAMLIndex()
    index.blocks = [Block("l0_b300", 0, {}, list(_ENTRIES))]
    return ModelingV2Rule(index, {}, repo_root=repo)


@pytest.mark.parametrize(
    ("changed", "selected"),
    (
        pytest.param(
            "models/fam/a/ops/norm/fused.py",
            {f"{_UNIT}/models/fam", f"{_GATE_A}::test_gsm8k[tp1]"},
            id="target-runs-its-family-unit-tests-and-only-its-own-gate",
        ),
        pytest.param(
            "models/fam/b_sm103/ops/gemm.py",
            {f"{_UNIT}/models/fam", f"{_GATE_A}::test_gsm8k[tp1]", f"{_GATE_B}::test_gsm8k[tp4]"},
            id="ops-only-dir-belongs-to-its-family",
        ),
        pytest.param(
            "models/fam/routing.py",
            {f"{_UNIT}/models/fam", f"{_GATE_A}::test_gsm8k[tp1]", f"{_GATE_B}::test_gsm8k[tp4]"},
            id="family-file-runs-every-target-of-the-family",
        ),
        pytest.param(
            "models/fam/a/cells.py",
            {f"{_UNIT}/norm/test_modeling_v2_rmsnorm.py"},
            id="cells-run-only-the-ops-they-certify",
        ),
        pytest.param("catalog/norm/rmsnorm.py", set(_ENTRIES), id="catalog-runs-everything"),
    ),
)
def test_change_selects_its_node(tmp_path: Path, changed: str, selected: set[str]) -> None:
    result = _rule(tmp_path).apply(PRInputs(changed_files=[f"{_SRC}/{changed}"], diffs={}))
    kept = {entry for waives in result.block_filters[("l0_b300", 0)].values() for entry in waives}
    assert kept == selected | {f"{_UNIT}/test_modeling_v2_claims.py"}


def test_cells_naming_an_op_without_a_mirrored_test_fall_back_to_the_node(tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    cells = tmp_path / _SRC / "models/fam/a/cells.py"
    cells.write_text("from ....catalog.gemm.unmirrored import Op\n")
    result = rule.apply(PRInputs(changed_files=[f"{_SRC}/models/fam/a/cells.py"], diffs={}))
    kept = {entry for waives in result.block_filters[("l0_b300", 0)].values() for entry in waives}
    assert kept == {
        f"{_UNIT}/test_modeling_v2_claims.py",
        f"{_UNIT}/models/fam",
        f"{_GATE_A}::test_gsm8k[tp1]",
    }
