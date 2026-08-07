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

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _attribute_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_path(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _contains_attribute(node: ast.AST, attribute_path: str) -> bool:
    return any(_attribute_path(child) == attribute_path for child in ast.walk(node))


def _forward_mlp_has_nvfp4_branch(model_file: str) -> ast.If:
    source = (_REPO_ROOT / "tensorrt_llm" / "_torch" / "models" / model_file).read_text()
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == "forward_mlp":
            for child in ast.walk(node):
                if (_attribute_path(getattr(child, "test", ast.Constant(None)))
                        == "self.mlp.gate_up_proj.has_nvfp4"):
                    assert isinstance(child, ast.If)
                    return child
    raise AssertionError(f"{model_file} does not guard PRE_MLP NVFP4 fusion")


@pytest.mark.parametrize(
    "model_file",
    [
        "modeling_deepseekv3.py",
        "modeling_glm.py",
        "modeling_exaone_moe.py",
    ],
)
def test_pre_mlp_nvfp4_fusion_guards_unquantized_dense_mlp(model_file: str) -> None:
    nvfp4_branch = _forward_mlp_has_nvfp4_branch(model_file)

    assert _contains_attribute(nvfp4_branch, "self.mlp.gate_up_proj.input_scale")
    assert _contains_attribute(nvfp4_branch, "AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4")

    for false_branch_node in nvfp4_branch.orelse:
        assert not _contains_attribute(false_branch_node, "self.mlp.gate_up_proj.input_scale")
        assert not _contains_attribute(false_branch_node, "AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4")
    assert any(
        _contains_attribute(false_branch_node, "AllReduceFusionOp.RESIDUAL_RMS_NORM")
        for false_branch_node in nvfp4_branch.orelse)
