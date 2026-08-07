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
import textwrap
from pathlib import Path

import pytest


def _attribute_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_path(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _contains_attribute(node: ast.AST, attribute_path: str) -> bool:
    return any(_attribute_path(child) == attribute_path for child in ast.walk(node))


def _mentions_nvfp4_flag(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == "has_nvfp4":
            return True
        if isinstance(child, ast.Name) and child.id == "has_nvfp4":
            return True
        if isinstance(child, ast.Constant) and child.value == "has_nvfp4":
            return True
    return False


def _forward_mlp_nvfp4_aliases(function_node: ast.FunctionDef) -> set[str]:
    aliases: set[str] = set()
    for stmt in function_node.body:
        if isinstance(stmt, ast.Assign) and _mentions_nvfp4_flag(stmt.value):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    aliases.add(target.id)
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None and _mentions_nvfp4_flag(stmt.value):
            target = stmt.target
            if isinstance(target, ast.Name):
                aliases.add(target.id)
    return aliases


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "tensorrt_llm" / "_torch" / "models").exists():
            return candidate
    raise AssertionError(f"Could not locate repo root from {start}")


def _forward_mlp_has_nvfp4_branch(model_file: str, repo_root: Path) -> ast.If:
    source = (repo_root / "tensorrt_llm" / "_torch" / "models" / model_file).read_text()
    module = ast.parse(source)

    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == "forward_mlp":
            aliases = _forward_mlp_nvfp4_aliases(node)
            for child in ast.walk(node):
                if not isinstance(child, ast.If):
                    continue
                if not (_mentions_nvfp4_flag(child.test) or (
                    isinstance(child.test, ast.Name) and child.test.id in aliases)):
                    continue
                if not _contains_attribute(child, "self.mlp.gate_up_proj.input_scale"):
                    continue
                if not _contains_attribute(child, "AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4"):
                    continue
                if not any(_contains_attribute(false_branch_node, "AllReduceFusionOp.RESIDUAL_RMS_NORM")
                           for false_branch_node in child.orelse):
                    continue
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
def test_pre_mlp_nvfp4_fusion_guards_unquantized_dense_mlp(pytestconfig, model_file: str) -> None:
    repo_root = _find_repo_root(pytestconfig.rootpath)
    nvfp4_branch = _forward_mlp_has_nvfp4_branch(model_file, repo_root)

    assert _contains_attribute(nvfp4_branch, "self.mlp.gate_up_proj.input_scale")
    assert _contains_attribute(nvfp4_branch, "AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4")

    for false_branch_node in nvfp4_branch.orelse:
        assert not _contains_attribute(false_branch_node, "self.mlp.gate_up_proj.input_scale")
        assert not _contains_attribute(false_branch_node, "AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4")
    assert any(
        _contains_attribute(false_branch_node, "AllReduceFusionOp.RESIDUAL_RMS_NORM")
        for false_branch_node in nvfp4_branch.orelse)


def test_forward_mlp_has_nvfp4_branch_supports_alias_and_getattr(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    model_dir = repo_root / "tensorrt_llm" / "_torch" / "models"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "modeling_alias.py"
    model_file.write_text(
        textwrap.dedent(
            '''
            class Dummy:
                def forward_mlp(self):
                    if self.fusion_config.PRE_MLP_FUSION:
                        gate_up_proj = self.mlp.gate_up_proj
                        has_nvfp4 = getattr(gate_up_proj, "has_nvfp4", False)
                        if has_nvfp4:
                            act_fp4, act_sf, residual = self.allreduce(
                                hidden_states,
                                all_reduce_params=AllReduceParams(
                                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                                    residual=residual,
                                    norm_weight=self.post_attention_layernorm.weight,
                                    scale=self.mlp.gate_up_proj.input_scale,
                                    eps=self.post_attention_layernorm.variance_epsilon,
                                ),
                            )
                            hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
                        else:
                            hidden_states, residual = self.allreduce(
                                hidden_states,
                                all_reduce_params=AllReduceParams(
                                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                                    residual=residual,
                                    norm_weight=self.post_attention_layernorm.weight,
                                    eps=self.post_attention_layernorm.variance_epsilon,
                                ),
                            )
            '''
        )
    )

    nvfp4_branch = _forward_mlp_has_nvfp4_branch("modeling_alias.py", repo_root)

    assert _contains_attribute(nvfp4_branch, "self.mlp.gate_up_proj.input_scale")
    assert _contains_attribute(nvfp4_branch, "AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4")
    assert any(
        _contains_attribute(false_branch_node, "AllReduceFusionOp.RESIDUAL_RMS_NORM")
        for false_branch_node in nvfp4_branch.orelse)
