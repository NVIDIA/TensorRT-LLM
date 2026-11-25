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

import tensorrt_llm._torch.models as _models


_MODELS_DIR = Path(_models.__file__).parent


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


def _nvfp4_aliases_assigned_by_statement(node: ast.AST) -> set[str]:
    aliases: set[str] = set()
    if isinstance(node, ast.Assign) and _mentions_nvfp4_flag(node.value):
        for target in node.targets:
            if isinstance(target, ast.Name):
                aliases.add(target.id)
    elif isinstance(node, ast.AnnAssign) and node.value is not None and _mentions_nvfp4_flag(node.value):
        target = node.target
        if isinstance(target, ast.Name):
            aliases.add(target.id)
    return aliases


def _is_pre_mlp_nvfp4_branch(node: ast.If, aliases: set[str]) -> bool:
    test_uses_nvfp4 = _mentions_nvfp4_flag(node.test) or (
        isinstance(node.test, ast.Name) and node.test.id in aliases)
    if not test_uses_nvfp4:
        return False
    if not _contains_attribute(node, "self.mlp.gate_up_proj.input_scale"):
        return False
    if not _contains_attribute(node, "AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4"):
        return False
    return any(
        _contains_attribute(false_branch_node, "AllReduceFusionOp.RESIDUAL_RMS_NORM")
        for false_branch_node in node.orelse)


def _scan_statement_list_for_nvfp4_branch(statements: list[ast.stmt], aliases: set[str]) -> ast.If | None:
    visible_aliases = set(aliases)
    for stmt in statements:
        if isinstance(stmt, ast.If):
            if _is_pre_mlp_nvfp4_branch(stmt, visible_aliases):
                return stmt
            for branch in (stmt.body, stmt.orelse):
                branch_match = _scan_statement_list_for_nvfp4_branch(branch, visible_aliases)
                if branch_match is not None:
                    return branch_match
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
            branch_match = _scan_statement_list_for_nvfp4_branch(stmt.body, visible_aliases)
            if branch_match is not None:
                return branch_match
            if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                branch_match = _scan_statement_list_for_nvfp4_branch(stmt.orelse, visible_aliases)
                if branch_match is not None:
                    return branch_match
        elif isinstance(stmt, ast.Try):
            for branch in (stmt.body, stmt.orelse, stmt.finalbody, *(handler.body for handler in stmt.handlers)):
                branch_match = _scan_statement_list_for_nvfp4_branch(branch, visible_aliases)
                if branch_match is not None:
                    return branch_match
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                branch_match = _scan_statement_list_for_nvfp4_branch(case.body, visible_aliases)
                if branch_match is not None:
                    return branch_match

        visible_aliases.update(_nvfp4_aliases_assigned_by_statement(stmt))
    return None


def _forward_mlp_has_nvfp4_branch(model_file: str, models_dir: Path = _MODELS_DIR) -> ast.If:
    source = (models_dir / model_file).read_text()
    module = ast.parse(source)

    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == "forward_mlp":
            nvfp4_branch = _scan_statement_list_for_nvfp4_branch(node.body, set())
            if nvfp4_branch is not None:
                return nvfp4_branch
    raise AssertionError(
        f"{model_file} does not guard PRE_MLP NVFP4 fusion. "
        "Expected a branch that gates RESIDUAL_RMS_NORM_QUANT_NVFP4 on has_nvfp4, "
        "uses self.mlp.gate_up_proj.input_scale in that branch, and falls back to "
        "RESIDUAL_RMS_NORM otherwise; the has_nvfp4 check supports direct attributes, "
        "getattr(..., \"has_nvfp4\", ...), or an alias assigned before the branch."
    )



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
                        use_nvfp4 = getattr(gate_up_proj, "has_nvfp4", False)
                        if use_nvfp4:
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

    nvfp4_branch = _forward_mlp_has_nvfp4_branch("modeling_alias.py", model_dir)

    assert _contains_attribute(nvfp4_branch, "self.mlp.gate_up_proj.input_scale")
    assert _contains_attribute(nvfp4_branch, "AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4")
    assert any(
        _contains_attribute(false_branch_node, "AllReduceFusionOp.RESIDUAL_RMS_NORM")
        for false_branch_node in nvfp4_branch.orelse)


def _write_model_fixture(repo_root: Path, model_name: str, forward_mlp_body: str) -> None:
    model_dir = repo_root / "tensorrt_llm" / "_torch" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / model_name).write_text(
        "class Dummy:\n" + textwrap.indent(textwrap.dedent(forward_mlp_body), "    ")
    )


def test_forward_mlp_alias_must_be_defined_before_candidate_branch(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    _write_model_fixture(
        repo_root,
        "modeling_alias_use_before_assignment.py",
        '''
        def forward_mlp(self):
            if use_nvfp4:
                self.allreduce(
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                        scale=self.mlp.gate_up_proj.input_scale,
                    ),
                )
            else:
                self.allreduce(
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    ),
                )
            use_nvfp4 = self.mlp.gate_up_proj.has_nvfp4
        ''',
    )

    model_dir = repo_root / "tensorrt_llm" / "_torch" / "models"
    with pytest.raises(AssertionError, match="does not guard PRE_MLP NVFP4 fusion"):
        _forward_mlp_has_nvfp4_branch("modeling_alias_use_before_assignment.py", model_dir)


def test_forward_mlp_alias_must_not_leak_from_sibling_branch(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    _write_model_fixture(
        repo_root,
        "modeling_alias_sibling_branch.py",
        '''
        def forward_mlp(self):
            if self.fusion_config.PRE_MLP_FUSION:
                use_nvfp4 = self.mlp.gate_up_proj.has_nvfp4
            if use_nvfp4:
                self.allreduce(
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                        scale=self.mlp.gate_up_proj.input_scale,
                    ),
                )
            else:
                self.allreduce(
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    ),
                )
        ''',
    )

    model_dir = repo_root / "tensorrt_llm" / "_torch" / "models"
    with pytest.raises(AssertionError, match="does not guard PRE_MLP NVFP4 fusion"):
        _forward_mlp_has_nvfp4_branch("modeling_alias_sibling_branch.py", model_dir)
