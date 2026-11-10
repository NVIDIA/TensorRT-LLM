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
"""CPU checks of the uGPU kernel port: consumer selection and tail ordering.

Execute production host definitions with CUDA imports replaced by metadata
stubs. These checks establish selection, ABI, and source ordering contracts;
they do not establish generated kernel numerics or GPU synchronization.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_OP = _ROOT / "tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py"
_MEGA = _ROOT / ("tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega")
_GEN = _MEGA / "block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py"
_EPI = _MEGA / "block_scaled_swap_ab_fc12_epilogue.py"


class _KernelMetadata:
    def __init__(self, problem: dict | None = None, implementation: dict | None = None) -> None:
        self.problem = problem
        self.implementation = implementation


class _GenphaseMetadata(_KernelMetadata):
    @classmethod
    def impl_desc_require(cls) -> dict:
        return {"die_aware": str}

    @property
    def token_comm(self) -> SimpleNamespace:
        # Compact rows, as the backend stages them; the construction check compares these.
        hidden = int(self.problem["hidden_size"])
        return SimpleNamespace(
            plain_activation_row_elements=hidden,
            plain_activation_sf_row_elements=-(-hidden // 64) * 4,
        )

    def get_workspace_sizes(self) -> tuple[int, int]:
        return 256, 512


class _GenericMetadata(_KernelMetadata):
    @classmethod
    def impl_desc_require(cls) -> dict:
        return {"weight_layout": str}

    def get_workspace_sizes(self) -> tuple[int, int]:
        return 384, 1024


@pytest.fixture
def op(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load host definitions without importing TRT-LLM or any CUDA package."""
    tree = ast.parse(_OP.read_text())
    selected = (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef)
    nodes = [ast.parse("from __future__ import annotations").body[0]]
    for node in tree.body:
        if isinstance(node, selected):
            nodes.append(node)
        elif (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "IS_MEGAMOE_OP_AVAILABLE"
        ):
            for child in node.body:
                # The torch op registration needs real torch; the host definitions do not.
                if isinstance(child, ast.Assign) and ast.unparse(child.value).startswith(
                    "torch.library."
                ):
                    continue
                if isinstance(child, selected):
                    if isinstance(child, ast.FunctionDef) and child.name in (
                        "cute_dsl_megamoe_nvfp4_blackwell",
                        "_",
                    ):
                        child.decorator_list = []
                    nodes.append(child)
    module = types.ModuleType("_megamoe_kernel_port_cpu")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        dataclasses=dataclasses,
        functools=functools,
        inspect=inspect,
        weakref=weakref,
        torch=SimpleNamespace(bfloat16="torch.bfloat16"),
        cutlass=SimpleNamespace(BFloat16="torch.bfloat16", Int32="int32", Int64="int64"),
        TunableRunner=object,
        logger=SimpleNamespace(debug=lambda *args: None),
        get_sm_version=lambda: 107,
        ceil_div=lambda a, b: (a + b - 1) // b,
        pad_up=lambda a, b: (a + b - 1) // b * b,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_OP), "exec"), module.__dict__)
    module.IS_MEGAMOE_OP_AVAILABLE = True
    module.RubinInferenceGenphaseMegaMoE = _GenphaseMetadata
    module.RubinInferenceMegaMoE = _GenericMetadata
    monkeypatch.setattr(module, "_max_active_clusters", lambda *args: 46)
    return module


def _problem(**overrides: object) -> dict:
    values = dict(
        world_size=4,
        local_rank=0,
        num_topk=2,
        num_experts_per_rank=2,
        hidden_size=1024,
        intermediate_size_per_partition=512,
        expand_intermediate_size_per_partition=1024,
        max_tokens_per_rank=64,
        output_dtype="torch.bfloat16",
        ugpu_weight_layout="none",
    )
    values.update(overrides)
    return values


@pytest.mark.parametrize("layout", ["expert_half", "n_split"])
@pytest.mark.parametrize("capacity", [64, 2048])
def test_bf16_localized_query_and_runner_share_consumer(
    op: types.ModuleType, monkeypatch: pytest.MonkeyPatch, layout: str, capacity: int
) -> None:
    package = "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe"
    imports = {
        "cutlass.cute.nvgpu": {"OperandMajorMode": SimpleNamespace(K="K")},
        "tensorrt_llm._torch.moe.kernels.megamoe_kernels": {
            "BlackwellInferenceMegaMoE": _GenericMetadata,
            "RubinInferenceMegaMoE": _GenericMetadata,
            "RubinInferenceGenphaseMegaMoE": _GenphaseMetadata,
        },
        package + ".api": {"ImplDesc": dict, "ProblemDesc": dict},
        package + ".quant_def": {
            "CombineFormat": SimpleNamespace(parse=lambda value: value),
            "QuantKind": SimpleNamespace(nvfp4=SimpleNamespace(instruction_k=lambda mode: 128)),
        },
    }
    for name, values in imports.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    calls = []
    original = op._construct_megamoe_kernel

    def construct(**kwargs: object) -> tuple:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(op, "_construct_megamoe_kernel", construct)
    problem = _problem(ugpu_weight_layout=layout, max_tokens_per_rank=capacity)
    query = {
        key: value for key, value in problem.items() if key not in {"local_rank", "output_dtype"}
    }
    assert op.query_megamoe_kernel_properties(**query) == (
        (512, True) if capacity <= 1024 else (1024, False)
    )
    runner = op.Sm100MegaMoENvfp4Runner(**problem)
    # Generic (prefill) buckets keep the baseline's default tactic; only genphase pins one.
    tactic = (
        op._megamoe_ugpu_tactic()
        if capacity <= 1024
        else op._default_megamoe_tactic_for_key(runner._problem_key(), num_tokens=capacity)
    )
    assert (tactic == op._megamoe_ugpu_tactic()) == (capacity <= 1024)
    kernel = runner._build_kernel_once(tactic)
    assert kernel.problem["expert_count"] == 8
    assert kernel.problem["hidden_size"] == kernel.problem["intermediate_gateup_size"] == 1024
    if capacity <= 1024:
        assert isinstance(kernel, _GenphaseMetadata)
        assert kernel.implementation["die_aware"] == (
            "on" if layout == "n_split" else "expert_half"
        )
    else:
        assert isinstance(kernel, _GenericMetadata)
        assert kernel.implementation["weight_layout"] == (
            "n_split" if layout == "n_split" else "standard"
        )
    assert len(calls) == 2
    # The query leaves helper_expert_count to the constructor default of 0.
    for call in calls:
        call["common"].setdefault("helper_expert_count", 0)
    assert calls[0] == calls[1]
    assert calls[0]["ugpu_weight_layout"] == layout
    ordinary = op.Sm100MegaMoENvfp4Runner(**{**problem, "ugpu_weight_layout": "none"})
    assert runner.unique_id() != ordinary.unique_id()


def _method(path: Path, name: str) -> ast.FunctionDef:
    class_name = "SwapABGatedActEpilogue" if path == _EPI else "BlockScaledSwapAbGenphaseMoeKernel"
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(node: ast.AST, function: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and ast.unparse(child.func) == function
    ]


def test_role_join_precedes_tail_reset() -> None:
    """Guard the tail ordering: all roles join before the last arriver resets the workspace."""
    kernel = _method(_GEN, "_kernel")
    tails = [
        node
        for node in kernel.body
        if isinstance(node, ast.If) and _calls(node, "self.tail.reset_tail_space")
    ]
    assert len(tails) == 1
    tail = tails[0]
    assert ast.unparse(tail.test) == "warp_idx < len(self.epilogue_warp_ids)"
    tail_index = kernel.body.index(tail)
    joins = [
        node
        for node in kernel.body[:tail_index]
        if isinstance(node, ast.Expr) and _calls(node, "cute.arch.sync_threads")
    ]
    # This join is outside every warp-role branch, including idle/gather/TMA roles.
    assert len(joins) == 1
    resets = _calls(tail, "self.tail.reset_tail_space")
    assert resets and all(call.lineno > joins[0].lineno for call in resets)
    # The tail helper clears both workspace spans.
    helper = next(
        node
        for node in ast.walk(ast.parse((_MEGA / "genphase_tail.py").read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "reset_tail_space"
    )
    spans = {
        ast.literal_eval(call.args[0]) for call in _calls(helper, "workspace.reset_tail_space")
    }
    assert spans == {"shared", "local"}
