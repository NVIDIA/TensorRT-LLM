# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_MODULE_PATH = _ROOT / "tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe/helpers/megamoe_aot.py"
_SPEC = importlib.util.spec_from_file_location("megamoe_aot", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Kernel:
    def __init__(self, uses_global_scale, helper_expert_count):
        self.quant_kind = SimpleNamespace(uses_global_scale=uses_global_scale)
        self.helper_expert_count = helper_expert_count

    def __call__(
        self,
        activation,
        stream,
        fc1_alpha=None,
        fc2_alpha=None,
        fc1_norm_const=None,
        hot_expert_weight_ready_flags=None,
        hot_expert_weight_ready_generation=None,
    ):
        pass


@pytest.mark.parametrize(
    ("uses_global_scale", "helper_expert_count", "expected"),
    [
        (False, 0, ["activation", "stream"]),
        (
            True,
            0,
            ["activation", "stream", "fc1_alpha", "fc2_alpha", "fc1_norm_const"],
        ),
        (
            False,
            2,
            [
                "activation",
                "stream",
                "hot_expert_weight_ready_flags",
                "hot_expert_weight_ready_generation",
            ],
        ),
        (
            True,
            2,
            [
                "activation",
                "stream",
                "fc1_alpha",
                "fc2_alpha",
                "fc1_norm_const",
                "hot_expert_weight_ready_flags",
                "hot_expert_weight_ready_generation",
            ],
        ),
    ],
)
def test_aot_argument_names_match_compiled_variant(
    uses_global_scale, helper_expert_count, expected
):
    kernel = _Kernel(uses_global_scale, helper_expert_count)
    assert _MODULE.megamoe_aot_argument_names(kernel) == expected


_KERNEL_PATHS = (
    "kernel_src/blackwell/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py",
    "kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py",
    "kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py",
    "kernel_src/rubin/inference/local_mega/block_scaled_swap_ab_local_mega_moe_kernel.py",
)


def test_aot_optional_arguments_are_built_in_signature_order():
    import ast

    expected = [
        "fc1_alpha",
        "fc2_alpha",
        "fc1_norm_const",
        "hot_expert_weight_ready_flags",
        "hot_expert_weight_ready_generation",
    ]
    source_root = _ROOT / "tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe"
    for relative_path in _KERNEL_PATHS:
        tree = ast.parse((source_root / relative_path).read_text())
        kernel_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(member, ast.FunctionDef) and member.name == "aot_compile"
                for member in node.body
            )
        )
        constructor = next(
            node
            for node in kernel_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        assert not any(
            isinstance(node, ast.Name) and node.id == "fake_arguments"
            for node in ast.walk(constructor)
        )

        aot_compile = next(
            node
            for node in kernel_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "aot_compile"
        )
        optional_arguments = []
        statements = []
        for statement in aot_compile.body:
            statements.extend(statement.body if isinstance(statement, ast.If) else (statement,))
        for statement in statements:
            call = statement.value if isinstance(statement, ast.Expr) else None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "update"
            ):
                optional_arguments.extend(keyword.arg for keyword in call.keywords)
        assert optional_arguments == expected
