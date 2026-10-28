# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference precision scopes preserve the surrounding test's matmul policy."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

__extra_import_path__ = ["."]
import _precision as precision  # noqa: E402

pytestmark = pytest.mark.cpu_only


class _MatmulPolicy:
    def __init__(self, precision: str) -> None:
        self.fp32_precision = precision

    @property
    def allow_tf32(self) -> bool:
        raise AssertionError("reference precision must not use the legacy TF32 API")

    @allow_tf32.setter
    def allow_tf32(self, value: bool) -> None:
        raise AssertionError("reference precision must not use the legacy TF32 API")


@pytest.fixture(params=["ieee", "tf32", "bfx9"])
def matmul_policy(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _MatmulPolicy:
    policy = _MatmulPolicy(request.param)
    monkeypatch.setattr(
        precision,
        "torch",
        SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=policy))),
    )
    return policy


@pytest.mark.parametrize("raises", [False, True], ids=["success", "failure"])
def test_reference_scope_restores_precision(matmul_policy: _MatmulPolicy, raises: bool) -> None:
    previous = matmul_policy.fp32_precision
    expected = pytest.raises(RuntimeError, match="reference failed") if raises else nullcontext()
    with expected:
        with precision.true_fp32_matmul():
            assert matmul_policy.fp32_precision == "ieee"
            if raises:
                raise RuntimeError("reference failed")
    assert matmul_policy.fp32_precision == previous


def test_nested_reference_failure_restores_the_outer_scope(matmul_policy: _MatmulPolicy) -> None:
    previous = matmul_policy.fp32_precision
    with precision.true_fp32_matmul():
        with pytest.raises(RuntimeError, match="nested reference failed"):
            with precision.true_fp32_matmul():
                assert matmul_policy.fp32_precision == "ieee"
                raise RuntimeError("nested reference failed")
        assert matmul_policy.fp32_precision == "ieee"
    assert matmul_policy.fp32_precision == previous


def test_reference_decorator_restores_after_repeated_calls(matmul_policy: _MatmulPolicy) -> None:
    previous = matmul_policy.fp32_precision

    @precision.true_fp32_matmul()
    def reference(raises: bool = False) -> int:
        assert matmul_policy.fp32_precision == "ieee"
        if raises:
            raise RuntimeError("decorated reference failed")
        return 42

    assert reference() == 42
    assert matmul_policy.fp32_precision == previous
    with pytest.raises(RuntimeError, match="decorated reference failed"):
        reference(raises=True)
    assert matmul_policy.fp32_precision == previous
    assert reference() == 42
    assert matmul_policy.fp32_precision == previous


@pytest.mark.parametrize("raises", [False, True], ids=["success", "failure"])
def test_torch_backend_precision_is_restored(raises: bool) -> None:
    matmul = precision.torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    expected = pytest.raises(RuntimeError, match="reference failed") if raises else nullcontext()
    try:
        with expected:
            with precision.true_fp32_matmul():
                assert matmul.fp32_precision == "ieee"
                if raises:
                    raise RuntimeError("reference failed")
        assert matmul.fp32_precision == previous
    finally:
        matmul.fp32_precision = previous


@pytest.mark.parametrize(
    "relative_path",
    [
        "moe/test_modeling_v2_fp4_block_scale_moe_runner.py",
        "moe/test_modeling_v2_mxe4m3_mxe2m1_block_scale_moe_runner.py",
        "models/deepseek_v3/r1_0528_nvfp4__sm_10x__default/moe/test_modeling_v2_fused_moe.py",
    ],
)
def test_moe_collection_does_not_assign_precision(relative_path: str) -> None:
    source = Path(__file__).parent / relative_path
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            assert not (
                isinstance(target, ast.Attribute)
                and target.attr in ("allow_tf32", "fp32_precision")
            ), f"{relative_path}:{node.lineno}: collection mutates process-wide precision"


@pytest.mark.parametrize(
    "relative_path",
    [
        "catalog/moe/fp4_block_scale_moe_runner.py",
        "catalog/moe/mxe4m3_mxe2m1_block_scale_moe_runner.py",
        "models/deepseek_v3/r1_0528_nvfp4__sm_10x__default/ops/moe/fused_moe.py",
    ],
)
def test_moe_reference_does_not_override_precision(relative_path: str) -> None:
    root = Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/_experimental/modeling_v2"
    tree = ast.parse((root / relative_path).read_text())
    references = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "reference"
    ]
    assert references, relative_path
    for reference in references:
        assert not any(
            isinstance(node, ast.Name) and node.id == "true_fp32_matmul"
            for node in ast.walk(reference)
        ), f"{relative_path}: reference precision belongs to the test caller"


@pytest.mark.parametrize(
    "relative_path",
    [
        "moe/test_modeling_v2_fp4_block_scale_moe_runner.py",
        "moe/test_modeling_v2_mxe4m3_mxe2m1_block_scale_moe_runner.py",
        "models/deepseek_v3/r1_0528_nvfp4__sm_10x__default/moe/test_modeling_v2_fused_moe.py",
        "models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__tp1/moe/"
        "test_modeling_v2_fp4_block_scale_moe_renorm_routed.py",
        "models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__tp1/moe/"
        "test_modeling_v2_fp4_block_scale_moe_renorm_permuted.py",
        "models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__tp1/moe/"
        "test_modeling_v2_fp4_moe_renorm_cute_dsl.py",
    ],
)
def test_moe_test_reference_calls_scope_precision(relative_path: str) -> None:
    tree = ast.parse((Path(__file__).parent / relative_path).read_text())
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reference"
    ]
    assert calls, relative_path
    for call in calls:
        node = call
        scoped = False
        while id(node) in parents:
            node = parents[id(node)]
            if isinstance(node, ast.With):
                scoped |= any(
                    isinstance(item.context_expr, ast.Call)
                    and isinstance(item.context_expr.func, ast.Name)
                    and item.context_expr.func.id == "true_fp32_matmul"
                    for item in node.items
                )
        assert scoped, (
            f"{relative_path}:{call.lineno}: reference needs a test-owned precision scope"
        )
