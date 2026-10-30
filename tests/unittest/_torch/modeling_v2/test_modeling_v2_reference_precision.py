# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference precision scopes preserve the surrounding test's matmul policy."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.catalog import _op

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
        _op,
        "torch",
        SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=policy))),
    )
    return policy


@pytest.mark.parametrize("raises", [False, True], ids=["success", "failure"])
def test_reference_scope_restores_precision(matmul_policy: _MatmulPolicy, raises: bool) -> None:
    previous = matmul_policy.fp32_precision
    expected = pytest.raises(RuntimeError, match="reference failed") if raises else nullcontext()
    with expected:
        with _op.true_fp32_matmul():
            assert matmul_policy.fp32_precision == "ieee"
            if raises:
                raise RuntimeError("reference failed")
    assert matmul_policy.fp32_precision == previous


def test_nested_reference_failure_restores_the_outer_scope(matmul_policy: _MatmulPolicy) -> None:
    previous = matmul_policy.fp32_precision
    with _op.true_fp32_matmul():
        with pytest.raises(RuntimeError, match="nested reference failed"):
            with _op.true_fp32_matmul():
                assert matmul_policy.fp32_precision == "ieee"
                raise RuntimeError("nested reference failed")
        assert matmul_policy.fp32_precision == "ieee"
    assert matmul_policy.fp32_precision == previous


def test_reference_decorator_restores_after_repeated_calls(matmul_policy: _MatmulPolicy) -> None:
    previous = matmul_policy.fp32_precision

    @_op.true_fp32_matmul()
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
    matmul = _op.torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    expected = pytest.raises(RuntimeError, match="reference failed") if raises else nullcontext()
    try:
        with expected:
            with _op.true_fp32_matmul():
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
