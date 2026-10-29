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

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.compilation.utils import inplace_info
from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)

requires_cute_dsl = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE,
    reason="cutlass-dsl is not available",
)
requires_rubin_cute_dsl = pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="the CuTe DSL Rubin helpers are not available",
)


def _make_fp4_operands() -> dict[str, torch.Tensor]:
    packed_k = 16
    scale_numel = 128 * 4
    weight_0 = torch.empty((8, packed_k), dtype=torch.uint8, device="cuda")
    weight_scale_0 = torch.empty(scale_numel, dtype=torch.uint8, device="cuda")
    return {
        "input": torch.empty((2, packed_k), dtype=torch.uint8, device="cuda"),
        "weight_0": weight_0,
        "weight_1": torch.empty_like(weight_0),
        "input_scale": torch.empty(scale_numel, dtype=torch.uint8, device="cuda"),
        "weight_scale_0": weight_scale_0,
        "weight_scale_1": torch.empty_like(weight_scale_0),
        "alpha": torch.empty((1,), dtype=torch.float32, device="cuda"),
        "output": torch.empty((2, 16), dtype=torch.bfloat16, device="cuda"),
    }


def _single_fp4_operands(
    operands: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        operands["input"],
        operands["weight_0"],
        operands["input_scale"],
        operands["weight_scale_0"],
        operands["alpha"],
    )


def _make_bf16_operands(
    batched: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if batched:
        input_shape = (2, 8, 32)
        weight_shape = (2, 16, 32)
        output_shape = (2, 8, 16)
    else:
        input_shape = (8, 32)
        weight_shape = (16, 32)
        output_shape = (8, 16)

    input_tensor = torch.empty(input_shape, dtype=torch.bfloat16, device="cuda")
    weight_0 = torch.empty(weight_shape, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(output_shape, dtype=torch.bfloat16, device="cuda")
    ugpu_output_shape = (*output_shape[:-1], output_shape[-1] * 2)
    return (
        input_tensor,
        weight_0,
        torch.empty_like(weight_0),
        output,
        torch.empty(ugpu_output_shape, dtype=torch.bfloat16, device="cuda"),
    )


def _schema_argument_names(schema: torch.FunctionSchema) -> list[str]:
    return [argument.name for argument in schema.arguments]


def _schema_defaults(schema: torch.FunctionSchema) -> dict[str, object]:
    return {
        argument.name: argument.default_value
        for argument in schema.arguments
        if argument.has_default_value()
    }


def _schema_mutated_arguments(schema: torch.FunctionSchema) -> set[str]:
    return {
        argument.name
        for argument in schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    }


@requires_rubin_cute_dsl
def test_cute_dsl_nvfp4_legacy_and_inplace_schemas():
    legacy_schema = torch.ops.trtllm.cute_dsl_nvfp4_gemm_rubin.default._schema
    assert _schema_argument_names(legacy_schema) == [
        "input",
        "weight",
        "input_scale",
        "weight_scale",
        "alpha",
        "output_dtype",
        "to_userbuffers",
        "use_tvm_ffi",
        "output_tensor",
        "partition_id",
    ]
    assert _schema_defaults(legacy_schema) == {
        "to_userbuffers": False,
        "use_tvm_ffi": True,
        "output_tensor": None,
        "partition_id": -1,
    }
    assert _schema_mutated_arguments(legacy_schema) == set()
    assert [str(result.type) for result in legacy_schema.returns] == ["Optional[Tensor]"]

    inplace_schema = torch.ops.trtllm.cute_dsl_nvfp4_gemm_inplace_rubin.default._schema
    assert _schema_argument_names(inplace_schema) == [
        "input",
        "weight",
        "input_scale",
        "weight_scale",
        "alpha",
        "output_dtype",
        "to_userbuffers",
        "use_tvm_ffi",
        "output_tensor",
        "partition_id",
        "precomputed_tactic",
    ]
    assert _schema_defaults(inplace_schema) == {"precomputed_tactic": None}
    assert _schema_mutated_arguments(inplace_schema) == {"output_tensor"}
    assert len(inplace_schema.returns) == 0

    ugpu_schema = torch.ops.trtllm.cute_dsl_nvfp4_gemm_ugpu_inplace_rubin.default._schema
    assert _schema_argument_names(ugpu_schema) == [
        "input",
        "weight_0",
        "weight_1",
        "input_scale",
        "weight_scale_0",
        "weight_scale_1",
        "alpha",
        "output_dtype",
        "to_userbuffers",
        "use_tvm_ffi",
        "output_tensor",
    ]
    assert _schema_mutated_arguments(ugpu_schema) == {"output_tensor"}
    assert len(ugpu_schema.returns) == 0
    mutation_map = inplace_info()
    assert mutation_map[torch.ops.trtllm.cute_dsl_nvfp4_gemm_inplace_rubin.default] == {
        1: "output_tensor"
    }
    assert mutation_map[torch.ops.trtllm.cute_dsl_nvfp4_gemm_ugpu_inplace_rubin.default] == {
        1: "output_tensor"
    }


@requires_rubin_cute_dsl
def test_cute_dsl_nvfp4_legacy_and_inplace_fakes():
    legacy_op = torch.ops.trtllm.cute_dsl_nvfp4_gemm_rubin
    inplace_op = torch.ops.trtllm.cute_dsl_nvfp4_gemm_inplace_rubin
    ugpu_op = torch.ops.trtllm.cute_dsl_nvfp4_gemm_ugpu_inplace_rubin

    with FakeTensorMode():
        operands = _make_fp4_operands()
        gemm_operands = _single_fp4_operands(operands)

        result = legacy_op(*gemm_operands, torch.bfloat16)
        assert result is not None
        assert result.shape == (2, 8)
        assert result.dtype == torch.bfloat16

        with pytest.raises(ValueError, match="cute_dsl_nvfp4_gemm_inplace_rubin"):
            legacy_op(
                *gemm_operands,
                torch.bfloat16,
                False,
                True,
                operands["output"],
                0,
            )

        with pytest.raises(ValueError, match="partition_id"):
            legacy_op(
                *gemm_operands,
                torch.bfloat16,
                False,
                True,
                None,
                0,
            )

        result = inplace_op(
            *gemm_operands,
            torch.bfloat16,
            False,
            True,
            operands["output"],
            0,
            repr(-1),
        )
        assert result is None

        result = ugpu_op(
            operands["input"],
            operands["weight_0"],
            operands["weight_1"],
            operands["input_scale"],
            operands["weight_scale_0"],
            operands["weight_scale_1"],
            operands["alpha"],
            torch.bfloat16,
            False,
            True,
            operands["output"],
        )
        assert result is None


@requires_cute_dsl
@pytest.mark.parametrize(
    ("op_kind", "legacy_name", "ugpu_name"),
    [
        (
            "gemm",
            "cute_dsl_bf16_gemm_rubin",
            "cute_dsl_bf16_gemm_ugpu_inplace_rubin",
        ),
        (
            "bmm",
            "cute_dsl_bf16_bmm_rubin",
            "cute_dsl_bf16_bmm_ugpu_inplace_rubin",
        ),
    ],
)
def test_cute_dsl_bf16_legacy_and_ugpu_inplace_schema_and_fake_contract(
    op_kind: str,
    legacy_name: str,
    ugpu_name: str,
):
    legacy_op = getattr(torch.ops.trtllm, legacy_name)
    ugpu_op = getattr(torch.ops.trtllm, ugpu_name)

    legacy_schema = legacy_op.default._schema
    assert _schema_argument_names(legacy_schema) == [
        "input",
        "weight",
        "output",
        "use_tvm_ffi",
    ]
    assert _schema_defaults(legacy_schema) == {"use_tvm_ffi": True}
    assert _schema_mutated_arguments(legacy_schema) == {"output"}
    assert len(legacy_schema.returns) == 0

    ugpu_schema = ugpu_op.default._schema
    assert _schema_argument_names(ugpu_schema) == [
        "input",
        "weight_0",
        "weight_1",
        "output",
        "use_tvm_ffi",
    ]
    assert _schema_defaults(ugpu_schema) == {"use_tvm_ffi": True}
    assert _schema_mutated_arguments(ugpu_schema) == {"output"}
    assert len(ugpu_schema.returns) == 0

    if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        mutation_map = inplace_info()
        assert mutation_map[legacy_op.default] == {1: "output"}
        assert mutation_map[ugpu_op.default] == {1: "output"}

    with FakeTensorMode():
        input_tensor, weight_0, weight_1, output, ugpu_output = _make_bf16_operands(
            batched=op_kind == "bmm"
        )
        assert legacy_op(input_tensor, weight_0, output) is None
        assert ugpu_op(input_tensor, weight_0, weight_1, ugpu_output) is None


@requires_cute_dsl
def test_bf16_gemm_runner_cache_identity_includes_output_dtype():
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    bf16_runner = cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner(
        use_tvm_ffi=True, output_dtype=torch.bfloat16
    )
    fp32_runner = cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner(
        use_tvm_ffi=True, output_dtype=torch.float32
    )

    assert bf16_runner.unique_id() != fp32_runner.unique_id()


@requires_rubin_cute_dsl
@pytest.mark.parametrize(
    "runner_name",
    ["CuteDSLNVFP4RubinLinear", "CuteDSLNVFP4InplaceRubinLinear"],
)
def test_nvfp4_runner_cache_identity_includes_execution_options(runner_name: str):
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    runner_class = getattr(cute_dsl_custom_ops, runner_name)
    default_runner = runner_class(torch.bfloat16, to_userbuffers=False, use_tvm_ffi=True)
    userbuffers_runner = runner_class(torch.bfloat16, to_userbuffers=True, use_tvm_ffi=True)
    torch_ffi_runner = runner_class(torch.bfloat16, to_userbuffers=False, use_tvm_ffi=False)

    assert (
        len(
            {
                default_runner.unique_id(),
                userbuffers_runner.unique_id(),
                torch_ffi_runner.unique_id(),
            }
        )
        == 3
    )
