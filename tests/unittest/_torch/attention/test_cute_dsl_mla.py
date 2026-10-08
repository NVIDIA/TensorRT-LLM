# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side contract tests for the CuTe DSL MLA FMHA library's FP8 scales."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE


def test_fp8_mla_scales_remain_tensors() -> None:
    """FP8 scales reach the kernel as the same device tensors, never as host floats."""
    q = torch.empty(1)
    bmm1_scale = torch.tensor([0.25, 0.25 * 1.442695], dtype=torch.float32)
    bmm2_scale = torch.tensor([4.0], dtype=torch.float32)

    actual_bmm1_scale, actual_bmm2_scale = CuteDslMlaFmha._get_fp8_scales(
        q,
        SimpleNamespace(mla_bmm1_scale=bmm1_scale, mla_bmm2_scale=bmm2_scale),
    )

    assert actual_bmm1_scale is bmm1_scale
    assert actual_bmm2_scale is bmm2_scale


@pytest.mark.parametrize(
    "bmm1_scale, bmm2_scale, match",
    [
        (None, torch.tensor([1.0]), "requires mla_bmm1_scale and mla_bmm2_scale"),
        (torch.tensor([1.0], dtype=torch.float16), torch.tensor([1.0]), "float32"),
        (torch.tensor([1.0]), torch.empty(0), "float32"),
    ],
)
def test_fp8_mla_scales_are_validated(bmm1_scale, bmm2_scale, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        CuteDslMlaFmha._get_fp8_scales(
            torch.empty(1),
            SimpleNamespace(mla_bmm1_scale=bmm1_scale, mla_bmm2_scale=bmm2_scale),
        )


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="nvidia-cutlass-dsl is not installed")
def test_fp8_mla_custom_op_accepts_tensor_scales() -> None:
    # Importing the module registers the CuTe DSL custom ops.
    import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops  # noqa: F401

    schema = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell.default._schema
    argument_types = {argument.name: str(argument.type) for argument in schema.arguments}

    assert argument_types["softmax_scale"] == "Tensor"
    assert argument_types["output_scale"] == "Tensor"
