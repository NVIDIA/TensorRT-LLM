# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.custom_ops import flashinfer_custom_ops  # noqa: F401
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE


@pytest.mark.skipif(
    not IS_FLASHINFER_AVAILABLE or not torch.cuda.is_available(),
    reason="FlashInfer and CUDA are required",
)
@pytest.mark.parametrize("op_name", ["flashinfer_silu_and_mul", "flashinfer_gelu_tanh_and_mul"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gated_activation_strided_input(op_name, dtype):
    # Padding between rows must not enter the activation calculation.
    base = torch.full((4, 160), 77, device="cuda", dtype=dtype)
    x = base[:, :128]
    x.copy_(torch.linspace(-2, 2, x.numel(), device=x.device).reshape(x.shape))
    assert not x.is_contiguous()

    op = getattr(torch.ops.trtllm, op_name)
    torch.testing.assert_close(op(x), op(x.contiguous()))
