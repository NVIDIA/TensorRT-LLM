# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.ops.cute_bf16_gemm import (
    cute_bf16_gemm,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires Blackwell SM100 or SM103",
)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("cell", cute_bf16_gemm.CELLS)
def test_cute_bf16_gemm(cell, compiled):
    x = torch.randint(-2, 3, (cell.spec["m"], cell.spec["k"]), device="cuda").bfloat16()
    weight = torch.randint(-2, 3, (cell.spec["n"], cell.spec["k"]), device="cuda").bfloat16()
    cute_bf16_gemm.is_valid(x, weight)
    op = torch.compile(cute_bf16_gemm, fullgraph=True) if compiled else cute_bf16_gemm
    torch.testing.assert_close(op(x, weight), cute_bf16_gemm.reference(x, weight), rtol=0, atol=0)
