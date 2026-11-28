# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention.cudnn_sdpa import (
    CudnnSdpaRunner,
    cudnn_sdpa,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="certified on SM100",
)
@pytest.mark.parametrize("cell", cudnn_sdpa.CELLS, ids=lambda cell: cell.why)
def test_cudnn_sdpa(cell):
    rows, quant = cell.spec["rows"], cell.spec["quant"]
    runner = CudnnSdpaRunner(
        num_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        qk_dtype=quant,
        v_dtype=quant,
        graphs={},
        scales={},
    )
    generator = torch.Generator(device="cuda").manual_seed(19)
    q, k, v = [
        torch.randn((2, rows, 4, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    ]
    output = cudnn_sdpa(q, k, v, runner=runner)
    expected = cudnn_sdpa.reference(q, k, v, runner=runner)
    assert output.shape == expected.shape and output.dtype == q.dtype
    if quant:
        assert (
            F.cosine_similarity(output.float().flatten(), expected.float().flatten(), dim=0) > 0.995
        )
    else:
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
