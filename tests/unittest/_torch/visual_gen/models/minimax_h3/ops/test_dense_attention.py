# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3.ops.attention import DenseAttentionOp


@pytest.mark.parametrize("cell", DenseAttentionOp.CELLS, ids=lambda cell: cell.spec["backend"])
def test_h3_dense_attention_masks_padded_keys(cell):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Requires SM100")
    spec = cell.spec
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(10)
    shape = (spec["batch"], spec["seq"], spec["heads"], spec["dim"])
    q, k, v = [
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    ]
    mask = (torch.arange(spec["seq"], device=device) < spec["seq"] - 2).expand(spec["batch"], -1)
    v[:, -2:] = 100
    if spec["backend"] == "VANILLA":
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
    kwargs = dict(
        backend=spec["backend"],
        scale=1 / math.sqrt(spec["dim"]),
        mask=mask,
        recipe=None,
        runner=None,
        sparse_params=None,
        timestep=None,
    )
    op = DenseAttentionOp()
    op.is_valid(q, k, v, **kwargs)
    expected = op.reference(q, k, v, **kwargs)
    torch.testing.assert_close(op(q, k, v, **kwargs), expected, rtol=0.02, atol=0.02)
