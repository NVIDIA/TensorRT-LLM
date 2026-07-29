# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fused Wan DupUp3D output mapping."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import DupUp3D


@pytest.mark.parametrize("first_chunk", [False, True])
@pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last_3d])
@pytest.mark.parametrize(
    ("in_channels", "out_channels", "factor_t"),
    [
        pytest.param(4, 4, 2, id="temporal-repeat-8"),
        pytest.param(8, 4, 2, id="temporal-repeat-4"),
        pytest.param(8, 2, 2, id="temporal-repeat-2"),
        pytest.param(4, 4, 1, id="spatial-repeat-4"),
        pytest.param(8, 4, 1, id="spatial-repeat-2"),
    ],
)
def test_fused_dup_up3d_matches_eager(
    first_chunk: bool,
    memory_format: torch.memory_format,
    in_channels: int,
    out_channels: int,
    factor_t: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the fused DupUp3D kernel")

    x = torch.randn(
        2,
        in_channels,
        3,
        4,
        5,
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous(memory_format=memory_format)
    module = DupUp3D(in_channels, out_channels, factor_t=factor_t, factor_s=2)

    expected = module(x.cpu(), first_chunk=first_chunk).cuda()
    actual = module(x, first_chunk=first_chunk)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
