# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3_image.text_encoder import _normalize, _rotate


def test_conditioning_norm_rounds_before_scale():
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(17, 128, generator=gen).bfloat16()
    scale = torch.randn(128, generator=gen).bfloat16()
    unit = x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    expected = unit.bfloat16() * scale
    assert not torch.equal(expected, (unit * scale.float()).bfloat16())
    norm = SimpleNamespace(variance_epsilon=1e-6, weight=scale)
    torch.testing.assert_close(_normalize(x, norm), expected, atol=0, rtol=0)


def test_conditioning_rotary_rounds_each_product():
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(11, 4, 128, generator=gen).bfloat16()
    angles = torch.randn(11, 1, 64, generator=gen).repeat(1, 1, 2)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    a, b = x.chunk(2, -1)
    c, s = cos[..., :64], sin[..., :64]
    expected = torch.cat((a * c - b * s, b * c + a * s), -1)
    torch.testing.assert_close(_rotate(x, cos, sin), expected, atol=0, rtol=0)
