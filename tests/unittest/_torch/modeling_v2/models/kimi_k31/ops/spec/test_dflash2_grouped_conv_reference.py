# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 block convolution: per-block causal taps of the torch convolution."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_grouped_conv import (
    _torch_dflash2_grouped_conv as dflash2_grouped_conv,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _reference_conv(x, delta, base, block_size, group_size):
    """``out[t, c] = sum_{tau < taps, t mod B >= tau} (base[tau, c] + delta[t, tau, c // g]) x[t - tau, c]``,
    one block at a time, FP32."""
    taps, hidden = base.shape
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for start in range(0, x.shape[0], block_size):
        for i in range(block_size):
            t = start + i
            for tau in range(min(taps, i + 1)):
                k = base[tau].float() + delta[t, tau].float().repeat_interleave(group_size)
                out[t] += k * x[t - tau].float()
    return out


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size):
    torch.manual_seed(0)
    batch, hidden, group, taps = 3, 64, 16, 2
    n = batch * block_size
    x = torch.randn(n, hidden, device=_DEVICE).to(torch.bfloat16)
    delta = (0.1 * torch.randn(n, taps, hidden // group, device=_DEVICE)).to(torch.bfloat16)
    base = torch.randn(taps, hidden, device=_DEVICE).to(torch.bfloat16)
    got = dflash2_grouped_conv(x, delta, base, block_size, group)
    assert got.dtype == x.dtype
    ref = _reference_conv(x, delta, base, block_size, group)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=1e-2)
