# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs and the tolerance band of the Kimi K3.1 attn_res mix tests: the family's entries and
kernel, the engine's persistent op and the targets' moe-add and mailbox entries."""

import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm._attn_res_common import (  # noqa: E501
    HIDDEN,
)

RMS_EPS, OUT_EPS = 1e-6, 1e-5


def mix_inputs(spec, seed):
    """(layer / prefix, addend, block, (res_w, rms_w, out_w)) of ``spec``'s rows and candidates, BF16."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, n = spec["rows"], spec["candidates"]

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.bfloat16)

    layer, addend, block = r(m, HIDDEN, scale=4.0), r(m, HIDDEN), r(n - 1, m, HIDDEN, scale=4.0)
    weights = (r(HIDDEN, scale=0.05), (1 + r(HIDDEN, scale=0.1).float()).bfloat16(), r(HIDDEN))
    return layer, addend, block, weights


def assert_mix_close(
    got: torch.Tensor,
    ref: torch.Tensor,
    out_w: torch.Tensor,
    max_fraction: float | None = 0.005,
) -> float:
    """The reference rounds where the kernel does (mixture, normalized row, product with the
    weight) from FP32 math summed in another order, so the mixture and the normalized row can each
    land one BF16 step away -- at the row's scale (unit RMS after the norm) where the mixture
    cancels: within ``2 ulp(max(1, |normed|)) * |out_w| + ulp(out)``, and in fewer than
    ``max_fraction`` of the elements (None: any number of them). Returns the fraction that differs.
    """
    g, w = got.float(), ref.float()
    weight = out_w.float().abs()

    def ulp(x):
        return torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(2.0**-126))) - 7)

    normed = (w.abs() / weight.clamp_min(2.0**-20)).clamp_min(1.0)
    tol = 2 * ulp(normed) * weight + ulp(torch.maximum(g.abs(), w.abs()))
    diff = (g - w).abs()
    assert torch.all(diff <= tol * 1.0001), f"max diff {diff.max().item()}"
    fraction = (diff > 0).float().mean().item()
    if max_fraction is not None:
        assert fraction < max_fraction, f"{fraction:.4%} of the elements differ"
    return fraction
