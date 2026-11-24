# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stored-form NVFP4 experts, and the layer the engine's NVFP4 CuteDSL MoE method makes of them on
SM107, for the family's MoE entry tests."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    _rubin_moe_common as common,
)

SF_VEC = common.SF_VEC


def experts(num: int, hidden: int, inter: int, seed: int, device="cuda") -> list:
    """Stored-form NVFP4 experts with gate / up / down outputs of order one. ``weight_scale_2``
    differs between experts, so an alpha read for the wrong expert shows."""
    g = torch.Generator(device=device).manual_seed(seed)

    def proj(n, k, scale_2):
        codes = torch.randint(0, 256, (n, k // 2), generator=g, device=device, dtype=torch.uint8)
        steps = torch.randint(120, 132, (n, k // SF_VEC), generator=g, device=device).float()
        scales = (steps / 126.0).to(torch.float8_e4m3fn)
        return codes, scales, scale_2, torch.tensor(1.0, device=device)

    out = []
    for _ in range(num):
        s13, s2 = 0.02 + 0.02 * torch.rand(2, generator=g, device=device)
        gate_up = dict(w1=proj(inter, hidden, s13), w3=proj(inter, hidden, s13))
        out.append(dict(gate_up, w2=proj(hidden, inter, s2)))
    return out


def fc1_rows(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """FC1 rows ``[..., 2 * I, cols]`` as the SM107 kernels read them: groups of 16 rows of the
    gate (w1), each followed by the up (w3) rows of the same 16 outputs."""
    *lead, inter, cols = gate.shape
    groups = (*lead, inter // 16, 16, cols)
    pairs = torch.stack((gate.reshape(groups), up.reshape(groups)), dim=-3)
    return pairs.reshape(*lead, 2 * inter, cols)


def layer(experts: list, hidden: int) -> dict:
    """The layer as the engine's NVFP4CuteDslFusedMoEMethod leaves it after loading on SM107: FC1
    rows interleaved, block scales 128x4 swizzled, global scales ``1 / max(input_scale)`` over the
    rank's experts and per-expert alphas in that method's arithmetic."""
    from tensorrt_llm._torch.utils import swizzle_sf

    local, inter = len(experts), experts[0]["w1"][0].shape[0]

    def stack(w, i):
        return torch.stack([e[w][i] for e in experts])

    g1 = stack("w1", 3).float().max().reciprocal()
    g2 = stack("w2", 3).float().max().reciprocal()
    fc1_s = swizzle_sf(
        fc1_rows(stack("w1", 1).view(torch.uint8), stack("w3", 1).view(torch.uint8)),
        2 * inter,
        hidden,
    )
    fc2_s = swizzle_sf(stack("w2", 1).view(torch.uint8), hidden, inter)
    return dict(
        fc1_w=fc1_rows(stack("w1", 0), stack("w3", 0)).view(torch.float4_e2m1fn_x2),
        fc1_s=fc1_s.view(local, 2 * inter, hidden // SF_VEC),
        fc2_w=stack("w2", 0).view(torch.float4_e2m1fn_x2),
        fc2_s=fc2_s.view(local, hidden, inter // SF_VEC),
        g1=g1,
        g2=g2,
        fc1_alpha=1.0 / (g1 * (1.0 / stack("w1", 2).float())),
        fc2_alpha=1.0 / (g2 * (1.0 / stack("w2", 2).float())),
    )
