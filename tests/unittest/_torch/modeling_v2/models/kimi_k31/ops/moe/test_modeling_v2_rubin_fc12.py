# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 MoE reference the family's MoE entries share (``rubin_fc12.reference``), off the GPU: it
equals a token-by-token loop of the same arithmetic.

The entry's cells run on SM107 in test_modeling_v2_rubin_fc12_cells.py, on operands in the layout the
kernel reads.
"""

import itertools

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe import _rubin_moe_common as common
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe import rubin_fc12 as mod
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_fc12 import rubin_fc12

__extra_import_path__ = ["../.."]
from _k31_utils import cpu_or_sm

pytestmark = [pytest.mark.cpu_only, cpu_or_sm(107)]


def _random_experts(num, hidden, inter, seed, input_scale=1.0):
    g = torch.Generator().manual_seed(seed)

    def proj(n, k):
        codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
        scales = (torch.rand(n, k // 16, generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
        return codes, scales, torch.tensor(2.0**-6), torch.tensor(input_scale)

    return [
        {"w1": proj(inter, hidden), "w3": proj(inter, hidden), "w2": proj(hidden, inter)}
        for _ in range(num)
    ]


def _loop_reference(x, ids, weights, experts, offset, beta, linear_beta):
    """Token by token and slot by slot (an independent formulation of the same arithmetic)."""
    g1 = 1.0 / max(float(e[w][3]) for e in experts for w in ("w1", "w3"))
    g2 = 1.0 / max(float(e["w2"][3]) for e in experts)
    out = torch.zeros(x.shape[0], experts[0]["w2"][0].shape[0], dtype=torch.float64)
    for t, k in itertools.product(range(x.shape[0]), range(ids.shape[1])):
        j = int(ids[t, k]) - offset
        if not 0 <= j < len(experts):
            continue
        e = experts[j]
        w = {n: (common.dequantize_nvfp4(e[n][0], e[n][1]) * e[n][2]).double() for n in e}
        xt = common.quantize_nvfp4(x[t : t + 1].float(), g1).double()[0]
        gate, up = w["w1"] @ xt, w["w3"] @ xt
        act = mod.situ(gate, up, beta, linear_beta).float().unsqueeze(0)
        out[t] += float(weights[t, k]) * (w["w2"] @ common.quantize_nvfp4(act, g2).double()[0])
    return out


@pytest.mark.parametrize("input_scale", [0.02])
def test_reference_matches_a_token_loop(input_scale):
    hidden, inter, num, top_k, rows = 64, 32, 6, 2, 5
    experts = _random_experts(num, hidden, inter, 1, input_scale)
    g = torch.Generator().manual_seed(2)
    x = (torch.randn(rows, hidden, generator=g) * 0.2).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(num + 2, generator=g)[:top_k] for _ in range(rows)])
    weights = torch.rand(rows, top_k, generator=g)
    for offset in (0, 2):
        ref = rubin_fc12.reference(
            x,
            ids,
            weights,
            experts,
            local_expert_offset=offset,
            situ_beta=8.0,
            situ_linear_beta=20.0,
        )
        loop = _loop_reference(x, ids, weights, experts, offset, 8.0, 20.0)
        assert ref.dtype == torch.float32
        torch.testing.assert_close(ref.double(), loop, rtol=1e-5, atol=1e-6)
