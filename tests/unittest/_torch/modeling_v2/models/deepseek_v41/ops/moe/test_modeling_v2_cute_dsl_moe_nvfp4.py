# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the cute_dsl_moe_nvfp4 entry: the engine's CuTe-DSL NVFP4 MoE chain against an fp32 reference.

Weights and activations are drawn as exact NVFP4 codes with power-of-two scales, so both sides read bit-identical
operands; the reference models the FC1-output NVFP4 requantization. The SwiGLU limit is set below the largest gate
value so the clamp is exercised. Gate: the shared MoE gate (`_assert_moe_close`: 16 bf16 ulp of the row's magnitude per
element, 2 ulp of relative RMS).
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = [
    "../../../..",
    "../../../../moe",
]  # _validating, _precision; _fp4_moe_test_utils
from _fp4_moe_test_utils import _assert_moe_close, _rand_activation, _rand_nvfp4  # noqa: E402
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    dequantize_nvfp4,  # noqa: E501
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.cute_dsl_moe_nvfp4 import (  # noqa: E501
    cute_dsl_moe_layout,
    cute_dsl_moe_nvfp4,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="CuTe-DSL kernels, SM103",
)


def _routing(t, experts, top_k, gen):
    ids = torch.argsort(torch.rand(t, experts, generator=gen, device=DEV), dim=1)[:, :top_k].to(
        torch.int32
    )
    w = torch.rand(t, top_k, generator=gen, device=DEV).to(torch.bfloat16)
    return ids, w


@pytest.mark.parametrize(
    "cell", cute_dsl_moe_nvfp4.CELLS, ids=[c.why[:44] for c in cute_dsl_moe_nvfp4.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(cute_dsl_moe_nvfp4)
    s = cell.spec
    gen = gen_for(cell)
    e, hid, inter = s["local"], s["hidden"], s["inter"]
    up_p, up_s, up_c = _rand_nvfp4(e, inter, hid, gen)
    gt_p, gt_s, gt_c = _rand_nvfp4(e, inter, hid, gen)
    dn_p, dn_s, dn_c = _rand_nvfp4(e, hid, inter, gen)
    lay = [
        cute_dsl_moe_layout(
            up_p[i],
            up_s[i].view(torch.uint8),
            gt_p[i],
            gt_s[i].view(torch.uint8),
            dn_p[i],
            dn_s[i].view(torch.uint8),
        )
        for i in range(e)
    ]
    fc1_w, fc1_s, fc2_w, fc2_s = (torch.stack([lw[j] for lw in lay]) for j in range(4))
    del lay

    g1 = 137.0
    data, sf_flat, x = _rand_activation(s["tokens"], hid, g1, gen)
    xq = x * g1
    ids, w = _routing(s["tokens"], s["experts"], s["top_k"], gen)
    a1 = torch.full((e,), 1.0 / g1, dtype=torch.float32, device=DEV) * 4.0  # a weight_scale_2 of 4
    up = [(up_c[i], up_s[i]) for i in range(e)]
    gate = [(gt_c[i], gt_s[i]) for i in range(e)]
    down = [(dn_c[i], dn_s[i]) for i in range(e)]

    # calibrate the FC2 input scale and a limit that clamps, from the unclamped activations of a sample
    with true_fp32_matmul():
        local = (ids[:512] >= s["offset"]) & (
            ids[:512] < s["offset"] + e
        )  # picks of this rank's experts
        tok = local.any(1).nonzero().flatten()[:256]
        le = (
            int(ids[tok[0]][(ids[tok[0]] >= s["offset"]) & (ids[tok[0]] < s["offset"] + e)][0])
            - s["offset"]
        )
        gs = (xq[tok] @ dequantize_nvfp4(*gate[le]).t()) * float(a1[le])
        us = (xq[tok] @ dequantize_nvfp4(*up[le]).t()) * float(a1[le])
    limit = 0.5 * float(gs.abs().max())
    gg = gs.clamp(max=limit)
    amax = float((us.clamp(-limit, limit) * gg * torch.sigmoid(gg)).abs().max())
    g2 = 448.0 * 6.0 / (2.0 * amax)
    fc2_global = torch.tensor([g2], dtype=torch.float32, device=DEV)
    a2 = torch.full((e,), 1.0 / g2, dtype=torch.float32, device=DEV) * 2.0  # a weight_scale_2 of 2

    y = cute_dsl_moe_nvfp4(
        data,
        sf_flat.view(s["tokens"], hid // 16),
        ids,
        w,
        fc1_w,
        fc1_s,
        fc2_w,
        fc2_s,
        a1,
        a2,
        fc2_global,
        limit,
        s["experts"],
        s["offset"],
        tile_size=s.get("tile", 128),
    )
    with true_fp32_matmul():
        ref = cute_dsl_moe_nvfp4.reference(
            xq, ids, w, up, gate, down, a1, a2, g2, limit, s["offset"]
        )
    assert y.shape == (s["tokens"], hid)
    _assert_moe_close(y, ref.to(torch.bfloat16))
    # into a caller's buffer (the all-to-all's combine region): zeroed first, the same result
    buf = torch.full_like(y, 3.0)
    y2 = cute_dsl_moe_nvfp4(
        data,
        sf_flat.view(s["tokens"], hid // 16),
        ids,
        w,
        fc1_w,
        fc1_s,
        fc2_w,
        fc2_s,
        a1,
        a2,
        fc2_global,
        limit,
        s["experts"],
        s["offset"],
        buf,
    )
    assert y2.data_ptr() == buf.data_ptr()
    _assert_moe_close(
        y2, ref.to(torch.bfloat16)
    )  # (the finalize adds top-k rows atomically: not bitwise)
    # rows routed nowhere on this rank (no tile to run): FC1 still zeroes the whole buffer
    none = torch.full_like(ids, -1)
    buf.fill_(3.0)
    y3 = cute_dsl_moe_nvfp4(
        data,
        sf_flat.view(s["tokens"], hid // 16),
        none,
        w,
        fc1_w,
        fc1_s,
        fc2_w,
        fc2_s,
        a1,
        a2,
        fc2_global,
        limit,
        s["experts"],
        s["offset"],
        buf,
    )
    assert y3.data_ptr() == buf.data_ptr(), "the result is the caller's buffer"
    assert torch.count_nonzero(buf) == 0, "the caller's buffer itself is zeroed"


def test_layout_is_a_permutation():
    """The FC1 layout regroups `[up | gate]` rows as `[up64, gate64, ...]` and keeps every byte."""
    gen = torch.Generator(device=DEV).manual_seed(7)
    inter, hid = 256, 512
    up = torch.randint(0, 256, (inter, hid // 2), dtype=torch.uint8, device=DEV, generator=gen)
    gate = torch.randint(0, 256, (inter, hid // 2), dtype=torch.uint8, device=DEV, generator=gen)
    s = torch.randint(0, 120, (inter, hid // 16), dtype=torch.uint8, device=DEV, generator=gen)
    fc1_w, _, _, _ = cute_dsl_moe_layout(
        up, s, gate, s, up.t().contiguous()[:hid].repeat(1, 1), s.repeat(2, 1)[:hid]
    )
    want = torch.cat(
        [blk for i in range(0, inter, 64) for blk in (up[i : i + 64], gate[i : i + 64])]
    )
    assert torch.equal(fc1_w.reshape(want.shape), want), "every row of [up64, gate64, ...] in order"
