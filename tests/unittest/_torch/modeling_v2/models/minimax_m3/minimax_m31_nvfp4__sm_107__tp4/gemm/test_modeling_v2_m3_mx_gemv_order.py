# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP4 decode GEMV order overrides (modeling.mx_gemv_cfg: QKV split 8, o_proj strided k-tiles):
for every overridden (projection, rows) the output is the same on 3 eager calls and on a CUDA graph replay
(deterministic), and its max abs / max rel error vs the fp64 product of the dequantized operands stays within the
shipped order's band (the shared table's config on the same operands, over several seeds)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4 import (
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.gemm.m3_mx_gemv import (  # noqa: E501
    O_PROJ,
    QKV,
    QKV_SPARSE,
    m3_mx_gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.moe.m3_moe_engine import (  # noqa: E501
    mxfp8_dequant,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP4 target runs on SM107 only",
)

CASES = [(nk, m) for nk in (QKV_SPARSE, QKV, O_PROJ) for m in (1, 2, 4, 8)]


def _err(y, ref):
    d = (y.double() - ref).abs()
    return float(d.max()), float((d / ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)).max())


@pytest.mark.parametrize("nk, m", CASES, ids=[f"{n}x{k}-m{m}" for (n, k), m in CASES])
def test_order_override_is_deterministic_and_in_band(nk, m) -> None:
    n, k = nk
    cfg = dispatch.mx_gemv_cfg(n, k, m)
    assert cfg is not None, "the override is off or does not cover this call"
    worst_new, worst_ship = [0.0, 0.0], [0.0, 0.0]
    for seed in range(4):
        g = torch.Generator(device="cuda").manual_seed(1000 * seed + m)
        w = (torch.randn(n, k, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
        ws = torch.randint(118, 124, (n * k // 32,), dtype=torch.uint8, device="cuda", generator=g)
        x = (torch.randn(m, k, device="cuda", generator=g) * (0.5 + seed)).to(torch.bfloat16)
        q, s = torch.ops.trtllm.mxfp8_quantize(x, True)
        ref = (
            mxfp8_dequant(q.view(torch.uint8), s).double()
            @ mxfp8_dequant(w.view(torch.uint8), ws).double().t()
        )
        ys = [m3_mx_gemv(q, s, w, ws, True, cfg, True, None) for _ in range(3)]
        assert all(y is not None for y in ys), "the GEMV declined the override"
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        out = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out.append(m3_mx_gemv(q, s, w, ws, True, cfg, True, None))
        graph.replay()
        torch.cuda.synchronize()
        for y in ys[1:] + [out[0]]:
            assert torch.equal(ys[0].view(torch.int16), y.view(torch.int16)), (
                "not deterministic (eager / graph)"
            )
        ship = m3_mx_gemv(q, s, w, ws, True, None, True, None)
        torch.cuda.synchronize()
        for acc, y in ((worst_new, ys[0]), (worst_ship, ship)):
            e = _err(y, ref)
            acc[0], acc[1] = max(acc[0], e[0]), max(acc[1], e[1])
    print(
        f"\nR1B1 tp4 {n}x{k} m={m} cfg {cfg}: max abs / rel vs fp64 {worst_new[0]:.3e} / {worst_new[1]:.3e} | shipped "
        f"order {worst_ship[0]:.3e} / {worst_ship[1]:.3e}",
        flush=True,
    )
    # The band: the shipped order's own error (bf16 output rounding dominates), with 25 % headroom.
    assert worst_new[0] <= 1.25 * worst_ship[0] and worst_new[1] <= 1.25 * worst_ship[1]
