# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's MoE finalize (``moe_finalize``).

``out[t] = bf16(sum_k w[t, k] * y[e2p[t * K + k]])`` over the local slots in slot order, FP32. With
values whose products and partial sums are exact in FP32 the result is checked bitwise against a
torch reference; with random values every launch configuration (rows per CTA, prefetched slots,
load group) must give the same bits, since they change only the load schedule.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the finalize is a CuTe DSL kernel", allow_module_level=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("the finalize runs on SM100 / SM103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe.moe_finalize import (  # noqa: E402
    run_moe_finalize,
)

HIDDEN, K = 3584, 16
CONFIGS = [(4, 2, 2), (4, 4, 2), (8, 1, 1), (4, 8, 1), (8, 4, 4)]  # group, rows_per_cta, prefetch


def _case(rows: int, local_frac: float, seed: int, exact: bool):
    """Routing with each slot local with probability ``local_frac`` (row 0: no local slot, row 1: all 16)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    loc = torch.rand(rows, K, generator=g, device="cuda") < local_frac
    loc[0] = False
    if rows > 1:
        loc[1] = True
    n = int(loc.sum())
    e2p = torch.full((rows * K,), -1, dtype=torch.int32, device="cuda")
    e2p[loc.reshape(-1)] = torch.randperm(n, generator=g, device="cuda").int()
    if (
        exact
    ):  # small integers times multiples of 1/16: every product and partial sum is exact in FP32
        y = torch.randint(-8, 9, (max(n, 1), HIDDEN), generator=g, device="cuda").to(torch.bfloat16)
        w = (torch.randint(0, 17, (rows, K), generator=g, device="cuda") / 16.0).to(torch.bfloat16)
    else:
        y = torch.randn(max(n, 1), HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        w = torch.rand(rows, K, generator=g, device="cuda").to(torch.bfloat16)
    return y, e2p, w


def _ref(y, e2p, w):
    rows = w.shape[0]
    idx = e2p.view(rows, K).long()
    acc = torch.zeros(rows, HIDDEN, dtype=torch.float32, device="cuda")
    for k in range(K):
        sel = idx[:, k] >= 0
        acc[sel] += w[sel, k].float().unsqueeze(-1) * y[idx[sel, k]].float()
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 2, 7, 96, 1537, 2688])
@pytest.mark.parametrize("local_frac", [0.06, 0.125, 0.5, 1.0])
def test_exact_values_bitwise(rows, local_frac):
    y, e2p, w = _case(rows, local_frac, seed=rows, exact=True)
    ref = _ref(y, e2p, w)
    for group, rpc, pre in CONFIGS:
        out = torch.full((rows, HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
        run_moe_finalize(y, e2p, w, out, group=group, rows_per_cta=rpc, prefetch=pre)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), (group, rpc, pre)


@pytest.mark.parametrize("rows", [5, 1536, 65537])
@pytest.mark.parametrize("local_frac", [0.0625, 0.5])
def test_configs_agree_and_match_reference(rows, local_frac):
    y, e2p, w = _case(rows, local_frac, seed=7 + rows, exact=False)
    ref = _ref(y, e2p, w).float()
    first = None
    for group, rpc, pre in CONFIGS:
        out = torch.full((rows, HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
        run_moe_finalize(y, e2p, w, out, group=group, rows_per_cta=rpc, prefetch=pre)
        torch.cuda.synchronize()
        if first is None:
            first = out
            err = (out.float() - ref).norm() / ref.norm().clamp(min=1e-6)
            assert err < 4e-3, float(err)
        else:
            assert torch.equal(out, first), (group, rpc, pre)


def test_graph_replay():
    y, e2p, w = _case(300, 0.125, seed=3, exact=False)
    out = torch.empty(300, HIDDEN, dtype=torch.bfloat16, device="cuda")
    run_moe_finalize(y, e2p, w, out)
    expect = out.clone()
    out.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_moe_finalize(y, e2p, w, out)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expect)
