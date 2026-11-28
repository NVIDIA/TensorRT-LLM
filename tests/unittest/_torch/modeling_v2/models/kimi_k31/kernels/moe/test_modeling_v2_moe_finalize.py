# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's MoE finalize (``moe_finalize``).

``out[t] = bf16(sum_k w[t, k] * y[e2p[t * K + k]])`` over the local slots in slot order, FP32. With
values whose products and partial sums are exact in FP32 the result is checked bitwise against a
torch reference.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the finalize is a CuTe DSL kernel", allow_module_level=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("the finalize runs on SM100 / SM103", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.moe_finalize import (  # noqa: E402
    run_moe_finalize,
)

HIDDEN, K = 3584, 16


def _case(rows: int, local_frac: float, seed: int):
    """Routing with each slot local with probability ``local_frac`` (row 0: no local slot, row 1: all 16)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    loc = torch.rand(rows, K, generator=g, device="cuda") < local_frac
    loc[0] = False
    if rows > 1:
        loc[1] = True
    n = int(loc.sum())
    e2p = torch.full((rows * K,), -1, dtype=torch.int32, device="cuda")
    e2p[loc.reshape(-1)] = torch.randperm(n, generator=g, device="cuda").int()
    # Small integers times multiples of 1/16: every product and partial sum is exact in FP32.
    y = torch.randint(-8, 9, (max(n, 1), HIDDEN), generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randint(0, 17, (rows, K), generator=g, device="cuda") / 16.0).to(torch.bfloat16)
    return y, e2p, w


def _ref(y, e2p, w):
    rows = w.shape[0]
    idx = e2p.view(rows, K).long()
    acc = torch.zeros(rows, HIDDEN, dtype=torch.float32, device="cuda")
    for k in range(K):
        sel = idx[:, k] >= 0
        acc[sel] += w[sel, k].float().unsqueeze(-1) * y[idx[sel, k]].float()
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize("rows", [7, 2688])
def test_exact_values_bitwise(rows):
    y, e2p, w = _case(rows, 0.125, seed=rows)
    ref = _ref(y, e2p, w)
    out = torch.full((rows, HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
    run_moe_finalize(y, e2p, w, out)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
