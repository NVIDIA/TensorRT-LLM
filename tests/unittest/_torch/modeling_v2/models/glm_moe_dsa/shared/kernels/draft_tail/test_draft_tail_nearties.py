# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Self-checks of the near-tie tooling used by the flip-rate study."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draft_tail_nearties as nt  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_bf16_step_walks_the_grid():
    x = torch.tensor([1.0, -1.0, 0.0, 3.0, 17.875], dtype=torch.bfloat16)
    up = nt.bf16_step(x, 1).double()
    down = nt.bf16_step(x, -1).double()
    assert bool((up > x.double()).all()) and bool((down < x.double()).all())
    # No BF16 value lies strictly between x and its neighbours.
    mid_up = ((x.double() + up) / 2).float().to(torch.bfloat16).double()
    assert bool(((mid_up == x.double()) | (mid_up == up)).all())
    assert nt.bf16_step(x, 1)[4].item() == 18.0
    assert nt.bf16_step(nt.bf16_step(x, 5), -5).tolist() == x.tolist()


def test_bf16_round_exact_is_nearest_without_double_rounding():
    gen = torch.Generator().manual_seed(0)
    v = torch.randn(200000, generator=gen, dtype=torch.float64) * 20
    r = nt.bf16_round_exact(v)
    lo = nt.bf16_step(r, -1).double()
    hi = nt.bf16_step(r, 1).double()
    rd = r.double()
    assert bool(((v - rd).abs() <= (v - lo).abs()).all())
    assert bool(((v - rd).abs() <= (v - hi).abs()).all())
    # Just above / below a midpoint, closer than FP32 can resolve.
    b = torch.tensor([17.875], dtype=torch.bfloat16)
    n = nt.bf16_step(b, 1)
    m = (b.double() + n.double()) / 2
    assert nt.bf16_round_exact(m + 1e-9).item() == n.item()
    assert nt.bf16_round_exact(m - 1e-9).item() == b.item()
    # Exactly at the midpoint: ties to even (17.875 has an odd mantissa).
    assert nt.bf16_round_exact(m).item() == n.item()


def test_bf16_key_orders_value_then_lower_id():
    vals = torch.tensor([[1.0, 2.0, 2.0, -0.0, 0.0, -3.0]], dtype=torch.bfloat16)
    ids = torch.arange(6)[None]
    assert int(nt.bf16_key(vals, ids).argmax()) == 1
    z = torch.tensor([[-0.0, 0.0]], dtype=torch.bfloat16)
    k = nt.bf16_key(z, torch.arange(2)[None])
    assert int(k.argmax()) == 0  # -0 == +0: lower id


def test_construct_straddle_reaches_targets():
    gen = torch.Generator(device="cuda").manual_seed(1)
    head = (torch.randn(64, 6144, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    hidden = torch.randn(16, 6144, generator=gen, device="cuda").to(torch.bfloat16)
    up = torch.full((16,), 3, device="cuda")
    down = torch.full((16,), 7, device="cuda")
    for tau in (1.0, 64.0):
        res = nt.construct_straddle(hidden, head, up, down, torch.full((16,), tau, device="cuda"))
        ok = (
            (res["dist_hi_fp32_ulps"] > 0)
            & (res["dist_lo_fp32_ulps"] < 0)
            & (res["dist_hi_fp32_ulps"] <= 1.1 * tau)
            & (res["dist_lo_fp32_ulps"] >= -1.1 * tau)
        )
        # Greedy single-element steps from large toy gaps (about 2): most
        # rows reach the band (real near-tie rows start close); callers
        # keep only rows that did.
        assert int(ok.sum()) >= 8, (tau, res["dist_hi_fp32_ulps"], res["dist_lo_fp32_ulps"])
        # The returned logits are those of the returned rows.
        h = res["hidden"].double()
        torch.testing.assert_close((h * head[3].double()).sum(-1), res["l_hi"], rtol=0, atol=1e-9)
