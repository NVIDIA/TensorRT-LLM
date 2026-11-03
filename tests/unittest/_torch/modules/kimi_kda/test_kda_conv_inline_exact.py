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
"""The fused KDA prefill convolution gives the same bits with the exact SiLU of flagged token groups done inline
(two register passes per group) as with the recompute after the loop."""

from itertools import accumulate

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda import fused_prefill_conv as fpc

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs SM100/SM103/SM107",
)


def _run(inline, src, weight, state, starts, slots, has_init, total, heads, monkeypatch):
    monkeypatch.setattr(fpc, "INLINE_EXACT", inline)
    pool = state.clone()
    qkv = tuple(
        torch.full((1, total, heads, 128), 7.0, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    fpc.fused_prefill_conv(src, weight, pool, starts, slots, has_init, total, *qkv)
    torch.cuda.synchronize()
    return qkv, pool


@pytest.mark.parametrize("heads", [3, 96])
@pytest.mark.parametrize("scale", [0.3, 1.0, 3.0])
@pytest.mark.parametrize("source_offset", [0, 1])
@torch.inference_mode()
def test_inline_exact_silu_is_bitwise(monkeypatch, heads, scale, source_offset):
    lengths = [0, 1, 7, 8, 9, 31, 33, 100, 1000, 2051]
    slot_ids = [4, -1, 0, 7, 2, 9, -1, 5, 1, 3]
    init = [True, False, False, True, True, False, False, True, False, True]
    total, channels = sum(lengths), 3 * heads * 128
    torch.manual_seed(heads + int(10 * scale) + source_offset)
    row = 4 * heads * 128 + 8 * source_offset
    backing = torch.randn(total * row + 8, device="cuda") * scale
    src = torch.as_strided(
        backing.to(torch.bfloat16), (total, 4 * heads * 128), (row, 1), source_offset
    )
    weight = (torch.randn(channels, 4, device="cuda") * 0.6).to(torch.bfloat16)
    state = (torch.randn(12, channels, 3, device="cuda") * scale).to(torch.bfloat16)
    starts = torch.tensor([0] + list(accumulate(lengths)), device="cuda", dtype=torch.int32)
    slots = torch.tensor(slot_ids, device="cuda", dtype=torch.int32)
    has_init = torch.tensor(init, device="cuda", dtype=torch.bool)
    args = (src, weight, state, starts, slots, has_init, total, heads, monkeypatch)
    ref_qkv, ref_pool = _run(False, *args)
    got_qkv, got_pool = _run(True, *args)
    for got, ref in zip(got_qkv, ref_qkv):
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    assert torch.equal(got_pool.view(torch.int16), ref_pool.view(torch.int16))
