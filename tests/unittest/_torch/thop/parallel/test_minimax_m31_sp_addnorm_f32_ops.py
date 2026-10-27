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
"""MiniMax-M3.1 agg TP2 prefill ops built in-tree, held bit for bit to the CTXNORMQ base op:

* minimax_m31_ctx_residual_rmsnorm_mxfp8_f32 = the base op with write_norm=True, plus norm_out.float();
* minimax_m31_tp2ctx_peer_add_normq_out = the peer add (bf16 sum of the local and peer rows) followed by the base op,
  every output in place. The "peer" rows are a second buffer on the same GPU, so one device suffices.
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="the CTXNORMQ kernels need SM100 or newer",
)

HIDDEN = 6144
EPS = 1e-6
ROWS = (1, 7, 128, 129, 1000)


def _inputs(rows, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    inter = (torch.randn(rows, HIDDEN, device="cuda", generator=g) * scale).bfloat16()
    residual = (torch.randn(rows, HIDDEN, device="cuda", generator=g) * scale).bfloat16()
    weight = (1 + 0.3 * torch.randn(HIDDEN, device="cuda", generator=g)).bfloat16()
    weight[::97] = -weight[::97]  # negative gamma: zero rows give -0 bytes
    return inter, residual, weight


def _same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("fp4", [False, True])
def test_f32_op_is_base_op_plus_exact_widen(rows, fp4):
    inter, residual, weight = _inputs(rows, 11 * rows + fp4)
    inter[0].zero_()
    residual[0].zero_()
    gs = torch.tensor([0.37], device="cuda") if fp4 else None
    ref_inter = inter.clone()
    ref = torch.ops.trtllm.minimax_m31_ctx_residual_rmsnorm_mxfp8(
        ref_inter, residual, weight, EPS, True, gs
    )
    got = torch.ops.trtllm.minimax_m31_ctx_residual_rmsnorm_mxfp8_f32(
        inter, residual, weight, EPS, gs
    )
    torch.cuda.synchronize()
    assert _same(inter, ref_inter)
    for name, a, b in zip(("norm", "q", "sf", "fp4", "fp4_sf"), got[:5], ref):
        assert _same(a, b), name
    assert _same(got[5], got[0].float())


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("variant", ["no_norm", "norm", "norm_fp4", "norm_f32", "norm_fp4_f32"])
@pytest.mark.parametrize("ctas_per_sm", [1, 2])
def test_sp_addnorm_is_peer_add_then_base_op(rows, variant, ctas_per_sm):
    local, residual, weight = _inputs(rows, 7 * rows + len(variant))
    peer, _, _ = _inputs(rows, 13 * rows + 1)
    local[0].zero_()
    peer[0].zero_()
    residual[0].zero_()
    write_norm = variant != "no_norm"
    fp4 = "fp4" in variant
    f32 = "f32" in variant
    gs = torch.tensor([0.61], device="cuda") if fp4 else None

    ref_inter = (local.float() + peer.float()).bfloat16()  # the peer add's rounding
    ref = torch.ops.trtllm.minimax_m31_ctx_residual_rmsnorm_mxfp8(
        ref_inter, residual, weight, EPS, write_norm, gs
    )

    inter_out = torch.empty_like(local)
    norm = torch.empty_like(local) if write_norm else None
    q = torch.empty(rows, HIDDEN, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty((rows + 127) // 128 * 128 * (HIDDEN // 32), dtype=torch.uint8, device="cuda")
    f4 = torch.empty(rows, HIDDEN // 2, dtype=torch.uint8, device="cuda") if fp4 else None
    f4sf = torch.empty(rows * (HIDDEN // 16), dtype=torch.uint8, device="cuda") if fp4 else None
    f32_out = torch.empty(rows, HIDDEN, dtype=torch.float32, device="cuda") if f32 else None
    torch.ops.trtllm.minimax_m31_tp2ctx_peer_add_normq_out(
        local.data_ptr(),
        peer.data_ptr(),
        inter_out,
        residual,
        weight,
        EPS,
        norm,
        q,
        sf,
        f4,
        f4sf,
        gs,
        f32_out,
        ctas_per_sm,
        False,
    )
    torch.cuda.synchronize()
    assert _same(inter_out, ref_inter)
    if write_norm:
        assert _same(norm, ref[0])
    assert _same(q, ref[1])
    assert _same(sf, ref[2])
    if fp4:
        assert _same(f4, ref[3]) and _same(f4sf, ref[4])
    if f32:
        assert _same(f32_out, norm.float())
