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
"""Wan's fused LayerNorm + NVFP4 ops on simulated token shards (Blackwell).

Each rank's shard, with its per-shard modulation table (``seq_len_per_batch =
rows_per_entry``), must give the same FP4 bytes per row and, after regrouping, the same
scaling factors as the op on all rows.
"""

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.modules.linear import Linear, NVFP4LinearMethod
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen.models.wan.utils_wan import (
    apply_fused_layernorm_adaln_quant,
    apply_fused_layernorm_affine_quant,
)
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
    TokenShardPlan,
    quantize_nvfp4,
    regroup_swizzled_sf,
    static_nvfp4_input_scale,
    swizzled_sf_numel,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# tests/unittest/_torch/visual_gen: shared SF-layout references and simulated ranks.
__extra_import_path__ = ["../.."]

from token_sharded_tp_test_utils import simulated_helper, unswizzle_ref

D = 5120
EPS = 1e-6
SF_COLS = D // 16

pytestmark = skip_pre_blackwell


def _real_rows(plan):
    """Local row indices of real tokens, and their global [B * S] row indices."""
    loc, glob = [], []
    i = 0
    for b, s0, s1 in plan.local_segments():
        for s in range(s0, s1):
            if s < plan.seq_len:
                loc.append(i)
                glob.append(b * plan.seq_len + s)
            i += 1
    return torch.tensor(loc, dtype=torch.long), torch.tensor(glob, dtype=torch.long)


def _check_shards_match_full(full, shards, plans):
    """Per-row FP4 bytes equal; regroup(cat(shard SFs)) == full SF on the valid region."""
    for q, plan in zip(shards, plans):
        loc, glob = _real_rows(plan)
        loc, glob = loc.to(full.fp4_tensor.device), glob.to(full.fp4_tensor.device)
        assert torch.equal(q.fp4_tensor[loc], full.fp4_tensor[glob]), plan
    num_tokens = plans[0].num_tokens
    sf = regroup_swizzled_sf(
        torch.cat([q.scaling_factor.reshape(-1) for q in shards]), plans[0], SF_COLS
    )
    assert sf.numel() == swizzled_sf_numel(num_tokens, SF_COLS)
    assert torch.equal(
        unswizzle_ref(sf, num_tokens, SF_COLS),
        unswizzle_ref(full.scaling_factor, num_tokens, SF_COLS),
    )


# (B, S, tp): straddling rank with m = 70; padded B = 2; SF fast path; padded B = 1 with
# m = 126 (last rank partly padded); padded B = 1 with m = 128 (zero-copy SF prefix built
# from the kernel's own SF output); padded B = 2 with m = 128 (must regroup, not slice).
_CASES = [(2, 105, 3), (2, 105, 4), (2, 256, 2), (1, 1001, 8), (1, 509, 4), (2, 255, 4)]


# =============================================================================
# Fused LN + AdaLN + NVFP4 on shards with per-shard tables == full-M op
# =============================================================================


@pytest.mark.parametrize("batch,seq,tp", _CASES)
def test_fused_adaln_quant_shards_match_full(batch, seq, tp):
    torch.manual_seed(0)
    x = torch.randn(batch, seq, D, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(batch, D, device="cuda") * 0.1
    shift = torch.randn(batch, D, device="cuda") * 0.1
    qscale = torch.tensor([448.0 * 6.0 / 12.0], device="cuda")
    full = apply_fused_layernorm_adaln_quant(
        x.reshape(batch * seq, D), scale, shift, seq, qscale, EPS
    )
    plans = [TokenShardPlan.build(batch, seq, tp, r) for r in range(tp)]
    shards = []
    for plan in plans:
        ts = simulated_helper(plan)
        q = apply_fused_layernorm_adaln_quant(
            ts.shard(x),
            ts.per_sample_table(scale),
            ts.per_sample_table(shift),
            plan.rows_per_entry,
            qscale,
            EPS,
        )
        assert isinstance(q, Fp4QuantizedTensor)
        assert q.fp4_tensor.shape == (plan.local_rows, D // 2)
        shards.append(q)
    _check_shards_match_full(full, shards, plans)


# =============================================================================
# norm2 (affine LN) + quant on the shard == full-M result sliced
# =============================================================================


@pytest.mark.parametrize("batch,seq,tp", _CASES)
def test_fused_affine_quant_shards_match_full(batch, seq, tp):
    torch.manual_seed(1)
    x = torch.randn(batch, seq, D, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(D, device="cuda") * 0.1 + 1.0
    bias = torch.randn(D, device="cuda") * 0.1
    qscale = torch.tensor([448.0 * 6.0 / 12.0], device="cuda")
    full = apply_fused_layernorm_affine_quant(x.reshape(batch * seq, D), weight, bias, qscale, EPS)
    full_bf16 = apply_fused_layernorm_affine_quant(
        x.reshape(batch * seq, D), weight, bias, None, EPS
    )
    plans = [TokenShardPlan.build(batch, seq, tp, r) for r in range(tp)]
    shards = []
    for plan in plans:
        ts = simulated_helper(plan)
        x_loc = ts.shard(x)
        shards.append(apply_fused_layernorm_affine_quant(x_loc, weight, bias, qscale, EPS))
        loc, glob = _real_rows(plan)
        h = apply_fused_layernorm_affine_quant(x_loc, weight, bias, None, EPS)
        assert torch.equal(h[loc.cuda()], full_bf16[glob.cuda()])
    _check_shards_match_full(full, shards, plans)


# =============================================================================
# quantize_nvfp4 == Linear._input_prepare; static_nvfp4_input_scale eligibility
# =============================================================================


def _nvfp4_linear(k_in=512, n_out=256, input_scale=True, **kwargs):
    torch.manual_seed(3)
    weight = torch.randn(n_out, k_in, dtype=torch.bfloat16, device="cuda") * 0.05
    ws2 = (weight.float().abs().amax() / (448.0 * 6.0)).reshape(1)
    fp4, sf = torch.ops.trtllm.fp4_quantize(weight, 1.0 / ws2, 16, False, False)
    ckpt = {
        "weight": fp4.cpu(),
        "weight_scale": sf.view(torch.float8_e4m3fn).reshape(n_out, -1).cpu(),
        "weight_scale_2": ws2.cpu(),
    }
    if input_scale:
        ckpt["input_scale"] = torch.tensor([4.0 / (448.0 * 6.0)])
    lin = Linear(
        k_in,
        n_out,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        **kwargs,
    ).cuda()
    lin.load_weights([ckpt])
    lin.post_load_weights()
    return lin


def test_quantize_nvfp4_matches_linear_input_prepare(monkeypatch):
    lin = _nvfp4_linear()
    scale = static_nvfp4_input_scale(lin)
    assert scale is not None
    x = torch.randn(300, 512, device="cuda", dtype=torch.bfloat16) * 2
    monkeypatch.setattr(NVFP4LinearMethod, "use_tunable_quantize", False)
    ref_fp4, ref_sf, _ = lin.quant_method._input_prepare(lin, x)
    got = quantize_nvfp4(x, scale)
    assert torch.equal(got.fp4_tensor, ref_fp4)
    assert torch.equal(got.scaling_factor.reshape(-1), ref_sf.reshape(-1))
    assert got.is_sf_swizzled


def test_quantize_nvfp4_is_pinned_to_the_plain_op(monkeypatch):
    """Even when VisualGen tunes the Linears' quantize: the tunable op may pick FlashInfer's
    kernel, which shuffles rows and scaling factors in 128-row tiles."""

    def tunable(*args, **kwargs):
        raise AssertionError("quantize_nvfp4 must not use trtllm::tunable_fp4_quantize")

    monkeypatch.setattr(NVFP4LinearMethod, "use_tunable_quantize", True)
    monkeypatch.setattr(torch.ops.trtllm, "tunable_fp4_quantize", tunable)
    x = torch.randn(300, 512, device="cuda", dtype=torch.bfloat16) * 2
    scale = torch.tensor([448.0 * 6.0 / 8.0], device="cuda")
    got = quantize_nvfp4(x, scale)
    ref_fp4, ref_sf = torch.ops.trtllm.fp4_quantize(x, scale, 16, False)
    assert torch.equal(got.fp4_tensor, ref_fp4)
    assert torch.equal(got.scaling_factor.reshape(-1), ref_sf.reshape(-1))


def test_static_nvfp4_input_scale_eligibility():
    assert static_nvfp4_input_scale(None) is None
    static = _nvfp4_linear()
    assert static_nvfp4_input_scale(static) is static.input_scale
    dynamic = _nvfp4_linear(input_scale=False, force_dynamic_quantization=True)
    assert static_nvfp4_input_scale(dynamic) is None
    awq = _nvfp4_linear()
    awq.pre_quant_scale = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    assert static_nvfp4_input_scale(awq) is None
    plain = Linear(512, 256, bias=False, dtype=torch.bfloat16).cuda()
    assert static_nvfp4_input_scale(plain) is None
