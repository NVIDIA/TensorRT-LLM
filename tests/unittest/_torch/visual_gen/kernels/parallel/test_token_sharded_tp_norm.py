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
"""Single-GPU tests of the token-sharded TP row-local kernels (fused LN + NVFP4).

TP ranks are simulated: every rank's shard of a full ``[B * S, 5120]`` input goes
through the helper's row-local norm/quantize, and the result (FP4 bytes per row, and
the regrouped scaling factors of all shards) must equal the same op on all rows.
"""

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.modules.linear import Linear, NVFP4LinearMethod
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen.modules.fused_norm_quant import (
    apply_fused_layernorm_adaln_quant,
    apply_fused_layernorm_affine_quant,
)
from tensorrt_llm._torch.visual_gen.parallel import token_sharded_tp
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
    RowNorm,
    TokenShardPlan,
    apply_row_norm,
    quantize_nvfp4,
    regroup_swizzled_sf,
    static_nvfp4_input_scale,
    swizzled_sf_numel,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# tests/unittest/_torch/visual_gen: shared SF-layout references and simulated ranks.
__extra_import_path__ = ["../.."]

from token_sharded_tp_test_utils import simulated_helper, swizzle_ref, unswizzle_ref

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
# C1. Fused LN + AdaLN + NVFP4 on shards with per-shard tables == full-M op
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
        sp = simulated_helper(plan)
        spec = RowNorm(
            eps=EPS,
            scale=sp.per_sample_table(scale),
            shift=sp.per_sample_table(shift),
            quant_scale=qscale,
        )
        q = sp.norm(sp.shard(x), spec)
        assert isinstance(q, Fp4QuantizedTensor)
        assert q.fp4_tensor.shape == (plan.local_rows, D // 2)
        shards.append(q)
    _check_shards_match_full(full, shards, plans)


# =============================================================================
# C2. norm2 (affine LN) + quant on the shard == full-M result sliced
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
        sp = simulated_helper(plan)
        x_loc = sp.shard(x)
        shards.append(
            sp.norm(x_loc, RowNorm(eps=EPS, weight=weight, bias=bias, quant_scale=qscale))
        )
        loc, glob = _real_rows(plan)
        h = sp.norm(x_loc, RowNorm(eps=EPS, weight=weight, bias=bias))
        assert torch.equal(h[loc.cuda()], full_bf16[glob.cuda()])
    _check_shards_match_full(full, shards, plans)


def test_row_norm_uses_fused_op_only_when_eligible(monkeypatch):
    """The fused op runs for bf16 CUDA D == 5120 with exactly one of AdaLN / affine only."""
    calls = []

    def spy(fn):
        def wrapped(*args, **kwargs):
            calls.append(fn.__name__)
            return fn(*args, **kwargs)

        return wrapped

    for name in ("apply_fused_layernorm_adaln_quant", "apply_fused_layernorm_affine_quant"):
        monkeypatch.setattr(token_sharded_tp, name, spy(getattr(token_sharded_tp, name)))

    torch.manual_seed(2)
    x = torch.randn(64, D, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(2, D, device="cuda") * 0.1
    shift = torch.randn(2, D, device="cuda") * 0.1
    weight = torch.randn(D, device="cuda") * 0.1 + 1.0
    bias = torch.randn(D, device="cuda") * 0.1
    adaln = RowNorm(eps=EPS, scale=scale, shift=shift)
    affine = RowNorm(eps=EPS, weight=weight, bias=bias)

    fused = apply_row_norm(x, adaln)
    assert calls == ["apply_fused_layernorm_adaln_quant"]
    assert torch.equal(fused, apply_fused_layernorm_adaln_quant(x, scale, shift, 32, None, EPS))
    calls.clear()
    apply_row_norm(x, affine)
    assert calls == ["apply_fused_layernorm_affine_quant"]

    # Ineligible: fp32 input, D != 5120, both AdaLN and affine, CPU -> eager path.
    calls.clear()
    eager = apply_row_norm(x.float(), adaln)
    torch.testing.assert_close(eager, fused.float(), rtol=2e-2, atol=2e-2)
    x_small = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    apply_row_norm(x_small, RowNorm(eps=EPS, scale=scale[:, :256], shift=shift[:, :256]))
    apply_row_norm(x, RowNorm(eps=EPS, weight=weight, bias=bias, scale=scale, shift=shift))
    apply_row_norm(x.cpu(), RowNorm(eps=EPS, scale=scale.cpu(), shift=shift.cpu()))
    assert calls == []


# =============================================================================
# C3. quantize_nvfp4 == Linear._input_prepare; static_nvfp4_input_scale eligibility
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


@pytest.mark.parametrize("use_tunable", [False, True])
def test_quantize_nvfp4_matches_linear_input_prepare(use_tunable):
    lin = _nvfp4_linear()
    scale = static_nvfp4_input_scale(lin)
    assert scale is not None
    x = torch.randn(300, 512, device="cuda", dtype=torch.bfloat16) * 2
    prev = NVFP4LinearMethod.use_tunable_quantize
    NVFP4LinearMethod.use_tunable_quantize = use_tunable
    try:
        ref_fp4, ref_sf, _ = lin.quant_method._input_prepare(lin, x)
        got = quantize_nvfp4(x, scale)
    finally:
        NVFP4LinearMethod.use_tunable_quantize = prev
    assert torch.equal(got.fp4_tensor, ref_fp4)
    assert torch.equal(got.scaling_factor.reshape(-1), ref_sf.reshape(-1))
    assert got.is_sf_swizzled


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


# =============================================================================
# C4. regroup_swizzled_sf == trtllm::reswizzle_sf (independent oracle, even splits)
# =============================================================================


@pytest.mark.parametrize("tp,m,k", [(2, 300, 5120), (4, 256, 5120), (8, 70, 1728), (3, 77, 256)])
def test_regroup_matches_reswizzle_sf(tp, m, k):
    from tensorrt_llm._torch.utils import reswizzle_sf

    sf_cols = k // 16
    lin = torch.randint(0, 256, (tp * m, sf_cols), dtype=torch.uint8, device="cuda")
    sf_cat = torch.cat([swizzle_ref(lin[r * m : (r + 1) * m], pad_value=0) for r in range(tp)])
    plan = TokenShardPlan.build(1, tp * m, tp, 0)
    assert not plan.is_padded and plan.local_rows == m
    got = regroup_swizzled_sf(sf_cat, plan, sf_cols)
    oracle = reswizzle_sf(sf_cat, m, k, 16)
    assert got.numel() == oracle.numel() == swizzled_sf_numel(tp * m, sf_cols)
    assert torch.equal(unswizzle_ref(got, tp * m, sf_cols), unswizzle_ref(oracle, tp * m, sf_cols))
    assert torch.equal(unswizzle_ref(got, tp * m, sf_cols), lin)
