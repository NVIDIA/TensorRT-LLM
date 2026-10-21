# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-side dispatch of the decode attention-residual kernel (``modeling_kimi_linear``).

Decode-sized fused attention-residual calls go to the CuTe DSL kernel
(``trtllm::kimi_k3_attn_res_*``) and stay within its declared tolerance of the C++ per-token
path; a MoE layer's residual update deferred into the next layer's mix gives the same prefix and
normalized mix, bit for bit, as materializing it first.
"""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)
from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

HIDDEN = 7168
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE
    or not torch.cuda.is_available()
    or not (
        torch.cuda.get_device_capability() in ((10, 0), (10, 3))
        or (torch.cuda.get_device_capability() == (10, 7) and IS_CUTLASS_DSL_RUBIN_AVAILABLE)
    ),
    reason="needs an SM100/SM103 GPU, or SM107 with a Rubin-capable CuTe DSL",
)


def _max_tokens():
    from tensorrt_llm._torch.kernels.blackwell import kimi_k3_attn_res

    return kimi_k3_attn_res.MAX_TOKENS


def _modules(seed: int):
    torch.manual_seed(seed)
    proj = nn.Linear(HIDDEN, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    nn.init.normal_(proj.weight, std=0.02)
    norm = mkl.KimiK3RMSNorm(HIDDEN, eps=EPS, dtype=torch.bfloat16).cuda()
    out_norm = RMSNorm(hidden_size=HIDDEN, eps=EPS, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN, device="cuda"))
        out_norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN, device="cuda"))
    return proj, norm, out_norm


def _rows(*shape):
    return torch.randn(*shape, device="cuda").to(torch.bfloat16)


def _close_to_cpp(actual: torch.Tensor, expected: torch.Tensor, what=None):
    # Statistics summed in another order: rare differences of one or two BF16 rounding steps.
    a, e = actual.float(), expected.float()
    bound = (
        torch.maximum(a.abs(), e.abs()) * 2.0**-6 + 2.0**-7 * e.pow(2).mean(-1, keepdim=True).sqrt()
    )
    over = (a - e).abs() - bound
    assert bool((over <= 0).all()), (float(over.max()), int((over > 0).sum()), what)
    assert float(((a - e) != 0).float().mean()) <= 2e-2, what


@pytest.mark.parametrize("snapshots", [1, 3, 4, 8, 11])
@pytest.mark.parametrize("m", [1, 7, 32, 64])
@torch.no_grad()
def test_dispatch_matches_cpp(m: int, snapshots: int, monkeypatch):
    proj, norm, out_norm = _modules(m + snapshots)
    prefix, addend, block = _rows(m, HIDDEN), _rows(m, HIDDEN), _rows(snapshots, m, HIDDEN)
    out = mkl._apply_attn_res_and_rmsnorm(prefix, block, proj, norm, out_norm)
    upd, add_out = mkl._apply_attn_res_add_and_rmsnorm(prefix, addend, block, proj, norm, out_norm)
    w = (proj.weight.reshape(-1), norm.weight, out_norm.weight, EPS, EPS)
    assert torch.equal(out, torch.ops.trtllm.kimi_k3_attn_res_rmsnorm(prefix, block, *w))

    # The C++ per-token kernels as the reference.
    monkeypatch.setattr(mkl, "_attn_res_decode_kernel", lambda *args: None)
    monkeypatch.setattr(mkl, "_ATTN_RES_TOPOLOGY", "per_token")
    monkeypatch.setattr(mkl, "_FUSED_ATTN_RES_MAX_TOKENS", _max_tokens())
    ref_out = mkl._apply_attn_res_and_rmsnorm(prefix, block, proj, norm, out_norm)
    ref_upd, ref_add_out = mkl._apply_attn_res_add_and_rmsnorm(
        prefix, addend, block, proj, norm, out_norm
    )
    _close_to_cpp(out, ref_out, "mix")
    _close_to_cpp(add_out, ref_add_out, "add+mix")
    assert torch.equal(upd, ref_upd)


def test_moe_residual_deferred_up_to_the_kernel_limit():
    assert mkl._defers_moe_residual(1)
    assert mkl._defers_moe_residual(_max_tokens())
    assert not mkl._defers_moe_residual(_max_tokens() + 1)


class _FakeMoE(nn.Module):
    """Stands in for ``KimiK3MoERuntime``: routed/shared outputs derived from the input."""

    def forward_parts(self, hidden_states, all_rank_num_tokens=None):
        routed = (hidden_states.float() * 0.5).to(torch.bfloat16)
        shared = (hidden_states.float() * -0.25).to(torch.bfloat16)
        return routed, shared


def _fake_layer(layer_idx: int, seed: int):
    layer = mkl.KimiLinearDecoderLayer.__new__(mkl.KimiLinearDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.attn_res_block_size = 12
    layer.is_kda = True
    layer.is_moe = True
    layer.token_sliced = False
    layer.linear_attn = lambda h, metadata: (h.float() * 0.75).to(torch.bfloat16)
    layer.block_sparse_moe = _FakeMoE()
    proj_a, norm_a, layer.input_layernorm = _modules(seed)
    proj_b, norm_b, layer.post_attention_layernorm = _modules(seed + 1)
    layer.self_attention_res_proj, layer.self_attention_res_norm = proj_a, norm_a
    layer.mlp_res_proj, layer.mlp_res_norm = proj_b, norm_b
    return layer


@pytest.mark.parametrize("m", [7, 32, 64])
@torch.no_grad()
def test_decoder_layers_with_deferred_moe_residual(m: int, monkeypatch):
    """Two decoder layers: the MoE residual update folded into the next mix equals materializing
    it first, bit for bit."""
    layers = [_fake_layer(1, 10), _fake_layer(2, 20)]
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)

    def run():
        h, snaps = hidden, 3
        for layer in layers:
            h, snaps = layer(h, bank, snaps, None)
        return h

    deferred = run()
    assert isinstance(deferred, mkl._DeferredMoEResidual)
    monkeypatch.setattr(mkl, "_defers_moe_residual", lambda num_tokens: False)
    materialized = run()
    assert isinstance(materialized, torch.Tensor)
    assert torch.equal(deferred.materialize(), materialized)
