# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-side dispatch of the decode attention-residual kernel (``modeling_kimi_linear``).

Decode-sized fused attention-residual calls go to the CuTe DSL kernel
(``trtllm::kimi_k3_attn_res_*``) and stay within its declared tolerance of the C++ per-token
ops; a MoE layer's residual update deferred into the next layer's mix
gives the same prefix and normalized mix, bit for bit, as materializing it first.
"""

import functools

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


# The kernel itself is compared with the C++ ops over its full range in
# kernels/parallel/test_kimi_k3_attn_res_decode.py; this checks the model-side wrappers.
@pytest.mark.parametrize("snapshots", [1, 4, 11])
@pytest.mark.parametrize("m", [1, 128])
@torch.no_grad()
def test_dispatch_matches_cpp(m: int, snapshots: int):
    proj, norm, out_norm = _modules(m + snapshots)
    prefix, addend, block = _rows(m, HIDDEN), _rows(m, HIDDEN), _rows(snapshots, m, HIDDEN)
    out = mkl._apply_attn_res_and_rmsnorm(prefix, block, proj, norm, out_norm)
    upd, add_out = mkl._apply_attn_res_add_and_rmsnorm(prefix, addend, block, proj, norm, out_norm)
    w = (proj.weight.reshape(-1), norm.weight, out_norm.weight, EPS, EPS)
    assert torch.equal(out, torch.ops.trtllm.kimi_k3_attn_res_rmsnorm(prefix, block, *w))

    # The C++ per-token ops as the reference.
    block4 = block.view(snapshots, m, 1, HIDDEN)
    ref_out = torch.ops.trtllm.attn_res_rmsnorm_fwd(prefix.view(m, 1, HIDDEN), block4, *w)
    ref_upd, ref_add_out = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        prefix.view(m, 1, HIDDEN), addend.view(m, 1, HIDDEN), block4, *w
    )
    _close_to_cpp(out, ref_out.view(m, HIDDEN), "mix")
    _close_to_cpp(add_out, ref_add_out.view(m, HIDDEN), "add+mix")
    assert torch.equal(upd, ref_upd.view(m, HIDDEN))


@pytest.mark.parametrize("raw, persistent", [("per_token", False), ("1", True), ("split", True)])
def test_attn_res_topology_switch(monkeypatch, raw: str, persistent: bool):
    """``KIMI_K3_ATTN_RES_TOPOLOGY``: ``split`` stays an alias of ``1``; ``persistent`` and unknown
    values fail loudly instead of silently measuring the default."""
    monkeypatch.setenv(mkl.KIMI_K3_ATTN_RES_TOPOLOGY_ENV, raw)
    assert mkl._read_persistent_attn_res() is persistent
    for bad in ("persistent", "on"):
        monkeypatch.setenv(mkl.KIMI_K3_ATTN_RES_TOPOLOGY_ENV, bad)
        with pytest.raises(ValueError):
            mkl._read_persistent_attn_res()


class _PlainKdaMixer:
    """Stands in for the KDA mixer without an MXFP8 input."""

    def takes_prequantized_input(self, num_rows, attn_metadata):
        return False

    def __call__(self, h, attn_metadata, prequantized_input=None):
        assert prequantized_input is None
        return (h.float() * 0.75).to(torch.bfloat16)


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
    layer.linear_attn = _PlainKdaMixer()
    layer.block_sparse_moe = _FakeMoE()
    proj_a, norm_a, layer.input_layernorm = _modules(seed)
    proj_b, norm_b, layer.post_attention_layernorm = _modules(seed + 1)
    layer.self_attention_res_proj, layer.self_attention_res_norm = proj_a, norm_a
    layer.mlp_res_proj, layer.mlp_res_norm = proj_b, norm_b
    # Each fake layer's result goes on to another decoder layer.
    layer.forward = functools.partial(
        mkl.KimiLinearDecoderLayer.forward, layer, defer_moe_residual=True
    )
    return layer


# 129: one row above the decode kernel's limit, where the next mix is not the decode kernel and the
# MoE layer must not defer its residual update.
@pytest.mark.parametrize("m", [7, 32, 64, 128, 129])
@torch.no_grad()
def test_decoder_layers_with_deferred_moe_residual(m: int):
    """Two decoder layers: the MoE residual update folded into the next mix equals materializing
    it first, bit for bit, and is deferred only where the decode kernel runs that mix. Without
    ``defer_moe_residual`` a layer returns a tensor."""
    layers = [_fake_layer(1, 10), _fake_layer(2, 20)]
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)

    def run(defer: bool):
        h, snaps = hidden, 3
        for layer in layers:
            h, snaps = layer(h, bank, snaps, None, defer_moe_residual=defer)
        return h

    deferred = run(defer=True)
    assert isinstance(deferred, mkl._DeferredMoEResidual) == (m <= _max_tokens())
    materialized = run(defer=False)
    assert isinstance(materialized, torch.Tensor)
    if isinstance(deferred, mkl._DeferredMoEResidual):
        deferred = deferred.materialize()
    assert torch.equal(deferred, materialized)


class _FakeKdaMixer:
    """Stands in for the KDA mixer: records the MXFP8 hand-off it is given for each call."""

    def __init__(self):
        self.seen = []

    def takes_prequantized_input(self, num_rows, attn_metadata):
        return attn_metadata.num_contexts == 0

    def __call__(self, h, attn_metadata, prequantized_input=None):
        self.seen.append((h, prequantized_input))
        return (h.float() * 0.75).to(torch.bfloat16)


@pytest.mark.parametrize("attention", ["kda", "mla"])
@pytest.mark.parametrize("m", [7, 32])
@torch.no_grad()
def test_decoder_layers_mxfp8_handoff(m: int, attention: str):
    """A KDA or MLA layer whose pre-attention mix folds in the previous MoE residual hands the
    attention the MXFP8 form of the normed mix, bitwise the FP8 low-M GEMM's quantization of the
    BF16 rows; the layer outputs do not change, and context batches get no hand-off."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    layers = [_fake_layer(1, 10), _fake_layer(2, 20)]
    mixers = []
    for layer in layers:
        mixers.append(_FakeKdaMixer())
        layer.is_kda = attention == "kda"
        if layer.is_kda:
            layer.linear_attn = mixers[-1]
        else:
            layer.self_attn = mixers[-1]
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)

    def run(num_contexts):
        metadata = SimpleNamespace(num_contexts=num_contexts)
        for mixer in mixers:
            mixer.seen.clear()
        h, snaps = hidden, 3
        for layer in layers:
            h, snaps = layer(h, bank, snaps, metadata)
        return h.materialize()

    ref = run(num_contexts=1)
    assert all(x is None for mixer in mixers for _, x in mixer.seen)
    assert torch.equal(run(num_contexts=0), ref)
    # The first layer's input is a plain row (no deferred MoE update), the second one's is not.
    (_, first), (rows, handoff) = mixers[0].seen[0], mixers[1].seen[0]
    assert first is None and handoff is not None
    ref_q, ref_s = low_m.mxfp8_quantize_reference(rows)
    xq, xs = handoff
    assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
    r = torch.arange(m, device="cuda").unsqueeze(1)
    c = torch.arange(HIDDEN // 32, device="cuda").unsqueeze(0)
    offsets = (c // 4) * 512 + (r % 32) * 16 + (r % 128) // 32 * 4 + c % 4
    assert torch.equal(xs[offsets], ref_s)


def _kda_mixer_taking_mxfp8(max_rows: int):
    from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention

    mixer = KimiKDALinearAttention.__new__(KimiKDALinearAttention)
    nn.Module.__init__(mixer)
    mixer.register_to_config = False
    mixer._qkvg_takes_prequantized = None
    return mixer, lambda num_rows: 1 <= num_rows <= max_rows


def test_kda_mixer_takes_mxfp8_only_for_verify_batches():
    """The producer predicate and the verify routing share one batch condition: plain
    single-token decode and context batches get no MXFP8 input, and neither does a mixer whose
    fused q/k/v/g projection was not declared to take one."""
    from types import SimpleNamespace

    mixer, takes = _kda_mixer_taking_mxfp8(max_rows=32)

    def metadata(num_contexts: int, num_tokens: int, num_requests: int):
        return SimpleNamespace(
            num_contexts=num_contexts, num_tokens=num_tokens, seq_lens=torch.empty(num_requests)
        )

    verify = metadata(0, 16, 4)
    assert not mixer.takes_prequantized_input(16, verify)
    mixer.accept_prequantized_qkvg_input(takes)
    assert mixer.takes_prequantized_input(16, verify)
    assert not mixer.takes_prequantized_input(64, metadata(0, 64, 8))
    assert not mixer.takes_prequantized_input(4, metadata(0, 4, 4))
    assert not mixer.takes_prequantized_input(16, metadata(1, 16, 4))
