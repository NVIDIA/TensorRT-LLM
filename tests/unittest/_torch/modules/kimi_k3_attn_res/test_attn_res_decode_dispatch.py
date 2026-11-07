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
@pytest.mark.parametrize("m", [1, 64])
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


@pytest.mark.parametrize(
    "raw, persistent",
    [("per_token", False), ("1", True), ("split", True), ("persistent", True)],
)
def test_attn_res_topology_switch(monkeypatch, raw: str, persistent: bool):
    """``KIMI_K3_ATTN_RES_TOPOLOGY``: ``split`` and ``persistent`` are aliases of ``1``; unknown
    values fail loudly instead of silently measuring the default."""
    monkeypatch.setenv(mkl.KIMI_K3_ATTN_RES_TOPOLOGY_ENV, raw)
    assert mkl._read_persistent_attn_res() is persistent
    for bad in ("on", "per-token"):
        monkeypatch.setenv(mkl.KIMI_K3_ATTN_RES_TOPOLOGY_ENV, bad)
        with pytest.raises(ValueError):
            mkl._read_persistent_attn_res()


def test_fused_max_tokens_env(monkeypatch):
    """``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS``: a positive integer, default 64 (the decode
    kernel's range); anything else fails loudly."""
    env = mkl.KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV
    monkeypatch.delenv(env, raising=False)
    assert mkl._read_token_limit(env, 64) == 64 == _max_tokens()
    monkeypatch.setenv(env, "1")
    assert mkl._read_token_limit(env, 64) == 1
    for bad in ("0", "-3", "1.5", "all"):
        monkeypatch.setenv(env, bad)
        with pytest.raises(ValueError):
            mkl._read_token_limit(env, 64)


def _kernel_for(m: int, num_candidates: int = 5):
    return mkl._attn_res_kernel(_rows(m, HIDDEN), num_candidates)


@pytest.mark.parametrize("persistent", [False, True])
def test_selector_token_ceiling(monkeypatch, persistent: bool):
    """Up to the ceiling the decode kernel takes the mix; above it the persistent kernel with
    ``KIMI_K3_ATTN_RES_TOPOLOGY=1``, else unfused. A ceiling of 1 leaves only single-token mixes
    on the decode kernel, and only they defer a MoE residual update."""
    K = mkl._AttnResKernel
    above = K.PERSISTENT if persistent else K.UNFUSED
    monkeypatch.setattr(mkl, "_PERSISTENT_ATTN_RES_ENABLED", persistent)
    assert [_kernel_for(m) for m in (1, 64, 65)] == [K.DECODE, K.DECODE, above]
    monkeypatch.setattr(mkl, "_FUSED_ATTN_RES_MAX_TOKENS", 1)
    assert [_kernel_for(m) for m in (1, 2, 64)] == [K.DECODE, above, above]
    assert mkl._defers_moe_residual(_rows(1, HIDDEN), 5)
    assert not mkl._defers_moe_residual(_rows(2, HIDDEN), 5)


@pytest.mark.parametrize("persistent", [False, True])
def test_selector_falls_back_to_cpp_per_token_ops(monkeypatch, persistent: bool):
    """Without the decode kernel on the device (no CuTe DSL, or SM107 without the Rubin helpers)
    the mixes up to the ceiling run on the C++ per-token ops, which have no MoE fold: no MoE
    layer defers its residual update."""
    K = mkl._AttnResKernel
    monkeypatch.setattr(mkl, "_decode_attn_res_on_device", lambda device_index: False)
    monkeypatch.setattr(mkl, "_PERSISTENT_ATTN_RES_ENABLED", persistent)
    above = K.PERSISTENT if persistent else K.UNFUSED
    assert [_kernel_for(m) for m in (1, 64, 65)] == [K.PER_TOKEN, K.PER_TOKEN, above]
    assert _kernel_for(8, num_candidates=12) is K.PER_TOKEN
    assert not mkl._defers_moe_residual(_rows(8, HIDDEN), 5)


@pytest.mark.parametrize("m", [1, 16])
@torch.no_grad()
def test_cpp_per_token_fallback_matches_cpp_ops(monkeypatch, m: int):
    """The fallback runs the C++ ops themselves (bitwise), and a decoder layer pair then returns
    the materialized residual."""
    monkeypatch.setattr(mkl, "_decode_attn_res_on_device", lambda device_index: False)
    proj, norm, out_norm = _modules(m)
    prefix, addend, block = _rows(m, HIDDEN), _rows(m, HIDDEN), _rows(4, m, HIDDEN)
    w = (proj.weight.reshape(-1), norm.weight, out_norm.weight, EPS, EPS)
    block4 = block.view(4, m, 1, HIDDEN)
    out = mkl._apply_attn_res_and_rmsnorm(prefix, block, proj, norm, out_norm)
    ref = torch.ops.trtllm.attn_res_rmsnorm_fwd(prefix.view(m, 1, HIDDEN), block4, *w)
    assert torch.equal(out, ref.view(m, HIDDEN))
    upd, add_out = mkl._apply_attn_res_add_and_rmsnorm(prefix, addend, block, proj, norm, out_norm)
    ref_upd, ref_out = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        prefix.view(m, 1, HIDDEN), addend.view(m, 1, HIDDEN), block4, *w
    )
    assert torch.equal(upd, ref_upd.view(m, HIDDEN))
    assert torch.equal(add_out, ref_out.view(m, HIDDEN))
    layers = [_fake_layer(1, 10), _fake_layer(2, 20)]
    h, snaps = _rows(m, HIDDEN), 3
    for layer in layers:
        h, snaps = layer(h, _rows(8, m, HIDDEN), snaps, None)
    assert isinstance(h, torch.Tensor)


class _PlainKdaMixer:
    """Stands in for the KDA mixer without an MXFP8 input."""

    def takes_prequantized_input(self, num_rows, attn_metadata):
        return False

    def __call__(self, h, attn_metadata, prequantized_input=None):
        assert prequantized_input is None
        return (h.float() * 0.75).to(torch.bfloat16)


class _FakeMoE(nn.Module):
    """Stands in for ``KimiK3MoERuntime``: routed/shared outputs derived from the input."""

    def forward(self, hidden_states, all_rank_num_tokens=None, return_parts=False):
        assert return_parts
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


# 65: one row above the decode kernel's limit, where the next mix is not the decode kernel and the
# MoE layer must not defer its residual update.
@pytest.mark.parametrize("m", [7, 32, 64, 65])
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


@pytest.mark.parametrize("m", [7, 32])
@torch.no_grad()
def test_decoder_layers_mxfp8_handoff(m: int):
    """A KDA layer whose pre-attention mix folds in the previous MoE residual hands the mixer the
    MXFP8 form of the normed mix, bitwise the FP8 low-M GEMM's quantization of the BF16 rows; the
    layer outputs do not change, and context batches get no hand-off."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    layers = [_fake_layer(1, 10), _fake_layer(2, 20)]
    for layer in layers:
        layer.linear_attn = _FakeKdaMixer()
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)

    def run(num_contexts):
        metadata = SimpleNamespace(num_contexts=num_contexts)
        for layer in layers:
            layer.linear_attn.seen.clear()
        h, snaps = hidden, 3
        for layer in layers:
            h, snaps = layer(h, bank, snaps, metadata)
        return h.materialize()

    ref = run(num_contexts=1)
    assert all(x is None for layer in layers for _, x in layer.linear_attn.seen)
    assert torch.equal(run(num_contexts=0), ref)
    # The first layer's input is a plain row (no deferred MoE update), the second one's is not.
    (_, first), (rows, handoff) = layers[0].linear_attn.seen[0], layers[1].linear_attn.seen[0]
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
    fused q/k/v/g projection was not declared to take one, nor a padded batch."""
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
    # CUDA-graph padding: more rows than the batch's tokens, so the mixer would not use it.
    assert not mixer.takes_prequantized_input(16, metadata(0, 12, 3))


def _schedule_layer(layer_idx: int, is_moe: bool, is_kda: bool, token_sliced: bool = False):
    from types import SimpleNamespace

    return SimpleNamespace(
        layer_idx=layer_idx,
        is_moe=is_moe,
        is_kda=is_kda,
        token_sliced=token_sliced,
        linear_attn=SimpleNamespace(qkvg_takes_prequantized=lambda num_rows: num_rows <= 32),
    )


def _variants(layers, max_tokens=64, helix_first=None, verify=True):
    return mkl._decode_attn_res_variants(
        layers,
        2,
        HIDDEN,
        max_tokens,
        torch.cuda.current_device(),
        len(layers) if helix_first is None else helix_first,
        verify,
    )


def test_prewarm_variants_follow_the_layer_schedule(monkeypatch):
    """The prewarmed decode-kernel variants are those the decoder layers launch: snapshot counts,
    add modes (plain, attention add, folded MoE residual), the MXFP8 output only for KDA layers
    after a MoE layer in verify batches, and cluster sizes over the token range."""
    V = mkl._AttnResVariant
    none, prefix, moe = 0, 1, 2
    monkeypatch.setattr(mkl, "_PERSISTENT_ATTN_RES_ENABLED", False)
    layers = [
        _schedule_layer(0, is_moe=False, is_kda=True),
        _schedule_layer(1, is_moe=True, is_kda=True),
        _schedule_layer(2, is_moe=True, is_kda=False),
        _schedule_layer(3, is_moe=True, is_kda=True),
        _schedule_layer(4, is_moe=True, is_kda=True),
    ]
    plain = {
        V(2, none, 4, False, False),
        V(2, prefix, 4, False, False),
        V(2, moe, 4, False, False),
        V(3, none, 4, False, False),
        V(3, moe, 4, False, False),
        V(3, prefix, 4, False, False),
        # Five candidates' output mix: eight CTAs per token up to 16 tokens.
        V(4, none, 8, False, False),
        V(4, none, 4, False, False),
    }
    assert _variants(layers, verify=False) == plain
    assert _variants(layers) == plain | {V(3, moe, 4, True, False)}
    # Up to 16 tokens only the eight-CTA cluster serves four or more candidates.
    assert _variants(layers, max_tokens=16, verify=False) == plain - {V(4, none, 4, False, False)}
    # A token ceiling of 1 behaves like a budget of one token.
    monkeypatch.setattr(mkl, "_FUSED_ATTN_RES_MAX_TOKENS", 1)
    assert _variants(layers, verify=False) == plain - {V(4, none, 4, False, False)}
    monkeypatch.setattr(mkl, "_decode_attn_res_on_device", lambda device_index: False)
    assert _variants(layers) == set()


def test_prewarm_variants_helix_a2a(monkeypatch):
    """helix_a2a: the first token-sliced layer's pre-attention mix reads the bank after the PDL
    wait, and sliced KDA layers get no MXFP8 hand-off."""
    V = mkl._AttnResVariant
    none, prefix, moe = 0, 1, 2
    monkeypatch.setattr(mkl, "_PERSISTENT_ATTN_RES_ENABLED", False)
    layers = [
        _schedule_layer(0, is_moe=False, is_kda=True),
        _schedule_layer(1, is_moe=False, is_kda=True),
        _schedule_layer(2, is_moe=True, is_kda=True, token_sliced=True),
        _schedule_layer(3, is_moe=True, is_kda=True, token_sliced=True),
    ]
    assert _variants(layers, helix_first=2) == {
        V(2, none, 4, False, False),
        V(2, prefix, 4, False, False),
        V(2, none, 4, False, True),
        V(3, none, 4, False, False),
        V(3, moe, 4, False, False),
        V(3, prefix, 4, False, False),
    }


@torch.no_grad()
def test_prewarm_compiles_what_the_mixes_launch(monkeypatch):
    """After ``prewarm_decode_kernels`` the model's mixes launch without compiling."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.kernels.blackwell import kimi_k3_attn_res

    monkeypatch.setattr(mkl, "_PERSISTENT_ATTN_RES_ENABLED", False)
    model = mkl.KimiLinearModel.__new__(mkl.KimiLinearModel)
    nn.Module.__init__(model)
    model.layers = [
        _schedule_layer(0, is_moe=False, is_kda=True),
        _schedule_layer(1, is_moe=False, is_kda=True),
    ]
    model._text_cfg = SimpleNamespace(attn_res_block_size=2, hidden_size=HIDDEN)
    model.helix_a2a_first_layer = len(model.layers)
    assert model.prewarm_decode_kernels(8, verify_batches=False) == 2
    misses = kimi_k3_attn_res._compiled.cache_info().misses
    proj, norm, out_norm = _modules(0)
    for m in (1, 8):
        prefix, addend, block = _rows(m, HIDDEN), _rows(m, HIDDEN), _rows(1, m, HIDDEN)
        mkl._apply_attn_res_and_rmsnorm(prefix, block, proj, norm, out_norm)
        mkl._apply_attn_res_add_and_rmsnorm(prefix, addend, block, proj, norm, out_norm)
    torch.cuda.synchronize()
    assert kimi_k3_attn_res._compiled.cache_info().misses == misses
