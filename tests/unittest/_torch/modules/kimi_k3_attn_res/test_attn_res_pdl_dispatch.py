# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-side dispatch of ``TRTLLM_K3_ATTN_RES_PDL`` (``modeling_kimi_linear``).

With the switch on, the decode-sized fused attention-residual calls go to the CuTe DSL kernel
(``trtllm::k3_attn_res_*``) and must return exactly what the C++ path returns with it off; the
deferred MoE residual update (mode 2) must give the same prefix and normalized mix as
materializing it first.
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

HIDDEN = 7168
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs an SM100/SM103/SM107 GPU",
)


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


def _patched(mode: str):
    return mock.patch.multiple(
        mkl,
        _K3_ATTN_RES_PDL=mode != "0",
        _K3_ATTN_RES_PDL_MODE=mode,
        _ATTN_RES_TOPOLOGY="split",
        _FUSED_ATTN_RES_MAX_TOKENS=32,
    )


def _close_to_cpp(actual: torch.Tensor, expected: torch.Tensor, exact: bool):
    if exact:
        assert torch.equal(actual, expected)
    else:
        # Statistics summed in another order: rare differences of one or two BF16 rounding steps.
        a, e = actual.float(), expected.float()
        bound = (
            torch.maximum(a.abs(), e.abs()) * 2.0**-6
            + 2.0**-8 * e.pow(2).mean(-1, keepdim=True).sqrt()
        )
        assert bool(((a - e).abs() <= bound).all())
        assert float(((a - e) != 0).float().mean()) <= 2e-2


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("snapshots", [1, 3, 4, 8])
@pytest.mark.parametrize("m", [7, 32, 56])
def test_dispatch_matches_cpp(m: int, snapshots: int, exact: bool, monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", exact)
    proj, norm, out_norm = _modules(m + snapshots)
    prefix, addend, block = _rows(m, HIDDEN), _rows(m, HIDDEN), _rows(snapshots, m, HIDDEN)
    with _patched("0"):
        ref_out = mkl._apply_attn_res_and_rmsnorm(prefix, block, proj, norm, out_norm)
        ref_upd, ref_add_out = mkl._apply_attn_res_add_and_rmsnorm(
            prefix, addend, block, proj, norm, out_norm
        )
    with _patched("1"):
        assert mkl._k3_attn_res_pdl(prefix, None, block, proj, norm, out_norm) is not None
        out = mkl._apply_attn_res_and_rmsnorm(prefix, block, proj, norm, out_norm)
        upd, add_out = mkl._apply_attn_res_add_and_rmsnorm(
            prefix, addend, block, proj, norm, out_norm
        )
    if m <= 32:
        # Below the split threshold the C++ path is the per-token kernel.
        _close_to_cpp(out, ref_out, exact)
        _close_to_cpp(add_out, ref_add_out, exact)
        assert torch.equal(upd, ref_upd)
    else:
        # Above it the C++ path is the persistent kernel (different summation order).
        for actual, expected in ((out, ref_out), (add_out, ref_add_out), (upd, ref_upd)):
            torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("snapshots", [1, 8])
@pytest.mark.parametrize("m", [7, 32])
@pytest.mark.parametrize("exact", [True, False])
def test_deferred_moe_residual(m: int, snapshots: int, exact: bool, monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", exact)
    proj, norm, out_norm = _modules(m * snapshots)
    deferred = mkl._DeferredMoEResidual(_rows(m, HIDDEN), _rows(m, HIDDEN), _rows(m, HIDDEN))
    block = _rows(snapshots, m, HIDDEN)
    with _patched("2"):
        assert mkl._defer_moe_residual(m)
        assert not mkl._defer_moe_residual(mkl._K3_ATTN_RES_PDL_MAX_M + 1)
        prefix, out = mkl._k3_attn_res_pdl_moe_add(deferred, block, proj, norm, out_norm)
    with _patched("0"):
        ref_prefix = mkl._materialize_prefix_sum(deferred)
        ref_out = mkl._apply_attn_res_and_rmsnorm(ref_prefix, block, proj, norm, out_norm)
    assert torch.equal(prefix, ref_prefix)
    _close_to_cpp(out, ref_out, exact)


class _FakeMoE(nn.Module):
    """Stands in for ``KimiK3MoERuntime``: routed/shared outputs derived from the input."""

    def accepts_fp8_input(self):
        return False

    def forward(
        self, hidden_states, all_rank_num_tokens=None, *, residual=None, defer_residual=False
    ):
        routed = (hidden_states.float() * 0.5).to(torch.bfloat16)
        shared = (hidden_states.float() * -0.25).to(torch.bfloat16)
        if defer_residual:
            return mkl._DeferredMoEResidual(residual, routed, shared)
        from tensorrt_llm._torch.modules.kimi_k3_moe_residual import _fused_moe_residual_add

        # The row cap ``KimiK3MoERuntime.forward`` passes.
        return _fused_moe_residual_add(
            routed, shared, residual, max_rows=mkl._moe_residual_max_rows()
        )


def _fake_layer(layer_idx: int, seed: int):
    layer = mkl.KimiLinearDecoderLayer.__new__(mkl.KimiLinearDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.attn_res_block_size = 12
    layer.is_kda = True
    layer.is_moe = True
    layer.linear_attn = lambda h, metadata: (h.float() * 0.75).to(torch.bfloat16)
    layer.block_sparse_moe = _FakeMoE()
    proj_a, norm_a, layer.input_layernorm = _modules(seed)
    proj_b, norm_b, layer.post_attention_layernorm = _modules(seed + 1)
    layer.self_attention_res_proj, layer.self_attention_res_norm = proj_a, norm_a
    layer.mlp_res_proj, layer.mlp_res_norm = proj_b, norm_b
    return layer


@pytest.mark.parametrize("m", [7, 32])
def test_decoder_layers_with_deferred_moe_residual(m: int, monkeypatch):
    """Two decoder layers: mode 2 (MoE add folded into the next mix) equals the C++ path."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", True)
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL", True)
    layers = [_fake_layer(1, 10), _fake_layer(2, 20)]
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)

    def run(mode):
        h, snaps = hidden, 3
        with _patched(mode):
            for layer in layers:
                h, snaps = layer(h, bank, snaps, None)
                if mode == "2":
                    assert isinstance(h, mkl._DeferredMoEResidual)
            return mkl._materialize_prefix_sum(h)

    assert torch.equal(run("2"), run("0"))


class _RecordingMoE(_FakeMoE):
    """``_FakeMoE`` that also serves the unfused layer path and records how it was called."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self, hidden_states, all_rank_num_tokens=None, *, residual=None, defer_residual=False
    ):
        self.calls.append((residual is not None, defer_residual))
        if residual is None:
            routed = (hidden_states.float() * 0.5).to(torch.bfloat16)
            shared = (hidden_states.float() * -0.25).to(torch.bfloat16)
            return routed + shared
        return super().forward(
            hidden_states, all_rank_num_tokens, residual=residual, defer_residual=defer_residual
        )


_CAP = mkl._MOE_RESIDUAL_MAX_FUSED_ROWS


@pytest.mark.parametrize("m", [7, _CAP, _CAP + 1, _CAP + 64])
def test_decoder_layers_with_ctx_fused_moe_residual(m: int, monkeypatch):
    """TRTLLM_K3_FUSE_MOE_RESIDUAL_CTX: above the generation cap the layer hands the residual to
    the MoE (never deferred), whose fused kernel runs uncapped; at or below it the layer is
    unchanged. The prefix is bitwise equal to the unfused layers'."""
    from tensorrt_llm._torch.modules import kimi_k3_moe_residual as residual_mod

    launches = []
    kernel = residual_mod._moe_residual_add_kernel

    class _Spy:
        def __getitem__(self, grid):
            launches.append(grid)
            return kernel[grid]

    monkeypatch.setattr(residual_mod, "_moe_residual_add_kernel", _Spy())
    layers = [_fake_layer(1, 10), _fake_layer(2, 20)]
    for layer in layers:
        layer.block_sparse_moe = _RecordingMoE()
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)

    def run(ctx_fused: bool, mode: str):
        monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL_CTX", ctx_fused)
        for layer in layers:
            layer.block_sparse_moe.calls.clear()
        launches.clear()
        h, snaps = hidden, 3
        # Inference has no autograd; the fused kernel leaves inputs that require grad to ATen.
        with _patched(mode), torch.no_grad():
            for layer in layers:
                h, snaps = layer(h, bank, snaps, None)
        return mkl._materialize_prefix_sum(h), [
            c for la in layers for c in la.block_sparse_moe.calls
        ]

    ref, ref_calls = run(False, "0")
    assert ref_calls == [(False, False)] * 2
    assert not launches
    fused = m > mkl._MOE_RESIDUAL_MAX_FUSED_ROWS
    for mode in ("0", "2") if fused else ("0",):
        out, calls = run(True, mode)
        assert calls == [(fused, False)] * 2
        assert len(launches) == (2 if fused else 0)
        assert torch.equal(out, ref)


@pytest.mark.parametrize("ctx_fused", [False, True])
def test_ctx_fused_moe_residual_keeps_the_pdl_deferral(ctx_fused: bool, monkeypatch):
    """With ``TRTLLM_K3_FUSE_MOE_RESIDUAL`` and PDL mode 2 deferring above the row cap
    (``TRTLLM_K3_ATTN_RES_PDL_MAX_M`` raised), the context flag changes nothing: both layers
    still defer their update to the next mix."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    m = _CAP + 44
    monkeypatch.setattr(k3_attn_res, "_EXACT", True)
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL", True)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PDL_MAX_M", 1024)
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL_CTX", ctx_fused)
    layers = [_fake_layer(1, 10), _fake_layer(2, 20)]
    for layer in layers:
        layer.block_sparse_moe = _RecordingMoE()
    h, snaps = _rows(m, HIDDEN), 3
    bank = _rows(8, m, HIDDEN)
    with _patched("2"):
        for layer in layers:
            h, snaps = layer(h, bank, snaps, None)
            assert isinstance(h, mkl._DeferredMoEResidual)
    assert [c for la in layers for c in la.block_sparse_moe.calls] == [(True, True)] * 2


# --------------------------------------------------------------------------------------------
# TRTLLM_K3_ATTN_RES_FP8_OUT: the PDL mix also writes the next GEMM's FP8 1x128 input.
# --------------------------------------------------------------------------------------------


def _patched_fp8(mode: str, post: bool = True):
    return mock.patch.multiple(
        mkl,
        _K3_ATTN_RES_PDL=mode != "0",
        _K3_ATTN_RES_PDL_MODE=mode,
        _K3_ATTN_RES_FP8_OUT=mode != "0",
        _K3_ATTN_RES_FP8_OUT_POST=post and mode != "0",
        _ATTN_RES_TOPOLOGY="split",
        _FUSED_ATTN_RES_MAX_TOKENS=32,
    )


def _assert_is_quantized(fp8: "mkl._Fp8Activation", normed: torch.Tensor):
    ref_act, ref_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(normed)
    assert torch.equal(fp8.act.view(torch.uint8), ref_act.view(torch.uint8))
    assert torch.equal(fp8.scale.view(torch.uint8).reshape(-1), ref_scale.view(torch.uint8))


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("snapshots", [1, 4, 8])
@pytest.mark.parametrize("m", [7, 32, 56])
def test_fp8_dispatch_matches_bf16_dispatch(m: int, snapshots: int, exact: bool, monkeypatch):
    """The FP8-output helpers return the BF16 PDL path's outputs bit for bit plus their FP8
    1x128 quantization (what _Fp8BlockScaleWeightReadLinear would compute)."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", exact)
    proj, norm, out_norm = _modules(3 * m + snapshots)
    prefix, addend, block = _rows(m, HIDDEN), _rows(m, HIDDEN), _rows(snapshots, m, HIDDEN)
    deferred = mkl._DeferredMoEResidual(_rows(m, HIDDEN), _rows(m, HIDDEN), _rows(m, HIDDEN))
    with _patched_fp8("2"):
        ref = mkl._k3_attn_res_pdl(prefix, None, block, proj, norm, out_norm)
        upd_none, normed, fp8 = mkl._k3_attn_res_fp8(prefix, None, block, proj, norm, out_norm)
        assert upd_none is None and torch.equal(normed, ref)
        _assert_is_quantized(fp8, normed)

        ref_upd, ref_out = mkl._k3_attn_res_pdl(prefix, addend, block, proj, norm, out_norm)
        upd, normed, fp8 = mkl._k3_attn_res_fp8(prefix, addend, block, proj, norm, out_norm)
        assert torch.equal(upd, ref_upd) and torch.equal(normed, ref_out)
        _assert_is_quantized(fp8, normed)

        ref_prefix, ref_out = mkl._k3_attn_res_pdl_moe_add(deferred, block, proj, norm, out_norm)
        new_prefix, normed, fp8 = mkl._k3_attn_res_moe_add_fp8(
            deferred, block, proj, norm, out_norm
        )
        assert torch.equal(new_prefix, ref_prefix) and torch.equal(normed, ref_out)
        _assert_is_quantized(fp8, normed)
    with _patched_fp8("0"):
        assert mkl._k3_attn_res_fp8(prefix, None, block, proj, norm, out_norm) is None
        assert mkl._k3_attn_res_moe_add_fp8(deferred, block, proj, norm, out_norm) is None


def _fp8_linear(out_features: int, in_features: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(out_features, in_features, device="cuda", generator=gen) * 0.02).to(
        torch.bfloat16
    )
    lin = mkl._Fp8BlockScaleWeightReadLinear(
        *mkl._Fp8BlockScaleWeightReadLinear.quantize_weight(w), out_features
    )
    if not lin.supports_prequantized_input:
        pytest.skip("needs the Rubin CuTe DSL MXFP8 GEMM (SM107)")
    return lin


@pytest.mark.parametrize("m", [7, 56])
def test_latent_down_prequantized_is_bitwise_identical(m: int):
    """KimiK3MoERuntime._latent_down from the FP8 pair equals the FP8 weight-read GEMM of the
    BF16 rows (its own quantize kernel produces the same pair)."""
    down = _fp8_linear(3584, HIDDEN, seed=m)
    fake = mock.Mock(spec=["routed_expert_down_proj", "_routed_projection"])
    fake.routed_expert_down_proj = down
    fake._routed_projection = mkl.KimiK3MoERuntime._routed_projection
    x = _rows(m, HIDDEN)
    fp8 = mkl._Fp8Activation(*torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x))
    assert mkl.KimiK3MoERuntime.accepts_fp8_input(fake)
    ref = mkl.KimiK3MoERuntime._latent_down(fake, x, None)
    out = mkl.KimiK3MoERuntime._latent_down(fake, x, fp8)
    assert torch.equal(out, ref) and torch.equal(ref, down(x))


class _FakeKDA:
    """Stands in for KimiKDALinearAttention and records the prequantized inputs it receives."""

    def __init__(self, consume: bool = True):
        self.received = []
        self.consume = consume

    def accepts_prequantized_input(self):
        return True

    def consumes_prequantized_input(self, metadata, num_rows):
        return self.consume

    def __call__(self, h, metadata, hidden_fp8=None):
        self.received.append((h, hidden_fp8))
        return (h.float() * 0.75).to(torch.bfloat16)


class _FakeFp8MoE(_FakeMoE):
    def __init__(self):
        super().__init__()
        self.received = []

    def accepts_fp8_input(self):
        return True

    def forward(
        self,
        hidden_states,
        all_rank_num_tokens=None,
        *,
        residual=None,
        defer_residual=False,
        hidden_fp8=None,
    ):
        self.received.append((hidden_states, hidden_fp8))
        return super().forward(
            hidden_states, all_rank_num_tokens, residual=residual, defer_residual=defer_residual
        )


def _run_layers(mode: str, m: int, *, fp8: bool, post: bool = True, consume: bool = True):
    """Two decoder layers (the second takes a snapshot) over a fixed bank; returns the prefix sum
    and the layers (their fake KDA / MoE record the operands they were handed)."""
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)
    metadata = mock.Mock(spec=["num_contexts"])
    metadata.num_contexts = 0
    layers = [_fake_layer(11, 10), _fake_layer(12, 20)]
    for layer in layers:
        layer.linear_attn = _FakeKDA(consume)
        layer.block_sparse_moe = _FakeFp8MoE()
    h, snaps = hidden, 3
    with _patched_fp8(mode, post) if fp8 else _patched(mode):
        for layer in layers:
            h, snaps = layer(h, bank, snaps, metadata)
    return mkl._materialize_prefix_sum(h), layers


@pytest.mark.parametrize("mode", ["1", "2"])
def test_post_attention_site_is_a_separate_switch(mode: str, monkeypatch):
    """Without TRTLLM_K3_ATTN_RES_FP8_OUT_POST only the pre-attention mix writes an FP8 operand."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", True)
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL", True)
    torch.manual_seed(7)
    state = torch.cuda.get_rng_state()
    ref, _ = _run_layers(mode, 7, fp8=False)
    torch.cuda.set_rng_state(state)
    out, layers = _run_layers(mode, 7, fp8=True, post=False)
    assert torch.equal(out, ref)
    for layer in layers:
        assert all(fp8 is not None for _, fp8 in layer.linear_attn.received)
        assert all(fp8 is None for _, fp8 in layer.block_sparse_moe.received)


@pytest.mark.parametrize("mode", ["1", "2"])
def test_no_fp8_operand_when_the_mixer_does_not_consume_it(mode: str, monkeypatch):
    """Plain decode and sequential verification ignore the operand: the epilogue is not paid."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", True)
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL", True)
    torch.manual_seed(8)
    state = torch.cuda.get_rng_state()
    ref, _ = _run_layers(mode, 7, fp8=False)
    torch.cuda.set_rng_state(state)
    out, layers = _run_layers(mode, 7, fp8=True, post=False, consume=False)
    assert torch.equal(out, ref)
    for layer in layers:
        assert all(fp8 is None for _, fp8 in layer.linear_attn.received)


def test_fp8_helpers_and_deferral_follow_the_pdl_gate():
    """KIMI_K3_FUSED_ATTN_RES=0 reaches the helpers through ``_K3_ATTN_RES_PDL``."""
    proj, norm, out_norm = _modules(5)
    m = 7
    prefix, block = _rows(m, HIDDEN), _rows(4, m, HIDDEN)
    deferred = mkl._DeferredMoEResidual(_rows(m, HIDDEN), _rows(m, HIDDEN), _rows(m, HIDDEN))
    with mock.patch.multiple(mkl, _K3_ATTN_RES_PDL=False, _K3_ATTN_RES_PDL_MODE="2"):
        assert mkl._k3_attn_res_pdl(prefix, None, block, proj, norm, out_norm) is None
        assert mkl._k3_attn_res_fp8(prefix, None, block, proj, norm, out_norm) is None
        assert mkl._k3_attn_res_moe_add_fp8(deferred, block, proj, norm, out_norm) is None
        assert not mkl._defer_moe_residual(m)
    with mock.patch.multiple(mkl, _K3_ATTN_RES_PDL=True, _K3_ATTN_RES_PDL_MODE="2"):
        assert mkl._defer_moe_residual(m)
        assert not mkl._defer_moe_residual(mkl._K3_ATTN_RES_PDL_MAX_M + 1)


def _import_flags(env: dict, snippet: str) -> str:
    """Run ``snippet`` in a fresh interpreter that imports the model module under ``env``."""
    base = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("KIMI_K3_FUSED_ATTN_RES", "TRTLLM_K3_ATTN_RES"))
    }
    code = "from tensorrt_llm._torch.models import modeling_kimi_linear as m\n" + snippet
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    return done.stdout.strip().splitlines()[-1]


_FLAGS = (
    "print(json.dumps([m._K3_ATTN_RES_PDL, m._K3_ATTN_RES_FP8_OUT, m._K3_ATTN_RES_FP8_OUT_POST,"
    " m._defer_moe_residual(8)]))"
)


@pytest.mark.parametrize(
    "env,expected",
    [
        (  # the kill switch beats every opt-in
            {
                "KIMI_K3_FUSED_ATTN_RES": "0",
                "TRTLLM_K3_ATTN_RES_PDL": "2",
                "TRTLLM_K3_ATTN_RES_FP8_OUT": "1",
                "TRTLLM_K3_ATTN_RES_FP8_OUT_POST": "1",
            },
            [False, False, False, False],
        ),
        (  # the kill switch beats mode 1 too
            {"KIMI_K3_FUSED_ATTN_RES": "0", "TRTLLM_K3_ATTN_RES_PDL": "1"},
            [False, False, False, False],
        ),
        # mode 2 alone defers the MoE residual; mode 1 does not
        ({"TRTLLM_K3_ATTN_RES_PDL": "2"}, [True, False, False, True]),
        ({"TRTLLM_K3_ATTN_RES_PDL": "1"}, [True, False, False, False]),
        ({}, [False, False, False, False]),
        (  # the post-attention site is opt-in on top of the pre-attention one
            {"TRTLLM_K3_ATTN_RES_PDL": "2", "TRTLLM_K3_ATTN_RES_FP8_OUT": "1"},
            [True, True, False, True],
        ),
        (  # FP8 output without the PDL kernels imports fine and stays off
            {"TRTLLM_K3_ATTN_RES_FP8_OUT": "1"},
            [False, False, False, False],
        ),
    ],
)
def test_env_flags(env: dict, expected: list):
    assert json.loads(_import_flags(env, "import json\n" + _FLAGS)) == expected


def test_fp8_output_without_pdl_is_rejected_at_model_build():
    snippet = (
        "try:\n    m._check_attn_res_fp8_flags()\n    print('accepted')\n"
        "except ValueError:\n    print('rejected')"
    )
    assert _import_flags({"TRTLLM_K3_ATTN_RES_FP8_OUT": "1"}, snippet) == "rejected"
    assert _import_flags(
        {"TRTLLM_K3_ATTN_RES_FP8_OUT": "1", "TRTLLM_K3_ATTN_RES_PDL": "1"}, snippet
    ) == ("accepted")


@pytest.mark.parametrize(
    "accepts,contexts,rows,replay,expected",
    [
        (True, 0, 24, True, True),  # fused verification: 4 requests x 6 tokens
        (True, 0, 4, True, False),  # one token per request: forward_decode quantizes itself
        (True, 1, 24, True, False),  # a context request in the batch
        (True, 0, 24, False, False),  # sequential verification (no KDA replay caches)
        (False, 0, 24, True, False),  # the projection takes no prequantized input
    ],
)
def test_mixer_consumes_prequantized_input_only_on_fused_verify(
    accepts, contexts, rows, replay, expected
):
    from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

    fake = SimpleNamespace(
        accepts_prequantized_input=lambda: accepts,
        layer_idx=3,
        _has_kda_replay_caches=lambda cache: replay and cache == "layer-cache",
    )
    metadata = SimpleNamespace(
        num_contexts=contexts,
        seq_lens=torch.empty(4, dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(mamba_layer_cache=lambda idx: "layer-cache"),
    )
    assert KimiKDALinearAttention.consumes_prequantized_input(fake, metadata, rows) is expected


@pytest.mark.parametrize("mode", ["1", "2"])
@pytest.mark.parametrize("m", [7, 32])
def test_decoder_layers_with_fp8_output(m: int, mode: str, monkeypatch):
    """Decoder layers with TRTLLM_K3_ATTN_RES_FP8_OUT: the prefix sums equal the BF16-output run
    and every attention / MoE input arrives with its exact FP8 1x128 quantization."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", True)
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL", True)
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)
    metadata = mock.Mock(spec=["num_contexts"])
    metadata.num_contexts = 0

    def run(fp8: bool):
        # Layer 12 takes a snapshot (no residual add before its MoE mix).
        layers = [_fake_layer(11, 10), _fake_layer(12, 20)]
        for layer in layers:
            layer.linear_attn = _FakeKDA()
            layer.block_sparse_moe = _FakeFp8MoE()
        h, snaps = hidden, 3
        with _patched_fp8(mode) if fp8 else _patched(mode):
            for layer in layers:
                h, snaps = layer(h, bank, snaps, metadata)
        return mkl._materialize_prefix_sum(h), layers

    ref, _ = run(False)
    out, layers = run(True)
    assert torch.equal(out, ref)
    for layer in layers:
        for h, fp8 in layer.linear_attn.received + layer.block_sparse_moe.received:
            assert fp8 is not None
            _assert_is_quantized(fp8, h)


# --------------------------------------------------------------------------------------------
# TRTLLM_K3_ATTN_RES_POST_FP8: the persistent post-attention mix also writes the MoE latent-down
# FP8 1x128 input (DEP16 decode: PDL kernels off, persistent topology, P8 on).
# --------------------------------------------------------------------------------------------


def _dep16_stack(post_fp8: bool, min_tokens: int = 0, kf: bool = False):
    return mock.patch.multiple(
        mkl,
        _K3_ATTN_RES_PDL=False,
        _K3_ATTN_RES_PDL_MODE="0",
        _K3_ATTN_RES_FP8_OUT=False,
        _K3_ATTN_RES_FP8_OUT_POST=False,
        _K3_FUSE_MOE_RESIDUAL=True,
        _K3_ATTN_RES_PERSISTENT_FUSE=True,
        _K3_ATTN_RES_PERSISTENT_FUSE_MAX_TOKENS=1024,
        _K3_ATTN_RES_POST_FP8=post_fp8,
        _K3_ATTN_RES_POST_FP8_MIN_TOKENS=min_tokens,
        _K3_ATTN_RES_KF=kf,
        _K3_ATTN_RES_KF_MAX_M=84,
        _FUSED_ATTN_RES_ENABLED=True,
        _ATTN_RES_TOPOLOGY="split",
        _FUSED_ATTN_RES_MAX_TOKENS=32,
    )


@pytest.fixture(scope="module")
def persistent_ext_op():
    try:
        return mkl._attn_res_persistent_ext_op()
    except (RuntimeError, OSError) as err:
        pytest.skip(f"trtllm::attn_res_persistent_ext_fwd unavailable: {err}")


def _run_dep16_layers(
    m: int,
    post_fp8: bool,
    min_tokens: int = 0,
    accepts: bool = True,
    kf: bool = False,
):
    """Layers 11 (post mix = prefix + attention output) and 12 (a snapshot layer: the attention
    output is the new prefix, no addend) over a fixed bank; returns the prefix sum and layers."""
    bank = _rows(8, m, HIDDEN)
    hidden = _rows(m, HIDDEN)
    metadata = mock.Mock(spec=["num_contexts"])
    metadata.num_contexts = 0
    layers = [_fake_layer(11, 10), _fake_layer(12, 20)]
    for layer in layers:
        layer.linear_attn = _FakeKDA()
        layer.block_sparse_moe = _FakeFp8MoE()
        if not accepts:
            layer.block_sparse_moe.accepts_fp8_input = lambda: False
    h, snaps = hidden, 3
    with _dep16_stack(post_fp8, min_tokens, kf), torch.no_grad():
        for layer in layers:
            h, snaps = layer(h, bank, snaps, metadata)
    return mkl._materialize_prefix_sum(h), layers


@pytest.mark.parametrize("m", [36, 84, 129, 192, 256, 280])
def test_decoder_layers_with_persistent_post_fp8(persistent_ext_op, m: int):
    """The post-attention mix of both an add layer and a snapshot layer hands the MoE its exact FP8
    1x128 quantization, and the prefix sum equals the run without the fusion bit for bit."""
    torch.manual_seed(m)
    state = torch.cuda.get_rng_state()
    ref, ref_layers = _run_dep16_layers(m, post_fp8=False)
    torch.cuda.set_rng_state(state)
    out, layers = _run_dep16_layers(m, post_fp8=True)
    assert torch.equal(out, ref)
    for layer in ref_layers:
        assert all(fp8 is None for _, fp8 in layer.block_sparse_moe.received)
    for layer, ref_layer in zip(layers, ref_layers):
        assert len(layer.block_sparse_moe.received) == 1
        h, fp8 = layer.block_sparse_moe.received[0]
        assert fp8 is not None
        _assert_is_quantized(fp8, h)
        # The BF16 rows the router and the shared expert read are the unfused path's, bit for bit.
        assert torch.equal(h, ref_layer.block_sparse_moe.received[0][0])


@pytest.mark.parametrize("m", [36, 84, 100, 192])
def test_persistent_post_fp8_yields_to_the_kf_kernel(persistent_ext_op, m: int):
    """Where TRTLLM_K3_ATTN_RES_KF serves the plain mix (up to its token cap) the layer keeps it
    and the MoE quantizes by itself; above the cap the persistent epilogue takes over."""
    torch.manual_seed(m)
    state = torch.cuda.get_rng_state()
    ref, _ = _run_dep16_layers(m, post_fp8=False, kf=True)
    torch.cuda.set_rng_state(state)
    out, layers = _run_dep16_layers(m, post_fp8=True, kf=True)
    assert torch.equal(out, ref)
    served = m > 84
    for layer in layers:
        assert all((fp8 is not None) == served for _, fp8 in layer.block_sparse_moe.received)


@pytest.mark.parametrize("m", [100, 127])
def test_persistent_post_fp8_off_below_min_tokens(persistent_ext_op, m: int):
    torch.manual_seed(m)
    state = torch.cuda.get_rng_state()
    ref, _ = _run_dep16_layers(m, post_fp8=False)
    torch.cuda.set_rng_state(state)
    out, layers = _run_dep16_layers(m, post_fp8=True, min_tokens=128)
    assert torch.equal(out, ref)
    for layer in layers:
        assert all(fp8 is None for _, fp8 in layer.block_sparse_moe.received)


def test_persistent_post_fp8_needs_a_moe_that_takes_it(persistent_ext_op):
    out, layers = _run_dep16_layers(192, post_fp8=True, accepts=False)
    for layer in layers:
        assert all(fp8 is None for _, fp8 in layer.block_sparse_moe.received)


def test_post_fp8_gate(monkeypatch):
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PERSISTENT_FUSE", True)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PERSISTENT_FUSE_MAX_TOKENS", 1024)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PDL", False)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_KF", False)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_KF_MAX_M", 84)
    monkeypatch.setattr(mkl, "_ATTN_RES_TOPOLOGY", "split")
    monkeypatch.setattr(mkl, "_FUSED_ATTN_RES_MAX_TOKENS", 32)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_POST_FP8", True)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_POST_FP8_MIN_TOKENS", 0)
    applies = mkl._post_fp8_persistent_applies
    assert applies(33, 2) and applies(128, 2) and applies(256, 9) and applies(1024, 5)
    assert not applies(32, 2)  # below the persistent topology's split
    assert not applies(1025, 2)  # above the persistent fuse cap
    assert not applies(256, 10)  # the persistent kernel's N range is [2, 9]
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_POST_FP8_MIN_TOKENS", 128)
    assert not applies(127, 2) and applies(128, 2)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_POST_FP8_MIN_TOKENS", 0)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_KF", True)
    assert not applies(36, 2) and not applies(
        84, 5
    )  # the generated attn_res kernel keeps its plain mixes
    assert applies(85, 2)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_KF", False)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PDL", True)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PDL_MAX_M", 256)
    assert not applies(256, 2)  # the PDL kernels serve this range
    assert applies(257, 2)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PDL", False)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_POST_FP8", False)
    assert not applies(512, 2)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_POST_FP8", True)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PERSISTENT_FUSE", False)
    assert not applies(512, 2)  # needs TRTLLM_K3_ATTN_RES_PERSISTENT_FUSE
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PERSISTENT_FUSE", True)
    with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
        assert not applies(512, 2)  # the op has no fake kernel


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, [True, 0]),
        ({"TRTLLM_K3_ATTN_RES_POST_FP8": "0"}, [False, 0]),
        ({"TRTLLM_K3_ATTN_RES_POST_FP8_MIN_TOKENS": "192"}, [True, 192]),
    ],
)
def test_post_fp8_env_flags(env: dict, expected: list):
    snippet = (
        "import json\n"
        "print(json.dumps([m._K3_ATTN_RES_POST_FP8, m._K3_ATTN_RES_POST_FP8_MIN_TOKENS]))"
    )
    assert json.loads(_import_flags(env, snippet)) == expected
