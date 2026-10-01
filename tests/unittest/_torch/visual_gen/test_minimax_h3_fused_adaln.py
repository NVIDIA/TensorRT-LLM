# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiniMax-H3 fused AdaLN kernels (RMSNorm + gathered modulation, gated residual)."""

import pytest
import torch
from test_minimax_h3_transformer import (  # noqa: I001  (sibling test module, pytest prepends its dir)
    _initialize_weights,
    _make_model_config,
    _model_inputs,
)

import tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 as h3
from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_adaln import (
    h3_adaln_supported,
    h3_gate_res_norm_mod,
    h3_norm_mod,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _rmsnorm_bf16(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The reference norm: FP32 statistics, output rounded to BF16 like the custom RMSNorm op."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(
        torch.bfloat16
    )


def _inputs(rows=1029, hidden=1536, n_mod=6, seed=3):
    torch.manual_seed(seed)
    x = torch.randn((1, rows, hidden), device="cuda", dtype=torch.bfloat16)
    a = torch.randn((1, rows, hidden), device="cuda", dtype=torch.bfloat16)
    w = (1 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
    mod = (0.5 * torch.randn((n_mod, 6 * hidden), device="cuda")).to(torch.bfloat16)
    idx = torch.randint(0, n_mod, (rows,), device="cuda")
    return x, a, w, mod, idx


@requires_cuda
def test_norm_mod_matches_reference_within_rounding():
    x, _, w, mod, idx = _inputs()
    hidden = x.shape[-1]
    scale, shift = mod[:, hidden : 2 * hidden], mod[:, :hidden]
    expected = (
        _rmsnorm_bf16(x, w, 1e-5).float() * (1.0 + scale.index_select(0, idx).float())
        + shift.index_select(0, idx).float()
    )
    actual = h3_norm_mod(x, w, mod, idx, 1e-5, 1, 0)
    assert actual.shape == x.shape and actual.dtype == torch.bfloat16
    # FP32 modulation with one rounding vs the FP32 formula: within one BF16 ulp of the output scale.
    assert (actual.float() - expected).abs().max() <= expected.abs().max() * 2**-7


@requires_cuda
def test_gate_res_norm_mod_matches_single_rounding_residual_and_norm():
    x, a, w, mod, idx = _inputs()
    hidden = x.shape[-1]
    gate = mod[:, 2 * hidden : 3 * hidden]
    scale = mod[:, 4 * hidden : 5 * hidden]
    shift = mod[:, 3 * hidden : 4 * hidden]
    # The kernel forms x + gate * a in FP32 and rounds once, matching the compiled region
    # (Inductor fuses the two BF16 ops into one FP32 chain); the eager path rounds twice.
    expected_x1 = (x.float() + gate.index_select(0, idx).float() * a.float()).to(torch.bfloat16)
    eager_x1 = x + gate.index_select(0, idx) * a
    expected_m = (
        _rmsnorm_bf16(expected_x1, w, 1e-5).float() * (1.0 + scale.index_select(0, idx).float())
        + shift.index_select(0, idx).float()
    )
    x1, m = h3_gate_res_norm_mod(x, a, w, mod, idx, 1e-5, 2, 4, 3)
    assert torch.equal(x1, expected_x1)
    assert (x1.float() - eager_x1.float()).abs().max() <= eager_x1.float().abs().max() * 2**-7
    assert (m.float() - expected_m).abs().max() <= expected_m.abs().max() * 2**-7


@requires_cuda
def test_supported_gate():
    assert h3_adaln_supported(5376, torch.bfloat16)
    assert not h3_adaln_supported(5376, torch.float16)
    assert not h3_adaln_supported(6 * 1024 + 1, torch.bfloat16)


@requires_cuda
def test_block_dispatch_matches_separate_path_within_rounding(monkeypatch):
    """Model output with the fused AdaLN kernels is close to the separate norm + modulation path.

    The fused kernels modulate in FP32 with one rounding where the eager path rounds after each
    BF16 op, so the outputs are not bit-identical; they agree to BF16 rounding noise.
    """
    inputs = _model_inputs("cuda")
    outputs = {}
    calls = []
    op = h3.h3_norm_mod

    def tracked(*args, **kwargs):
        calls.append(True)
        return op(*args, **kwargs)

    monkeypatch.setattr(h3, "h3_norm_mod", tracked)
    for fuse in (True, False):
        monkeypatch.setattr(h3, "FUSE_ADALN", fuse)
        torch.manual_seed(0)
        model = (
            h3.MiniMaxH3Transformer3DModel(_make_model_config(num_layers=1, num_refiner_layers=1))
            .to("cuda")
            .eval()
        )
        _initialize_weights(model, scale=0.1)
        model.requires_grad_(False)
        before = len(calls)
        with torch.inference_mode():
            outputs[fuse] = model(**inputs)
        assert (len(calls) > before) == fuse
    for key in ("sample", "audio_sample"):
        fused_out = getattr(outputs[True], key).float()
        ref = getattr(outputs[False], key).float()
        assert fused_out.shape == ref.shape
        assert (fused_out - ref).abs().max() <= ref.abs().max() * 2**-5
