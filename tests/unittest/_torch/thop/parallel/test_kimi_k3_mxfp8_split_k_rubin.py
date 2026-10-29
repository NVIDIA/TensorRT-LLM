# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-K tactics and PDL of the Kimi K3 MXFP8 Rubin GEMM.

``TRTLLM_K3_MXFP8_SPLIT_K=1`` adds split-K tactics to ``CuteDSLKimiK3MXFP8RubinLinear``: the
GEMM writes one FP32 partial per K slice and ``split_k_reduce_bf16`` sums them in a fixed
order. Only the FP32 summation order differs from the unsplit GEMM, so outputs agree within
one BF16 ulp and are deterministic. ``TRTLLM_CUTEDSL_RUBIN_GEMM_PDL=1`` changes only the
launch (programmatic dependent launch), so outputs must be bitwise identical.
"""

import pytest
import torch

from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
    _rubin_gemm_grid_ctas,
    _rubin_gemm_use_pdl,
    _split_k_factors,
)

# (N, K, M): K3 decode shapes (TP8 latent down / fused_a / KDA o_proj; DEP16 o_proj).
GEMM_CASES = [(3584, 7168, 12), (2112, 7168, 6), (7168, 1536, 96), (7168, 12288, 256)]
# Split-K tactics checked per shape (the smallest and the largest split): each tactic is a
# separate kernel compile.
MAX_SPLIT_K_TACTICS = 2


def test_split_k_factors_divide_k_and_fill_one_wave():
    # 56 K blocks (K = 7168), 28 tiles, 212 CTAs: 7 fits (196), 8 overflows (224).
    assert _split_k_factors(56, 28, 212) == [4, 7, 8]
    # 12 K blocks (K = 1536), 56 tiles: 3 fits (168); 4 overflows (224).
    assert _split_k_factors(12, 56, 212) == [2, 3, 4]
    # Every factor divides the K blocks.
    for k_blocks in (12, 28, 56, 96):
        assert all(k_blocks % s == 0 for s in _split_k_factors(k_blocks, 7, 212))
    # A prime number of K blocks has no split.
    assert _split_k_factors(13, 4, 212) == []


def test_rubin_gemm_grid_ctas_and_pdl_bound(monkeypatch):
    # TP8 latent_down at M = 12 (swap AB: kernel M = 3584 weight rows), 128 x 64 tile, split 7.
    assert _rubin_gemm_grid_ctas(3584, 12, 7, (128, 64), 128, (1, 1), 212) == 196
    # g_proj + fused_a (3648 rows) in 2-CTA clusters, split 7: 29 tiles round up to 30.
    assert _rubin_gemm_grid_ctas(3648, 12, 7, (128, 64), 128, (2, 1), 106) == 210
    # 2-CTA instructions: a 256-row MMA tile is 128 rows per CTA.
    assert _rubin_gemm_grid_ctas(6144, 12, 1, (256, 64), 256, (2, 1), 106) == 48
    # The static scheduler caps the grid at one wave; the CLC dynamic one launches every tile.
    assert _rubin_gemm_grid_ctas(49152, 128, 1, (128, 128), 128, (1, 1), 212) == 212
    assert _rubin_gemm_grid_ctas(49152, 128, 1, (128, 128), 128, (1, 1), None) == 384

    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "1")
    monkeypatch.delenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL_MAX_CTAS", raising=False)
    assert _rubin_gemm_use_pdl(190)
    assert _rubin_gemm_use_pdl(192), "the qkvg split-K grid keeps PDL"
    assert not _rubin_gemm_use_pdl(196), "near-full-wave grids launch without PDL"
    assert not _rubin_gemm_use_pdl(196, 196), "a single wave is bounded"
    # DEP16 qkvg at M = 128: a 212-CTA persistent grid over 384 tiles keeps PDL.
    assert _rubin_gemm_use_pdl(212, 384)
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL_MAX_CTAS", "256")
    assert _rubin_gemm_use_pdl(210)
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL_MAX_CTAS", "-1")
    with pytest.raises(ValueError):
        _rubin_gemm_use_pdl(48)
    with pytest.raises(ValueError):
        _rubin_gemm_use_pdl(212, 384)
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL_MAX_CTAS", "256")
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "0")
    assert not _rubin_gemm_use_pdl(48)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("split_k", [2, 3, 7])
def test_split_k_reduce_is_the_ordered_fp32_sum(split_k, use_pdl):
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.split_k_reduce import split_k_reduce_bf16

    if use_pdl and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("PDL needs SM90+")
    torch.manual_seed(0)
    partial = torch.randn(split_k, 37, 3584, device="cuda") * 8
    out = torch.empty(37, 3584, dtype=torch.bfloat16, device="cuda")
    split_k_reduce_bf16(partial, out, use_pdl=use_pdl)
    expected = partial[0].clone()
    for s in range(1, split_k):
        expected = expected + partial[s]
    assert torch.equal(out, expected.to(torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_split_k_reduce_rejects_bad_tensors():
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.split_k_reduce import split_k_reduce_bf16

    partial = torch.zeros(2, 4, 8, device="cuda")
    with pytest.raises(ValueError):
        split_k_reduce_bf16(partial.half(), torch.empty(4, 8, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError):
        split_k_reduce_bf16(partial, torch.empty(4, 9, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError):
        split_k_reduce_bf16(partial, torch.empty(4, 8, dtype=torch.float32, device="cuda"))


def _rubin_available() -> bool:
    from tensorrt_llm._torch.models.modeling_kimi_linear import _rubin_mxfp8_is_available

    return _rubin_mxfp8_is_available()


def _tactics(runner, inputs):
    from tensorrt_llm._torch.autotuner import OptimizationProfile, StaticDim

    profile = OptimizationProfile([[StaticDim(x) for x in t.size()] for t in inputs])
    return list(runner.get_valid_tactics(inputs, profile))


def _k3_inputs(n, k, m):
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import _get_mxfp8_gemm_alpha
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    torch.manual_seed(2026)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    w_fp8, w_sf = _Fp8BlockScaleWeightReadLinear.quantize_weight(w)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x)
    return [a, w_fp8, a_sf, w_sf, _get_mxfp8_gemm_alpha(a.device)]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _rubin_available(),
    reason="needs SM107 with the CuTe DSL Rubin helpers",
)
def test_pdl_and_split_k_are_enabled_only_for_the_k3_runner():
    # The early griddepcontrol.launch_dependents was checked against the consumers of the K3
    # projections only; every other Rubin runner (NVFP4, plain MXFP8, DSpark) must not get it.
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLBlockScaledRubinLinear,
        CuteDSLKimiK3MXFP8RubinLinear,
        CuteDSLMXFP8RubinLinear,
    )

    assert CuteDSLKimiK3MXFP8RubinLinear.pdl_supported
    assert CuteDSLKimiK3MXFP8RubinLinear.split_k_supported
    for runner_cls in (CuteDSLBlockScaledRubinLinear, CuteDSLMXFP8RubinLinear):
        assert not runner_cls.pdl_supported, runner_cls.__name__
        assert not runner_cls.split_k_supported, runner_cls.__name__


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _rubin_available(),
    reason="needs SM107 with the CuTe DSL Rubin helpers",
)
def test_split_k_tactic_with_k_not_divisible_by_the_split_is_rejected(monkeypatch):
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLKimiK3MXFP8RubinLinear

    inputs = _k3_inputs(3584, 7168, 12)  # 56 K tiles: 3 does not divide them
    monkeypatch.setenv("TRTLLM_K3_MXFP8_SPLIT_K", "1")
    runner = CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    tactic = next(t for t in _tactics(runner, inputs) if t[0] == "splitk")
    with pytest.raises(ValueError, match="divisible"):
        runner(inputs, tactic=tactic[:-1] + (3,))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _rubin_available(),
    reason="needs SM107 with the CuTe DSL Rubin helpers",
)
@pytest.mark.parametrize("n,k,m", GEMM_CASES)
def test_k3_mxfp8_split_k_matches_unsplit_and_pdl_is_bitwise(monkeypatch, n, k, m):
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLKimiK3MXFP8RubinLinear

    inputs = _k3_inputs(n, k, m)

    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "0")
    monkeypatch.setenv("TRTLLM_K3_MXFP8_SPLIT_K", "0")
    runner = CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    tactics = _tactics(runner, inputs)
    assert not any(t[0] == "splitk" for t in tactics), "split-K must be opt-in"
    base_tactic = next(t for t in tactics if t[0] == "base")
    base = runner(inputs, tactic=base_tactic)

    monkeypatch.setenv("TRTLLM_K3_MXFP8_SPLIT_K", "1")
    split_tactics = sorted(
        (t for t in _tactics(runner, inputs) if t[0] == "splitk"), key=lambda t: t[-1]
    )
    assert split_tactics, f"no split-K tactic for N={n} K={k} M={m}"
    checked = [split_tactics[0], split_tactics[-1]][:MAX_SPLIT_K_TACTICS]
    scale = base.float().abs().max().item()
    for tactic in checked:
        out = runner(inputs, tactic=tactic)
        assert torch.equal(out, runner(inputs, tactic=tactic)), f"{tactic}: not deterministic"
        # One BF16 ulp (2**-7 relative) plus an absolute floor of 2**-17 of the largest output.
        torch.testing.assert_close(
            out.float(), base.float(), rtol=2.0**-7, atol=2.0**-17 * scale, msg=str(tactic)
        )

        monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "1")
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
        # Default grid bound, then PDL on every grid (the PDL kernel itself is checked too).
        for max_ctas in ("", "100000"):
            monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL_MAX_CTAS", max_ctas)
            assert torch.equal(runner(inputs, tactic=tactic), out), (
                f"{tactic}: PDL changed the output"
            )
        assert torch.equal(runner(inputs, tactic=base_tactic), base), "PDL changed the base output"
        monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "0")
