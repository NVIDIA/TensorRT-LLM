# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA output gate fused with the o_proj FP8 quantization on SM100/103 (TRTLLM_K3_MLA_GATE_FP8_DG).

Bitwise: the fused kernel's codes and scales against ``fp8_quantize_1x128_packed_ue8m0`` of
``attn_out * gate.sigmoid()`` (PyTorch's BF16 sigmoid and multiply), and the o_proj projection of
both operands. Needs an SM100/SM103 GPU."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103 GPU",
)

K = 1536  # 12 heads x 128 per TP8 rank


def _inputs(m, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    a = torch.randn(m, K, generator=g, device="cuda") * 3.0
    gate = torch.randn(m, K, generator=g, device="cuda") * 4.0
    # exact zeros, tiny, huge and inf/NaN-free extremes
    a[:, :37] = 0.0
    a[:, 128:140] *= 1e-6
    gate[:, 300:310] = 40.0
    gate[:, 400:410] = -40.0
    a[0, 700:760] *= 3e3
    return a.bfloat16(), gate.bfloat16()


def _scale_bytes(scale, m):
    """[m, K/128] uint8 scales of an int32 [m, K/512] packed MN-major scale."""
    words = scale.as_strided((scale.shape[1], m), (scale.stride(1), 1))  # [K/512, m]
    b = words.contiguous().view(torch.uint8).view(scale.shape[1], m, 4)
    return b.permute(1, 0, 2).reshape(m, -1)


@pytest.mark.parametrize("m", [1, 2, 6, 8, 12, 16, 24, 32, 40, 64, 96, 128, 1024, 8192])
def test_gate_mul_fp8_deepgemm_bitwise(m):
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_deepgemm

    for seed in range(3):
        a, gate = _inputs(m, seed)
        ref_codes, ref_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
            a * gate.sigmoid(), False
        )
        codes, scale = gate_mul_fp8_deepgemm(a, gate)
        assert scale.dtype == torch.int32 and scale.shape == ref_scale.shape
        assert scale.stride() == ref_scale.stride()
        assert (codes.view(torch.uint8) == ref_codes.view(torch.uint8)).all()
        assert (_scale_bytes(scale, m) == _scale_bytes(ref_scale, m)).all()


@pytest.mark.parametrize("m", [6, 8, 16, 24, 32])
def test_o_proj_deepgemm_prequantized_matches(m):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_deepgemm

    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("the K3 weight-read projection is the DeepGEMM one on SM103")
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    lin = torch.nn.Linear(K, 7168, bias=False, device="cuda", dtype=torch.bfloat16)
    lin.weight.data.copy_((torch.randn(7168, K, generator=g, device="cuda") * 0.02).bfloat16())
    mod = _Fp8BlockScaleWeightReadLinear.from_linear(lin)
    a, gate = _inputs(m, 11)
    ref = mod(a * gate.sigmoid())
    got = mod.forward_prequantized_deepgemm(*gate_mul_fp8_deepgemm(a, gate))
    assert (ref.view(torch.int16) == got.view(torch.int16)).all()


def _baseline_o_proj(mod, x, tactic):
    """The served o_proj of the BF16 operand (``fp8_swap_ab_gemm``) with its quantizer tactic
    forced: 0 = the CUDA op, 1 = the Triton quantizer (the autotuner picks per M)."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import fp8SwapABGemmRunner

    runner = fp8SwapABGemmRunner(torch.bfloat16, True, quant_tactic=tactic)
    return runner.forward([x, mod.weight, mod.weight_scale])


@pytest.mark.parametrize("m", [33, 97, 129, 257, 1000, 2049, 4097, 8191, 8192])
def test_o_proj_context_m_matches_cuda_tactic(m):
    """Large and odd M (context steps): bitwise equal to the CUDA-quantizer baseline (tactic 0).
    The Triton-quantizer baseline (tactic 1) is compared and reported, not asserted."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear
    from tensorrt_llm._torch.modules.kimi_k3_mla.output_gate_fp8 import gate_mul_fp8_deepgemm

    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("the K3 weight-read projection is the DeepGEMM one on SM103")
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    lin = torch.nn.Linear(K, 7168, bias=False, device="cuda", dtype=torch.bfloat16)
    lin.weight.data.copy_((torch.randn(7168, K, generator=g, device="cuda") * 0.02).bfloat16())
    mod = _Fp8BlockScaleWeightReadLinear.from_linear(lin)
    a, gate = _inputs(m, 11)
    x = a * gate.sigmoid()
    got = mod.forward_prequantized_deepgemm(*gate_mul_fp8_deepgemm(a, gate))
    ref0 = _baseline_o_proj(mod, x, 0)
    ref1 = _baseline_o_proj(mod, x, 1)
    n1 = int((ref1.view(torch.int16) != got.view(torch.int16)).sum())
    print(f"M={m}: vs CUDA tactic: equal={bool((ref0.view(torch.int16) == got.view(torch.int16)).all())}; "
          f"vs Triton tactic: {n1} of {got.numel()} outputs differ", flush=True)
    assert (ref0.view(torch.int16) == got.view(torch.int16)).all()
