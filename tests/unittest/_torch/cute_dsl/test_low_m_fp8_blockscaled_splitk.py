# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-level tests of the low-M FP8 block-scaled split-K GEMM against DeepGEMM 1d1d.

Both GEMMs get the same FP8 operands: the weight quantized once (E4M3, 128x128 block scales
resmoothed to UE8M0, DeepGEMM's per-row packed int32 layout) and the activation from
``fp8_quantize_1x128_packed_ue8m0`` (E4M3, packed 1x128 UE8M0).

- ``split_k == 1`` keeps DeepGEMM's MMA sequence and K order: the output must be bitwise equal.
- ``split_k > 1`` sums the K slices in cluster-rank order: deterministic, with an error against
  an FP64 dequantized reference no larger than DeepGEMM's.
- Rows and columns of the output buffer outside ``[M, N]`` are never written.
"""

import pytest
import torch

from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() == 0 or not is_sm_100f(),
    reason="requires an SM100/SM103 GPU (tcgen05 block-scaled MMA, DeepGEMM 1d1d)",
)

# (N, K): an N tail inside one weight tile with an odd K split, and the Kimi K3 TP8 latent
# up / down projections.
_SHAPES = [(200, 640), (7168, 3584), (3584, 7168)]
_ROWS = [1, 3, 8, 13, 32]


def _quantized_weight(n: int, k: int, generator: torch.Generator):
    """FP8 weight, DeepGEMM-packed scales and the FP32 block scales, for any N."""
    from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8
    from tensorrt_llm.quantization.utils.fp8_utils import (
        resmooth_to_fp8_e8m0,
        transform_sf_into_required_layout,
    )

    n_pad = -(-n // 128) * 128
    weight = (torch.randn(n_pad, k, generator=generator, device="cuda") * 0.02).to(torch.bfloat16)
    weight_fp8, block_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
    weight_fp8, block_scale = resmooth_to_fp8_e8m0(
        weight_fp8.contiguous(), block_scale.contiguous().float()
    )
    weight_fp8 = weight_fp8[:n].contiguous()
    packed = transform_sf_into_required_layout(
        block_scale.float(), mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
    )
    return weight_fp8, packed, block_scale


def _unpack_rows(packed: torch.Tensor, rows: int, k_blocks: int) -> torch.Tensor:
    """int32 MN-major packed UE8M0 ``[rows, words]`` -> FP32 ``[rows, k_blocks]``."""
    out = torch.empty(rows, k_blocks, dtype=torch.float32, device=packed.device)
    for block in range(k_blocks):
        byte = (packed[:rows, block // 4] >> (8 * (block % 4))) & 0xFF
        out[:, block] = torch.exp2(byte.float() - 127.0)
    return out


@pytest.mark.parametrize("n,k", _SHAPES)
@pytest.mark.parametrize("rows", _ROWS)
def test_matches_deepgemm(n: int, k: int, rows: int) -> None:
    from tensorrt_llm import deep_gemm
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import low_m_fp8_blockscaled_splitk as lowm

    generator = torch.Generator(device="cuda").manual_seed(1234 + n + k + rows)
    weight_fp8, weight_scale, block_scale = _quantized_weight(n, k, generator)
    k_blocks = k // 128
    weight_ref = (
        weight_fp8.double().view(n, k_blocks, 128)
        * block_scale.double().repeat_interleave(128, 0)[:n, :, None]
    ).view(n, k)

    x = (torch.randn(rows, k, generator=generator, device="cuda") * 4).to(torch.bfloat16)
    act, act_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, False)
    ref = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    deep_gemm.fp8_gemm_nt(
        (act, act_scale), (weight_fp8, weight_scale), ref, disable_ue8m0_cast=True
    )
    act_ref = (
        act.double().view(rows, k_blocks, 128)
        * _unpack_rows(act_scale, rows, k_blocks).double()[:, :, None]
    ).view(rows, k)
    exact = act_ref @ weight_ref.t()
    ref_err = (ref.double() - exact).abs()

    tactics = lowm.candidate_tactics(rows, n, k)
    assert any(t.split_k == 1 for t in tactics)
    for tactic in tactics:
        buffer = torch.full((rows + 2, n + 64), -7.0, dtype=torch.bfloat16, device="cuda")
        out = buffer[1 : rows + 1, 32 : 32 + n]
        for pdl in (False, True):
            lowm.run_fp8_blockscaled_splitk(
                act, act_scale, weight_fp8, weight_scale, out, pdl, tactic
            )
            torch.cuda.synchronize()
            first = out.clone()
            lowm.run_fp8_blockscaled_splitk(
                act, act_scale, weight_fp8, weight_scale, out, pdl, tactic
            )
            torch.cuda.synchronize()
            assert torch.equal(first, out), f"{tactic} pdl={pdl}: not deterministic"
            if tactic.split_k == 1:
                assert torch.equal(out, ref), f"{tactic} pdl={pdl}: differs from DeepGEMM"
            else:
                err = (out.double() - exact).abs()
                assert err.max() <= 1.5 * ref_err.max() + 1e-6, f"{tactic}: max error"
                assert err.pow(2).mean() <= 1.44 * ref_err.pow(2).mean() + 1e-12, f"{tactic}: rms"
        assert (buffer[0] == -7).all() and (buffer[rows + 1] == -7).all()
        assert (buffer[:, :32] == -7).all() and (buffer[:, 32 + n :] == -7).all()


def test_rejects_unsupported_operands() -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import low_m_fp8_blockscaled_splitk as lowm

    n, k = 256, 512
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device="cuda")
    weight_scale = torch.zeros(1, n, dtype=torch.int32, device="cuda").t()
    act = torch.zeros(lowm.MAX_M + 1, k, dtype=torch.float8_e4m3fn, device="cuda")
    act_scale = torch.zeros(1, lowm.MAX_M + 1, dtype=torch.int32, device="cuda").t()
    out = torch.empty(lowm.MAX_M + 1, n, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="M <="):
        lowm.run_fp8_blockscaled_splitk(act, act_scale, weight, weight_scale, out, False)
    with pytest.raises(ValueError, match="multiple of 128"):
        lowm.validate_tactic(lowm.Fp8SplitKTactic(8, 1, 2), 8, n, k + 64)


def _refused_layouts(act: torch.Tensor, act_scale: torch.Tensor):
    """``(reason fragment, activation, scale)`` for two layouts of ``act`` the kernel refuses: an
    activation at an 8-byte offset (TMA needs 16-byte alignment) and a 40-row scale buffer for M
    rows (the kernel takes exactly ``[M, ceil(K / 512)]``)."""
    rows, k = act.shape
    shifted = torch.empty(rows * k + 8, dtype=act.dtype, device=act.device)[8:].view(rows, k)
    shifted.copy_(act)
    padded = torch.zeros(act_scale.shape[1], 40, dtype=torch.int32, device=act.device).t()
    padded[:rows].copy_(act_scale)
    return [("16-byte", shifted, act_scale), ("activation scale", act, padded)]


def test_supports_is_the_non_raising_check() -> None:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import low_m_fp8_blockscaled_splitk as lowm

    rows, n, k = 8, 256, 3584
    k_words = k // 512
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device="cuda")
    weight_scale = torch.zeros(k_words, n, dtype=torch.int32, device="cuda").t()
    act = torch.zeros(rows, k, dtype=torch.float8_e4m3fn, device="cuda")
    act_scale = torch.zeros(k_words, rows, dtype=torch.int32, device="cuda").t()
    out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    assert lowm.supports(act, act_scale, weight, weight_scale)
    assert lowm.supports(act, act_scale, weight, weight_scale, out)
    for match, bad_act, bad_scale in _refused_layouts(act, act_scale):
        assert not lowm.supports(bad_act, bad_scale, weight, weight_scale, out)
        assert match in lowm.unsupported_reason(bad_act, bad_scale, weight, weight_scale)
        with pytest.raises(ValueError, match=match):
            lowm.run_fp8_blockscaled_splitk(bad_act, bad_scale, weight, weight_scale, out, False)
    many = torch.zeros(lowm.MAX_M + 1, k, dtype=torch.float8_e4m3fn, device="cuda")
    many_scale = torch.zeros(k_words, lowm.MAX_M + 1, dtype=torch.int32, device="cuda").t()
    assert "rows" in lowm.unsupported_reason(many, many_scale, weight, weight_scale)


def test_kimi_k3_projection_keeps_deepgemm() -> None:
    """The Kimi K3 FP8 projection leaves the GEMM to DeepGEMM instead of raising: per call for
    operands the kernel refuses, and for good when the arm-time build fails."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import low_m_fp8_blockscaled_splitk as lowm
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    rows, n, k = 8, 512, 3584
    generator = torch.Generator(device="cuda").manual_seed(4321)
    weight_fp8, weight_scale, _ = _quantized_weight(n, k, generator)
    x = (torch.randn(rows, k, generator=generator, device="cuda") * 4).to(torch.bfloat16)
    act, act_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, False)
    reference = torch.ops.trtllm.fp8_prequantized_swap_ab_gemm(
        act,
        act_scale,
        weight_fp8,
        weight_scale,
        output_dtype=torch.bfloat16,
        disable_ue8m0_cast=True,
    )

    projection = _Fp8BlockScaleWeightReadLinear(weight_fp8, weight_scale, n)
    assert projection.arm_low_m_fp8() is None
    assert projection._low_m_fp8_gemm(act, act_scale) is not None
    # Refused operands leave the call to DeepGEMM instead of raising here. (DeepGEMM needs the
    # same 16-byte alignment and exact [M, ceil(K / 512)] scale, so it refuses these two too.)
    for _, bad_act, bad_scale in _refused_layouts(act, act_scale):
        assert projection._low_m_fp8_gemm(bad_act, bad_scale) is None

    def failing_build(*args, **kwargs):
        raise RuntimeError("simulated CuTe DSL compile failure")

    run = lowm.run_fp8_blockscaled_splitk
    lowm.run_fp8_blockscaled_splitk = failing_build
    try:
        unarmed = _Fp8BlockScaleWeightReadLinear(weight_fp8, weight_scale, n)
        blocker = unarmed.arm_low_m_fp8()
    finally:
        lowm.run_fp8_blockscaled_splitk = run
    assert blocker is not None and "simulated CuTe DSL compile failure" in blocker
    assert unarmed._low_m_fp8 is None
    assert torch.equal(unarmed.forward_prequantized_deepgemm(act, act_scale), reference)
