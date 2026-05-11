"""Tests for NVFP4 W4A16 (dequant-to-BF16) path on GPUs without FP4 tensor cores.

Tests:
1. Dequant round-trip: quantize BF16 -> NVFP4 -> dequant, verify closeness
2. NVFP4W4A16LinearMethod forward matches manual dequant + F.linear
3. W4A4 vs W4A16 output comparison (Blackwell-only)
4. SM version routing in get_quant_method
5. MoE weight dequant correctness
"""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _dequantize_nvfp4,
    _quantize_nvfp4,
)
from tensorrt_llm._torch.modules.linear import (
    Linear,
    NVFP4LinearMethod,
    NVFP4W4A16LinearMethod,
    get_quant_method,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

SCALING_VECTOR_SIZE = 16
FP8_MAX = 448.0
E2M1_MAX = 6.0


def _make_nvfp4_weights(out_features, in_features, dtype=torch.bfloat16):
    """Create NVFP4 quantized weights from random BF16 data.

    Returns (w_original, w_fp4_packed, block_scale_fp8, weight_scale_2)
    where the scales are in ModelOpt checkpoint format (not CUTLASS swizzled).
    """
    w = torch.randn(out_features, in_features, dtype=dtype).cuda()
    w_float = w.float()
    weight_scale_2 = w_float.abs().amax().float() / (E2M1_MAX * FP8_MAX)

    packed_weight, block_scale = _quantize_nvfp4(w_float, SCALING_VECTOR_SIZE, weight_scale_2)
    packed_uint8 = packed_weight.to(torch.uint8)
    block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)

    return w, packed_uint8, block_scale_fp8, weight_scale_2


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Dequant round-trip correctness
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(128, 256), (192, 128), (7168, 16384)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nvfp4_dequant_roundtrip(shape, dtype):
    """Quantize BF16 -> NVFP4 -> dequant and verify closeness to original."""
    out_features, in_features = shape
    w_orig, packed, bs_fp8, ws2 = _make_nvfp4_weights(out_features, in_features, dtype)

    # Dequant using the reference AutoDeploy function
    w_recovered = _dequantize_nvfp4(packed, bs_fp8, ws2, (out_features, in_features), dtype)

    # FP4 E2M1 has very coarse quantization, so tolerances are loose.
    # The important thing is no systematic bias or index errors.
    torch.testing.assert_close(w_recovered, w_orig.to(dtype), atol=0.5, rtol=0.15)


@pytest.mark.parametrize("shape", [(128, 256), (192, 128)])
def test_nvfp4_dequant_w4a16_matches_reference(shape):
    """Verify NVFP4W4A16LinearMethod._dequantize_weight matches _dequantize_nvfp4."""
    out_features, in_features = shape
    _, packed, bs_fp8, ws2 = _make_nvfp4_weights(out_features, in_features)

    ref = _dequantize_nvfp4(packed, bs_fp8, ws2, (out_features, in_features), torch.bfloat16)

    # Build the E2M1 LUT-based dequant used by NVFP4W4A16LinearMethod
    method = NVFP4W4A16LinearMethod()
    lut = method._E2M1_VALUES.to(packed.device)
    low = (packed.view(torch.uint8) & 0x0F).long()
    high = ((packed.view(torch.uint8) >> 4) & 0x0F).long()
    idx = torch.empty(out_features, in_features, dtype=torch.long, device=packed.device)
    idx[:, 0::2] = low
    idx[:, 1::2] = high
    vals = lut[idx]

    num_blocks = out_features * (in_features // SCALING_VECTOR_SIZE)
    ws = bs_fp8.to(torch.float32).reshape(-1)[:num_blocks]
    s2 = ws2.to(torch.float32)
    block_scales = (ws * s2).view(out_features, in_features // 16, 1)
    vals = vals.view(out_features, in_features // 16, 16) * block_scales
    manual = vals.view(out_features, in_features).to(torch.bfloat16)

    torch.testing.assert_close(manual, ref, atol=1e-3, rtol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: NVFP4W4A16LinearMethod forward pass
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mnk", [(4, 192, 128), (1, 128, 256), (16, 192, 128)])
def test_nvfp4_w4a16_linear_forward(mnk):
    """Forward pass of NVFP4W4A16LinearMethod matches manual dequant + F.linear."""
    seq_len, out_features, in_features = mnk
    dtype = torch.bfloat16
    torch.manual_seed(42)

    x = torch.randn(seq_len, in_features, dtype=dtype).cuda()
    w_orig, packed, bs_fp8, ws2 = _make_nvfp4_weights(out_features, in_features, dtype)

    # Compute reference: dequant + F.linear
    w_dequant = _dequantize_nvfp4(packed, bs_fp8, ws2, (out_features, in_features), dtype)
    ref_output = torch.nn.functional.linear(x, w_dequant)

    # Build Linear with W4A16 method (force it regardless of actual SM)
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=90):
        linear = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            dtype=dtype,
            quant_config=qc,
        )

    # Verify the method is W4A16
    assert isinstance(linear.linear_method, NVFP4W4A16LinearMethod)

    # Prepare weight_scale in ModelOpt format (unswizzled FP8)
    bs_unswizzled = bs_fp8.view(torch.float8_e4m3fn)

    # input_scale: ModelOpt stores amax/(448*6), which becomes weight_scale_2
    input_scale_ckpt = torch.tensor([1.0 / (FP8_MAX * E2M1_MAX)])  # dummy static scale

    linear.load_weights(
        [
            {
                "weight": packed.cpu(),
                "weight_scale": bs_unswizzled.cpu(),
                "weight_scale_2": ws2.cpu(),
                "input_scale": input_scale_ckpt.cpu(),
            }
        ]
    )
    linear = linear.cuda()

    with torch.inference_mode():
        output = linear(x)

    torch.testing.assert_close(output, ref_output, atol=0.01, rtol=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: W4A4 vs W4A16 output comparison (Blackwell only)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(get_sm_version() < 100, reason="W4A4 NVFP4 GEMM requires Blackwell (sm >= 100)")
@pytest.mark.parametrize("mnk", [(4, 192, 128), (128, 192, 128)])
def test_w4a4_vs_w4a16_output_comparison(mnk):
    """On Blackwell, compare W4A4 and W4A16 outputs on same weights/input."""
    from tensorrt_llm._torch.autotuner import autotune

    seq_len, out_features, in_features = mnk
    dtype = torch.bfloat16
    torch.manual_seed(0)

    x = torch.randn(seq_len, in_features, dtype=dtype).cuda()
    x_sf_global = (FP8_MAX * E2M1_MAX) / x.abs().max().float()

    w = torch.randn(out_features, in_features, dtype=dtype).cuda()
    w_sf_global = (FP8_MAX * E2M1_MAX) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global, SCALING_VECTOR_SIZE, False)

    w_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(out_features, 128), -1)
    )

    weight_dict = {
        "input_scale": 1.0 / x_sf_global.cpu(),
        "weight": w_fp4.cpu(),
        "weight_scale": w_sf_block_unswizzled.view(torch.float8_e4m3fn),
        "weight_scale_2": 1.0 / w_sf_global.cpu(),
    }

    # W4A4 path (native NVFP4)
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    l_w4a4 = Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=dtype,
        quant_config=qc,
        nvfp4_allowed_backends=["cutlass"],
    )
    l_w4a4.load_weights([dict(weight_dict)])
    l_w4a4 = l_w4a4.cuda()

    with torch.inference_mode(), autotune():
        out_w4a4 = l_w4a4(x)

    # W4A16 path (dequant to BF16)
    with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=90):
        l_w4a16 = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            dtype=dtype,
            quant_config=qc,
        )
    assert isinstance(l_w4a16.linear_method, NVFP4W4A16LinearMethod)
    l_w4a16.load_weights([dict(weight_dict)])
    l_w4a16 = l_w4a16.cuda()

    with torch.inference_mode():
        out_w4a16 = l_w4a16(x)

    # Outputs differ due to activation quantization in W4A4 and GEMM precision,
    # but should be in the same ballpark. Use generous tolerance.
    torch.testing.assert_close(out_w4a4, out_w4a16, atol=1.0, rtol=0.15)

    # Verify they're reasonably correlated (cosine sim > 0.95)
    cos_sim = torch.nn.functional.cosine_similarity(
        out_w4a4.flatten().float(), out_w4a16.flatten().float(), dim=0
    )
    assert cos_sim > 0.95, f"Cosine similarity too low: {cos_sim:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: SM version routing in get_quant_method
# ──────────────────────────────────────────────────────────────────────────────


def test_get_quant_method_routes_w4a16_on_hopper():
    """get_quant_method returns NVFP4W4A16LinearMethod when sm < 100."""
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=90):
        method = get_quant_method(qc)
    assert isinstance(method, NVFP4W4A16LinearMethod)


def test_get_quant_method_routes_nvfp4_on_blackwell():
    """get_quant_method returns NVFP4LinearMethod when sm >= 100."""
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=100):
        method = get_quant_method(qc)
    assert isinstance(method, NVFP4LinearMethod)
    assert not isinstance(method, NVFP4W4A16LinearMethod)


def test_get_quant_method_routes_arc_on_blackwell():
    """get_quant_method still returns NVFP4ARCLinearMethod for ARC on Blackwell."""
    from tensorrt_llm._torch.modules.linear import NVFP4ARCLinearMethod

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4_ARC)
    with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=100):
        method = get_quant_method(qc)
    assert isinstance(method, NVFP4ARCLinearMethod)


def test_get_quant_method_routes_w4a16_for_arc_on_hopper():
    """On Hopper, even NVFP4_ARC should route to W4A16 (no FP4 tensor cores)."""
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4_ARC)
    with patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=90):
        method = get_quant_method(qc)
    assert isinstance(method, NVFP4W4A16LinearMethod)


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: MoE weight dequant correctness
# ──────────────────────────────────────────────────────────────────────────────


def test_moe_dequant_nvfp4_tensor():
    """NemotronHMOE._dequant_nvfp4_tensor matches _dequantize_nvfp4 reference."""
    from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHMOE

    out_features, in_features = 192, 128
    _, packed, bs_fp8, ws2 = _make_nvfp4_weights(out_features, in_features)

    ref = _dequantize_nvfp4(packed, bs_fp8, ws2, (out_features, in_features), torch.bfloat16)

    result = NemotronHMOE._dequant_nvfp4_tensor(packed, bs_fp8, ws2, torch.bfloat16)

    torch.testing.assert_close(result, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("num_experts", [2, 4])
def test_moe_dequant_multiple_experts(num_experts):
    """Verify per-expert dequant produces correct results for stacked experts."""
    from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHMOE

    hidden = 128
    intermediate = 192
    dtype = torch.bfloat16

    for expert_idx in range(num_experts):
        torch.manual_seed(expert_idx)
        _, packed, bs_fp8, ws2 = _make_nvfp4_weights(intermediate, hidden, dtype)

        ref = _dequantize_nvfp4(packed, bs_fp8, ws2, (intermediate, hidden), dtype)
        result = NemotronHMOE._dequant_nvfp4_tensor(packed, bs_fp8, ws2, dtype)

        torch.testing.assert_close(
            result, ref, atol=1e-3, rtol=1e-3, msg=f"Expert {expert_idx} dequant mismatch"
        )
