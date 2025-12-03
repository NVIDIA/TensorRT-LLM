"""
This file contains test functions copied from:
https://github.com/flashinfer-ai/flashinfer/blob/main/tests/moe/test_trtllm_cutlass_fused_moe.py
"""

from typing import Callable

import pytest
import torch
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available  # noqa: F401
from torch.nn import functional as F
from utils.util import skip_pre_hopper

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.utils import ActivationType

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FLOAT4_E2M1_MAX = 6.0
FP8_DTYPE = torch.float8_e4m3fn
NVFP4_BLOCK_SIZE = 16


def dynamic_per_tensor_fp8_quant(x: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
    fp8_traits_max = FLOAT8_E4M3_MAX
    fp8_traits_min = -FLOAT8_E4M3_MAX
    fp8_max = torch.tensor(fp8_traits_max).float()
    one = torch.tensor(1.0).float()

    x_max = x.abs().max().float()
    scale = x_max / fp8_max
    iscale = one / scale
    out = (x.float() * iscale).clamp(fp8_traits_min, fp8_traits_max).to(FP8_DTYPE)
    return out, scale.view((1,))


def gen_tensor(shape, dtype, stype=None, scale=1.0):
    x = torch.randn(*shape, dtype=dtype).cuda() * scale
    return x.to(stype) if stype else x


def cast_to_representable(x):
    """
    Convert a tensor of floats to exactly representable in FP8 format to reduce quantization error in the test.

    returns:
        x_dq: A tensor of floats that is exactly representable in FP8 format.
              x_dq = dq(q(x, x_scale), x_scale)
              where x_scale is computed using min-max range clipping.
    """
    x_q, x_scale = dynamic_per_tensor_fp8_quant(x)
    x_dq = x_q.to(x.dtype) * x_scale.to(x.dtype)
    return x_dq


def compute_routing(router_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts from router logits.

    Args:
        router_logits (torch.Tensor): Router logits of shape [batch_size, num_experts]
        top_k (int): Number of experts to route to per token

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - routing_weights: Expert weights of shape [batch_size, top_k]
            - selected_experts: Expert indices of shape [batch_size, top_k]
    """
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def compute_with_experts(
    num_experts,
    x,
    w31_weight,
    w2_weight,
    selected_experts,
    routing_weights,
    alpha=None,
    beta=None,
    limit=None,
    activation_func="silu",
):
    def relu2(x: torch.Tensor) -> torch.Tensor:
        return torch.square(F.relu(x))

    results = torch.zeros_like(x)
    for expert_id in range(num_experts):
        mask = selected_experts == expert_id
        if not mask.sum():
            continue
        batch_idx, nth_expert = torch.where(mask)
        w31_expert = w31_weight[expert_id]  # [2 * intermediate_size, hidden_size]
        w2_expert = w2_weight[expert_id]  # [hidden_size, intermediate_size]

        # Split w13 into w1 and w3
        w3_expert, w1_expert = torch.chunk(w31_expert, 2, dim=0)

        expert_inputs = x[batch_idx]
        if alpha is not None and limit is not None and beta is not None:
            # SwiGLUBias
            x1 = expert_inputs @ w1_expert.t()
            x1 = x1.clamp_(min=None, max=limit)
            x1_scaled = x1 * torch.sigmoid(alpha * x1)
            x2 = expert_inputs @ w3_expert.t()
            x2 = x2.clamp_(min=-limit, max=limit) + beta

            inter = x1_scaled * x2
        else:
            if activation_func == "swiglu" or activation_func == "silu":
                inter = F.silu(expert_inputs @ w1_expert.t()) * (expert_inputs @ w3_expert.t())
            else:
                inter = relu2(expert_inputs @ w1_expert.t())

        output = inter @ w2_expert.t()
        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output
    return results.view_as(x)


def _get_test_data(
    otype, wtype, batch_size, hidden_size, num_experts, intermediate_size, X_GEN_SCALE, W_GEN_SCALE
):
    input_shape = (batch_size, hidden_size)
    w31_shape = (num_experts, 2 * intermediate_size, hidden_size)
    w2_shape = (num_experts, hidden_size, intermediate_size)

    x = cast_to_representable(gen_tensor(input_shape, otype, scale=X_GEN_SCALE))
    router_logits = gen_tensor((batch_size, num_experts), otype)
    w31_weight = gen_tensor(w31_shape, otype, wtype, W_GEN_SCALE)
    w2_weight = gen_tensor(w2_shape, otype, wtype, W_GEN_SCALE)
    w31_empty_scales = torch.empty(num_experts, 2, dtype=otype).cuda()
    w2_empty_scales = torch.empty(num_experts, 1, dtype=otype).cuda()
    return x, router_logits, w31_weight, w2_weight, w31_empty_scales, w2_empty_scales


def _activation_type_from_str(activation_func: str) -> ActivationType:
    return ActivationType.Swiglu if activation_func in ["swiglu", "silu"] else ActivationType.Relu2


def _print_diff_if(
    condition: Callable[[torch.Tensor], bool],
    diff: torch.Tensor,
    ad_test_output: torch.Tensor,
    ref_output: torch.Tensor,
):
    if condition(diff):
        print("diff: " + "-" * 20)
        print(f"{diff[:10]}")
        print("test_output: " + "-" * 20)
        print(f"{ad_test_output[:10]}")
        print("ref_output: " + "-" * 20)
        print(f"{ref_output[:10]}")


# Test configurations
BATCH_SIZES = [
    1,
]
HIDDEN_SIZES = [
    128,
]
NUM_EXPERTS = [2]
TOP_K_VALUES = [2]
INTERMEDIATE_SIZES = [
    128,
]
EP_NUM_EXPERTS = [8]
EP_TOP_K = [2]


F16_TEST_DTYPES = [
    (torch.float16, torch.float16, torch.float16),
    (torch.bfloat16, torch.bfloat16, torch.bfloat16),
]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("itype, otype, wtype", F16_TEST_DTYPES)
@pytest.mark.parametrize("activation_func", ["silu", "relu2"])
@skip_pre_hopper
def test_trtllm_fused_moe(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    itype,
    otype,
    wtype,
    activation_func,
):
    # Skip invalid configurations
    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})")

    torch.manual_seed(42)
    if activation_func in ["swiglu", "silu"]:
        X_GEN_SCALE = 1.0
    else:
        X_GEN_SCALE = 0.5
    W_GEN_SCALE = 0.1

    x, router_logits, w31_weight, w2_weight, w31_scales, w2_scales = _get_test_data(
        otype,
        wtype,
        batch_size,
        hidden_size,
        num_experts,
        intermediate_size,
        X_GEN_SCALE,
        W_GEN_SCALE,
    )

    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    ref_output = compute_with_experts(
        num_experts,
        x,
        w31_weight,
        w2_weight,
        selected_experts,
        routing_weights,
        activation_func=activation_func,
    )

    assert itype == torch.bfloat16 or itype == torch.float16, (
        "F16 test only supports bfloat16 or float16"
    )
    assert otype == torch.bfloat16 or otype == torch.float16, (
        "F16 test only supports bfloat16 or float16"
    )
    assert wtype == torch.bfloat16 or wtype == torch.float16, (
        "F16 test only supports bfloat16 or float16"
    )

    activation_type = _activation_type_from_str(activation_func)

    def get_fc1_expert_weights(
        activation_func: str, w31_weight: torch.Tensor, w1_weight: torch.Tensor
    ) -> torch.Tensor:
        if activation_func == "relu2":
            return w1_weight.contiguous()
        else:
            return w31_weight

    # (num_experts, 2 * intermediate_size, hidden_size) => (num_experts, intermediate_size, hidden_size)
    _, w1_weight = torch.chunk(w31_weight, 2, dim=1)
    mlp_style = "mlp" if activation_func == "relu2" else "gated_mlp"

    torch.cuda.synchronize()
    ad_test_output = torch.ops.auto_deploy.trtllm_moe_fused(
        x,
        selected_experts.to(torch.int),
        routing_weights,
        w3_w1_stacked_weight=get_fc1_expert_weights(activation_func, w31_weight, w1_weight),
        w2_stacked_weight=w2_weight,
        mlp_style=mlp_style,
        act_fn=activation_func,
    )
    trtllm_test_output = torch.ops.trtllm.fused_moe(
        x,
        selected_experts.to(torch.int),
        routing_weights,
        fc1_expert_weights=get_fc1_expert_weights(activation_func, w31_weight, w1_weight),
        fc1_expert_biases=None,
        fc2_expert_weights=w2_weight,
        fc2_expert_biases=None,
        output_dtype=otype,
        quant_scales=[],
        activation_type=activation_type,
    )[0].view(x.shape)

    torch.cuda.synchronize()
    if mlp_style == "mlp":
        with torch.inference_mode():
            output_triton_moe = torch.ops.auto_deploy.triton_moe_fused(
                x,
                selected_experts,
                routing_weights,
                w1_weight.contiguous(),
                w2_weight.contiguous(),
            )[0].view(x.shape)
            torch.testing.assert_close(output_triton_moe, ad_test_output, rtol=1e-2, atol=1e-2)

    diff = (ref_output - ad_test_output).abs()
    print(f"max diff: {diff.max()}")
    torch.testing.assert_close(ad_test_output, trtllm_test_output, rtol=1e-6, atol=1e-6)

    _print_diff_if(lambda diff: diff.max() > 1e-1, diff, ad_test_output, ref_output)
    torch.testing.assert_close(ref_output, ad_test_output, rtol=1e-2, atol=1e-2)


FP8_TEST_DTYPES = [
    (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn),
    (torch.float8_e4m3fn, torch.float16, torch.float8_e4m3fn),
]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("itype, otype, wtype", FP8_TEST_DTYPES)
@pytest.mark.parametrize("activation_func", ["silu", "relu2"])
@pytest.mark.skipif(
    not fp8_compatible() or not trtllm_ops_available(),
    reason="Requires fp8 and trtllm support",
)
def test_trtllm_fused_moe_fp8(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    itype,
    otype,
    wtype,
    activation_func,
):
    # Skip invalid configurations
    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})")

    assert itype == torch.float8_e4m3fn and wtype == torch.float8_e4m3fn, (
        "FP8 test only supports float8_e4m3fn"
    )
    assert otype == torch.bfloat16 or otype == torch.float16, (
        "FP8 test only supports bfloat16 or float16 output type"
    )

    torch.manual_seed(42)
    if activation_func in ["swiglu", "silu"]:
        X_GEN_SCALE = 1.0
    else:
        X_GEN_SCALE = 0.5

    W_GEN_SCALE = 0.1

    def dequantize_weights(w31_weight, w2_weight, w31_scales, w2_scales, W_GEN_SCALE):
        w31_shape = (num_experts, 2 * intermediate_size, hidden_size)
        w2_shape = (num_experts, hidden_size, intermediate_size)

        w31_dequantized = gen_tensor(w31_weight.shape, otype)
        w2_dequantized = gen_tensor(w2_weight.shape, otype)
        for expert_id in range(num_experts):
            w31 = cast_to_representable(gen_tensor(w31_shape[1:], otype, scale=W_GEN_SCALE))
            w2 = cast_to_representable(gen_tensor(w2_shape[1:], otype, scale=W_GEN_SCALE))
            w31_quant, s31 = dynamic_per_tensor_fp8_quant(w31)
            w2_quant, s2 = dynamic_per_tensor_fp8_quant(w2)
            w31_weight.data[expert_id].copy_(w31_quant)
            w2_weight.data[expert_id].copy_(w2_quant)
            w31_scales.data[expert_id].copy_(s31)
            w2_scales.data[expert_id].copy_(s2)
            w31_dequantized.data[expert_id].copy_(torch.mul(w31_quant.to(dtype=otype), s31))
            w2_dequantized.data[expert_id].copy_(torch.mul(w2_quant.to(dtype=otype), s2))
        return w31_dequantized, w2_dequantized

    x, router_logits, w31_weight, w2_weight, w31_scales, w2_scales = _get_test_data(
        otype,
        wtype,
        batch_size,
        hidden_size,
        num_experts,
        intermediate_size,
        X_GEN_SCALE,
        W_GEN_SCALE,
    )

    w31_dequantized, w2_dequantized = dequantize_weights(
        w31_weight, w2_weight, w31_scales, w2_scales, W_GEN_SCALE
    )

    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    ref_output = compute_with_experts(
        num_experts,
        x,
        w31_dequantized,
        w2_dequantized,
        selected_experts,
        routing_weights,
        activation_func=activation_func,
    )

    # For fp8, the hidden_state expects quantized.
    w3_scales, w1_scales = torch.chunk(w31_scales, 2, dim=-1)

    _, hidden_states_scale = dynamic_per_tensor_fp8_quant(x)
    hidden_states_scale = hidden_states_scale[0].detach().clone().cuda()

    w3_input_scale = torch.tensor([1.0]).cuda()
    w2_input_scale = torch.tensor([1.0]).cuda()

    # (num_experts, 2 * intermediate_size, hidden_size) => (num_experts, intermediate_size, hidden_size)
    w3_weight, w1_weight = torch.chunk(w31_weight, 2, dim=1)
    mlp_style = "mlp" if activation_func == "relu2" else "gated_mlp"

    # compute quant_scales
    gemm1_dequant = (w1_scales * hidden_states_scale).contiguous().squeeze().to(torch.float32)
    gemm2_act_quant = (1.0 / w2_input_scale[0]).contiguous().to(torch.float32)
    gemm2_dequant = (w2_scales * w2_input_scale[0]).contiguous().squeeze().to(torch.float32)

    print("before fused_moe.cutlass_fused_moe")
    torch.cuda.synchronize()
    ad_test_output = torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused(
        x,  # Note! unquantized input is expected
        selected_experts.to(torch.int),
        routing_weights,
        w1_weight=w1_weight.contiguous(),
        w2_weight=w2_weight.contiguous(),
        w3_weight=w3_weight.contiguous(),
        w1_input_scale=hidden_states_scale.unsqueeze(0),
        w2_input_scale=w2_input_scale,
        w3_input_scale=w3_input_scale,
        w1_weight_scale=w1_scales,
        w2_weight_scale=w2_scales,
        w3_weight_scale=w3_scales,
        gemm1_dequant=gemm1_dequant,
        gemm2_act_quant=gemm2_act_quant,
        gemm2_dequant=gemm2_dequant,
        mlp_style=mlp_style,
        act_fn=activation_func,
    )

    torch.cuda.synchronize()

    if mlp_style == "mlp":
        with torch.inference_mode():
            output_triton_fp8_moe = torch.ops.auto_deploy.triton_quant_fp8_moe(
                x,
                selected_experts,
                routing_weights,
                w1_weight,
                w2_weight,
                w3_weight,
                hidden_states_scale.unsqueeze(0),
                w2_input_scale,
                w3_input_scale,
                w1_scales,
                w2_scales,
                w3_scales,
                mlp_style=mlp_style,
                act_fn=activation_func,
            )
            torch.testing.assert_close(output_triton_fp8_moe, ref_output, rtol=1e-1, atol=1e-1)

    diff = (ref_output - ad_test_output).abs()
    print(f"max diff: {diff.max()}")

    _print_diff_if(lambda diff: diff.max() > 1e-1, diff, ad_test_output, ref_output)
    torch.testing.assert_close(ref_output, ad_test_output, rtol=1e-1, atol=1e-1)


# Originally from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/moe/test_trtllm_cutlass_fused_moe.py
def torch_moe_nvfp4(a, w1, w2, topk, topk_weight, topk_ids, activation_type):
    """Reference implementation of NVFP4 MoE.

    The intermediate activations are quantized and dequantized to emulate the precision loss of a real
    quantized operation.
    """
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    if activation_type == ActivationType.Swiglu:

        def act(weight, mask):
            m = weight.shape[0]
            assert m % 2 == 0
            w1_expert, w3_expert = weight[m // 2 :, :], weight[: m // 2, :]
            return F.silu(a[mask] @ w1_expert.t()) * (a[mask] @ w3_expert.t())
    elif activation_type == ActivationType.Relu2:

        def act(weight, mask):
            return F.relu(a[mask] @ weight.t()) ** 2
    else:
        raise ValueError(f"Unsupported activation type {activation_type}")

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter = act(w1[i], mask)
            inter_gs = torch.tensor(1.0).cuda()
            inter_q, inter_blockscale = torch.ops.trtllm.fp4_quantize(
                inter, inter_gs, NVFP4_BLOCK_SIZE
            )
            inter = dequantize_nvfp4_to_dtype(
                inter_q,
                inter_blockscale,
                inter_gs,
                dtype=inter.dtype,
                device=inter.device,
                block_size=NVFP4_BLOCK_SIZE,
            ).cuda()
            out[mask] = inter @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


# Originally from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/moe/test_trtllm_cutlass_fused_moe.py
def dequantize_nvfp4_to_dtype(tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16):
    """Dequantize the fp4 tensor back to high precision."""

    def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
        m_tiles = (m + 128 - 1) // 128
        f = block_size * 4
        k_tiles = (k + f - 1) // f
        tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
        tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
        out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
        return out[0:m, 0:k]

    # Originally from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/moe/test_trtllm_cutlass_fused_moe.py
    def break_fp4_bytes(a, dtype):
        assert a.dtype == torch.uint8
        m, n = a.shape

        # Vectorized nibble processing
        a_flat = a.flatten()
        high = (a_flat & 0xF0) >> 4  # Upper nibbles
        low = a_flat & 0x0F  # Lower nibbles

        # Combine nibbles for batch processing
        combined = torch.stack((low, high), dim=1).flatten()

        # Vectorized sign and magnitude extraction
        signs = (combined & 0x08).to(torch.bool)  # Sign bits
        abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

        # Device-aware lookup and sign application
        kE2M1ToFloat = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
        kE2M1 = kE2M1ToFloat.to(device=a.device)
        values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

        # Reshape to final form
        return values.reshape(m, n * 2).to(dtype=dtype)

    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


NVFP4_TEST_DTYPES = [
    (torch.float16, torch.float8_e4m3fn),
    (torch.bfloat16, torch.float8_e4m3fn),
]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("otype, wtype", NVFP4_TEST_DTYPES)
@pytest.mark.parametrize("activation_func", ["silu", "relu2"])
@pytest.mark.skipif(
    not fp4_compatible() or not trtllm_ops_available(),
    reason="Requires fp4 and trtllm support",
)
def test_trtllm_fused_moe_nvfp4(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    otype,
    wtype,
    activation_func,
):
    # In the code below:
    #   sf := block scale factors for NVFP4
    #   blockscale := block scale factor for NVFP4
    #   gs := global scale for NVFP4

    # Skip invalid configurations
    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})")
    torch.manual_seed(42)

    def _get_test_data(
        otype,
        batch_size,
        hidden_size,
        num_experts,
        intermediate_size,
    ):
        x = gen_tensor((batch_size, hidden_size), otype)
        w1_shape = (num_experts, intermediate_size, hidden_size)
        w3_shape = w1_shape
        w1 = gen_tensor(w1_shape, otype, scale=0.1)
        w2 = gen_tensor((num_experts, hidden_size, intermediate_size), otype, scale=0.1)
        w3 = gen_tensor(w3_shape, otype, scale=0.1)
        router_logits = torch.randn(batch_size, num_experts, dtype=otype).cuda()
        return x, w1, w2, w3, router_logits

    def _quantize_weights(w1, w2, w3):
        def round_up(x, y):
            return (x + y - 1) // y * y

        w1_n = w1.shape[1]
        w3_n = w3.shape[1]
        sf_w1_n = round_up(w1_n, 128)
        sf_w3_n = round_up(w3_n, 128)
        sf_w1_k = round_up(hidden_size // NVFP4_BLOCK_SIZE, 4)
        w1_blockscale = torch.empty(
            (num_experts, sf_w1_n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
        )
        sf_w2_k = round_up(hidden_size, 128)
        sf_w2_n = round_up(intermediate_size // NVFP4_BLOCK_SIZE, 4)
        w2_blockscale = torch.empty(
            (num_experts, sf_w2_k, sf_w2_n), device="cuda", dtype=torch.float8_e4m3fn
        )
        w3_blockscale = torch.empty(
            (num_experts, sf_w3_n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
        )
        w1_q = torch.empty((num_experts, w1_n, hidden_size // 2), device="cuda", dtype=torch.uint8)
        w2_q = torch.empty(
            (num_experts, hidden_size, intermediate_size // 2), device="cuda", dtype=torch.uint8
        )
        w3_q = torch.empty((num_experts, w3_n, hidden_size // 2), device="cuda", dtype=torch.uint8)

        w1_gs = torch.empty((num_experts,), device="cuda", dtype=torch.float32)
        w2_gs = torch.empty((num_experts,), device="cuda", dtype=torch.float32)
        w3_gs = torch.empty((num_experts,), device="cuda", dtype=torch.float32)

        for expert in range(num_experts):
            w1_amax = torch.abs(w1[expert]).max().to(torch.float32)
            w2_amax = torch.abs(w2[expert]).max().to(torch.float32)
            w3_amax = torch.abs(w3[expert]).max().to(torch.float32)
            w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
            w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
            w3_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w3_amax

            nvfp4_vals, fp8_block_scales = torch.ops.trtllm.fp4_quantize(
                w1[expert], w1_gs[expert], NVFP4_BLOCK_SIZE, isSfSwizzledLayout=True
            )
            w1_q[expert] = nvfp4_vals
            w1_blockscale[expert] = fp8_block_scales.reshape(w1_blockscale[expert].shape)

            nvfp4_vals, fp8_block_scales = torch.ops.trtllm.fp4_quantize(
                w2[expert], w2_gs[expert], NVFP4_BLOCK_SIZE, isSfSwizzledLayout=True
            )
            w2_q[expert] = nvfp4_vals
            w2_blockscale[expert] = fp8_block_scales.reshape(w2_blockscale[expert].shape)

            nvfp4_vals, fp8_block_scales = torch.ops.trtllm.fp4_quantize(
                w3[expert], w3_gs[expert], NVFP4_BLOCK_SIZE, isSfSwizzledLayout=True
            )
            w3_q[expert] = nvfp4_vals
            w3_blockscale[expert] = fp8_block_scales.reshape(w3_blockscale[expert].shape)

        return w1_q, w2_q, w3_q, w1_blockscale, w2_blockscale, w3_blockscale, w1_gs, w2_gs, w3_gs

    x, w1, w2, w3, router_logits = _get_test_data(
        otype, batch_size, hidden_size, num_experts, intermediate_size
    )

    (
        w1_q_fp4,
        w2_q_fp4,
        w3_q_fp4,
        w1_blockscale,
        w2_blockscale,
        w3_blockscale,
        w1_gs,
        w2_gs,
        w3_gs,
    ) = _quantize_weights(w1, w2, w3)

    fc1_activation_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    fc2_activation_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    fc1_weight_gs = torch.max(w3_gs, w1_gs)
    fc1_alpha = 1.0 / (fc1_activation_gs * fc1_weight_gs)
    fc2_alpha = 1.0 / (fc2_activation_gs * w2_gs)

    mlp_style = "mlp" if activation_func == "relu2" else "gated_mlp"
    if mlp_style == "gated_mlp":
        # For gated MLP, concatenate w1 and w3 as [w3, w1]
        fc1_expert_weights_fp4 = torch.cat([w3_q_fp4, w1_q_fp4], dim=1).contiguous()
        fc1_weight_blockscale_fp8 = torch.cat([w3_blockscale, w1_blockscale], dim=1)
        fc1_weight_gs = torch.max(w3_gs, w1_gs)
        if activation_func != "silu":
            raise ValueError(
                f"Unsupported activation '{activation_func}' for gated_mlp. Use 'silu'."
            )
    elif mlp_style == "mlp":
        # For non-gated MLP with ReLU^2
        fc1_expert_weights_fp4 = w1_q_fp4
        fc1_weight_blockscale_fp8 = w1_blockscale.view(torch.long)
        fc1_weight_gs = w1_gs
        if activation_func != "relu2":
            raise ValueError(f"Unsupported activation '{activation_func}' for mlp. Use 'relu2'.")
    else:
        raise ValueError(f"Unknown mlp_style '{mlp_style}'. Use 'gated_mlp' or 'mlp'.")

    fc2_expert_weights_fp4 = w2_q_fp4.view(torch.long)
    fc2_weight_blockscale_fp8 = w2_blockscale.view(torch.long)
    fc1_expert_weights_fp4 = fc1_expert_weights_fp4.view(torch.long)

    trtllm_output = torch.ops.auto_deploy.trtllm_quant_nvfp4_moe_fused(
        x,
        selected_experts.to(torch.int),
        routing_weights,
        fc1_expert_weights_fp4,
        fc2_expert_weights_fp4,
        fc1_weight_blockscale_fp8,
        fc2_weight_blockscale_fp8,
        fc1_activation_gs,
        fc2_activation_gs,
        fc1_alpha,
        fc2_alpha,
        mlp_style=mlp_style,
        act_fn=activation_func,
    )

    def compute_ref_output(w1_gs, w3_gs):
        # Quantize then dequantize the input to emulate the precision loss.
        a_fp4, a_scale_interleaved = torch.ops.trtllm.fp4_quantize(
            x, fc1_activation_gs, NVFP4_BLOCK_SIZE
        )
        x_dq = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            fc1_activation_gs,
            dtype=otype,
            device=x.device,
            block_size=NVFP4_BLOCK_SIZE,
        )

        concat_w3_w1 = mlp_style == "gated_mlp"
        if concat_w3_w1:
            w1_gs = w3_gs = torch.max(w1_gs, w3_gs)

        w1_dq = torch.empty(w1.shape, device="cuda", dtype=otype)
        w3_dq = torch.empty(w3.shape, device="cuda", dtype=otype)
        w2_dq = torch.empty(w2.shape, device="cuda", dtype=otype)

        # Dequantize the weights to emulate the precision loss.
        for idx in range(0, num_experts):
            w1_dq[idx] = dequantize_nvfp4_to_dtype(
                w1_q_fp4[idx],
                w1_blockscale[idx],
                w1_gs[idx],
                dtype=w1.dtype,
                device=w1.device,
                block_size=NVFP4_BLOCK_SIZE,
            )
            w2_dq[idx] = dequantize_nvfp4_to_dtype(
                w2_q_fp4[idx],
                w2_blockscale[idx],
                w2_gs[idx],
                dtype=w2.dtype,
                device=w2.device,
                block_size=NVFP4_BLOCK_SIZE,
            )
            w3_dq[idx] = dequantize_nvfp4_to_dtype(
                w3_q_fp4[idx],
                w3_blockscale[idx],
                w3_gs[idx],
                dtype=w3.dtype,
                device=w3.device,
                block_size=NVFP4_BLOCK_SIZE,
            )

        ref_output = torch_moe_nvfp4(
            x_dq,
            torch.cat([w3_dq, w1_dq], dim=1) if concat_w3_w1 else w1_dq,
            w2_dq,
            top_k,
            routing_weights,
            selected_experts,
            _activation_type_from_str(activation_func),
        )
        return ref_output

    ref_output = compute_ref_output(w1_gs, w3_gs)
    print(f"max diff: {(ref_output - trtllm_output).abs().max()}")
    print(f"diff = {ref_output - trtllm_output}")
    print(f"ref_output = {ref_output}")
    print(f"flash_output = {trtllm_output}")
    torch.testing.assert_close(ref_output, trtllm_output, rtol=2e-1, atol=2e-1)
