import pytest
import torch
from torch.nn import functional as F

from tensorrt_llm._torch.custom_ops.torch_custom_ops import ActivationType

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_DTYPE = torch.float8_e4m3fn

# ACT_FUNC = "relu2"
# ACT_FUNC = "swiglu"
# ACT_FUNC = "silu"
# TEST_DTYPE = torch.float16
# TEST_DTYPE = torch.bfloat16
# TEST_DTYPE = torch.float8_e4m3fn
# O_TYPE = torch.bfloat16 if TEST_DTYPE == torch.float8_e4m3fn else TEST_DTYPE


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
    x = torch.randn(*shape, dtype=dtype).cuda() * scale  # * 0.1
    return x.to(stype) if stype else x


def cast_to_representable(x):
    x_q, x_scale = dynamic_per_tensor_fp8_quant(x)
    x = x_q.to(x.dtype) * x_scale.to(x.dtype)
    return x


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


TEST_DTYPES = [
    # Todo: separate to two tests: float and fp8
    # (torch.float16, torch.float16, torch.float16),
    # (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16, torch.float8_e4m3fn),
    (torch.bfloat16, torch.bfloat16, torch.float8_e4m3fn),
    (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn),
    (torch.float8_e4m3fn, torch.float16, torch.float8_e4m3fn),
]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("itype, otype, wtype", TEST_DTYPES)
@pytest.mark.parametrize("activation_func", ["silu", "relu2"])
def test_trtllm_cutlass_fused_moe(
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
    input_shape = (batch_size, hidden_size)
    w31_shape = (num_experts, 2 * intermediate_size, hidden_size)
    w2_shape = (num_experts, hidden_size, intermediate_size)
    if activation_func in ["swiglu", "silu"]:
        X_GEN_SCALE = 1.0
    else:
        X_GEN_SCALE = 0.5

    x = cast_to_representable(gen_tensor(input_shape, otype, scale=X_GEN_SCALE))
    router_logits = gen_tensor((batch_size, num_experts), otype)

    # Create weight tensors
    w31_weight = gen_tensor(w31_shape, otype, wtype)
    w2_weight = gen_tensor(w2_shape, otype, wtype)
    w31_scales = torch.empty(num_experts, 2, dtype=otype).cuda()
    w2_scales = torch.empty(num_experts, 1, dtype=otype).cuda()

    w31_dequantized = gen_tensor(w31_shape, otype)
    w2_dequantized = gen_tensor(w2_shape, otype)
    for expert_id in range(num_experts):
        w31 = cast_to_representable(gen_tensor(w31_shape[1:], otype, scale=0.1))
        w2 = cast_to_representable(gen_tensor(w2_shape[1:], otype, scale=0.09))

        w31_quant, s31 = dynamic_per_tensor_fp8_quant(w31)
        w2_quant, s2 = dynamic_per_tensor_fp8_quant(w2)

        w31_weight.data[expert_id].copy_(w31_quant)
        w2_weight.data[expert_id].copy_(w2_quant)
        w31_scales.data[expert_id].copy_(s31)
        w2_scales.data[expert_id].copy_(s2)
        w31_dequantized.data[expert_id].copy_(torch.mul(w31_quant.to(dtype=otype), s31))
        w2_dequantized.data[expert_id].copy_(torch.mul(w2_quant.to(dtype=otype), s2))

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

    x_quant, hidden_states_scale = dynamic_per_tensor_fp8_quant(x)
    hidden_states_scale = hidden_states_scale[0].detach().clone().cuda()

    w3_input_scale = torch.tensor(1.0).cuda()
    w2_input_scale = torch.tensor(1.0).cuda()
    quant_scales = [
        torch.squeeze(w1_scales * hidden_states_scale).float(),  # gemm1 dequant scale
        w3_input_scale,  # gemm2 activation quant scale
        torch.squeeze(1.0 * w2_scales).float(),  # gemm2 dequant scale
        hidden_states_scale,  # gemm1 input dequant scale
    ]

    # (num_experts, 2 * intermediate_size, hidden_size) => (num_experts, intermediate_size, hidden_size)
    w3_weight, w1_weight = torch.chunk(w31_weight, 2, dim=1)
    w3_weight_dequantized, w1_weight_dequantized = torch.chunk(w31_dequantized, 2, dim=1)
    torch.cuda.synchronize()
    print("before fused_moe.cutlass_fused_moe")
    activation_type = (
        ActivationType.Swiglu if activation_func in ["swiglu", "silu"] else ActivationType.Relu2
    )

    if itype == torch.bfloat16 or itype == torch.float16:
        ad_test_output = torch.ops.auto_deploy.trtllm_moe_fused(
            x,
            selected_experts.to(torch.int),
            routing_weights,
            w3_w1_stacked_weight=w1_weight_dequantized.contiguous()
            if activation_func == "relu2"
            else w31_dequantized,
            w2_stacked_weight=w2_dequantized,
            mlp_style="mlp" if activation_func == "relu2" else "gated_mlp",
            act_fn=activation_func,
        )
        trtllm_test_output = torch.ops.trtllm.fused_moe(
            x,
            selected_experts.to(torch.int),
            routing_weights,
            fc1_expert_weights=w1_weight_dequantized.contiguous()
            if activation_func == "relu2"
            else w31_dequantized,
            fc1_expert_biases=None,
            fc2_expert_weights=w2_dequantized,
            fc2_expert_biases=None,
            output_dtype=otype,
            quant_scales=quant_scales,
            activation_type=activation_type,
        )[0].view(input_shape)

    elif itype == torch.float8_e4m3fn:
        # FP8
        ad_test_output = torch.ops.auto_deploy.trtllm_quant_fp8moe_fused(
            x,  # Note! unquantized input is expected
            selected_experts.to(torch.int),
            routing_weights,
            w1_weight=w1_weight.contiguous(),
            w2_weight=w2_weight.contiguous(),
            w3_weight=w3_weight.contiguous(),
            w1_input_scale=hidden_states_scale,
            w2_input_scale=w2_input_scale,
            w3_input_scale=w3_input_scale,
            w1_weight_scale=w1_scales,
            w2_weight_scale=w2_scales,
            w3_weight_scale=w3_scales,
            mlp_style="mlp" if activation_func == "relu2" else "gated_mlp",
            act_fn=activation_func,
        )

        trtllm_test_output = torch.ops.trtllm.fused_moe(
            x_quant,  # Note! quantized input is expected
            selected_experts.to(torch.int),
            routing_weights,
            fc1_expert_weights=w1_weight.contiguous() if activation_func == "relu2" else w31_weight,
            fc1_expert_biases=None,
            fc2_expert_weights=w2_weight,
            fc2_expert_biases=None,
            output_dtype=otype,
            quant_scales=quant_scales,
            activation_type=activation_type,
        )[0].view(input_shape)
    torch.cuda.synchronize()

    diff = (ref_output - ad_test_output).abs()
    print(f"max diff: {diff.max()}")
    assert trtllm_test_output is not None
    # torch.testing.assert_close(ad_test_output, trtllm_test_output, rtol=1e-6, atol=1e-6)

    if diff.max() > 1e-1:
        print("diff: " + "-" * 20)
        print(f"{diff[:10]}")
        print("test_output: " + "-" * 20)
        print(f"{ad_test_output[:10]}")
        print("ref_output: " + "-" * 20)
        print(f"{ref_output[:10]}")
    torch.testing.assert_close(ref_output, ad_test_output, rtol=1e-1, atol=1e-1)
