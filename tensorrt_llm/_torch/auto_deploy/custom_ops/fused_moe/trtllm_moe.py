import torch

from tensorrt_llm._torch.utils import ActivationType


@torch.library.custom_op("auto_deploy::trtllm_moe_fused", mutates_args=())
def trtllm_moe_fused(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    routing_weights = routing_weights.to(torch.float32)
    selected_experts = selected_experts.to(torch.int32)
    quant_scales = []

    # Determine activation type
    mlp_style = mlp_style.lower()
    act_fn = act_fn.lower()

    activation_type = ActivationType.Swiglu
    if mlp_style == "gated_mlp":
        # Gated MLP uses Silu: silu(x @ w1.T) * (x @ w3.T)
        if act_fn == "silu":
            activation_type = ActivationType.Swiglu
        else:
            raise ValueError(f"Unsupported activation '{act_fn}' for gated_mlp. Use 'silu'.")
    elif mlp_style == "mlp":
        # For non-gated MLP with ReLU^2
        if act_fn == "relu2":
            activation_type = ActivationType.Relu2
        else:
            raise ValueError(f"Unsupported activation '{act_fn}' for mlp. Use 'relu2'.")
    else:
        raise ValueError(f"Unknown mlp_style '{mlp_style}'. Use 'gated_mlp' or 'mlp'.")

    return torch.ops.trtllm.fused_moe(
        x,
        selected_experts,
        routing_weights,
        fc1_expert_weights=w3_w1_stacked_weight,
        fc1_expert_biases=None,
        fc2_expert_weights=w2_stacked_weight,
        fc2_expert_biases=None,
        output_dtype=x.dtype,
        quant_scales=quant_scales,
        activation_type=activation_type,
    )[0].view(x_shape)


@trtllm_moe_fused.register_fake
def trtllm_moe_fused_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    return torch.empty_like(x)


# Todo: refactor this repeating code block
def _quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP8 with clamping (matches torch_quant_fp8_linear)."""
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    return (x / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)


def _validate_mlp_style_and_act_fn(mlp_style: str, act_fn: str) -> None:
    supported_combinations = {
        "gated_mlp": ["silu"],
        "mlp": ["relu2"],
    }
    supported_act_fns = [
        act_fn for act_fn_list in supported_combinations.values() for act_fn in act_fn_list
    ]
    assert mlp_style in supported_combinations.keys(), (
        f"Unknown mlp_style '{mlp_style}'. Use {supported_combinations.keys()}."
    )
    assert act_fn in supported_act_fns, f"Unknown act_fn '{act_fn}'. Use {supported_act_fns}."
    assert act_fn in supported_combinations[mlp_style], (
        f"Unsupported combination: mlp_style='{mlp_style}', act_fn='{act_fn}'. "
        f"Supported combinations: {supported_combinations}"
    )


@torch.library.custom_op("auto_deploy::trtllm_quant_fp8_moe_fused", mutates_args=())
def trtllm_quant_fp8_moe_fused(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: torch.Tensor,  # [E, I, H] stacked FP8 weights
    w2_weight: torch.Tensor,  # [E, H, I] stacked FP8 weights
    w3_weight: torch.Tensor,  # [E, I, H] for gated_mlp, unused for mlp
    w1_input_scale: torch.Tensor,  # [E] stacked input scales
    w2_input_scale: torch.Tensor,  # [E] stacked input scales
    w3_input_scale: torch.Tensor,  # [E] or unused
    w1_weight_scale: torch.Tensor,  # [E] stacked weight scales
    w2_weight_scale: torch.Tensor,  # [E] stacked weight scales
    w3_weight_scale: torch.Tensor,  # [E] or unused
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    """
    TensorRT-LLM Cutlass FP8 W8A8 MoE for gated and non-gated MLP.
    Parameters:
        x: BF16/FP16 input tensor of shape (B, H) or (B, S, H)
        selected_experts: Expert indices (B*S, TOP_K)
        routing_weights: Routing weights (B*S, TOP_K)
        w1_weight: FP8 w1 weights [E, I, H]
        w2_weight: FP8 w2 weights [E, H, I]
        w3_weight: FP8 w3 weights [E, I, H] (for gated_mlp)
        w1_input_scale: Input scales for w1 [E]
        w2_input_scale: Input scales for w2 [E]
        w3_input_scale: Input scales for w3 [E]
        w1_weight_scale: Weight scales for w1 [E]
        w2_weight_scale: Weight scales for w2 [E]
        w3_weight_scale: Weight scales for w3 [E]
        mlp_style: "gated_mlp" or "mlp"
        act_fn: "silu" for gated_mlp, "relu2" for mlp

    Non-Gated MLP:
        activation_fn(expert_inputs @ w1_expert.t())@ w2_expert.t()

    Gated MLP:
        activation_fn(expert_inputs @ w1_expert.t()) * (expert_inputs @ w3_expert.t()) @ w2_expert.t()
    """

    _validate_mlp_style_and_act_fn(mlp_style, act_fn)

    # Store original shape and flatten to 2D
    x_shape = x.shape
    x2d = x.view(-1, x_shape[-1])
    # Quantize input
    x_q_fp8 = _quantize_fp8(x2d, w1_input_scale[0])

    # Scales are stored in float32
    w1_weight_scale = w1_weight_scale.to(torch.float32)
    w2_weight_scale = w2_weight_scale.to(torch.float32)
    w1_input_scale = w1_input_scale.to(torch.float32)[0]
    w2_input_scale = w2_input_scale.to(torch.float32)[0]

    # Prepare quant_scales for TensorRT-LLM FP8 format:
    # [gemm1_dequant_scale, gemm2_act_quant_scale, gemm2_dequant_scale, gemm1_input_dequant_scale]
    # For gated MLP:
    # - gemm1_dequant_scale: w1_weight_scale * w1_input_scale (combined for w1 and w3)
    # - gemm2_act_quant_scale: 1 / w2_input_scale
    # - gemm2_dequant_scale: w2_weight_scale * w2_input_scale
    # - gemm1_input_dequant_scale: w1_input_scale

    # Compute combined scales
    gemm1_dequant = (w1_weight_scale * w1_input_scale).contiguous().squeeze()
    gemm2_act_quant = (1.0 / w2_input_scale).contiguous().to(torch.float32)
    gemm2_dequant = (w2_weight_scale * w2_input_scale).contiguous().squeeze()
    gemm1_input_dequant = w1_input_scale.contiguous()

    assert gemm1_dequant.ndim == 1, "gemm1_dequant must be 1D"
    assert gemm2_dequant.ndim == 1, "gemm2_dequant must be 1D"
    quant_scales = [gemm1_dequant, gemm2_act_quant, gemm2_dequant, gemm1_input_dequant]

    # Ensure contiguous tensors
    selected_experts = selected_experts.int().contiguous()
    routing_weights = routing_weights.contiguous()

    # Todo: refactor this repeating code block

    # Determine activation type
    mlp_style = mlp_style.lower()
    act_fn = act_fn.lower()

    activation_type = ActivationType.Swiglu
    if mlp_style == "gated_mlp":
        # Gated MLP uses Silu: silu(x @ w1.T) * (x @ w3.T)
        # For gated MLP, concatenate w1 and w3 as [w3, w1]
        w3_w1_stacked = torch.cat([w3_weight, w1_weight], dim=1).contiguous()  # [E, 2*I, H]
        fc1_expert_weights = w3_w1_stacked
        if act_fn == "silu":
            activation_type = ActivationType.Swiglu
        else:
            raise ValueError(f"Unsupported activation '{act_fn}' for gated_mlp. Use 'silu'.")
    elif mlp_style == "mlp":
        # For non-gated MLP with ReLU^2
        fc1_expert_weights = w1_weight.contiguous()
        if act_fn == "relu2":
            activation_type = ActivationType.Relu2
        else:
            raise ValueError(f"Unsupported activation '{act_fn}' for mlp. Use 'relu2'.")
    else:
        raise ValueError(f"Unknown mlp_style '{mlp_style}'. Use 'gated_mlp' or 'mlp'.")

    # Note! Outputting Float8_e4m3fn directly is not currently supported
    output = torch.ops.trtllm.fused_moe(
        x_q_fp8,
        selected_experts,
        routing_weights,
        fc1_expert_weights=fc1_expert_weights,
        fc1_expert_biases=None,
        fc2_expert_weights=w2_weight.contiguous(),
        fc2_expert_biases=None,
        output_dtype=x.dtype,
        quant_scales=quant_scales,
        activation_type=activation_type,
    )

    # Restore original shape
    return output[0].view(x_shape)


@trtllm_quant_fp8_moe_fused.register_fake
def trtllm_quant_fp8_moe_fused_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    w1_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
    w3_input_scale: torch.Tensor,
    w1_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w3_weight_scale: torch.Tensor,
    mlp_style: str,
    act_fn: str,
) -> torch.Tensor:
    return torch.empty_like(x)
