import torch


@torch.library.custom_op("auto_deploy::trtllm_moe_fused", mutates_args=())
def trtllm_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    routing_weights = routing_weights.to(torch.float32)
    selected_experts = selected_experts.to(torch.int32)
    quant_scales = []

    return torch.ops.trtllm.fused_moe(
        x,
        selected_experts,
        routing_weights,
        w3_w1_stacked_weight,
        None,  # w3_w1_stacked_bias
        w2_stacked_weight,
        None,  # w2_stacked_bias
        x.dtype,
        quant_scales,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        enable_alltoall=False,
    )[0].view(x_shape)


@trtllm_fused_moe.register_fake
def trtllm_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)
