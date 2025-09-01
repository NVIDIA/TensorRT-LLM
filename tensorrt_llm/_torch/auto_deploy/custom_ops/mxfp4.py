import torch

_triton_kernels_hub = None


def _hub():
    global _triton_kernels_hub
    if _triton_kernels_hub is None:
        from kernels import get_kernel

        _triton_kernels_hub = get_kernel("kernels-community/triton_kernels")
    return _triton_kernels_hub


# copied from transformers.integrations.mxfp4::swizzle_mxfp4
def _swizzle_mxfp4(w, w_scale):
    triton_kernels_hub = _hub()

    FP4, convert_layout, wrap_torch_tensor = (
        triton_kernels_hub.tensor.FP4,
        triton_kernels_hub.tensor.convert_layout,
        triton_kernels_hub.tensor.wrap_torch_tensor,
    )
    layout = triton_kernels_hub.tensor_details.layout
    StridedLayout = triton_kernels_hub.tensor_details.layout.StridedLayout

    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
    # TODO : add that when we are actually sure that it works on B200
    # if torch.cuda.get_device_capability()[0] == 10:
    #     constraints = {
    #         "is_persistent": True,
    #         "epilogue_subtile": 1,
    #     }
    #     opt_flags.update_opt_flags_constraints(constraints)
    # # transpose the tensor so that the quantization axis is on dim1

    # TODO: there is still an issue with the scales on hopper
    # scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=8)
    # w_scale = convert_layout(wrap_torch_tensor(w_scale), scale_layout, **scale_layout_opts)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale


@torch.library.custom_op("auto_deploy::mxfp4_mlp", mutates_args=())
def mxfp4_mlp(
    hidden_states: torch.Tensor,  # [B, S, H] or [T, H]
    # router
    router_weight: torch.Tensor,  # [E, H]
    router_bias: torch.Tensor,  # [E]
    top_k: int,
    # gate_up path
    gate_up_blocks: torch.Tensor,  # usually [E, 2I, H] or [E, H, 2I]
    gate_up_bias: torch.Tensor,  # [E, 2I]
    gate_up_scales: torch.Tensor,  # any broadcastable scale shape (raw)
    alpha: float,
    limit: float,
    # down path
    down_blocks: torch.Tensor,  # usually [E, I, H] or [E, H, I]
    down_bias: torch.Tensor,  # [E, H]
    down_scales: torch.Tensor,  # any broadcastable scale shape (raw)
) -> torch.Tensor:
    """
    Wrapper that forwards to your Python reference implementation.
    Return:
      routed_out:   same leading shape as hidden_states, last dim = H
      router_logits: [T, E] (T = number of tokens = prod(hidden_states.shape[:-1]))
    """

    hub = _hub()
    routing = hub.routing.routing

    batch_size = hidden_states.shape[0]
    intermediate_size = gate_up_blocks.shape[1] // 2
    hidden_size = hidden_states.shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_size)
    router_logits = torch.nn.functional.linear(hidden_states, router_weight, router_bias)

    with torch.cuda.device(router_logits.device):
        routing_data, gather_idx, scatter_idx = routing(router_logits, top_k)

    FnSpecs, FusedActivation, matmul_ogs = (
        hub.matmul_ogs.FnSpecs,
        hub.matmul_ogs.FusedActivation,
        hub.matmul_ogs.matmul_ogs,
    )
    FlexCtx, InFlexData, PrecisionConfig = (
        hub.matmul_ogs.FlexCtx,
        hub.matmul_ogs.InFlexData,
        hub.matmul_ogs.PrecisionConfig,
    )
    swiglu_fn = hub.swiglu.swiglu_fn

    local_experts = gate_up_blocks.size(0)
    gate_up_blocks = gate_up_blocks.view(local_experts, intermediate_size * 2, -1)
    triton_gate_up_w, gate_up_w_scale_raw = _swizzle_mxfp4(
        gate_up_blocks.transpose(-2, -1), gate_up_scales.transpose(-2, -1)
    )
    triton_gate_up_w.shape = torch.Size([local_experts, hidden_size, intermediate_size * 2])
    down_blocks = down_blocks.view(local_experts, -1, intermediate_size // 2)
    triton_down_w, down_w_scale_raw = _swizzle_mxfp4(
        down_blocks.transpose(-2, -1), down_scales.transpose(-2, -1)
    )
    triton_down_w.shape = torch.Size([local_experts, intermediate_size, hidden_size])

    gate_pc = PrecisionConfig(
        weight_scale=gate_up_w_scale_raw, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )

    down_pc = PrecisionConfig(
        weight_scale=down_w_scale_raw, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )

    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (float(alpha), float(limit)), 2
    )

    intermediate_cache1 = matmul_ogs(
        hidden_states,
        triton_gate_up_w,
        gate_up_bias.to(torch.float32),
        routing_data,
        gather_indx=gather_idx,
        precision_config=gate_pc,
        gammas=None,
        fused_activation=act,
    )

    routed_out = matmul_ogs(
        intermediate_cache1,
        triton_down_w,
        down_bias.to(torch.float32),
        routing_data,
        scatter_indx=scatter_idx,
        precision_config=down_pc,
        gammas=routing_data.gate_scal,
    )

    routed_out = routed_out.reshape(batch_size, -1, hidden_size)
    return routed_out


@mxfp4_mlp.register_fake
def _mxfp4_mlp_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
):
    return torch.empty_like(hidden_states)
