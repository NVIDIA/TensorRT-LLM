# Triton-kernels-based MXFP4 MoE ops (GPT-OSS style) with routing, swizzling, and fused activation

from typing import Callable, Tuple

import torch
import torch.nn.functional as F

IS_TRITON_KERNELS_AVAILABLE = True
TRITON_KERNELS_UNAVAILABLE_REASON = ""

try:
    from triton_kernels.matmul_ogs import (
        FlexCtx,
        FnSpecs,
        FusedActivation,
        PrecisionConfig,
        matmul_ogs,
    )
    from triton_kernels.numerics import InFlexData
    from triton_kernels.routing import RoutingData, routing
    from triton_kernels.swiglu import swiglu_fn
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout
    from triton_kernels.tensor_details.layout import StridedLayout

    from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import TritonEPRouter

except Exception as _e:
    IS_TRITON_KERNELS_AVAILABLE = False
    TRITON_KERNELS_UNAVAILABLE_REASON = f"{type(_e).__name__}: {_e}"

    FlexCtx = FnSpecs = FusedActivation = PrecisionConfig = matmul_ogs = None
    InFlexData = RoutingData = routing = swiglu_fn = None
    FP4 = convert_layout = wrap_torch_tensor = None
    layout = StridedLayout = None
    TritonEPRouter = None


# copied from transformers.integrations.mxfp4::swizzle_mxfp4 with minor modification
def _swizzle_mxfp4(w, w_scale):
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale


RouteFn = Callable[[torch.Tensor], Tuple[RoutingData, torch.Tensor, torch.Tensor]]


def _prepare_weights_scales(
    hidden_size: int,
    gate_up_blocks: torch.Tensor,  # [E_local, 2I, H//32, 16] in unit8
    gate_up_scales: torch.Tensor,  # [E_local, 2I, H//32] in unit8
    down_blocks: torch.Tensor,  # [E_local, H, I//32, 16] in uint8
    down_scales: torch.Tensor,  # [E_local, H, I//32] in uint8
):
    local_experts = gate_up_blocks.size(0)
    intermediate_size = gate_up_blocks.shape[1] // 2

    # canon shapes for swizzling (use last two dims as [K, N] style)
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

    return (
        triton_gate_up_w,
        gate_up_w_scale_raw,
        triton_down_w,
        down_w_scale_raw,
    )


def _run_mxfp4_mlp_core(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    route_fn: RouteFn,  # injects routing variant
) -> torch.Tensor:
    """
    Shared core for both triton_mxfp4_moe and triton_mxfp4_moe_ep.
    - route_fn encapsulates the only difference: how we produce (routing_data, gather_idx, scatter_idx).
    """
    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)

    router_logits = F.linear(x, router_weight, router_bias)
    # route (global vs EP-aware)
    with torch.cuda.device(router_logits.device):
        routing_data, gather_idx, scatter_idx = route_fn(router_logits)

    (
        triton_gate_up_w,
        gate_up_w_scale_raw,
        triton_down_w,
        down_w_scale_raw,
    ) = _prepare_weights_scales(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )

    gate_pc = PrecisionConfig(
        weight_scale=gate_up_w_scale_raw, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )
    down_pc = PrecisionConfig(
        weight_scale=down_w_scale_raw, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )

    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (float(alpha), float(limit)), 2
    )

    # gate_up (with SWiGLU fused)
    inter = matmul_ogs(
        x,
        triton_gate_up_w,
        gate_up_bias.to(torch.float32),
        routing_data,
        gather_indx=gather_idx,
        precision_config=gate_pc,
        gammas=None,
        fused_activation=act,
    )

    # down
    y = matmul_ogs(
        inter,
        triton_down_w,
        down_bias.to(torch.float32),
        routing_data,
        scatter_indx=scatter_idx,
        precision_config=down_pc,
        gammas=routing_data.gate_scal,
    )

    y = y.reshape(*leading_shape, hidden_size)
    return y


@torch.library.custom_op("auto_deploy::triton_mxfp4_moe", mutates_args=())
def triton_mxfp4_moe(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    # router
    router_weight: torch.Tensor,  # [E, H]
    router_bias: torch.Tensor,  # [E]
    top_k: int,
    # gate_up path
    gate_up_blocks: torch.Tensor,  # [E, 2I, H//32, 16] in unit8
    gate_up_bias: torch.Tensor,  # [E, 2I]
    gate_up_scales: torch.Tensor,  # [E, 2I, H//32] in unit8
    alpha: float,
    limit: float,
    # down path
    down_blocks: torch.Tensor,  # [E, H, I//32, 16] in uint8
    down_bias: torch.Tensor,  # [E, H]
    down_scales: torch.Tensor,  # [E, H, I//32] in uint8
) -> torch.Tensor:
    def _global_route_fn(logits: torch.Tensor):
        return routing(logits, top_k)

    return _run_mxfp4_mlp_core(
        hidden_states,
        router_weight,
        router_bias,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        route_fn=_global_route_fn,
    )


@triton_mxfp4_moe.register_fake
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


@torch.library.custom_op("auto_deploy::triton_mxfp4_moe_ep", mutates_args=())
def triton_mxfp4_moe_ep(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    # router (replicated across EP)
    router_weight: torch.Tensor,  # [E_total, H]
    router_bias: torch.Tensor,  # [E_total]
    top_k: int,
    # expert params (already sharded along dim 0)
    gate_up_blocks: torch.Tensor,  # [E_local, 2I, H//32, 16] in unit8
    gate_up_bias: torch.Tensor,  # [E_local, 2I]
    gate_up_scales: torch.Tensor,  # [E_local, 2I, H//32] in unit8
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,  # [E_local, H, I//32, 16] in uint8
    down_bias: torch.Tensor,  # [E_local, H]
    down_scales: torch.Tensor,  # [E_local, H, I//32] in uint8
    # EP topology
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
    triton_ep_router = TritonEPRouter()

    def _ep_route_fn(logits: torch.Tensor):
        return triton_ep_router(logits, top_k, ep=ep_size, node_idx=ep_rank)

    return _run_mxfp4_mlp_core(
        hidden_states,
        router_weight,
        router_bias,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        route_fn=_ep_route_fn,
    )


@triton_mxfp4_moe_ep.register_fake
def _mxfp4_mlp_ep_fake(
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
    ep_size: int,
    ep_rank: int,
):
    return torch.empty_like(hidden_states)
