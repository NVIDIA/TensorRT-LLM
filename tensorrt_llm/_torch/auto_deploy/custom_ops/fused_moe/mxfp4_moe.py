# Triton-kernels-based MXFP4 MoE ops (GPT-OSS style) with routing, swizzling, and fused activation

import math
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


# adapted from gpt-oss official implementation:
# https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py#L68
FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _dequantize_mxfp4(
    blocks: torch.Tensor,  # [..., G, B] in uint8
    scales: torch.Tensor,  # [..., G] in uint8
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 16384 * 512,
) -> torch.Tensor:
    """
    Dequantize MXFP4 format weights to bfloat16.
    Based on the reference implementation in gpt-oss official repo.

    Args:
        blocks: Packed FP4 values (2 per byte) in uint8 format
        scales: Exponent scales in uint8 format (offset by 127)
        dtype: Target dtype for dequantized output
        rows_per_chunk: Number of rows to process at once (for memory efficiency)

    Returns:
        Dequantized tensor in target dtype
    """
    scales_int = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks_flat = blocks.reshape(rows_total, B)
    scales_flat = scales_int.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks_flat[r0:r1]
        exp = scales_flat[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.long)  # Lower nibble
        idx_hi = (blk >> 4).to(torch.long)  # Upper nibble

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]  # Even positions: lower nibble
        sub[:, 1::2] = lut[idx_hi]  # Odd positions: upper nibble

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


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


@torch.library.custom_op("auto_deploy::torch_mxfp4_moe", mutates_args=())
def torch_mxfp4_moe(
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
    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)  # (num_tokens, hidden_size)
    num_tokens = x.shape[0]
    num_experts = gate_up_blocks.shape[0]
    intermediate_size = gate_up_blocks.shape[1] // 2

    # Dequantize gate_up weights
    # gate_up_blocks: [E, 2I, KB, BS] where KB*BS*2 = H
    # After dequant: [E, 2I, H]
    gate_up_w = []
    for e in range(num_experts):
        w_dequant = _dequantize_mxfp4(
            gate_up_blocks[e],  # [2I, KB, BS]
            gate_up_scales[e],  # [2I, KB]
            dtype=hidden_states.dtype,
        )  # [2I, H]
        gate_up_w.append(w_dequant)
    gate_up_w = torch.stack(gate_up_w, dim=0)  # [E, 2I, H]
    gate_up_w = gate_up_w.transpose(-2, -1)  # [E, H, 2I]

    # Dequantize down weights
    # down_blocks: [E, H, KB, BS] -> dequantize to get enough elements for [E, I, H]
    # Note: Due to the complex MXFP4 layout, we dequantize and extract the needed portion
    down_w = []
    for e in range(num_experts):
        # Dequantize: [H, KB, BS] -> [H, KB*BS*2] = [H, H] (since KB*BS*2 = H typically)
        w_dequant = _dequantize_mxfp4(
            down_blocks[e],
            down_scales[e],
            dtype=hidden_states.dtype,
        )  # [H, H]

        # Flatten and extract first I*H elements
        w_flat = w_dequant.reshape(-1)
        w_flat = w_flat[: intermediate_size * hidden_size]
        w_dequant = w_flat.reshape(intermediate_size, hidden_size)
        down_w.append(w_dequant)
    down_w = torch.stack(down_w, dim=0)  # [E, I, H]

    # Router: compute logits and select top-k experts per token
    router_logits = F.linear(x, router_weight, router_bias)  # [num_tokens, E]
    topk_weights, topk_indices = torch.topk(router_logits, top_k, dim=-1)  # [num_tokens, top_k]
    topk_weights = F.softmax(topk_weights, dim=-1)  # Normalize only top-k weights

    # Process experts in order (similar to triton's expert-grouped approach)
    # Use float32 accumulation to reduce precision errors, then cast back to input dtype
    output = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=x.device)
    for expert_idx in range(num_experts):
        expert_mask = topk_indices == expert_idx  # [num_tokens, top_k]
        if not expert_mask.any():
            continue
        token_indices, k_indices = torch.where(expert_mask)
        expert_tokens = x[token_indices]  # [n_tokens_using_expert, H]
        expert_weights = topk_weights[token_indices, k_indices]  # [n_tokens_using_expert]
        gate_up = F.linear(
            expert_tokens, gate_up_w[expert_idx].T, gate_up_bias[expert_idx]
        )  # [n, 2I]
        gate = gate_up[..., 0::2]  # [n, I]
        up = gate_up[..., 1::2]  # [n, I]
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        expert_out = F.linear((up + 1) * glu, down_w[expert_idx].T, down_bias[expert_idx])  # [n, H]
        expert_out = (expert_out * expert_weights.unsqueeze(-1)).to(torch.float32)  # [n, H]
        output.index_add_(0, token_indices, expert_out)

    output = output.to(hidden_states.dtype)
    output = output.reshape(*leading_shape, hidden_size)
    return output


@torch_mxfp4_moe.register_fake
def _torch_mxfp4_moe_fake(
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
