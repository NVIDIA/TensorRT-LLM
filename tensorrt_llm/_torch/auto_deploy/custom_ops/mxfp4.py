import torch
from triton_kernels.matmul_ogs import FlexCtx, FnSpecs, FusedActivation, PrecisionConfig, matmul_ogs
from triton_kernels.numerics import InFlexData
from triton_kernels.routing import routing
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout
from triton_kernels.tensor_details.layout import StridedLayout

from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import TritonEPRouter


# copied from transformers.integrations.mxfp4::swizzle_mxfp4 with minor modification
def _swizzle_mxfp4(w, w_scale):
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
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

    batch_size = hidden_states.shape[0]
    intermediate_size = gate_up_blocks.shape[1] // 2
    hidden_size = hidden_states.shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_size)
    router_logits = torch.nn.functional.linear(hidden_states, router_weight, router_bias)

    with torch.cuda.device(router_logits.device):
        routing_data, gather_idx, scatter_idx = routing(router_logits, top_k)

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


@torch.library.custom_op("auto_deploy::mxfp4_mlp_ep", mutates_args=())
def mxfp4_mlp_ep(
    hidden_states: torch.Tensor,  # [B,S,H] or [T,H]
    # router (replicated across EP)
    router_weight: torch.Tensor,  # [E_total, H]
    router_bias: torch.Tensor,  # [E_total]
    top_k: int,
    # expert params (already sharded along dim 0)
    gate_up_blocks: torch.Tensor,  # [E_local, 2I, H] or [E_local, H, 2I]
    gate_up_bias: torch.Tensor,  # [E_local, 2I]
    gate_up_scales: torch.Tensor,  # broadcastable; first dim = E_local
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,  # [E_local, I, H] or [E_local, H, I]
    down_bias: torch.Tensor,  # [E_local, H]
    down_scales: torch.Tensor,  # broadcastable; first dim = E_local
    # local expert-id range in GLOBAL indexing (set by the transform)
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
    """
    EP local-shard op:
      - runs global routing via hub.routing.routing
      - slices routing to this rank's expert range [local_lo, local_hi)
      - repacks X using gather_idx for only local rows
      - builds a minimal local RoutingData (ragged) with n_expts_act=1
      - runs gate_up -> swiglu -> down via matmul_ogs (no gather/scatter)
      - index_add_ into token positions -> partial y (sum across ranks later)
    """
    batch_size = hidden_states.shape[0]
    intermediate_size = gate_up_blocks.shape[1] // 2
    hidden_size = hidden_states.shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_size)
    router_logits = torch.nn.functional.linear(hidden_states, router_weight, router_bias)

    routing_data, gather_idx, scatter_idx = TritonEPRouter()(
        router_logits, top_k, ep=ep_size, node_idx=ep_rank
    )

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


@mxfp4_mlp_ep.register_fake
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
    local_lo: int,
    local_hi: int,
):
    return torch.empty_like(hidden_states)
