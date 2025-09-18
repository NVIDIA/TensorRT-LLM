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
    local_lo: int,
    local_hi: int,
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
    hub = _hub()
    routing = hub.routing.routing

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

    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)  # [T, H]
    T = x.shape[0]

    # router logits and global routing
    router_logits = torch.nn.functional.linear(x, router_weight, router_bias)
    with torch.cuda.device(router_logits.device):
        routing_data, gather_idx, scatter_idx = routing(router_logits, top_k)

    # quick exits for empty shards
    if local_hi <= local_lo:
        return torch.zeros_like(hidden_states)
    E_local = int(local_hi - local_lo)

    # global meta
    K = int(routing_data.n_expts_act)  # top_k from routing
    # sum of counts for local experts gives how many rows we compute here
    hist_global = routing_data.expt_hist  # [E_total]
    # offsets over global expert order (prefix-sum over hist)
    offs_raw_global = routing_data.expt_data.token_offs_raw  # length >= E_total+1

    # slice local histogram and offsets; renormalize to start at 0
    hist_local = hist_global[local_lo:local_hi].to(torch.int32)  # [E_local]
    start = int(offs_raw_global[local_lo].item())
    end = int(offs_raw_global[local_hi].item())
    if end - start == 0:
        return torch.zeros_like(hidden_states)

    offs_local = (
        offs_raw_global[local_lo : local_hi + 1] - offs_raw_global[local_lo]
    )  # [E_local+1], 0-based

    # gather_idx contains the token mapping for ALL assignments; take local segment
    # convert from assignment indexing to token indexing (divide by K)
    tok_idx_local = (gather_idx.src_indx[start:end] // K).to(torch.int64)  # [L_local] in [0..T)
    x_local = x.index_select(0, tok_idx_local)  # [L_local, H]

    # per-assignment gates; take the same local segment
    gamma_local = routing_data.gate_scal[start:end].to(torch.float32)  # [L_local]

    # wizzle local shard weights to Triton layout
    intermediate_size = gate_up_blocks.shape[1] // 2
    gate_up_blocks = gate_up_blocks.contiguous().view(E_local, intermediate_size * 2, -1)
    gate_up_scales = gate_up_scales.contiguous().view(E_local, intermediate_size * 2, -1)
    triton_gate_up_w, gate_up_w_scale_raw = _swizzle_mxfp4(
        gate_up_blocks.transpose(-2, -1),
        gate_up_scales.transpose(-2, -1),
    )
    triton_gate_up_w.shape = torch.Size([E_local, hidden_size, intermediate_size * 2])
    down_blocks = down_blocks.contiguous().view(E_local, -1, intermediate_size // 2)
    down_scales = down_scales.contiguous().view(E_local, -1, intermediate_size // 2)
    triton_down_w, down_w_scale_raw = _swizzle_mxfp4(
        down_blocks.transpose(-2, -1),
        down_scales.transpose(-2, -1),
    )
    triton_down_w.shape = torch.Size([E_local, intermediate_size, hidden_size])

    gate_pc = PrecisionConfig(
        weight_scale=gate_up_w_scale_raw, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )
    down_pc = PrecisionConfig(
        weight_scale=down_w_scale_raw, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )
    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (float(alpha), float(limit)), 2
    )

    # --- build minimal local RoutingData (ragged), n_expts_act=1 ---
    ExptDataCls = type(routing_data.expt_data)
    RoutingDataCls = type(routing_data)

    # Slice padded offsets if available; rebase to 0 like offs_local
    token_offs_pad_local = {}
    # TODO: check if sharding block_pid_map is required
    block_pid_map_local = {}
    if getattr(routing_data.expt_data, "token_offs_pad", None):
        for bm, v in routing_data.expt_data.token_offs_pad.items():
            # local, rebased cumulative block counts (length = E_local+1)
            tpad = (v[local_lo : local_hi + 1] - v[local_lo]).to(torch.int32)
            token_offs_pad_local[bm] = tpad

            # build local PID map with expert ids in [0..E_local-1]
            counts = (tpad[1:] - tpad[:-1]).tolist()  # per-expert number of PIDs
            pid_list = []
            for e, nb in enumerate(counts):
                pid_list.extend([e] * int(nb))
            # add a few -1s for occupancy; not required for correctness
            block_pid_map_local[bm] = torch.tensor(pid_list, device=x.device, dtype=torch.int32)

    # build local expt_data with LOCAL maps
    expt_data_local = ExptDataCls(
        hist=hist_local,
        token_offs_raw=offs_local.to(torch.int32),
        token_offs_pad=token_offs_pad_local,
        block_pid_map=block_pid_map_local,
    )

    routing_local = RoutingDataCls(
        gate_scal=torch.nan_to_num(gamma_local, nan=0.0),  #  nan=0.0 required?
        expt_hist=hist_local,
        n_expts_tot=E_local,
        n_expts_act=1,
        expt_data=expt_data_local,
        expected_tokens_per_expt=getattr(routing_data, "expected_tokens_per_expt", None),
    )

    # local compute (no gather/scatter)
    inter = matmul_ogs(
        x_local,
        triton_gate_up_w,
        gate_up_bias.to(torch.float32),
        routing_data=routing_local,
        gather_indx=None,
        scatter_indx=None,
        precision_config=gate_pc,
        gammas=None,
        fused_activation=act,
    )

    y_local = matmul_ogs(
        inter,
        triton_down_w,
        down_bias.to(torch.float32),
        routing_data=routing_local,
        gather_indx=None,
        scatter_indx=None,
        precision_config=down_pc,
        gammas=gamma_local,  # apply gating on down path
    )  # [L_local, H]

    # accumulate into token positions on this rank; reduce across ranks later
    y_partial = torch.zeros((T, hidden_size), dtype=torch.float32, device=x.device)
    y_partial.index_add_(0, tok_idx_local, y_local.float())
    y_partial = y_partial.to(hidden_states.dtype).reshape_as(hidden_states)
    return y_partial


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
