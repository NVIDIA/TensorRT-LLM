# Copyright (c) 2024, Tri Dao.
# causal_conv1d_update, adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# causal_conv1d_varlen_states, adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_varlen.py

from typing import Optional

import torch


def causal_conv1d_fwd(xBC: torch.Tensor, conv1d_weight: torch.Tensor,
                      conv1d_bias: torch.Tensor) -> torch.Tensor:

    slot_mapping = None
    remove_padding = True
    apply_silu = True
    is_paged_state = False

    conv_dim = conv1d_weight.shape[0]
    d_conv = conv1d_weight.shape[1]
    seq_len = [xBC.shape[2]]

    host_context_lengths = torch.as_tensor(seq_len, dtype=torch.int32).cuda()
    host_request_types = torch.zeros_like(host_context_lengths).cuda()
    last_token_ids = torch.cumsum(host_context_lengths,
                                  dim=0,
                                  dtype=torch.int32).cuda()

    conv_states_in = torch.zeros(1, d_conv - 1, conv_dim).cuda()

    y_new, _ = torch.ops.trtllm.mamba_conv1d(
        # xBC is [S, dim]
        xBC.squeeze(0).permute(1, 0).contiguous(),
        # conv_weight is [1, d_conv, dim]
        conv1d_weight.unsqueeze(0).permute(0, 2, 1).contiguous(),
        conv1d_bias,
        conv_states_in,
        host_request_types,
        last_token_ids,
        host_context_lengths,
        slot_mapping,
        conv_dim,
        d_conv,
        0,
        0,
        remove_padding,
        apply_silu,
        is_paged_state,
    )

    y_new = y_new.unsqueeze(0).permute(0, 2, 1).contiguous()
    return y_new


PAD_SLOT_ID = -1000000  # TODO: check if this matters. resource_manager.py::add_dummy_requests uses similar value to pad requests to batch size.


def causal_conv1d_fn(x: torch.Tensor,
                     weight: torch.Tensor,
                     bias: Optional[torch.Tensor] = None,
                     query_start_loc: Optional[torch.Tensor] = None,
                     cache_indices: Optional[torch.Tensor] = None,
                     has_initial_state: Optional[torch.Tensor] = None,
                     conv_states: Optional[torch.Tensor] = None,
                     activation: Optional[str] = "silu",
                     pad_slot_id: int = PAD_SLOT_ID):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3


    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    torch.ops.trtllm.causal_conv1d_fwd(x, weight, bias, conv_states,
                                       query_start_loc, cache_indices,
                                       has_initial_state, activation
                                       in ["silu", "swish"], pad_slot_id)
    return x


def causal_conv1d_update(x: torch.Tensor,
                         conv_state: torch.Tensor,
                         weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None,
                         activation: Optional[str] = None,
                         cache_seqlens: Optional[torch.Tensor] = None,
                         conv_state_indices: Optional[torch.Tensor] = None,
                         pad_slot_id: int = PAD_SLOT_ID):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    torch.ops.trtllm.causal_conv1d_update(x, conv_state, weight, bias,
                                          activation_val, cache_seqlens,
                                          conv_state_indices, pad_slot_id)
    if unsqueeze:
        x = x.squeeze(-1)
    return x
