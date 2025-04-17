# Copyright (c) 2024, Tri Dao.
# causal_conv1d_update, adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# causal_conv1d_varlen_states, adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_varlen.py

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]

    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)

    x_new = torch.cat([conv_state, x], dim=-1).to(
        weight.dtype)  # (batch, dim, state_len + seqlen)
    conv_state.copy_(x_new[:, :, -state_len:])

    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0,
                   groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


@triton.jit
def _causal_conv1d_varlen_states(
    X,
    CU_SEQLENS,
    STATES,
    state_len,
    dim,
    stride_x_seqlen,
    stride_x_dim,
    stride_states_batch,
    stride_states_seqlen,
    stride_states_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(2)
    STATES += batch_idx * stride_states_batch
    end_idx = tl.load(CU_SEQLENS + batch_idx + 1)
    start_idx = tl.maximum(tl.load(CU_SEQLENS + batch_idx), end_idx - state_len)
    rows = end_idx - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(
        X + rows[:, None] * stride_x_seqlen + cols[None, :] * stride_x_dim,
        mask=(rows[:, None] >= start_idx) & (cols[None, :] < dim),
        other=0,
    )
    rows_states = state_len - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(
        0, BLOCK_M)
    tl.store(
        STATES + rows_states[:, None] * stride_states_seqlen +
        cols[None, :] * stride_states_dim,
        x,
        mask=(rows_states[:, None] >= 0) & (cols[None, :] < dim),
    )


def causal_conv1d_varlen_states(x: torch.Tensor, cu_seqlens: torch.Tensor,
                                state_len: int) -> torch.Tensor:
    """
    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state_len: int. For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    states = torch.empty(batch, state_len, dim, dtype=x.dtype,
                         device=x.device).transpose(1, 2)
    BLOCK_M = min(triton.next_power_of_2(state_len), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 256)
    grid = (triton.cdiv(dim, BLOCK_N), triton.cdiv(state_len, BLOCK_M), batch)
    with torch.cuda.device(x.device.index):
        _causal_conv1d_varlen_states[grid](
            x,
            cu_seqlens,
            states,
            state_len,
            dim,
            x.stride(0),
            x.stride(1),
            states.stride(0),
            states.stride(2),
            states.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    return states
