# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# Copyright (c) 2024, Tri Dao.
#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID


def causal_conv1d_fn(x: torch.Tensor,
                     weight: torch.Tensor,
                     bias: Optional[torch.Tensor] = None,
                     query_start_loc: Optional[torch.Tensor] = None,
                     cache_indices: Optional[torch.Tensor] = None,
                     has_initial_state: Optional[torch.Tensor] = None,
                     conv_states: Optional[torch.Tensor] = None,
                     activation: Optional[str] = "silu",
                     pad_slot_id: int = PAD_SLOT_ID,
                     out: Optional[torch.Tensor] = None):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
        Either axis may be the contiguous one. A channel-last (token-major) x -- i.e.
        ``x.stride(0) == 1`` for the varlen layout -- selects a dedicated kernel and lets
        callers holding [tokens, channels] activations pass a transposed view directly
        instead of materialising a channel-major copy.
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


    out: optional destination with the same shape/dtype as x. Defaults to writing back into
        x in place. Must not overlap x when x is channel-last, because that kernel chunks
        each sequence along the token axis and every chunk reads a halo written by the
        previous one; a matching channel-last buffer is allocated automatically if omitted.

    returns: the tensor that was written (x when out is None)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    channel_last = x.stride(-2) == 1 and x.stride(-1) > 1
    if x.stride(-1) != 1 and not channel_last:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None
    if channel_last and out is None:
        # Match x's channel-last layout so the caller keeps a token-major result.
        # (torch.empty_like would give a channel-major buffer whenever x is a strided view.)
        if x.dim() == 2:
            out = torch.empty(x.shape[1],
                              x.shape[0],
                              dtype=x.dtype,
                              device=x.device).t()
        else:
            out = torch.empty(x.shape[0],
                              x.shape[2],
                              x.shape[1],
                              dtype=x.dtype,
                              device=x.device).transpose(1, 2)

    torch.ops.trtllm.causal_conv1d_fwd(x, weight, bias, conv_states,
                                       query_start_loc, cache_indices,
                                       has_initial_state, activation
                                       in ["silu", "swish"], pad_slot_id, out)
    return x if out is None else out


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
