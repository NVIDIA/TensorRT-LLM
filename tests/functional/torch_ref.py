# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def geglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * torch.nn.functional.gelu(b)


def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * x


def generate_qkv(x, Wqkv, nheads, kvpacked=False, qkvpacked=False):
    """
    Arguments:
        x: (batch_size, seqlen, nheads * d)
        Wqkv: nn.Linear(nheads * d, 3 * nheads * d)
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen, dim = x.shape
    q, k, v = Wqkv(x).chunk(3, dim=-1)

    q_unpad = rearrange(q, 'b s (h d) -> (b s) h d', h=nheads)
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen,
                                step=seqlen,
                                dtype=torch.int32,
                                device=q_unpad.device)
    max_seqlen_q = seqlen
    output_pad_fn = lambda output_unpad: rearrange(
        output_unpad, '(b s) h d -> b s h d', b=batch_size)

    k_unpad = rearrange(k, 'b s (h d) -> (b s) h d', h=nheads)
    v_unpad = rearrange(v, 'b s (h d) -> (b s) h d', h=nheads)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen,
                                step=seqlen,
                                dtype=torch.int32,
                                device=q_unpad.device)
    max_seqlen_k = seqlen

    if qkvpacked:
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = rearrange(torch.stack([q, k, v], dim=2),
                        'b s t (h d) -> b s t h d',
                        h=nheads)
        dqkv_pad_fn = lambda dqkv_unpad: rearrange(
            dqkv_unpad, '(b s) t h d -> b s t h d', b=batch_size)
        return (qkv_unpad, cu_seqlens_q, max_seqlen_q, qkv, output_pad_fn,
                dqkv_pad_fn)
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        q = rearrange(q, 'b s (h d) -> b s h d', h=nheads)
        kv = rearrange(torch.stack([k, v], dim=2),
                       'b s t (h d) -> b s t h d',
                       h=nheads)
        dq_pad_fn = output_pad_fn
        dkv_pad_fn = lambda dkv_unpad: rearrange(
            dkv_unpad, '(b s) t h d -> b s t h d', b=batch_size)
        return (q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                max_seqlen_k, q, kv, output_pad_fn, dq_pad_fn, dkv_pad_fn)
    else:
        q, k, v = [
            rearrange(z, 'b s (h d) -> b s h d', h=nheads) for z in [q, k, v]
        ]
        dq_pad_fn = output_pad_fn
        dk_pad_fn = lambda dk_unpad: rearrange(
            dk_unpad, '(b s) h d -> b s h d', b=batch_size)
        return (q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, q, k, v, output_pad_fn, dq_pad_fn,
                dk_pad_fn)


def attention_ref(q,
                  k,
                  v,
                  causal=False,
                  bias=None,
                  upcast=True,
                  reorder_ops=False):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads, head_dim)
        v: (batch_size, seqlen_k, nheads, head_dim)
        bias: (batch_size, nheads, seqlen_q, seqlen_k)
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum('bthd,bshd->bhts', q / math.sqrt(d), k)
    else:
        scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    if bias is not None:
        scores = (scores + bias).to(dtype=scores.dtype)
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device),
            1)
        scores.masked_fill_(causal_mask, float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    output = torch.einsum('bhts,bshd->bthd', attention, v)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_kvpacked_ref(q, kv, causal=False, upcast=True, reorder_ops=False):
    return attention_ref(q,
                         kv[:, :, 0],
                         kv[:, :, 1],
                         upcast=upcast,
                         causal=causal,
                         reorder_ops=reorder_ops)


def attention_qkvpacked_ref(qkv,
                            causal=False,
                            bias=None,
                            upcast=True,
                            reorder_ops=False):
    return attention_ref(qkv[:, :, 0],
                         qkv[:, :, 1],
                         qkv[:, :, 2],
                         upcast=upcast,
                         causal=causal,
                         bias=bias,
                         reorder_ops=reorder_ops)


def mamba_conv1d_ref(x, past_conv_state, conv_weight, conv_bias, apply_silu):
    """
    Arguments:
        x: [batch_size, dim, seq_len]
        past_conv_state: [batch_size, dim, dconv-1]
        conv_weight: [dim, 1, dconv]
        conv_bias: [dim]
    Output:
        y: [batch_size, dim, seq_len]
        present_conv_state: [batch_size, dim, dconv-1]
    """
    assert x.dim() == 3
    assert past_conv_state.dim() == 3
    assert conv_weight.dim() == 3
    assert conv_bias.dim() == 1
    batch_size, dim, seq_len = x.shape
    assert past_conv_state.shape[0] == batch_size
    assert past_conv_state.shape[1] == dim
    dconv = past_conv_state.shape[2] + 1
    assert conv_weight.shape[0] == dim
    assert conv_weight.shape[1] == 1
    assert conv_weight.shape[2] == dconv
    assert conv_weight.shape[0] == dim

    padded_x = torch.cat([past_conv_state, x], dim=2)
    present_conv_state = padded_x[:, :, -(dconv - 1):]
    x_conv = F.conv1d(padded_x, conv_weight, bias=conv_bias, groups=dim)

    y = F.silu(x_conv) if apply_silu else x_conv
    return y, present_conv_state


def selective_scan_ref(u,
                       delta,
                       A,
                       B,
                       C,
                       D=None,
                       z=None,
                       delta_bias=None,
                       delta_softplus=False):
    """
    u: (B L D)
    delta: (B L D)
    A: (N D)
    B: (B L N)
    C: (B L N)
    D: (D)
    z: (B L D)
    delta_bias: (D), fp32

    out: (B L D)
    last_state (optional): (B dstate D), fp32
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(1).float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dstate, dim = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dstate, dim))
    ys = []
    deltaA = torch.exp(torch.einsum('bld,nd->blnd', delta, A))
    deltaB_u = torch.einsum('bld,bln,bld->blnd', delta, B, u)
    last_state = None
    for i in range(u.shape[1]):
        x = deltaA[:, i, :] * x + deltaB_u[:, i, :]
        y = torch.einsum('bnd,bn->bd', x, C[:, i, :])
        if i == u.shape[1] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=1)  # (batch L dim)
    out = y if D is None else y + u * rearrange(D, "d -> 1 d")
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype=dtype_in)
    last_state = last_state.to(dtype=dtype_in)
    return out, last_state


def selective_state_update_ref(state,
                               x,
                               dt,
                               A,
                               B,
                               C,
                               D=None,
                               z=None,
                               dt_bias=None,
                               dt_softplus=False):
    """
    Argument:
        state: (batch, dstate, dim)
        x: (batch, dim)
        dt: (batch, dim)
        A: (dstate, dim)
        B: (batch, dstate)
        C: (batch, dstate)
        D: (dim,)
        z: (batch, dim)
        dt_bias: (dim,)
    Return:
        out: (batch, dim)
    """
    batch, dstate, dim = state.shape
    assert x.shape == (batch, dim)
    assert dt.shape == x.shape
    assert A.shape == (dstate, dim)
    assert B.shape == (batch, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (dim, )
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (dim, )
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, "b d -> b 1 d") * A)  # (batch, dstate, dim)
    dB = rearrange(dt, "b d -> b 1 d") * rearrange(
        B.float(), "b n -> b n 1")  # (batch, dstate, dim)
    state_new = state * dA + dB * rearrange(
        x, "b d -> b 1 d")  # (batch, dstate, dim)
    state.copy_(state_new.to(state.dtype))
    out = torch.einsum("bnd,bn->bd", state_new, C.float())
    if D is not None:
        out += x * D
    return (out if z is None else out * F.silu(z.float())).to(x.dtype)


class mamba_ref(nn.Module):

    def __init__(self,
                 d_model,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dt_rank="auto",
                 conv_bias=True,
                 bias=False,
                 device=None,
                 dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model /
                                 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model,
                                 self.d_inner * 2,
                                 bias=bias,
                                 **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner,
                                self.dt_rank + self.d_state * 2,
                                bias=False,
                                **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank,
                                 self.d_inner,
                                 bias=True,
                                 **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner,
                                  self.d_model,
                                  bias=bias,
                                  **factory_kwargs)

        # S4D real initialization
        A = repeat(torch.arange(1,
                                self.d_state + 1,
                                dtype=torch.float32,
                                device=device),
                   "n -> d n",
                   d=self.d_inner).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A = nn.Parameter(-torch.exp(A_log.float()))

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner,
                                         device=device))  # Keep in fp32

    def forward(self,
                hidden_states,
                last_token_ids,
                conv_state,
                ssm_state,
                remove_padding,
                batch_size,
                seqlen_offset=0):
        out, present_conv_state, present_ssm_state = [], [], []
        for i in range(batch_size):
            start_id = 0 if (i == 0
                             or not remove_padding) else last_token_ids[i - 1]
            end_id = last_token_ids[i]
            if remove_padding:
                hidden_states_i = hidden_states[start_id:end_id, :].unsqueeze(0)
            else:
                hidden_states_i = hidden_states[i:i + 1, start_id:end_id, :]
            conv_state_i = conv_state[i:i + 1, :]
            ssm_state_i = ssm_state[i:i + 1, :]
            out_i, conv_state_i, ssm_state_i = self.forward_impl(
                hidden_states_i, conv_state_i, ssm_state_i, seqlen_offset)
            if remove_padding:
                out_i = out_i.squeeze(0)
            else:
                padding_num = hidden_states.shape[1] - out_i.shape[1]
                out_i = F.pad(out_i, (0, 0, 0, padding_num, 0, 0), value=0)
            out.append(out_i)
            present_conv_state.append(conv_state_i)
            present_ssm_state.append(ssm_state_i)
        out = torch.concat(out, dim=0)
        present_conv_state = torch.concat(present_conv_state, dim=0)
        present_ssm_state = torch.concat(present_ssm_state, dim=0)
        return out, present_conv_state, present_ssm_state

    def forward_impl(self,
                     hidden_states,
                     conv_state,
                     ssm_state,
                     seqlen_offset=0):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape

        if seqlen_offset > 0:
            # The states are updated inplace
            out, conv_state, ssm_state = self.step(hidden_states, conv_state,
                                                   ssm_state)
            return out, conv_state, ssm_state

        # in_proj
        xz = torch.nn.functional.linear(hidden_states, self.in_proj.weight)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype),
                                "d -> 1 d")

        # Conv
        x, z = xz.chunk(2, dim=2)
        x = x.permute(0, 2, 1)
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
        x_conv = self.conv1d(x)[..., :seqlen]
        x = self.act(x_conv)

        # Get A, dt, B, and C
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl,
                               [self.dt_rank, self.d_state, self.d_state],
                               dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b l d", l=seqlen).contiguous()
        B = rearrange(B, "(b l) dstate -> b l dstate", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b l dstate", l=seqlen).contiguous()

        # Selective scan
        x = x.permute(0, 2, 1)
        y, last_state = selective_scan_ref(x,
                                           dt,
                                           self.A,
                                           B,
                                           C,
                                           self.D.float(),
                                           z=z,
                                           delta_bias=self.dt_proj.bias.float(),
                                           delta_softplus=True)
        ssm_state.copy_(last_state)

        # out_proj
        out = self.out_proj(y)
        return out, conv_state, ssm_state

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1

        # in_proj
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        # Conv step
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x

        x = torch.sum(conv_state *
                      rearrange(self.conv1d.weight, "d 1 w -> d w"),
                      dim=-1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = self.act(x).to(dtype=dtype)

        # Get dt, B, and C
        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state],
                               dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)

        # SSM step
        y = selective_state_update_ref(ssm_state,
                                       x,
                                       dt,
                                       self.A,
                                       B,
                                       C,
                                       D=self.D.float(),
                                       z=z,
                                       dt_bias=self.dt_proj.bias.float(),
                                       dt_softplus=True)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


def rnn_scan(x: torch.Tensor, a: torch.Tensor, reset: torch.Tensor,
             h0: torch.Tensor):
    """Runs the recurrence of a linear RNN."""
    assert x.ndim == 3
    assert a.shape == x.shape[-a.ndim:]
    assert a.dtype == x.dtype
    assert type(a) is type(x)

    # Multiply `a` by the reset
    a = a * (1 - reset)

    if x.shape[1] == 1:
        # Using scan in sampling mode.
        y = a * h0[:, None] + x
    else:
        # Using scan in linear mode.
        h_t = h0
        y = torch.zeros_like(x)
        for t in range(x.shape[1]):
            y[:, t] = a[:, t] * h_t + x[:, t]
            h_t = y[:, t].type_as(x)
    h_last = y[:, -1]

    return y, h_last


def rg_lru_ref(x,
               input_gate_x,
               a_gate_x,
               y,
               y_bias,
               segment_pos,
               prev_h,
               a_param,
               gate_x_bias=None,
               gate_a_bias=None):

    bs, l, d = x.shape
    assert segment_pos.shape == (bs, l)
    reset = (segment_pos == 0).type(torch.int32)[..., None]
    prev_h = torch.zeros(size=(bs, d)) if prev_h is None else prev_h

    # Gates for x and a.
    if gate_x_bias is not None:
        input_gate_x += gate_x_bias.reshape(1, 1, d)
    if gate_a_bias is not None:
        a_gate_x += gate_a_bias.reshape(1, 1, d)
    gate_x = torch.sigmoid(input_gate_x.float())
    gate_a = torch.sigmoid(a_gate_x.float())

    # Compute the parameter `A` of the recurrence
    c = -8.0 * nn.functional.softplus(a_param.float())
    log_a = c * gate_a
    a = torch.exp(log_a)

    # Gate the input
    gated_x = x * gate_x

    # Apply gamma normalization to the input
    multiplier = torch.sqrt(1 - torch.exp(2 * log_a))
    multiplier = reset + (1 - reset) * multiplier
    normalized_x = gated_x * multiplier

    # rnn scan
    out, last_h = rnn_scan(
        x=normalized_x,
        a=a,
        reset=reset,
        h0=prev_h,
    )

    # y branch
    if y_bias is not None:
        out = out * torch.nn.functional.gelu(y + y_bias)
    elif y is not None:
        out = out * y
    else:
        out = out
    return out.type(x.dtype), last_h


def rg_lru_batch_ref(x,
                     input_gate_x,
                     a_gate_x,
                     y,
                     y_bias,
                     segment_pos,
                     prev_h,
                     a_param,
                     batch_size,
                     remove_padding,
                     last_token_ids,
                     gate_x_bias=None,
                     gate_a_bias=None):
    outputs, lru_states = [], []
    for i in range(batch_size):
        start_id = 0 if (i == 0 or not remove_padding) else last_token_ids[i -
                                                                           1]
        end_id = last_token_ids[i]
        if remove_padding:
            x_i = x[start_id:end_id, :].unsqueeze(0)
            input_gate_x_i = input_gate_x[start_id:end_id, :].unsqueeze(0)
            a_gate_x_i = a_gate_x[start_id:end_id, :].unsqueeze(0)
            y_i = y[start_id:end_id, :].unsqueeze(0) if y is not None else None
        else:
            x_i = x[i:i + 1, start_id:end_id, :]
            input_gate_x_i = input_gate_x[i:i + 1, start_id:end_id, :]
            a_gate_x_i = a_gate_x[i:i + 1, start_id:end_id, :]
            y_i = y[i:i + 1, start_id:end_id, :] if y is not None else None
        segment_pos_i = segment_pos[i:i + 1, 0:end_id - start_id]
        prev_h_i = prev_h[i:i + 1, :]

        out_i, lru_state_i = rg_lru_ref(x_i, input_gate_x_i, a_gate_x_i, y_i,
                                        y_bias, segment_pos_i, prev_h_i,
                                        a_param, gate_x_bias, gate_a_bias)
        if remove_padding:
            out_i = out_i.squeeze(0)
        else:
            padding_num = x.shape[1] - out_i.shape[1]
            out_i = F.pad(out_i, (0, 0, 0, padding_num, 0, 0), value=0)
        outputs.append(out_i)
        lru_states.append(lru_state_i)
    out = torch.concat(outputs, dim=0)
    last_h = torch.concat(lru_states, dim=0)
    return out, last_h


class BlockDiagonalLinear(nn.Module):
    """Block-diagonal linear layer."""

    def __init__(self,
                 num_blocks: int,
                 width: int,
                 fuse_bias=False,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_blocks = num_blocks
        self.width = width
        self.fuse_bias = fuse_bias
        block_width = self.width // self.num_blocks

        # Parameters
        self.w = nn.Parameter(
            torch.randn([self.num_blocks, block_width, block_width],
                        **factory_kwargs))
        self.b = nn.Parameter(
            torch.randn([self.num_blocks, block_width], **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split x to blocks
        x = rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

        # Linear layer over each block + bias
        y = torch.einsum("... h i, h i j -> ... h j", x, self.w)
        if not self.fuse_bias:
            y += self.b

        # Flatten the output
        return rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


class recurrent_ref(nn.Module):

    def __init__(self,
                 width,
                 lru_width,
                 num_heads,
                 d_conv,
                 device=None,
                 dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_conv = d_conv
        self.recurrent_param = nn.Parameter(
            torch.randn([lru_width], device=device))

        self.linear_x = nn.Linear(width, lru_width, **factory_kwargs)
        self.linear_y = nn.Linear(width,
                                  lru_width,
                                  bias=False,
                                  **factory_kwargs)
        self.y_bias = nn.Parameter(torch.randn([1, 1, lru_width],
                                               device=device))

        self.conv1d = nn.Conv1d(
            in_channels=lru_width,
            out_channels=lru_width,
            bias=True,
            kernel_size=d_conv,
            groups=lru_width,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.input_gate = BlockDiagonalLinear(
            num_blocks=num_heads,
            width=lru_width,
            fuse_bias=True,
            **factory_kwargs,
        )
        self.recurrent_gate = BlockDiagonalLinear(
            num_blocks=num_heads,
            width=lru_width,
            fuse_bias=True,
            **factory_kwargs,
        )

        self.linear_out = nn.Linear(lru_width, width, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        segment_pos: torch.Tensor,
        batch_size: int,
        remove_padding: bool,
        last_token_ids: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        lru_state: Optional[torch.Tensor] = None,
        conv_idx: Optional[torch.Tensor] = None,
    ):
        outputs, conv_states, lru_states = [], [], []
        for i in range(batch_size):
            start_id = 0 if (i == 0
                             or not remove_padding) else last_token_ids[i - 1]
            end_id = last_token_ids[i]
            if remove_padding:
                x_i = x[start_id:end_id, :].unsqueeze(0)
            else:
                x_i = x[i:i + 1, start_id:end_id, :]
            segment_pos_i = segment_pos[i:i + 1, 0:end_id - start_id]
            conv_state_i = None if conv_state is None else conv_state[i:i + 1, ]
            lru_state_i = None if lru_state is None else lru_state[i:i + 1, ]
            conv_idx_i = None if conv_idx is None else conv_idx[i:i + 1, ]
            out_i, conv_state_i, lru_state_i = self.forward_impl(
                x_i, segment_pos_i, conv_state_i, lru_state_i, conv_idx_i)
            if remove_padding:
                out_i = out_i.squeeze(0)
            else:
                padding_num = x.shape[1] - out_i.shape[1]
                out_i = F.pad(out_i, (0, 0, 0, padding_num, 0, 0), value=0)
            outputs.append(out_i)
            conv_states.append(conv_state_i)
            lru_states.append(lru_state_i)
        out = torch.concat(outputs, dim=0)
        conv_state = torch.concat(conv_states, dim=0)
        lru_state = torch.concat(lru_states, dim=0)
        return out, conv_state, lru_state

    def forward_impl(
        self,
        x: torch.Tensor,
        segment_pos: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        lru_state: Optional[torch.Tensor] = None,
        conv_indices: Optional[torch.Tensor] = None,
    ):
        _, seqlen, _ = x.shape

        # y branch
        y = self.linear_y(x)

        # x branch
        x = self.linear_x(x)

        # conv1d
        if conv_state is None:
            x = x.permute([0, 2, 1])
            conv_state = F.pad(x, (self.d_conv - 1, 0))
            conv_state = torch.gather(conv_state,
                                      dim=2,
                                      index=conv_indices.type(torch.int64))
            x = self.conv1d(x)[..., :seqlen].permute([0, 2, 1])
        else:
            x = x.squeeze(1)
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state *
                          rearrange(self.conv1d.weight, "d 1 w -> d w"),
                          dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = x.unsqueeze(1)

        # rg lru
        gate_x = self.input_gate(x)
        gate_a = self.recurrent_gate(x)

        x, lru_state = rg_lru_ref(x, gate_x, gate_a, y, self.y_bias,
                                  segment_pos, lru_state, self.recurrent_param,
                                  self.input_gate.b, self.recurrent_gate.b)

        # Join branches.
        x = self.linear_out(x)

        return x, conv_state, lru_state
