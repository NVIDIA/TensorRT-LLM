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
    u: (B D L)
    delta: (B D L)
    A: (D N)
    B: (B N L)
    C: (B N L)
    D: (D)
    z: (B D L)
    delta_bias: (D), fp32

    out: (B D L)
    last_state (optional): (B D dstate), fp32
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype=dtype_in)
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
        state: (batch, dim, dstate)
        x: (batch, dim)
        dt: (batch, dim)
        A: (dim, dstate)
        B: (batch, dstate)
        C: (batch, dstate)
        D: (dim,)
        z: (batch, dim)
        dt_bias: (dim,)
    Return:
        out: (batch, dim)
    """
    batch, dim, dstate = state.shape
    assert x.shape == (batch, dim)
    assert dt.shape == x.shape
    assert A.shape == (dim, dstate)
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
    dA = torch.exp(rearrange(dt, "b d -> b d 1") * A)  # (batch, dim, dstate)
    dB = rearrange(dt, "b d -> b d 1") * rearrange(
        B.float(), "b n -> b 1 n")  # (batch, dim, dstate)
    state_new = state * dA + dB * rearrange(
        x, "b d -> b d 1")  # (batch, dim, dstate)
    state.copy_(state_new.to(state.dtype))
    out = torch.einsum("bdn,bn->bd", state_new, C.float())
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
                conv_state=None,
                ssm_state=None,
                seqlen_offset=0):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        if seqlen_offset > 0:
            # The states are updated inplace
            out, conv_state, ssm_state = self.step(hidden_states, conv_state,
                                                   ssm_state)
            return out, conv_state, ssm_state

        # in_proj
        xz = torch.nn.functional.linear(hidden_states, self.in_proj.weight)
        xz = xz.permute(0, 2, 1)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype),
                                "d -> d 1")

        # Conv
        x, z = xz.chunk(2, dim=1)
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
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        # Selective scan
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
        y = rearrange(y, "b d l -> b l d")
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
