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
from einops import rearrange


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
