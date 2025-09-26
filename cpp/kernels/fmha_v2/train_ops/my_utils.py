# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math

import numpy as np
import torch


def widen_cells():
    from IPython.core.display import HTML, display
    display(HTML("<style>.container { width:85% !important; }</style>"))
    np.set_printoptions(edgeitems=1000, linewidth=1000000)


def unpad(x: torch.Tensor, cu_seqlens: torch.IntTensor) -> torch.Tensor:
    """interpret the first two dims as BxS and unpad"""
    shape = list(x.shape)
    b = len(cu_seqlens) - 1
    assert b == shape[0]
    total = cu_seqlens[-1].item()
    new_shape = [total] + shape[2:]
    y = torch.empty(new_shape, dtype=x.dtype, device=x.device)
    for bi in range(b):
        start = cu_seqlens[bi]
        end = cu_seqlens[bi + 1]
        si = end - start
        y[start:end, :] = x[bi, :si, :]
    return y


def pad(x: torch.Tensor, cu_seqlens: torch.Tensor, s: int) -> torch.Tensor:
    # interpret the first two dims as BxS and unpad
    shape = list(x.shape)
    b = len(cu_seqlens) - 1
    total = cu_seqlens[-1].item()
    assert total == shape[0]
    new_shape = [b, s] + shape[1:]
    y = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
    for bi in range(b):
        start = cu_seqlens[bi]
        end = cu_seqlens[bi + 1]
        si = end - start
        y[bi, :si, :] = x[start:end, :]
    return y


def mask_softmax(S: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    Snew = torch.zeros_like(S)
    shape = S.shape
    b = len(cu_seqlens) - 1
    assert b == shape[0]
    for bi in range(b):
        start = cu_seqlens[bi]
        end = cu_seqlens[bi + 1]
        si = end - start
        Snew[bi, :, :si, :si] = S[bi, :, :si, :si]
    return Snew


def reshape_softmax(S: torch.Tensor,
                    b: int,
                    s: int,
                    h: int,
                    d: int,
                    warps_m: int,
                    warps_n: int,
                    extract_dmask: bool = True) -> [torch.Tensor, torch.Tensor]:
    m = 16
    if warps_m == 1:
        m = 16
    if warps_n == 1 and s == 1024:
        m = 64
    if warps_n == 1 and s == 2048:
        m = 64
    if warps_n == 1 and s == 4096:
        m = 128
    n = s
    m_per_cta, n_per_cta = warps_m * 16, warps_n * 16
    mmas_m, mmas_n = m // m_per_cta, n // n_per_cta
    loops = s // (mmas_m * m_per_cta)
    assert (loops == 8   and s == 128 ) or \
           (loops == 16  and s == 256 ) or \
           (loops == 32  and s == 512 ) or \
           (loops == 24  and s == 384 ) or \
           (loops == 64  and s == 1024) or \
           (loops == 128 and s == 2048) or \
           (loops == 256 and s == 4096) or \
           (loops == 32  and s == 4096) or \
           (loops == 16  and s == 1024) or \
           (loops == 32  and s == 2048)
    quads, lohi, lr, vals = 8, 2, 2, 2
    # 0   1       2        3        4         5         6       7   8      9   10  11
    # B x H x LOOPS x MMAS_M x MMAS_N x WARPS_N x WARPS_M x QUADS x 4 x LOHI x LR x 2:
    #   MMA register layout
    # 0   1       2        3         6      9       7        4         5   10   8  11
    # B x H x LOOPS x MMAS_M x WARPS_M x LOHI x QUADS x MMAS_N x WARPS_N x LR x 4 x 2
    #   Expected format B x H x S x S
    S = S.reshape((b, h, loops, mmas_m, mmas_n, warps_n, warps_m, quads, 4, lohi, lr, vals)) \
         .permute(0, 1, 2, 3, 6, 9, 7, 4, 5, 10, 8, 11) \
         .reshape((b,h,s,s))
    if not extract_dmask:
        return S, None
    # torch.signbit does not work on -0, need to do it in numpy!
    # positive is True
    SS = S.to(torch.float16)
    dmask = torch.tensor(np.logical_not(np.signbit(SS.cpu().numpy())),
                         dtype=SS.dtype,
                         device=SS.device)
    dmask = dmask.to(S.dtype)
    #S = S.abs()
    return S, dmask


def reshape_softmax_new(S: torch.Tensor, b: int, s: int, h: int, d: int,
                        warps_m: int,
                        warps_n: int) -> [torch.Tensor, torch.Tensor]:
    m = s if s == 128 else 16
    n = s
    m_per_cta, n_per_cta = warps_m * 16, warps_n * 16
    mmas_m, mmas_n = m // m_per_cta, n // n_per_cta
    loops = s // (mmas_m * m_per_cta)
    assert (loops == 1   and s == 128 ) or \
           (loops == 16  and s == 256 ) or \
           (loops == 32  and s == 512 ) or \
           (loops == 24  and s == 384 ) or \
           (loops == 64  and s == 1024) or \
           (loops == 128 and s == 2048) or \
           (loops == 256 and s == 4096)
    quads, lohi, lr, vals = 8, 2, 2, 2
    # 0   1       2        3        7         8         4       6   10      5   9  11
    # B x H x LOOPS x MMAS_M x MMAS_N x WARPS_N x WARPS_M x QUADS x 4 x LOHI x LR x 2:
    #   MMA register layout
    # 0   1       2        3         4      5       6        7         8    9   10  11
    # B x H x LOOPS x MMAS_M x WARPS_M x LOHI x QUADS x MMAS_N x WARPS_N x LR x 4 x 2:
    #   Expected format B x H x S x S

    S = S.reshape((b, h, loops, mmas_m, warps_m, lohi, quads, mmas_n, warps_n, lr, 4, vals)) \
         .permute(0, 1, 2, 3, 7, 8, 4, 6, 10, 5, 9, 11) \
         .reshape(-1)

    return S


def mha_ref(qkv, amask, D, b, s, h, d, p_dropout, is_causal, alibi_bias):
    qkv = qkv.view(b, s, h, 3, d)
    # [b, h, s, d]
    q = qkv[:, :, :, 0, :].permute(0, 2, 1, 3)
    k = qkv[:, :, :, 1, :].permute(0, 2, 1, 3)
    v = qkv[:, :, :, 2, :].permute(0, 2, 1, 3)

    #like this multiplications will be done in FP32 too, but we avoid one downcast to FP16
    p = torch.matmul(q.float(), k.permute(0, 1, 3, 2).float())
    # [b, h, s, s]
    p_masked = p / math.sqrt(d) + (1.0 - amask) * -10000.0

    if (is_causal):
        causal_mask = torch.triu(
            torch.ones(s, s, dtype=torch.bool, device=qkv.device), 1)
        p_masked.masked_fill_(causal_mask, float('-inf'))

    if alibi_bias is not None:
        # [b, h, s, s] + [b, h, 1, s]
        p_masked = p_masked + alibi_bias

    mm = torch.max(p_masked, -1)[0].to(torch.float)  # [b, h, s]
    mm_kd = torch.max(p_masked, -1, True)  # [b, h, s, 1]
    exp = torch.exp(p_masked - mm_kd[0])
    ll = torch.sum(exp, -1).to(torch.float)
    softmax_lse = mm + torch.log(ll)  # [b, h, s]

    sm = torch.softmax(p_masked, -1)
    sm = sm.to(qkv.dtype)

    rp_keep = 1.0 / (1.0 - p_dropout)
    #d = sm * rp_keep
    d = sm * D.to(sm.dtype) * rp_keep  # TODO
    ctx = torch.matmul(d, v)
    ctx = ctx.permute(0, 2, 1, 3).contiguous()

    ctx.retain_grad()
    p.retain_grad()
    sm.retain_grad()

    return ctx, p, sm, softmax_lse


def perr(a, b):
    a, b = a.float(), b.float()
    diff = (a - b)
    return (diff.abs().sum() / a.abs().sum()).item()


def mae(a, b):
    a, b = a.float(), b.float()
    diff = (a - b)
    return diff.abs().mean().item()


def build_alibi_tensor(max_seq_len, num_attention_heads, batch_size):
    # Based on https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    """Returns tensor shaped (batch_size * num_attention_heads, 1, max_seq_len)"""

    def get_slopes(n):

        def get_slopes_power_of_2(n):
            # 2 ** (-8/n)
            start = (2**(-2**-(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    # [h]
    slopes = torch.Tensor(get_slopes(num_attention_heads))
    # [h, 1, 1] * [1, 1, s] --> [h, 1, s]
    alibi = slopes.unsqueeze(1).unsqueeze(
        1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
            num_attention_heads, -1, -1)

    #Select the part of the tensor that corresponds to our tensor parallel index.
    # [b, h, 1, s]
    alibi = alibi.repeat(batch_size, 1, 1, 1)

    return alibi
