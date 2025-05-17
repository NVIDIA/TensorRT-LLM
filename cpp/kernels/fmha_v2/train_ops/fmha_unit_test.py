# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# testing fixed sequence length

import math
import os
import sys
from time import time

import numpy as np
import torch

current_script_dir = os.path.dirname(os.path.abspath(__file__))
my_path = current_script_dir + os.path.sep + 'build'
if my_path not in sys.path:
    sys.path.append(my_path)

# exception is suppressed on purpose:
# - often we don't build train_ops yet, but would like to run tests for other components
# - pytest complain about missing train_ops module during test collection
# - we don't want false alarms so we silence it during test collection
try:
    import bert_mha_train as mha
except ImportError:
    pass

try:
    # for running via pytest module
    from .my_utils import *
except ImportError:
    # for direct running as script
    try:
        from my_utils import *
        widen_cells()
    except ImportError:
        raise ImportError("my_utils.py not found")


def run_test(s, d, heads_interleaved=False, seqs_interleaved=False):
    if heads_interleaved:
        print("[HEADS_INTERLEAVED]")
    if seqs_interleaved:
        print("[SEQUENCES_INTERLEAVED]")
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    dtype = torch.float16
    device = torch.device('cuda')
    warps_m = 4
    warps_n = 1
    if s == 128:
        runs = 1
        warps_m = 1
    elif s == 256:
        runs = 1
        warps_m = 1
    elif s == 384:
        runs = 1
        warps_m = 1
        warps_n = 8
    elif s == 512:
        runs = 1
        warps_m = 1
        warps_n = 8

    b = 2
    h = 2
    # d = 40
    runs = 100

    p_dropout = 0.0
    is_training = True
    is_causal = False
    has_alibi = False

    print(
        "\n#### TEST fp16 = {}, is_causal = {}, has_alibi = {}, b = {}, h = {}, s = {}, d = {}, runs = {}, p_drop = {} ####\n"
        .format(dtype == torch.float16, is_causal, has_alibi, b, h, s, d, runs,
                p_dropout))

    alibi_bias = None
    if (has_alibi):
        alibi_bias = build_alibi_tensor(s, h, b).to(dtype=dtype, device=device)

    slens = [s] * b  # fixed sequence lengths, full mask

    a = torch.tensor(np.array([0] + slens), dtype=torch.int32)
    amask = torch.ones(b, h, s, s, dtype=dtype, device=device)

    seqlens = torch.tensor(slens, dtype=torch.int32, device=device)
    cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=device)
    total = cu_seqlens[-1].item()

    assert cu_seqlens.numel() == b + 1, "ahh"
    assert dtype in [torch.float16, torch.bfloat16]

    qkv = torch.randn((b, s, h, 3, d), device=device, dtype=dtype)
    # Convert to bxsx3xhxd!
    qkv_vs = unpad(qkv.permute(0, 1, 3, 2, 4).contiguous(), cu_seqlens)
    if seqs_interleaved:
        qkv_vs = qkv_vs.reshape(b, s, 3, h,
                                d).permute(1, 0, 2, 3,
                                           4).reshape(b * s, 3, h, d)

    if heads_interleaved:
        qkv_vs = qkv_vs.permute(0, 2, 1, 3).detach().contiguous()
    #qkv_vs = unpad(qkv, cu_seqlens)

    qkv.requires_grad = True
    is_nl = True

    for _ in range(runs):
        ctx, lse, random_seed, S_mma = mha.fwd(qkv_vs, cu_seqlens, p_dropout, s,
                                               is_training, is_nl,
                                               seqs_interleaved, is_causal,
                                               has_alibi, None)

    torch.cuda.synchronize()
    time0 = time()

    for _ in range(runs):
        ctx, lse, random_seed, S_mma = mha.fwd(qkv_vs, cu_seqlens, p_dropout, s,
                                               is_training, is_nl,
                                               seqs_interleaved, is_causal,
                                               has_alibi, None)

    torch.cuda.synchronize()
    time1 = time()
    print("Average %f ms for forward pass\n" % ((time1 - time0) / runs * 1000))

    if seqs_interleaved:
        ctx = pad(
            ctx.reshape(s, b, h, d).permute(1, 0, 2, 3).reshape(s * b, h, d),
            cu_seqlens, s)
    else:
        ctx = pad(ctx, cu_seqlens, s)
    S, D = reshape_softmax(S_mma, b, s, h, d, warps_m, warps_n, True)

    ctx_ref, Pref, Sref, lse_ref = mha_ref(qkv, amask, D, b, s, h, d, p_dropout,
                                           is_causal, alibi_bias)

    print("= Testing Fused Forward =")
    print("[Softmax_lse] AC: {}; MAE: {:.4e}; PERR: {:.4e}".format(
        torch.allclose(lse.float(), lse_ref.float(), atol=1e-3),
        mae(lse_ref, lse), perr(lse_ref, lse)))
    print("[CTX]     AC: {}; MAE: {:.4e}; PERR: {:.4e}".format(
        torch.allclose(ctx_ref.float(), ctx.float(), atol=1e-3),
        mae(ctx_ref, ctx), perr(ctx_ref, ctx)))
    print()

    labels = torch.randn_like(ctx_ref)
    diff = ctx_ref - labels
    l = (diff * diff).sum() / b
    l.backward()

    dw = ctx_ref.grad.permute(0, 2, 1, 3)

    q = qkv[:, :, :, 0, :].permute(0, 2, 1, 3)
    k = qkv[:, :, :, 1, :].permute(0, 2, 1, 3)
    v = qkv[:, :, :, 2, :].permute(0, 2, 1, 3)

    ##### Backward equations
    # BMM1
    dU = torch.matmul(dw, v.permute(0, 1, 3, 2))  # this is like QxK'
    # dS = dU / (1-p_dropout)
    dS = dU * D / (1 - p_dropout)
    scale = 1.0 / math.sqrt(d)
    # Reduction
    tmp = (dS * Sref).sum(-1,
                          keepdims=True)  # one reduction, similar to softmax
    dP = (dS - tmp) * Sref * scale  # some elementwise stuff similar to softmax

    dP = dP.to(dtype)
    # BMM4
    #dV = torch.matmul((Sref / (1-p_dropout)).permute(0,1,3,2),dw)
    dV = torch.matmul((Sref * D / (1 - p_dropout)).permute(0, 1, 3, 2), dw)
    # BMM2
    dQ = torch.matmul(dP, k)
    # BMM3
    dK = torch.matmul(dP.permute(0, 1, 3, 2), q)

    ##### Compare equations to PyT reference grad
    dqkv = qkv.grad
    dq = dqkv[:, :, :, 0, :].permute(0, 2, 1, 3).float()
    dk = dqkv[:, :, :, 1, :].permute(0, 2, 1, 3).float()
    dv = dqkv[:, :, :, 2, :].permute(0, 2, 1, 3).float()
    print("= Testing Backward Equations =")
    print("[dQ eq]   AC: {}; MAE: {:.4e}; PERR: {:.4e}".format(
        torch.allclose(dq, dQ.float(), atol=1e-3), mae(dq, dQ), perr(dq, dQ)))
    print("[dK eq]   AC: {}; MAE: {:.4e}; PERR: {:.4e}".format(
        torch.allclose(dk, dK.float(), atol=1e-3), mae(dk, dK), perr(dk, dK)))
    print("[dV eq]   AC: {}; MAE: {:.4e}; PERR: {:.4e}".format(
        torch.allclose(dv, dV.float(), atol=1e-3), mae(dv, dV), perr(dv, dV)))
    print()

    ##### Fused Bwd
    if seqs_interleaved:
        dw2 = dw.permute(2, 0, 1, 3).clone().detach().contiguous()
        ctx = ctx.reshape(b, s, h, d).permute(1, 0, 2, 3).detach().contiguous()
    else:
        dw2 = dw.permute(0, 2, 1, 3).clone().detach().contiguous()

    for _ in range(10):
        dqkv2, _ = mha.bwd(dw2, ctx, qkv_vs, lse_ref, random_seed, cu_seqlens,
                           p_dropout, s, False, seqs_interleaved, is_causal,
                           has_alibi, None)

    torch.cuda.synchronize()
    time0 = time()
    for _ in range(runs):
        dqkv2, dq_acc = mha.bwd(dw2, ctx, qkv_vs, lse_ref, random_seed,
                                cu_seqlens, p_dropout, s, False,
                                seqs_interleaved, is_causal, has_alibi, None)

    torch.cuda.synchronize()
    time1 = time()
    print("Average %f ms for backward pass\n" % ((time1 - time0) / runs * 1000))

    #Convert sum_sxhx3xd => sum_sx3xhxd if heads_interleaved
    if heads_interleaved:
        dqkv2 = dqkv2.permute(0, 2, 1, 3)

    #Convert sum_sx3xhxd => sum_sxhx3xd
    if seqs_interleaved:
        dqkv2 = dqkv2.reshape(s, b, 3, h, d).permute(1, 0, 3, 2,
                                                     4).reshape(b * s, h, 3, d)
    else:
        dqkv2 = dqkv2.permute(0, 2, 1, 3)

    #dp, _ = reshape_softmax(dp_mma, b,s,h,d, warps_m, warps_n, False)
    dqkv2 = pad(dqkv2, cu_seqlens, s)
    dq2 = dqkv2[:, :, :, 0, :].permute(0, 2, 1, 3).float()
    dk2 = dqkv2[:, :, :, 1, :].permute(0, 2, 1, 3).float()
    dv2 = dqkv2[:, :, :, 2, :].permute(0, 2, 1, 3).float()
    print("= Testing Fused Backward =")
    #print("[dP]      AC: {}; MAE: {:.4e}; PERR: {:.4e}".format(torch.allclose(dP.float(), dp.float(), atol=1e-3), mae(dP, dp), perr(dP, dp)))
    print("[dQ]      AC: {}; MAE: {:.4e}; PERR: {:.4e}".format(
        torch.allclose(dq, dq2, atol=1.5e-3), mae(dq, dq2), perr(dq, dq2)))
    print("[dK]      AC: {}; MAE: {:.4e}; PERR: {:.4e}".format(
        torch.allclose(dk, dk2, atol=1.5e-3), mae(dk, dk2), perr(dk, dk2)))
    print("[dV]      AC: {}; MAE: {:.4e}; PERR: {:.4e}".format(
        torch.allclose(dv, dv2, atol=1.5e-3), mae(dv, dv2), perr(dv, dv2)))
    print("[dQKV]    AC: {}; MAE: {:.4e}; PERR: {:.4e}".format(
        torch.allclose(dqkv.float(), dqkv2.float(), atol=1.5e-3),
        mae(dqkv, dqkv2), perr(dqkv, dqkv2)))
    print()

    ## for debug
    #dv_np = dk2.cpu().float().detach().numpy()
    #np.savetxt("dv.txt", dv_np.reshape(-1))
    #dV_np = dk.cpu().float().detach().numpy()
    #np.savetxt("dV.txt", dV_np.reshape(-1))

    return locals()


if __name__ == "__main__":
    all_locals = {}
    # Note that head_interleaved == True or sequence_interleaved == True
    #   needs to generate new kernels with flags on
    for s in (512, ):
        for d in (40, 64, 80, 96, 128):
            all_locals[s] = run_test(s, d, False, False)
