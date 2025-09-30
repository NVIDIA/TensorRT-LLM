# SPDX-FileCopyrightText: Copyright (c) 1000-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:85% !important; }</style>"))
#import numpy as np
#np.set_printoptions(edgeitems=1000, linewidth=1000000)

import sys

my_path = '/data/projects/fmha_v2/train_ops/build/'
if my_path not in sys.path:
    sys.path.insert(0, my_path)
#my_path = '/data/projects/apex_gitlab/apex/contrib/csrc/fmha/build'
#if my_path not in sys.path:
#    sys.path.insert(0, my_path)

import math

import apex_mha
import bert_mha_train as mha
import numpy as np
import torch

#import fmhalib as mha

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


class GPUTimer:

    def __init__(self, stream):
        self.start_ = torch.cuda.Event(enable_timing=True)
        self.stop_ = torch.cuda.Event(enable_timing=True)
        self.stream_ = stream

    def start(self):
        self.stream_.record_event(self.start_)

    def stop(self):
        self.stream_.record_event(self.stop_)

    def sync(self):
        self.stream_.synchronize()

    def millis(self):
        return self.start_.elapsed_time(self.stop_)


def reshape_softmax(S, b, s, h, d, warps_m, warps_n):
    m = s if s == 128 else 16
    n = s
    m_per_cta = warps_m * 16
    n_per_cta = warps_n * 16
    mmas_m = m // m_per_cta
    mmas_n = n // n_per_cta
    loops = s // (mmas_m * m_per_cta)
    print(loops, )
    assert (loops == 1 and s == 128) or (loops == 16 and s == 256) or (
        loops == 32 and s == 512) or (loops == 24 and s == 384), "no.."
    quads = 8
    lohi = 2
    lr = 2
    vals = 2
    # B x H x LOOPS x MMAS_M x MMAS_N x THREADS_PER_CTA x LOHI x LR x 2
    # B x H x LOOPS x MMAS_M x MMAS_N x WARPS_N x WARPS_M x 32 x LOHI x LR x 2
    # 0   1       2        3        4         5         6       7   8      9   10  11
    # B x H x LOOPS x MMAS_M x MMAS_N x WARPS_N x WARPS_M x QUADS x 4 x LOHI x LR x 2
    # 0   1       2        3         6      9       7        4         5   10   8  11
    # B x H x LOOPS x MMAS_M x WARPS_M x LOHI x QUADS x MMAS_N x WARPS_N x LR x 4 x 2

    S = S.reshape((b, h, loops, mmas_m, mmas_n, warps_n, warps_m, quads, 4,
                   lohi, lr, vals)).permute(0, 1, 2, 3, 6, 9, 7, 4, 5, 10, 8,
                                            11).reshape((b, h, s, s))
    Snp = S.cpu().numpy()
    dmask = torch.tensor(np.logical_not(np.signbit(Snp)),
                         dtype=torch.float32,
                         device=device)
    S = S.abs()
    return S, dmask


def mha_ref(qkv, D, b, s, h, d, p_dropout):
    qkv = qkv.view(b, s, h, 3, d)
    q = qkv[:, :, :, 0, :].permute(0, 2, 1, 3)
    k = qkv[:, :, :, 1, :].permute(0, 2, 1, 3)
    v = qkv[:, :, :, 2, :].permute(0, 2, 1, 3)
    p = torch.matmul(q.float(), k.permute(0, 1, 3, 2).float()) / math.sqrt(d)
    s = torch.softmax(p, -1).half()
    d = s * D.half() * (1 / (1 - p_dropout))
    #d = s
    ctx = torch.matmul(d, v)

    return ctx, p, s


runs = 1

s = 512
b = 32

warps_m = 1
warps_n = 4
if s == 256:
    runs == 20000
if s == 384:
    runs == 20000
elif s == 512:
    warps_n = 8
    runs == 5000
elif s == 128:
    runs = 20000

#runs = 1

h = 16
d = 64
p_dropout = 0.1
dtype = torch.float16
device = torch.device('cuda')

if b <= 4: runs *= 10

#s_valid = int(s * 0.5)
#s_valid = int(s * 0.97)
s_valid = 246  # average per batch
s_valid = s

#slens = [s_valid] * b
#a = torch.tensor(np.array([0] + slens), dtype=torch.int32)
#cu_seqlens =torch.cumsum(a, 0).to(dtype=torch.int32, device=device)

seqlens = torch.linspace(1, s, b, dtype=torch.int32, device=device)
seqlens = torch.ones(b) * s

cu_seqlens = torch.zeros(b + 1, dtype=torch.int32, device=device)
cu_seqlens[1:] = torch.cumsum(seqlens, 0)
total = cu_seqlens[-1].item()
print(seqlens)
print(cu_seqlens)

assert cu_seqlens.numel() == b + 1, "ahh"

qkv = torch.randn((b, s, h, 3, d), device=device, dtype=dtype)
#qkv_vs = qkv[:, :s_valid, :, : ,:].contiguous().view(total, h, 3, d).permute(0,2,1,3).contiguous()
qkv_vs = torch.empty((total, h, 3, d), dtype=dtype, device=device)
for bi in range(b):
    begin = cu_seqlens[bi]
    end = cu_seqlens[bi + 1]
    si = end - begin
    qkv_vs[begin:end, ...] = qkv[bi, :si, ...]
qkv_vs = qkv_vs.contiguous().view(total, h, 3, d).permute(0, 2, 1,
                                                          3).contiguous()
qkvt = qkv.view((b, s, h, 3, d)).permute(1, 0, 2, 3, 4).contiguous()
mask = torch.ones((b, s), device=device, dtype=dtype)

stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    assert torch.cuda.current_stream() == stream

    timer = GPUTimer(stream)

    for it in range(runs):
        Pa, Sa, Da, Ca = apex_mha.fwd(qkvt, mask, p_dropout)
    is_nl = b < 4
    is_training = True

    timer.start()
    for it in range(runs):
        ctx, S = mha.fwd(qkv_vs, cu_seqlens, p_dropout, s, is_training, is_nl,
                         None)
    timer.stop()
    timer.sync()
    ms_fused = timer.millis() / runs
    # S will be overwritten in the backward pass, so reshape it here already.
    Snew, D = reshape_softmax(S, b, s, h, d, warps_m, warps_n)

    timer.start()
    for it in range(runs):
        Pa, Sa, Da, Ca = apex_mha.fwd(qkvt, mask, p_dropout)
    timer.stop()
    timer.sync()
    ms_apex = timer.millis() / runs

    print(Ca.shape, qkvt.shape)
    timer.start()
    for it in range(runs):
        dqkv_a, dU_a = apex_mha.bwd(h, Ca, Ca, Sa, Pa, mask, qkvt, Da,
                                    p_dropout)
    timer.stop()
    timer.sync()
    ms_apex_bwd = timer.millis() / runs

    timer.start()
    for it in range(runs):
        if b < 4 and b > 1:
            _ = mha.bwd_nl(ctx, qkv_vs, S, cu_seqlens, p_dropout, s)
        else:
            dqkv2, dp_mma = mha.bwd(ctx, qkv_vs, S, cu_seqlens, p_dropout, s)
    timer.stop()
    timer.sync()
    ms_fused_bwd = timer.millis() / runs

ctx_ref, Pref, Sref = mha_ref(qkv, D, b, s, h, d, p_dropout)
ctx_ref = ctx_ref.permute(0, 2, 1, 3)
#ctx = ctx.view((b,s,h,d))

ctx_pad = torch.zeros((b, s, h, d), dtype=dtype, device=device)
for bi in range(b):
    begin = cu_seqlens[bi]
    end = cu_seqlens[bi + 1]
    si = end - begin
    ctx_pad[bi, :si, ...] = ctx[begin:end, ...]

print(torch.allclose(Snew.float(), Sref.float(), atol=1e-4))
print(torch.allclose(ctx_ref.float(), ctx_pad.float(), atol=1e-3))

print(
    '[FWD s={:d}, b={:d}] Fused {:.3f}ms Apex {:.3f}ms Diff {:.3f}ms Speedup {:.2f}x'
    .format(s, b, ms_fused, ms_apex, ms_apex - ms_fused, ms_apex / ms_fused))
print(
    '[BWD s={:d}, b={:d}] Fused {:.3f}ms Apex {:.3f}ms Diff {:.3f}ms Speedup {:.2f}x'
    .format(s, b, ms_fused_bwd, ms_apex_bwd, ms_apex_bwd - ms_fused_bwd,
            ms_apex_bwd / ms_fused_bwd))
