# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def push_qkv(
    Q,
    K,
    V,
    O0,
    O1,
    O2,
    O3,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chunk = tl.program_id(0)
    peer = tl.program_id(1)
    kind = tl.program_id(2)
    peer = (peer + R) % 4
    i = chunk * BLOCK + tl.arange(0, BLOCK)
    hp: tl.constexpr = H // 4
    total: tl.constexpr = N * hp * D
    seq = i // (hp * D)
    hd = i % (hp * D)
    if kind == 0:
        val = tl.load(Q + seq * QS + peer * hp * D + hd, i < total, 0)
    elif kind == 1:
        val = tl.load(K + seq * KS + peer * hp * D + hd, i < total, 0)
    else:
        val = tl.load(V + seq * VS + peer * hp * D + hd, i < total, 0)
    dst = (kind * 4 + R) * total + i
    out = tl.where(peer == 0, O0, tl.where(peer == 1, O1, tl.where(peer == 2, O2, O3)))
    tl.store(out + dst, val, i < total)


@triton.jit
def push_output(
    X,
    O0,
    O1,
    O2,
    O3,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chunk = tl.program_id(0)
    peer = tl.program_id(1)
    peer = (peer + R) % 4
    i = chunk * BLOCK + tl.arange(0, BLOCK)
    total: tl.constexpr = N * H * D
    val = tl.load(X + peer * total + i, i < total, 0)
    seq = i // (H * D)
    hd = i % (H * D)
    dst = seq * (H * 4) * D + R * H * D + hd
    out = tl.where(peer == 0, O0, tl.where(peer == 1, O1, tl.where(peer == 2, O2, O3)))
    tl.store(out + dst, val, i < total)
