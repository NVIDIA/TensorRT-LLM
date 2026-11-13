# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-row split descriptors (which stream CTA streams which 256-position splits and pages of a row), built on the
device by ``build_desc_device``.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32

from . import topk_select_lib as L

K_TOP = 512
SPLIT_TOK = 256
TILE = 128
MAXS = 8
N_MAX = 1 << 18
PLAN = dict(aim_base=1024, sfac=16, amin=1792, sd_en=0, r_const=64, scap=8192)
DESC_WORDS = 32
MAGIC = 20912


@cute.jit
def _or_reduce(v):
    return Int32(cute.arch.warp_redux_sync(v, "or"))


@cute.kernel
def _desc_kernel(
    mCtx: cute.Tensor,
    mBt: cute.Tensor,
    mDesc: cute.Tensor,
    ns_ctas: cutlass.Constexpr[int],
    rot: cutlass.Constexpr[int] = 0,
    zlast: cutlass.Constexpr[bool] = False,
    zc: cutlass.Constexpr[int] = -1,
):
    c, _, _ = cute.arch.block_idx()
    lane, _, _ = cute.arch.thread_idx()
    out = Int32(0)
    if c < Int32(ns_ctas):
        n = mCtx[0]
        bt_cols = cute.size(mBt, mode=[1])
        n_ok = n
        if n < Int32(0) or n > Int32(N_MAX):
            n_ok = Int32(0)
        nsplit = (n_ok + Int32(SPLIT_TOK - 1)) // Int32(SPLIT_TOK)
        # this row's range is the one of virtual CTA (c + rot) mod ns_ctas
        cv = (c + Int32(rot)) % Int32(ns_ctas)
        s0 = cv * nsplit // Int32(ns_ctas)
        s1 = (cv + Int32(1)) * nsplit // Int32(ns_ctas)
        if cutlass.const_expr(zc >= 0):
            # ranges over ns_ctas - 1 CTAs; CTA zc gets an empty range
            cz_ = c
            if c > Int32(zc):
                cz_ = c - Int32(1)
            s0 = cz_ * nsplit // Int32(ns_ctas - 1)
            s1 = (cz_ + Int32(1)) * nsplit // Int32(ns_ctas - 1)
            if c == Int32(zc):
                s0 = nsplit
                s1 = nsplit
        if cutlass.const_expr(zlast):
            # ranges over ns_ctas - 1 CTAs; CTA ns_ctas - 1 (the early T server) gets an empty range
            s0 = c * nsplit // Int32(ns_ctas - 1)
            s1 = (c + Int32(1)) * nsplit // Int32(ns_ctas - 1)
            if c == Int32(ns_ctas - 1):
                s0 = nsplit
                s1 = nsplit
        ns = s1 - s0
        ntile = (n_ok + Int32(TILE - 1)) // Int32(TILE)
        SMP, SS2, TGT, TGT2 = L.gvr_ladder(
            n_ok,
            Int32(K_TOP),
            Int32(PLAN["aim_base"]),
            Int32(PLAN["sfac"]),
            Int32(PLAN["amin"]),
            Int32(PLAN["sd_en"]),
            PLAN["r_const"],
            PLAN["scap"],
        )
        wst = Int32(8) * SS2
        t = lane
        gt = Int32(2) * s0 + t
        inchunk = t < Int32(2) * ns
        valid = inchunk and gt < ntile
        j0 = (Int32(TILE) * gt + wst - Int32(1)) // wst
        win = valid and j0 < SMP and (j0 * wst < Int32(TILE) * (gt + Int32(1)))
        lm = Int32(cute.arch.lanemask_lt())
        bw = Int32(cute.arch.vote_ballot_sync(win))
        bv = Int32(cute.arch.vote_ballot_sync(valid and (not win)))
        bi = Int32(cute.arch.vote_ballot_sync(inchunk and (not valid)))
        nW = Int32(cute.arch.popc(bw))
        m = Int32(cute.arch.popc(bv))
        nv_ = Int32(cute.arch.popc(bw | bv))
        pos = Int32(0)
        if win:
            pos = Int32(cute.arch.popc(bw & lm))
        elif valid:
            r = m - Int32(1) - Int32(cute.arch.popc(bv & lm))
            k = r % Int32(4)
            base = Int32(0)
            for kk in cutlass.range_constexpr(3):
                ck = (m - Int32(kk) + Int32(3)) // Int32(4)
                if Int32(kk) < k:
                    base = base + ck
            pos = nW + base + r // Int32(4)
        else:
            pos = nW + m + Int32(cute.arch.popc(bi & lm))
        g = pos % Int32(2)
        ti = pos // Int32(2)
        o0 = Int32(0)
        o1 = Int32(0)
        if inchunk:
            if g == Int32(0):
                o0 = t << Int32(4) * ti
            else:
                o1 = t << Int32(4) * ti
        ord0 = _or_reduce(o0)
        ord1 = _or_reduce(o1)
        wl = lane - Int32(8)
        task = wl // Int32(8)
        grp = wl % Int32(8) // Int32(4)
        kp = wl % Int32(4)
        tsel = Int32(0)
        if grp == Int32(0):
            tsel = ord0 >> Int32(4) * task & Int32(15)
        else:
            tsel = ord1 >> Int32(4) * task & Int32(15)
        gts = Int32(2) * s0 + tsel
        pg = Int32(4) * gts + kp
        page = Int32(0)
        if lane >= Int32(8):
            if task < ns:
                if gts < ntile:
                    if pg < bt_cols:
                        page = mBt[0, pg]
        if lane == Int32(0):
            out = s0
        elif lane == Int32(1):
            out = ns
        elif lane == Int32(2):
            out = n
        elif lane == Int32(3):
            out = ord0
        elif lane == Int32(4):
            out = ord1
        elif lane == Int32(5):
            out = nW
        elif lane == Int32(6):
            out = nv_
        elif lane == Int32(7):
            out = Int32(MAGIC) | Int32(ns_ctas << 16)
        else:
            out = page
    mDesc[c * Int32(DESC_WORDS) + lane] = out


@cute.jit
def _launch_desc(
    ctx: cute.Tensor,
    bt: cute.Tensor,
    desc: cute.Tensor,
    num_sms: cutlass.Constexpr[int],
    ns_ctas: cutlass.Constexpr[int],
    rot: cutlass.Constexpr[int],
    zlast: cutlass.Constexpr[bool],
    zc: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    _desc_kernel(ctx, bt, desc, ns_ctas, rot, zlast, zc).launch(
        grid=(num_sms, 1, 1), block=[32, 1, 1], stream=stream
    )


_COMPILED = {}


def compile_desc(num_sms: int, ns_ctas: int, rot: int = 0, zlast: bool = False, zc: int = -1):
    key = (num_sms, ns_ctas, rot, zlast, zc)
    if key not in _COMPILED:
        rt = cute.runtime
        s = cute.sym_int
        fk = [
            rt.make_fake_compact_tensor(cutlass.Int32, (1,), stride_order=(0,)),
            rt.make_fake_compact_tensor(cutlass.Int32, (1, s()), stride_order=(1, 0)),
            rt.make_fake_compact_tensor(cutlass.Int32, (s(),), stride_order=(0,)),
        ]
        _COMPILED[key] = cute.compile(
            _launch_desc,
            *fk,
            num_sms,
            ns_ctas,
            rot,
            zlast,
            zc,
            rt.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def build_desc_device(
    ctx_lens: torch.Tensor,
    bt_row: torch.Tensor,
    desc: torch.Tensor,
    num_sms: int,
    ns_ctas: int,
    rot: int = 0,
    zlast: bool = False,
    zc: int = -1,
):
    """Write one row's split descriptor into ``desc`` on the device from its compressed length and indexer
    block-table row.
    """
    assert desc.numel() >= num_sms * DESC_WORDS and desc.dtype == torch.int32
    compile_desc(num_sms, ns_ctas, rot % ns_ctas, bool(zlast), int(zc))(ctx_lens[0:1], bt_row, desc)


def build_desc_host(
    n: int, bt_row: torch.Tensor, ns_ctas: int, num_sms: int, smp: int, ss2: int
) -> torch.Tensor:
    n_ok = n if 0 <= n <= N_MAX else 0
    nsplit = (n_ok + SPLIT_TOK - 1) // SPLIT_TOK
    ntile = (n_ok + TILE - 1) // TILE
    wst = 8 * ss2
    bt = bt_row.reshape(-1).cpu().tolist()
    d = torch.zeros((num_sms, DESC_WORDS), dtype=torch.int64)
    for c in range(ns_ctas):
        s0 = c * nsplit // ns_ctas
        ns = (c + 1) * nsplit // ns_ctas - s0
        tiles = list(range(2 * ns))
        valid = [2 * s0 + t < ntile for t in tiles]
        win = []
        for t in tiles:
            gt = 2 * s0 + t
            j0 = (TILE * gt + wst - 1) // wst
            win.append(valid[t] and j0 < smp and (j0 * wst < TILE * (gt + 1)))
        W = [t for t in tiles if win[t]]
        R = [t for t in tiles if valid[t] and (not win[t])][::-1]
        R = [x for k in range(4) for x in R[k::4]]
        Inv = [t for t in tiles if not valid[t]]
        order = W + R + Inv
        o = [0, 0]
        for p, t in enumerate(order):
            o[p % 2] |= t << 4 * (p // 2)
        d[c, 0:8] = torch.tensor(
            [s0, ns, n, o[0], o[1], len(W), len(W) + len(R), MAGIC | ns_ctas << 16]
        )
        for task in range(3):
            for g in range(2):
                p = 2 * task + g
                if task < ns and p < len(order):
                    t = order[p]
                    gt = 2 * s0 + t
                    for k in range(4):
                        pg = 4 * gt + k
                        d[c, 8 + 8 * task + 4 * g + k] = (
                            bt[pg] if gt < ntile and pg < len(bt) else 0
                        )
    d = d.to(torch.int64)
    d = torch.where(d >= 2**31, d - 2**32, d)
    return d.to(torch.int32).view(-1)
