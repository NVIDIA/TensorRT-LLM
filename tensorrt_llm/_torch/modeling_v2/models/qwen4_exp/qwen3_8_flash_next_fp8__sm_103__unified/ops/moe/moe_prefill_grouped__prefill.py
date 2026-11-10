# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A prefill chunk's routed experts over a rank's window, pre-routed: the rows permuted into
expert-major slots padded to 128-row M tiles, FC1 and FC2 as contiguous grouped FP8 block-scale
GEMMs (`blockwise_gemm_grouped`, the checkpoint's 1x128 / 128x128 scales as they are), then each
row's routing-weighted sum over its slots (fp32, a fixed tree).

No host synchronization, so a call is CUDA-graph capturable: buffers are sized for the worst case
of the call's rows and the GEMMs skip tiles past the live count. Slot ranks depend on scheduling,
the results do not."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HIDDEN = 2560
INTER = 640
FC1_ROWS = 2 * INTER
TOPK = 10
BLOCK = 128
#: Rows of a grouped GEMM's M tile; every expert's rows are padded to a multiple of it.
M_TILE = 128
E4M3_MAX = 448.0
#: The runner's floor of a 1x128 block's amax before its scale (`fp8_block_scale_moe_runner`).
ACT_AMAX_FLOOR = 1.0e-10
#: Columns of a finalize chunk (`_finalize_kernel`, one row a program).
FIN_BH = 512
#: Rows of an activation program (`_act_q_kernel`).
ACT_BM = 16
#: Counter stride (int32 words) of `_count_kernel`'s atomics: one 128-byte line an expert.
CNT_STRIDE = 32
#: Whether the finalize keeps two chunks in flight (`_finalize_pipe_kernel`, the same sums).
FIN_PIPE = True

#: (mma_tiler_mn, cluster_shape_mn) of FC1 / FC2, on 1-CTA instructions.
TACTICS = (((128, 128), (1, 1)), ((128, 128), (1, 1)))
#: FC1's epilogue stores the 1x128 e4m3 round trip (`epi_quant`), not fp32: same bits, 1/4 bytes.
EPI_QUANT = True


@triton.jit(do_not_specialize=["rows", "offset", "local"])
def _count_kernel(
    ids_ptr,
    cnt_ptr,
    rank_ptr,
    rows,
    offset,
    local,
    K: tl.constexpr,
    KP: tl.constexpr,
    RB: tl.constexpr,
    CS: tl.constexpr,
):
    """rank[r, k] = the next free rank of expert ids[r, k] (atomic), or -1 outside the window."""
    r = tl.program_id(0) * RB + tl.arange(0, RB)
    kk = tl.arange(0, KP)
    m = (r < rows)[:, None] & (kk < K)[None, :]
    e = tl.load(ids_ptr + r[:, None] * K + kk[None, :], mask=m, other=-1) - offset
    inw = m & (e >= 0) & (e < local)
    ec = tl.where(inw, e, 0)
    pos = tl.atomic_add(cnt_ptr + ec * CS, tl.where(inw, 1, 0), mask=inw, sem="relaxed")
    tl.store(rank_ptr + r[:, None] * K + kk[None, :], tl.where(inw, pos, -1), mask=m)


@triton.jit(do_not_specialize=["local"])
def _layout_kernel(
    cnt_ptr,
    cnt_out_ptr,
    roff_ptr,
    group_ptr,
    ntiles_ptr,
    local,
    LP: tl.constexpr,
    TB: tl.constexpr,
    MT: tl.constexpr,
    CS: tl.constexpr,
):
    """Program e: expert e's rows start at roff[e] (a multiple of MT), its M tiles name it in
    `group`; program 0 stores the live tile count; with CS > 1, e's count goes to `cnt_out`."""
    pe = tl.program_id(0)
    e = tl.arange(0, LP)
    c = tl.load(cnt_ptr + e * CS, mask=e < local, other=0)
    if CS > 1:
        tl.store(cnt_out_ptr + pe, tl.sum(tl.where(e == pe, c, 0), axis=0))
    t = (c + MT - 1) // MT
    tend = tl.cumsum(t, axis=0)
    mine = tl.sum(tl.where(e == pe, t, 0), axis=0)
    start = tl.sum(tl.where(e == pe, tend, 0), axis=0) - mine
    tl.store(roff_ptr + pe, start * MT)
    if pe == 0:
        tl.store(ntiles_ptr, tl.sum(t, axis=0))
    for j0 in range(0, mine, TB):
        j = j0 + tl.arange(0, TB)
        tl.store(group_ptr + start + j, tl.full([TB], 0, tl.int32) + pe, mask=j < mine)


@triton.jit(do_not_specialize=["xs_kb", "xs_t", "sf_m", "offset", "local"])
def _permute_kernel(
    xq_ptr,
    xsf_ptr,
    xs_kb,
    xs_t,
    ids_ptr,
    rank_ptr,
    roff_ptr,
    a_ptr,
    sfa_ptr,
    sf_m,
    offset,
    local,
    K: tl.constexpr,
    HW: tl.constexpr,
    HWP: tl.constexpr,
    NKB: tl.constexpr,
    NKBP: tl.constexpr,
):
    """Row t's e4m3 codes and NKB scales to each of its window slots (row roff[e] + rank)."""
    t = tl.program_id(0).to(tl.int64)
    w = tl.arange(0, HWP)
    wm = w < HW
    v = tl.load(xq_ptr + t * HW + w, mask=wm, other=0)
    kb = tl.arange(0, NKBP)
    kbm = kb < NKB
    sc = tl.load(xsf_ptr + kb * xs_kb + t * xs_t, mask=kbm, other=0.0)
    for k in tl.static_range(K):
        e = tl.load(ids_ptr + t * K + k) - offset
        if (e >= 0) & (e < local):
            dst = (tl.load(roff_ptr + e) + tl.load(rank_ptr + t * K + k)).to(tl.int64)
            tl.store(a_ptr + dst * HW + w, v, mask=wm)
            tl.store(sfa_ptr + kb.to(tl.int64) * sf_m + dst, sc, mask=kbm)


@triton.jit
def _requant(x, AMAX_MIN: tl.constexpr, QMAX: tl.constexpr):
    """1x128 e4m3 round trip along axis 1."""
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), AMAX_MIN)
    sc = tl.math.div_rn(amax, tl.full(amax.shape, QMAX, tl.float32))
    inv = tl.math.div_rn(tl.full(amax.shape, QMAX, tl.float32), amax)
    q = tl.minimum(tl.maximum(x * inv[:, None], -QMAX), QMAX).to(tl.float8e4nv)
    return q, sc


@triton.jit(do_not_specialize=["sf_m"])
def _act_kernel(
    c1_ptr,
    group_ptr,
    roff_ptr,
    cnt_ptr,
    ntiles_ptr,
    a2_ptr,
    sfa2_ptr,
    sf_m,
    I_: tl.constexpr,
    NB: tl.constexpr,
    NBP: tl.constexpr,
    QB: tl.constexpr,
    BM: tl.constexpr,
    MT: tl.constexpr,
    AMAX_MIN: tl.constexpr,
    QMAX: tl.constexpr,
):
    """BM rows of one M tile: fp32 up / gate -> e4m3 round trip -> silu(gate) * up -> e4m3 codes
    `a2[r, I_]` and M-major scales `sfa2[b, r]`; rows and tiles past the live ones skipped."""
    r0 = tl.program_id(0) * BM
    tile = r0 // MT
    if tile < tl.load(ntiles_ptr):
        g = tl.load(group_ptr + tile)
        lim = tl.load(roff_ptr + g) + tl.load(cnt_ptr + g)
        if r0 < lim:
            r2 = tl.arange(0, BM * NBP)
            r = r0 + r2 // NBP
            b = r2 % NBP
            ok = (r < lim) & (b < NB)
            row = r.to(tl.int64)
            cols = tl.arange(0, QB)
            off = row[:, None] * (2 * I_) + b[:, None] * QB + cols[None, :]
            up = tl.load(c1_ptr + off, mask=ok[:, None], other=0.0)
            gate = tl.load(c1_ptr + off + I_, mask=ok[:, None], other=0.0)
            uq, us = _requant(up, AMAX_MIN, QMAX)
            gq, gs = _requant(gate, AMAX_MIN, QMAX)
            u = uq.to(tl.float32) * us[:, None]
            gv = gq.to(tl.float32) * gs[:, None]
            h = gv * tl.sigmoid(gv) * u
            hq, hs = _requant(h, AMAX_MIN, QMAX)
            tl.store(
                a2_ptr + row[:, None] * I_ + b[:, None] * QB + cols[None, :],
                hq.to(tl.uint8, bitcast=True),
                mask=ok[:, None],
            )
            tl.store(sfa2_ptr + b.to(tl.int64) * sf_m + row, hs, mask=ok)


@triton.jit(do_not_specialize=["sf_m"])
def _act_q_kernel(
    q1_ptr,
    s1_ptr,
    group_ptr,
    roff_ptr,
    cnt_ptr,
    ntiles_ptr,
    a2_ptr,
    sfa2_ptr,
    sf_m,
    I_: tl.constexpr,
    NB: tl.constexpr,
    QB: tl.constexpr,
    BM: tl.constexpr,
    MT: tl.constexpr,
    AMAX_MIN: tl.constexpr,
    QMAX: tl.constexpr,
):
    """`_act_kernel` over FC1's stored codes `q1[r, 2 I_]` and M-major scales `s1[b, r]`."""
    r0 = tl.program_id(0) * BM
    tile = r0 // MT
    if tile < tl.load(ntiles_ptr):
        g = tl.load(group_ptr + tile)
        lim = tl.load(roff_ptr + g) + tl.load(cnt_ptr + g)
        if r0 < lim:
            r = r0 + tl.arange(0, BM)
            ok = r < lim
            row = r.to(tl.int64)
            cols = tl.arange(0, QB)
            for b in tl.static_range(NB):
                off = row[:, None] * (2 * I_) + b * QB + cols[None, :]
                uq = tl.load(q1_ptr + off, mask=ok[:, None], other=0).to(
                    tl.float8e4nv, bitcast=True
                )
                gq = tl.load(q1_ptr + off + I_, mask=ok[:, None], other=0).to(
                    tl.float8e4nv, bitcast=True
                )
                us = tl.load(s1_ptr + b * sf_m + row, mask=ok, other=0.0)
                gs = tl.load(s1_ptr + (b + NB) * sf_m + row, mask=ok, other=0.0)
                u = uq.to(tl.float32) * us[:, None]
                gv = gq.to(tl.float32) * gs[:, None]
                h = gv * tl.sigmoid(gv) * u
                hq, hs = _requant(h, AMAX_MIN, QMAX)
                tl.store(
                    a2_ptr + row[:, None] * I_ + b * QB + cols[None, :],
                    hq.to(tl.uint8, bitcast=True),
                    mask=ok[:, None],
                )
                tl.store(sfa2_ptr + b * sf_m + row, hs, mask=ok)


@triton.jit(do_not_specialize=["offset", "local", "row0"])
def _finalize_kernel(
    c2_ptr,
    ids_ptr,
    w_ptr,
    rank_ptr,
    roff_ptr,
    out_ptr,
    offset,
    local,
    row0,
    H: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    BH: tl.constexpr,
    NCH: tl.constexpr,
):
    """Row t = row0 + program: its window slots' FC2 rows times their routing weights, summed in
    fp32 (a fixed tree), stored bf16 (zero with no slot in the window). A row reads only its own
    slots, so a launch over any row range writes the whole launch's bits there."""
    t = row0 + tl.program_id(0).to(tl.int64)
    kk = tl.arange(0, KP)
    km = kk < K
    e = tl.load(ids_ptr + t * K + kk, mask=km, other=-1) - offset
    inw = km & (e >= 0) & (e < local)
    ec = tl.where(inw, e, 0)
    rows = (
        tl.load(roff_ptr + ec, mask=inw, other=0)
        + tl.load(rank_ptr + t * K + kk, mask=inw, other=0)
    ).to(tl.int64)
    w = tl.where(inw, tl.load(w_ptr + t * K + kk, mask=km, other=0.0).to(tl.float32), 0.0)
    for ch in tl.static_range(NCH):
        cols = ch * BH + tl.arange(0, BH)
        cm = cols < H
        v = tl.load(
            c2_ptr + rows[:, None] * H + cols[None, :], mask=inw[:, None] & cm[None, :], other=0.0
        )
        acc = tl.sum(v.to(tl.float32) * w[:, None], axis=0)
        tl.store(out_ptr + t * H + cols, acc.to(tl.bfloat16), mask=cm)


@triton.jit(do_not_specialize=["offset", "local", "row0"])
def _finalize_pipe_kernel(
    c2_ptr,
    ids_ptr,
    w_ptr,
    rank_ptr,
    roff_ptr,
    out_ptr,
    offset,
    local,
    row0,
    H: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    BH: tl.constexpr,
    NCH: tl.constexpr,
):
    """`_finalize_kernel` with chunk c + 1's loads issued before chunk c's store; the same sums."""
    t = row0 + tl.program_id(0).to(tl.int64)
    kk = tl.arange(0, KP)
    km = kk < K
    e = tl.load(ids_ptr + t * K + kk, mask=km, other=-1) - offset
    inw = km & (e >= 0) & (e < local)
    ec = tl.where(inw, e, 0)
    rows = (
        tl.load(roff_ptr + ec, mask=inw, other=0)
        + tl.load(rank_ptr + t * K + kk, mask=inw, other=0)
    ).to(tl.int64)
    w = tl.where(inw, tl.load(w_ptr + t * K + kk, mask=km, other=0.0).to(tl.float32), 0.0)
    cols = tl.arange(0, BH)
    v = tl.load(
        c2_ptr + rows[:, None] * H + cols[None, :],
        mask=inw[:, None] & (cols < H)[None, :],
        other=0.0,
    )
    for ch in tl.static_range(1, NCH):
        nc = ch * BH + tl.arange(0, BH)
        vn = tl.load(
            c2_ptr + rows[:, None] * H + nc[None, :],
            mask=inw[:, None] & (nc < H)[None, :],
            other=0.0,
        )
        pc = (ch - 1) * BH + tl.arange(0, BH)
        acc = tl.sum(v.to(tl.float32) * w[:, None], axis=0)
        tl.store(out_ptr + t * H + pc, acc.to(tl.bfloat16), mask=pc < H)
        v = vn
    lc = (NCH - 1) * BH + tl.arange(0, BH)
    acc = tl.sum(v.to(tl.float32) * w[:, None], axis=0)
    tl.store(out_ptr + t * H + lc, acc.to(tl.bfloat16), mask=lc < H)


_GEMMS: dict = {}


def _grouped_gemm(a, w, sfa, sfw, c, group, ntiles, tiles, sf_m, tactic):
    """c[r, n] = sum_k a[r, k] w[group(r // M_TILE), n, k] with a's 1x128 (M-major `sfa[kb, r]`)
    and w's 128x128 fp32 scales, over `tiles` M tiles of which the first `ntiles[0]` are live."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_ptr

    from .blockwise_gemm_grouped import Sm100BlockwiseGemmKernel

    groups, n, k = w.shape
    sf_n, sf_k = -(-n // BLOCK), -(-k // BLOCK)
    tiler, cluster = tactic
    c3 = c.view(tiles, M_TILE, n).permute(1, 2, 0)
    key = (c.dtype, tactic)
    fn = _GEMMS.get(key)
    if fn is None:
        gemm = Sm100BlockwiseGemmKernel(mma_tiler_mn=tiler, cluster_shape_mn=cluster)
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])

        def ptr(t, dt, align=16):
            return make_ptr(dt, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

        fn = cute.compile(
            gemm.wrapper_grouped,
            tiles,
            n,
            k,
            sf_m,
            sf_n,
            sf_k,
            groups,
            ptr(a, cutlass.Float8E4M3FN),
            ptr(w, cutlass.Float8E4M3FN),
            ptr(sfa, cutlass.Float32),
            ptr(sfw, cutlass.Float32),
            from_dlpack(c3).mark_layout_dynamic(leading_dim=1),
            ptr(group, cutlass.Int32, 4),
            ptr(ntiles, cutlass.Int32, 4),
            max_active_clusters=mac,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _GEMMS[key] = fn
    fn(
        tiles,
        n,
        k,
        sf_m,
        sf_n,
        sf_k,
        groups,
        a.data_ptr(),
        w.data_ptr(),
        sfa.data_ptr(),
        sfw.data_ptr(),
        c3,
        group.data_ptr(),
        ntiles.data_ptr(),
    )


def _grouped_gemm_q(a, w, sfa, sfw, cq, cs, group, ntiles, tiles, sf_m, tactic):
    """`_grouped_gemm` with the 1x128 e4m3 round trip in the epilogue (`epi_quant`): `cq[r, n]`
    e4m3 codes, `cs[n // 128, r]` fp32 scales (row-major `[n / 128, sf_m]`)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_ptr

    from .blockwise_gemm_grouped import Sm100BlockwiseGemmKernel

    groups, n, k = w.shape
    sf_n, sf_k = -(-n // BLOCK), -(-k // BLOCK)
    key = ("q", tactic)
    fn = _GEMMS.get(key)
    if fn is None:
        tiler, cluster = tactic
        gemm = Sm100BlockwiseGemmKernel(
            mma_tiler_mn=tiler, cluster_shape_mn=cluster, epi_quant=True
        )
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])

        def ptr(t, dt, align=16):
            return make_ptr(dt, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

        fn = cute.compile(
            gemm.wrapper_grouped_quant,
            tiles,
            n,
            k,
            sf_m,
            sf_n,
            sf_k,
            groups,
            ptr(a, cutlass.Float8E4M3FN),
            ptr(w, cutlass.Float8E4M3FN),
            ptr(sfa, cutlass.Float32),
            ptr(sfw, cutlass.Float32),
            ptr(cq, cutlass.Float8E4M3FN),
            from_dlpack(cs).mark_layout_dynamic(leading_dim=1),
            ptr(group, cutlass.Int32, 4),
            ptr(ntiles, cutlass.Int32, 4),
            max_active_clusters=mac,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _GEMMS[key] = fn
    fn(
        tiles,
        n,
        k,
        sf_m,
        sf_n,
        sf_k,
        groups,
        a.data_ptr(),
        w.data_ptr(),
        sfa.data_ptr(),
        sfw.data_ptr(),
        cq.data_ptr(),
        cs,
        group.data_ptr(),
        ntiles.data_ptr(),
    )


def tile_bound(rows: int, local: int, topk: int = TOPK) -> int:
    """M tiles that hold any routing of `rows` rows over a `local`-expert window."""
    return -(-rows * min(topk, local) // M_TILE) + local


def _plan(xq, xsf, ids, wts, fc1, fc1_scale, fc2, fc2_scale, offset, tactics=None, fin_first=None):
    """One call's buffers and its launches in order as (name, launch) pairs, the last writing `out`;
    `fin_first = (r0, r1)` finalizes rows `[r0, r1)` before the rest (the same bits)."""
    t1, t2 = tactics or TACTICS
    rows = xq.shape[0]
    local = fc1.shape[0]
    dev = xq.device
    assert xq.shape[1] == HIDDEN and xq.is_contiguous() and ids.shape == (rows, TOPK)
    ids = ids.contiguous()
    wts = wts.contiguous()
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    if rows == 0:
        return out, []
    tiles = tile_bound(rows, local)
    m = tiles * M_TILE
    cs = CNT_STRIDE
    cntp = torch.empty(local * cs, dtype=torch.int32, device=dev)
    cnt = cntp if cs == 1 else torch.empty(local, dtype=torch.int32, device=dev)
    rank = torch.empty(rows, TOPK, dtype=torch.int32, device=dev)
    roff = torch.empty(local, dtype=torch.int32, device=dev)
    group = torch.empty(tiles, dtype=torch.int32, device=dev)
    ntiles = torch.empty(1, dtype=torch.int32, device=dev)
    # A (later the FC2 operand) and C (FC1's codes or fp32 accumulator, later FC2's bf16) are
    # reused in place: each is dead by the time its second use writes it (one stream).
    a = torch.empty(m * HIDDEN, dtype=torch.uint8, device=dev)
    sfa = torch.empty((HIDDEN // BLOCK) * m, dtype=torch.float32, device=dev)
    c = torch.empty(m * FC1_ROWS, dtype=torch.float32, device=dev)
    a1 = a.view(torch.float8_e4m3fn).view(m, HIDDEN)
    a2 = a[: m * INTER].view(m, INTER)
    sfa2 = sfa[: (INTER // BLOCK) * m].view(INTER // BLOCK, m)
    c2 = c.view(torch.bfloat16)[: m * HIDDEN].view(m, HIDDEN)
    epiq = EPI_QUANT and t1[0][1] == BLOCK
    q1 = c.view(torch.uint8)[: m * FC1_ROWS].view(m, FC1_ROWS)
    s1q = torch.empty(FC1_ROWS // BLOCK, m, dtype=torch.float32, device=dev) if epiq else None
    c1 = c.view(m, FC1_ROWS)
    rb = 16
    bm = 4

    def count():
        cntp.zero_()
        _count_kernel[(triton.cdiv(rows, rb),)](
            ids, cntp, rank, rows, offset, local, K=TOPK, KP=16, RB=rb, CS=cs, num_warps=4
        )

    def layout():
        _layout_kernel[(local,)](
            cntp,
            cnt,
            roff,
            group,
            ntiles,
            local,
            LP=triton.next_power_of_2(local),
            TB=128,
            MT=M_TILE,
            CS=cs,
            num_warps=4,
        )

    def permute():
        _permute_kernel[(rows,)](
            xq.view(torch.int32),
            xsf,
            xsf.stride(0),
            xsf.stride(1),
            ids,
            rank,
            roff,
            a.view(torch.int32),
            sfa,
            m,
            offset,
            local,
            K=TOPK,
            HW=HIDDEN // 4,
            HWP=triton.next_power_of_2(HIDDEN // 4),
            NKB=HIDDEN // BLOCK,
            NKBP=32,
            num_warps=4,
        )

    def fc1g():
        if epiq:
            _grouped_gemm_q(
                a1,
                fc1,
                sfa.view(HIDDEN // BLOCK, m),
                fc1_scale,
                q1,
                s1q,
                group,
                ntiles,
                tiles,
                m,
                t1,
            )
        else:
            _grouped_gemm(
                a1, fc1, sfa.view(HIDDEN // BLOCK, m), fc1_scale, c1, group, ntiles, tiles, m, t1
            )

    def act():
        if epiq:
            _act_q_kernel[(m // ACT_BM,)](
                q1,
                s1q,
                group,
                roff,
                cnt,
                ntiles,
                a2,
                sfa2,
                m,
                I_=INTER,
                NB=INTER // BLOCK,
                QB=BLOCK,
                BM=ACT_BM,
                MT=M_TILE,
                AMAX_MIN=ACT_AMAX_FLOOR,
                QMAX=E4M3_MAX,
                num_warps=4,
            )
        else:
            _act_kernel[(m // bm,)](
                c1,
                group,
                roff,
                cnt,
                ntiles,
                a2,
                sfa2,
                m,
                I_=INTER,
                NB=INTER // BLOCK,
                NBP=8,
                QB=BLOCK,
                BM=bm,
                MT=M_TILE,
                AMAX_MIN=ACT_AMAX_FLOOR,
                QMAX=E4M3_MAX,
                num_warps=4,
            )

    def fc2g():
        _grouped_gemm(
            a2.view(torch.float8_e4m3fn), fc2, sfa2, fc2_scale, c2, group, ntiles, tiles, m, t2
        )

    fin = _finalize_pipe_kernel if FIN_PIPE else _finalize_kernel

    def finalize_rows(r0, r1):
        if r1 <= r0:
            return
        fin[(r1 - r0,)](
            c2,
            ids,
            wts,
            rank,
            roff,
            out,
            offset,
            local,
            r0,
            H=HIDDEN,
            K=TOPK,
            KP=16,
            BH=FIN_BH,
            NCH=triton.cdiv(HIDDEN, FIN_BH),
            num_warps=4,
        )

    stages = [
        ("count", count),
        ("layout", layout),
        ("permute", permute),
        ("fc1", fc1g),
        ("act", act),
        ("fc2", fc2g),
    ]
    if fin_first is None:
        return out, stages + [("finalize", lambda: finalize_rows(0, rows))]
    r0, r1 = fin_first
    assert 0 <= r0 <= r1 <= rows and (r0 == 0 or r1 == rows), (fin_first, rows)
    rest = (r1, rows) if r0 == 0 else (0, r0)
    return out, stages + [
        ("finalize_first", lambda: finalize_rows(r0, r1)),
        ("finalize_rest", lambda: finalize_rows(*rest)),
    ]


def moe_prefill_grouped(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    ids: torch.Tensor,
    wts: torch.Tensor,
    fc1: torch.Tensor,
    fc1_scale: torch.Tensor,
    fc2: torch.Tensor,
    fc2_scale: torch.Tensor,
    offset: int,
    tactics=None,
    fin_first=None,
    after_first=None,
) -> torch.Tensor:
    """This window's partial of the routed experts, `[rows, HIDDEN]` bf16: `xq` `[rows, HIDDEN]`
    e4m3 (contiguous), `xsf` `[HIDDEN / 128, rows]` fp32, `ids` / `wts` `[rows, TOPK]` over the
    whole routing space (ids outside `[offset, offset + local)` add nothing), the window's `fc1`
    `[local, 2I, H]` / `fc2` `[local, H, I]` e4m3 with fp32 128x128 scales. `fin_first = (r0, r1)`
    (rows at either end) finalizes those rows, calls `after_first(out)`, then finalizes the rest."""
    out, stages = _plan(
        xq, xsf, ids, wts, fc1, fc1_scale, fc2, fc2_scale, offset, tactics, fin_first
    )
    for name, launch in stages:
        launch()
        if name == "finalize_first" and after_first is not None:
            after_first(out)
    return out


_READY: set = set()


def prefill_grouped_ready(local: int) -> bool:
    """Whether `warm_prefill_grouped` compiled this window's kernels (a call can be captured)."""
    return (TACTICS, local, EPI_QUANT) in _READY


def warm_prefill_grouped(device, local: int) -> None:
    """Compile every kernel of `moe_prefill_grouped` for a `local`-expert window with one eager
    call whose rows all go to expert 0 (the GEMMs' shapes are dynamic). Idempotent."""
    if (TACTICS, local, EPI_QUANT) in _READY:
        return
    dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    rows = 2 * M_TILE
    fc1 = torch.zeros(1, FC1_ROWS, HIDDEN, dtype=torch.float8_e4m3fn, device=dev).expand(
        local, -1, -1
    )
    fc2 = torch.zeros(1, HIDDEN, INTER, dtype=torch.float8_e4m3fn, device=dev).expand(local, -1, -1)
    s1 = torch.ones(1, FC1_ROWS // BLOCK, HIDDEN // BLOCK, dtype=torch.float32, device=dev).expand(
        local, -1, -1
    )
    s2 = torch.ones(1, HIDDEN // BLOCK, INTER // BLOCK, dtype=torch.float32, device=dev).expand(
        local, -1, -1
    )
    xq = torch.zeros(rows, HIDDEN, dtype=torch.float8_e4m3fn, device=dev)
    xsf = torch.ones(HIDDEN // BLOCK, rows, dtype=torch.float32, device=dev)
    ids = torch.zeros(rows, TOPK, dtype=torch.int32, device=dev)
    ids[:, 1:] = -1
    wts = torch.ones(rows, TOPK, dtype=torch.bfloat16, device=dev)
    moe_prefill_grouped(xq, xsf, ids, wts, fc1, s1, fc2, s2, 0)
    torch.cuda.synchronize(dev)
    _READY.add((TACTICS, local, EPI_QUANT))
