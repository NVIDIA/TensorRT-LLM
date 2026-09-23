# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import AddressSpace

DIM = 128
VEC = 8
WVEC = VEC // 4
NCH = DIM // VEC
NT = 256
MW = DIM // 32
NW = NT // 32 // MW
MSTEP = NT // NCH
STRIPW = 64
NSTRIP = DIM // STRIPW
NJ = DIM * NCH // NT
PAD = 136
LOG2E = 1.4426950408889634
CTAS_PER_SM = 2


def _load_x_vec(gX, layer, position, token, chunk):
    r = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
    cute.autovec_copy(
        cute.recast_tensor(gX[layer, position, token, chunk, None], cutlass.BFloat16),
        r,
    )
    return r


def _interleave3_store(a, b, c, dst):
    out = cute.make_rmem_tensor((3 * VEC,), cutlass.BFloat16)
    for q in range(VEC):
        out[3 * q] = a[q]
        out[3 * q + 1] = b[q]
        out[3 * q + 2] = c[q]
    cute.autovec_copy(out, dst)


def _conv_accept_one(gX, layer, position, chunk, dst):
    old = cute.make_rmem_tensor((3 * VEC,), cutlass.BFloat16)
    cute.autovec_copy(dst, old)
    x0 = _load_x_vec(gX, layer, position, cutlass.Int32(0), chunk)
    out = cute.make_rmem_tensor((3 * VEC,), cutlass.BFloat16)
    for q in range(VEC):
        out[3 * q] = old[3 * q + 1]
        out[3 * q + 1] = old[3 * q + 2]
        out[3 * q + 2] = x0[q]
    cute.autovec_copy(out, dst)


def _conv_accept_two(gX, layer, position, chunk, dst):
    old = cute.make_rmem_tensor((3 * VEC,), cutlass.BFloat16)
    cute.autovec_copy(dst, old)
    x0 = _load_x_vec(gX, layer, position, cutlass.Int32(0), chunk)
    x1 = _load_x_vec(gX, layer, position, cutlass.Int32(1), chunk)
    out = cute.make_rmem_tensor((3 * VEC,), cutlass.BFloat16)
    for q in range(VEC):
        out[3 * q] = old[3 * q + 2]
        out[3 * q + 1] = x0[q]
        out[3 * q + 2] = x1[q]
    cute.autovec_copy(out, dst)


@cute.kernel
def _combined_kernel(
    gStateRaw: cute.Tensor,
    gStateDesc: cute.Tensor,
    gConvRaw: cute.Tensor,
    gConvDesc: cute.Tensor,
    gX: cute.Tensor,
    gU: cute.Tensor,
    gK: cute.Tensor,
    gG: cute.Tensor,
    gLen: cute.Tensor,
    gWI: cute.Tensor,
    gAT: cute.Tensor,
    gDM: cute.Tensor,
    STATE_BASE_ADDR: cutlass.Int64,
    CONV_BASE_ADDR: cutlass.Int64,
    L: cutlass.Constexpr,
    H: cutlass.Constexpr,
    RH: cutlass.Constexpr,
    W: cutlass.Constexpr,
    C: cutlass.Constexpr,
    NCHC: cutlass.Constexpr,
    TABLE_CAPACITY: cutlass.Constexpr,
    GDN_INDIRECT: cutlass.Constexpr,
    CONV_INDIRECT: cutlass.Constexpr,
    GDN_BASE: cutlass.Constexpr,
    GDN_LAYER_STRIDE: cutlass.Constexpr,
    GDN_SLOT_STRIDE: cutlass.Constexpr,
    CONV_BASE: cutlass.Constexpr,
    CONV_LAYER_STRIDE: cutlass.Constexpr,
    CONV_SLOT_STRIDE: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    batch = cute.size(gWI, mode=[0])

    smem = utils.SmemAllocator()
    sAccP = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(DIM * PAD), byte_alignment=16)
    sUP = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(W * PAD), byte_alignment=16)
    sKP = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(W * PAD), byte_alignment=16)
    sGD = smem.allocate_tensor(
        cutlass.Int64,
        cute.make_layout(2 * L if GDN_INDIRECT else 1),
        byte_alignment=16,
    )
    sCD = smem.allocate_tensor(
        cutlass.Int64,
        cute.make_layout(2 * L if CONV_INDIRECT else 1),
        byte_alignment=16,
    )
    sTab = smem.allocate_tensor(
        cutlass.Int32,
        cute.make_layout((3, TABLE_CAPACITY), stride=(TABLE_CAPACITY, 1)),
        byte_alignment=16,
    )

    sAccS = cute.make_tensor(
        sAccP.iterator,
        cute.make_layout((NSTRIP, DIM, STRIPW), stride=(STRIPW, PAD, 1)),
    )
    sAccV = cute.make_tensor(
        sAccP.iterator, cute.make_layout((DIM, NCH, VEC), stride=(PAD, VEC, 1))
    )
    sUA = cute.make_tensor(sUP.iterator, cute.make_layout((DIM, W), stride=(1, PAD)))
    sUW = cute.make_tensor(sUP.iterator, cute.make_layout((W, NCH, VEC), stride=(PAD, VEC, 1)))
    sKB = cute.make_tensor(
        sKP.iterator,
        cute.make_layout((NSTRIP, STRIPW, W), stride=(STRIPW, 1, PAD)),
    )
    sKW = cute.make_tensor(sKP.iterator, cute.make_layout((W, NCH, VEC), stride=(PAD, VEC, 1)))

    # Resolve every per-item and per-layer indirection once per CTA.  The work
    # metadata is three dependent global loads deep (work item -> position ->
    # dummy/accepted); doing it per tile serialises the whole CTA behind that
    # chain on every grid-stride step.
    for i in cutlass.range(tid, batch, NT):
        position = gWI[i, 0]
        slot = gWI[i, 1]
        start = gWI[i, 2]
        dummy = gDM[position].to(cutlass.Int32)
        accepted = gAT[position]
        meta = cutlass.Int32(-1)
        if dummy == 0:
            meta = accepted + ((start + accepted) << 8)
        sTab[0, i] = position
        sTab[1, i] = slot
        sTab[2, i] = meta
    if cutlass.const_expr(GDN_INDIRECT):
        for i in cutlass.range(tid, 2 * L, NT):
            sGD[i] = gStateDesc[i // 2, i % 2]
    if cutlass.const_expr(CONV_INDIRECT):
        for i in cutlass.range(tid, 2 * L, NT):
            sCD[i] = gConvDesc[i // 2, i % 2]
    cute.arch.barrier()

    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(
        mma_op, cute.make_layout((MW, NW, 1)), permutation_mnk=(64, 32, 16)
    )
    thr_mma = tiled_mma.get_slice(tid)
    ld_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), cutlass.BFloat16)
    tc_a = cute.make_tiled_copy_A(ld_atom, tiled_mma)
    tc_b = cute.make_tiled_copy_B(ld_atom, tiled_mma)
    thr_a = tc_a.get_slice(tid)
    thr_b = tc_b.get_slice(tid)
    st_atom = cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 4), cutlass.BFloat16)
    st_c_atom = cute.make_tiled_copy_C_atom(st_atom, tiled_mma)
    st_r2s = cute.make_tiled_copy_S(st_atom, st_c_atom)
    thr_st = st_r2s.get_slice(tid)

    i16 = tid // NCH
    cvec = tid % NCH

    # Conv4 raw-input checkpoint.  One thread owns one 8-channel packet of one
    # (work item, layer); the flat packet space keeps every thread busy for any
    # channel count and keeps the candidate_x reads perfectly coalesced.
    conv_units = batch * L * NCHC
    for cbase in cutlass.range(bid * NT, conv_units, grid * NT):
        ci = cbase + tid
        if ci < conv_units:
            il = ci // NCHC
            chunk = ci % NCHC
            item = il // L
            layer = il % L
            meta = sTab[2, item]
            if meta >= 0:
                accepted = meta & 255
                if accepted > 0:
                    position = sTab[0, item]
                    slot = sTab[1, item]
                    layer_offset = cutlass.Int64(0)
                    slot_stride = cutlass.Int64(0)
                    if cutlass.const_expr(CONV_INDIRECT):
                        layer_offset = sCD[layer * 2]
                        slot_stride = sCD[layer * 2 + 1]
                    else:
                        layer_offset = (
                            cutlass.Int64(CONV_BASE) + cutlass.Int64(layer) * CONV_LAYER_STRIDE
                        )
                        slot_stride = cutlass.Int64(CONV_SLOT_STRIDE)
                    if cutlass.const_expr(CONV_INDIRECT):
                        conv_addr = (
                            layer_offset
                            + (cutlass.Int64(slot) * slot_stride + cutlass.Int64(chunk) * (3 * VEC))
                            * 2
                        )
                    else:
                        conv_addr = (
                            CONV_BASE_ADDR
                            + (
                                layer_offset
                                + cutlass.Int64(slot) * slot_stride
                                + cutlass.Int64(chunk) * (3 * VEC)
                            )
                            * 2
                        )
                    conv_ptr = cute.make_ptr(
                        cutlass.BFloat16,
                        conv_addr,
                        AddressSpace.gmem,
                        assumed_align=16,
                    )
                    dst = cute.make_tensor(conv_ptr, cute.make_layout(3 * VEC))
                    if accepted >= 3:
                        x0 = _load_x_vec(gX, layer, position, accepted - 3, chunk)
                        x1 = _load_x_vec(gX, layer, position, accepted - 2, chunk)
                        x2 = _load_x_vec(gX, layer, position, accepted - 1, chunk)
                        _interleave3_store(x0, x1, x2, dst)
                    elif accepted == 2:
                        _conv_accept_two(gX, layer, position, chunk, dst)
                    else:
                        _conv_accept_one(gX, layer, position, chunk, dst)

    # One CTA owns the small scalar length update.  This has no ordering
    # dependence with either the Conv stores or the rollover tiles.
    if bid == 0:
        for item in cutlass.range(tid, batch, NT):
            meta = sTab[2, item]
            if meta >= 0:
                total = meta >> 8
                new_len = total
                if total > W:
                    new_len = total - W
                gLen[sTab[1, item]] = new_len

    # Boundary-crossing GDN commit.  Runtime B appears only in this grid-stride
    # bound, so the same compiled kernel serves every positive batch size.
    gdn_units = batch * L * H
    for it in cutlass.range(bid, gdn_units, grid):
        item = it // (L * H)
        rem = it % (L * H)
        layer = rem // H
        h = rem % H
        meta = sTab[2, item]
        if meta >= 0:
            total = meta >> 8
            if total > W:
                tail = total - W
                slot = sTab[1, item]

                layer_offset = cutlass.Int64(0)
                layer_slot_stride = cutlass.Int64(0)
                if cutlass.const_expr(GDN_INDIRECT):
                    layer_offset = sGD[layer * 2]
                    layer_slot_stride = sGD[layer * 2 + 1]
                else:
                    layer_offset = cutlass.Int64(GDN_BASE) + cutlass.Int64(layer) * GDN_LAYER_STRIDE
                    layer_slot_stride = cutlass.Int64(GDN_SLOT_STRIDE)

                if cutlass.const_expr(GDN_INDIRECT):
                    state_addr = layer_offset + cutlass.Int64(slot) * layer_slot_stride * 2
                else:
                    state_addr = (
                        STATE_BASE_ADDR
                        + (layer_offset + cutlass.Int64(slot) * layer_slot_stride) * 2
                    )
                state_ptr = cute.make_ptr(
                    cutlass.Int64,
                    state_addr,
                    AddressSpace.gmem,
                    assumed_align=16,
                )
                gSt0 = cute.make_tensor(
                    state_ptr,
                    cute.make_layout(
                        (H * RH, DIM, NCH, WVEC),
                        stride=(DIM * NCH * WVEC, NCH * WVEC, WVEC, 1),
                    ),
                )

                hv0 = h * RH

                # The checkpoint tile is by far the longest load; issue all of
                # it before any of the dependent decay math.
                rs = [cute.make_rmem_tensor((WVEC,), cutlass.Int64) for _ in range(NJ)]
                gS0 = gSt0[hv0, None, None, None]
                for jj in cutlass.range_constexpr(NJ):
                    cute.autovec_copy(gS0[i16 + MSTEP * jj, cvec, None], rs[jj])

                gKt = gK[layer, slot, None, h, None, None]
                gUt = gU[layer, slot, None, None, None, None]
                gGt = gG[layer, slot, None, None]
                gend_cur = gGt[hv0, W - 1]

                # Overflow rows are read here and rewritten at the very end, so
                # their loads can ride along with the checkpoint stream.
                u_units = tail * RH * NCH
                k_units = tail * NCH
                have_u = tid < u_units
                have_k = tid < k_units
                rtU = cute.make_rmem_tensor((WVEC,), cutlass.Int64)
                rtK = cute.make_rmem_tensor((WVEC,), cutlass.Int64)
                uj = tid // (RH * NCH)
                urem = tid % (RH * NCH)
                uhv = hv0 + urem // NCH
                uc = urem % NCH
                kj = tid // NCH
                kc = tid % NCH
                if have_u:
                    cute.autovec_copy(gUt[W + uj, uhv, uc, None], rtU)
                if have_k:
                    cute.autovec_copy(gKt[W + kj, kc, None], rtK)

                # Stage K once and the first value head's decay-scaled U.
                for wb in cutlass.range_constexpr(W // 16):
                    irow = i16 + wb * 16
                    rk = cute.make_rmem_tensor((WVEC,), cutlass.Int64)
                    ru = cute.make_rmem_tensor((WVEC,), cutlass.Int64)
                    cute.autovec_copy(gKt[irow, cvec, None], rk)
                    cute.autovec_copy(gUt[irow, hv0, cvec, None], ru)
                    gi = gGt[hv0, irow]
                    dec = cute.exp2((gend_cur - gi) * LOG2E, approx=True, ftz=True)
                    cute.autovec_copy(
                        cute.recast_tensor(rk, cutlass.BFloat16),
                        sKW[irow, cvec, None],
                    )
                    rud = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
                    rud.store(
                        (
                            cute.recast_tensor(ru, cutlass.BFloat16).load().to(cutlass.Float32)
                            * dec
                        ).to(cutlass.BFloat16)
                    )
                    cute.autovec_copy(rud, sUW[irow, cvec, None])

                scale = cute.exp2(gend_cur * LOG2E, approx=True, ftz=True)
                cute.arch.barrier()

                for hvl in cutlass.range_constexpr(RH):
                    hv = hv0 + hvl
                    gSt = gSt0[hv, None, None, None]

                    if cutlass.const_expr(hvl + 1 < RH):
                        hvn = hv + 1
                        gend_next = gGt[hvn, W - 1]
                        run0 = cute.make_rmem_tensor((WVEC,), cutlass.Int64)
                        cute.autovec_copy(gUt[i16, hvn, cvec, None], run0)
                        gi0_next = gGt[hvn, i16]
                        if cutlass.const_expr(W == 32):
                            run1 = cute.make_rmem_tensor((WVEC,), cutlass.Int64)
                            cute.autovec_copy(gUt[i16 + 16, hvn, cvec, None], run1)
                            gi1_next = gGt[hvn, i16 + 16]

                    tAs = thr_mma.partition_A(sUA)
                    tCrA = tiled_mma.make_fragment_A(tAs)
                    cute.copy(tc_a, thr_a.partition_S(sUA), thr_a.retile(tCrA))
                    for strip in cutlass.range_constexpr(NSTRIP):
                        sB = sKB[strip, None, None]
                        tBs = thr_mma.partition_B(sB)
                        tCrB = tiled_mma.make_fragment_B(tBs)
                        cute.copy(tc_b, thr_b.partition_S(sB), thr_b.retile(tCrB))
                        acc = cute.make_rmem_tensor(
                            tiled_mma.partition_shape_C((DIM, STRIPW)),
                            cutlass.Float32,
                        )
                        acc.fill(0.0)
                        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                            cute.gemm(
                                tiled_mma,
                                acc,
                                tCrA[None, None, kb],
                                tCrB[None, None, kb],
                                acc,
                            )
                        sAcc_strip = sAccS[strip, None, None]
                        tRS_sAcc = thr_st.partition_D(sAcc_strip)
                        tRS_rAcc = st_r2s.retile(acc)
                        rC = cute.make_rmem_tensor(tRS_rAcc.shape, cutlass.BFloat16)
                        rC.store(tRS_rAcc.load().to(cutlass.BFloat16))
                        cute.copy(st_r2s, rC, tRS_sAcc)
                    cute.arch.barrier()

                    # The next head was prefetched while this head's MMA ran.
                    if cutlass.const_expr(hvl + 1 < RH):
                        dec0_next = cute.exp2(
                            (gend_next - gi0_next) * LOG2E,
                            approx=True,
                            ftz=True,
                        )
                        rud0 = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
                        rud0.store(
                            (
                                cute.recast_tensor(run0, cutlass.BFloat16)
                                .load()
                                .to(cutlass.Float32)
                                * dec0_next
                            ).to(cutlass.BFloat16)
                        )
                        cute.autovec_copy(rud0, sUW[i16, cvec, None])
                        if cutlass.const_expr(W == 32):
                            dec1_next = cute.exp2(
                                (gend_next - gi1_next) * LOG2E,
                                approx=True,
                                ftz=True,
                            )
                            rud1 = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
                            rud1.store(
                                (
                                    cute.recast_tensor(run1, cutlass.BFloat16)
                                    .load()
                                    .to(cutlass.Float32)
                                    * dec1_next
                                ).to(cutlass.BFloat16)
                            )
                            cute.autovec_copy(rud1, sUW[i16 + 16, cvec, None])

                    if cutlass.const_expr(hvl + 1 < RH):
                        gStn = gSt0[hv + 1, None, None, None]
                    for jj in cutlass.range_constexpr(NJ):
                        ra = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
                        cute.autovec_copy(sAccV[i16 + MSTEP * jj, cvec, None], ra)
                        ro = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
                        ro.store(
                            (
                                cute.recast_tensor(rs[jj], cutlass.BFloat16)
                                .load()
                                .to(cutlass.Float32)
                                * scale
                                + ra.load().to(cutlass.Float32)
                            ).to(cutlass.BFloat16)
                        )
                        cute.autovec_copy(
                            cute.recast_tensor(ro, cutlass.Int64),
                            gSt[i16 + MSTEP * jj, cvec, None],
                        )
                        if cutlass.const_expr(hvl + 1 < RH):
                            cute.autovec_copy(gStn[i16 + MSTEP * jj, cvec, None], rs[jj])
                    if cutlass.const_expr(hvl + 1 < RH):
                        scale = cute.exp2(gend_next * LOG2E, approx=True, ftz=True)
                    if cutlass.const_expr(hvl + 1 < RH):
                        cute.arch.barrier()

                # Compact only the accepted overflow rows.
                if have_u:
                    cute.autovec_copy(rtU, gUt[uj, uhv, uc, None])
                for ui in cutlass.range(tid + NT, u_units, NT):
                    j = ui // (RH * NCH)
                    urem2 = ui % (RH * NCH)
                    hv2 = hv0 + urem2 // NCH
                    uc2 = urem2 % NCH
                    rt = cute.make_rmem_tensor((WVEC,), cutlass.Int64)
                    cute.autovec_copy(gUt[W + j, hv2, uc2, None], rt)
                    cute.autovec_copy(rt, gUt[j, hv2, uc2, None])

                if have_k:
                    cute.autovec_copy(rtK, gKt[kj, kc, None])
                for ki in cutlass.range(tid + NT, k_units, NT):
                    kj2 = ki // NCH
                    kc2 = ki % NCH
                    rt2 = cute.make_rmem_tensor((WVEC,), cutlass.Int64)
                    cute.autovec_copy(gKt[W + kj2, kc2, None], rt2)
                    cute.autovec_copy(rt2, gKt[kj2, kc2, None])

                g_units = tail * RH
                for gidx in cutlass.range(tid, g_units, NT):
                    gj = gidx // RH
                    hvl3 = gidx % RH
                    hv3 = hv0 + hvl3
                    eg = gGt[hv3, W - 1]
                    gGt[hv3, gj] = gGt[hv3, W + gj] - eg


@cute.jit
def _launch(
    mStateRaw: cute.Tensor,
    mStateDesc: cute.Tensor,
    mConvRaw: cute.Tensor,
    mConvDesc: cute.Tensor,
    mX: cute.Tensor,
    mU: cute.Tensor,
    mK: cute.Tensor,
    mG: cute.Tensor,
    mLen: cute.Tensor,
    mWI: cute.Tensor,
    mAT: cute.Tensor,
    mDM: cute.Tensor,
    L: cutlass.Constexpr,
    H: cutlass.Constexpr,
    RH: cutlass.Constexpr,
    W: cutlass.Constexpr,
    C: cutlass.Constexpr,
    NCHC: cutlass.Constexpr,
    TABLE_CAPACITY: cutlass.Constexpr,
    RUNTIME_GRID: cutlass.Int32,
    GDN_INDIRECT: cutlass.Constexpr,
    CONV_INDIRECT: cutlass.Constexpr,
    GDN_BASE: cutlass.Constexpr,
    GDN_LAYER_STRIDE: cutlass.Constexpr,
    GDN_SLOT_STRIDE: cutlass.Constexpr,
    CONV_BASE: cutlass.Constexpr,
    CONV_LAYER_STRIDE: cutlass.Constexpr,
    CONV_SLOT_STRIDE: cutlass.Constexpr,
    stream,
):
    state_addr = mStateRaw.iterator.toint()
    conv_addr = mConvRaw.iterator.toint()
    _combined_kernel(
        mStateRaw,
        mStateDesc,
        mConvRaw,
        mConvDesc,
        mX,
        mU,
        mK,
        mG,
        mLen,
        mWI,
        mAT,
        mDM,
        state_addr,
        conv_addr,
        L,
        H,
        RH,
        W,
        C,
        NCHC,
        TABLE_CAPACITY,
        GDN_INDIRECT,
        CONV_INDIRECT,
        GDN_BASE,
        GDN_LAYER_STRIDE,
        GDN_SLOT_STRIDE,
        CONV_BASE,
        CONV_LAYER_STRIDE,
        CONV_SLOT_STRIDE,
    ).launch(
        grid=[RUNTIME_GRID, 1, 1],
        block=[NT, 1, 1],
        stream=stream,
    )


_CACHE = {}
_SM_COUNTS = {}
_DUMMY_DESCRIPTORS = {}


def _sm_count(device: torch.device) -> int:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    count = _SM_COUNTS.get(device_index)
    if count is None:
        count = torch.cuda.get_device_properties(device_index).multi_processor_count
        _SM_COUNTS[device_index] = count
    return count


def _dummy_descriptors(device: torch.device) -> torch.Tensor:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    result = _DUMMY_DESCRIPTORS.get(device_index)
    if result is None:
        result = torch.zeros((1, 2), dtype=torch.int64, device=device)
        _DUMMY_DESCRIPTORS[device_index] = result
    return result


@torch.no_grad()
def gdn_mtp_commit(
    ssm_states: torch.Tensor,
    conv_states: torch.Tensor,
    candidate_x: torch.Tensor,
    history_u: torch.Tensor,
    history_k: torch.Tensor,
    history_G: torch.Tensor,
    history_len: torch.Tensor,
    replay_work_items: torch.Tensor,
    accepted_tokens: torch.Tensor,
    is_dummy: torch.Tensor,
    *,
    ssm_state_descriptors: torch.Tensor | None = None,
    conv_state_descriptors: torch.Tensor | None = None,
) -> None:
    """Commit accepted Conv and GDN histories for every local layer.

    Affine Cache Manager V2 storage is supplied as slot-major views
    [S,L,HV,128,128] and [S,L,C,3]. Indirect storage supplies the first
    layer view plus absolute-byte-address descriptors [L,2] whose second
    column is the BF16 slot stride. ``candidate_x`` is the full server-capacity
    buffer; the runtime batch comes from ``replay_work_items`` and is
    intentionally absent from the compilation key.
    """
    layers, slots, capacity, value_heads, value_dim = history_u.shape
    history_layers, history_slots, _, key_heads, key_dim = history_k.shape
    candidate_layers, candidate_capacity, replay_width, channels = candidate_x.shape
    batch = replay_work_items.shape[0]
    window = capacity - replay_width

    if batch == 0:
        return

    if candidate_layers != layers or history_layers != layers:
        raise ValueError("candidate_x and GDN histories must have the same layer count")
    if history_slots != slots:
        raise ValueError("K and U histories must have the same slot count")
    if value_dim != DIM or key_dim != DIM:
        raise ValueError("GDN MTP commit requires key_dim=value_dim=128")
    if window not in (16, 32):
        raise ValueError("GDN MTP commit requires a history window of 16 or 32")
    if not 1 <= replay_width <= 8:
        raise ValueError("GDN MTP commit requires 1 <= tokens_per_step <= 8")
    if not 1 <= layers <= 128:
        raise ValueError("GDN MTP commit supports 1 <= layers <= 128")
    if key_heads <= 0 or value_heads <= 0 or value_heads % key_heads != 0:
        raise ValueError("value_heads must be divisible by key_heads")
    if channels != (2 * key_heads + value_heads) * DIM:
        raise ValueError("candidate_x has an unexpected channel count")
    if slots < batch or candidate_capacity < batch:
        raise ValueError("batch must be no larger than the slot pool or candidate buffer")
    if history_G.shape != (layers, slots, value_heads, capacity):
        raise ValueError("history_G has an unexpected shape")
    if history_len.shape != (slots,) or history_len.dtype != torch.int32:
        raise ValueError("history_len must be int32 with shape [slots]")
    if replay_work_items.ndim != 2 or replay_work_items.shape[0] != batch:
        raise ValueError("replay_work_items must have one row per batch item")
    if replay_work_items.shape[1] < 3 or replay_work_items.dtype != torch.int32:
        raise ValueError("replay_work_items must be int32 with at least three columns")
    if accepted_tokens.shape != (batch,) or accepted_tokens.dtype != torch.int32:
        raise ValueError("accepted_tokens must be int32 with shape [batch]")
    if is_dummy.shape != (batch,) or is_dummy.dtype != torch.bool:
        raise ValueError("is_dummy must be bool with shape [batch]")
    if candidate_x.dtype != torch.bfloat16 or not candidate_x.is_contiguous():
        raise ValueError("candidate_x must be a contiguous BF16 capacity buffer")
    if history_u.dtype != torch.bfloat16 or history_k.dtype != torch.bfloat16:
        raise TypeError("GDN K/U histories must be BF16")
    if history_G.dtype != torch.float32:
        raise TypeError("GDN cumulative-G history must be FP32")
    if (
        not history_u.is_contiguous()
        or not history_k.is_contiguous()
        or not history_G.is_contiguous()
    ):
        raise ValueError("GDN histories must be contiguous")
    if not replay_work_items.is_contiguous():
        raise ValueError("replay_work_items must be contiguous")
    if ssm_states.dtype != torch.bfloat16 or conv_states.dtype != torch.bfloat16:
        raise TypeError("GDN and Conv checkpoints must be BF16")

    gdn_indirect = ssm_state_descriptors is not None
    conv_indirect = conv_state_descriptors is not None
    if gdn_indirect:
        if ssm_states.shape != (slots, value_heads, DIM, DIM):
            raise ValueError("Indirect SSM state must be the first rank-4 layer view")
        if (
            ssm_state_descriptors.shape != (layers, 2)
            or ssm_state_descriptors.dtype != torch.int64
            or not ssm_state_descriptors.is_contiguous()
        ):
            raise ValueError("SSM descriptors must be contiguous int64 [layers, 2]")
        gdn_base = gdn_layer_stride = gdn_slot_stride = 0
        state_desc = ssm_state_descriptors
    else:
        if ssm_states.shape != (slots, layers, value_heads, DIM, DIM):
            raise ValueError("Affine SSM state must have shape [slots, layers, HV, 128, 128]")
        gdn_base = 0
        gdn_layer_stride = ssm_states.stride(1) if layers > 1 else 0
        gdn_slot_stride = ssm_states.stride(0)
        state_desc = _dummy_descriptors(ssm_states.device)

    if conv_indirect:
        if conv_states.shape != (slots, channels, 3):
            raise ValueError("Indirect Conv state must be the first rank-3 layer view")
        if (
            conv_state_descriptors.shape != (layers, 2)
            or conv_state_descriptors.dtype != torch.int64
            or not conv_state_descriptors.is_contiguous()
        ):
            raise ValueError("Conv descriptors must be contiguous int64 [layers, 2]")
        conv_base = conv_layer_stride = conv_slot_stride = 0
        conv_desc = conv_state_descriptors
    else:
        if conv_states.shape != (slots, layers, channels, 3):
            raise ValueError("Affine Conv state must have shape [slots, layers, channels, 3]")
        conv_base = 0
        conv_layer_stride = conv_states.stride(1) if layers > 1 else 0
        conv_slot_stride = conv_states.stride(0)
        conv_desc = _dummy_descriptors(conv_states.device)

    tensors = (
        ssm_states,
        conv_states,
        candidate_x,
        history_u,
        history_k,
        history_G,
        history_len,
        replay_work_items,
        accepted_tokens,
        is_dummy,
        state_desc,
        conv_desc,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("GDN MTP commit tensors must be CUDA tensors")
    if any(tensor.device != candidate_x.device for tensor in tensors):
        raise ValueError("GDN MTP commit tensors must be on the same device")

    ratio = value_heads // key_heads
    n_chunks = channels // VEC
    x_vec = candidate_x.view(torch.int64).view(
        layers, candidate_capacity, replay_width, n_chunks, WVEC
    )
    hu = history_u.view(torch.int64).view(layers, slots, capacity, value_heads, NCH, WVEC)
    hk = history_k.view(torch.int64).view(layers, slots, capacity, key_heads, NCH, WVEC)
    dm = is_dummy.view(torch.uint8)

    from cuda.bindings import driver as _drv

    stream = _drv.CUstream(torch.cuda.current_stream().cuda_stream)
    mx = from_dlpack(x_vec, assumed_align=16)
    mwi = from_dlpack(replay_work_items, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1)
    )
    mat = from_dlpack(accepted_tokens, assumed_align=4).mark_compact_shape_dynamic(
        mode=0, stride_order=(0,)
    )
    mdm = from_dlpack(dm, assumed_align=1).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
    args = (
        from_dlpack(ssm_states, assumed_align=16),
        from_dlpack(state_desc, assumed_align=16),
        from_dlpack(conv_states, assumed_align=16),
        from_dlpack(conv_desc, assumed_align=16),
        mx,
        from_dlpack(hu, assumed_align=16),
        from_dlpack(hk, assumed_align=16),
        from_dlpack(history_G, assumed_align=16),
        from_dlpack(history_len, assumed_align=16),
        mwi,
        mat,
        mdm,
    )

    conv_ctas = (batch * layers * n_chunks + NT - 1) // NT
    max_gdn_grid = batch * layers * key_heads
    grid = min(
        max(conv_ctas, max_gdn_grid),
        _sm_count(candidate_x.device) * CTAS_PER_SM,
    )
    key = (
        candidate_x.device.index,
        candidate_capacity,
        slots,
        replay_width,
        layers,
        key_heads,
        value_heads,
        window,
        channels,
        capacity,
        gdn_indirect,
        conv_indirect,
        gdn_base,
        gdn_layer_stride,
        gdn_slot_stride,
        conv_base,
        conv_layer_stride,
        conv_slot_stride,
    )
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(
            _launch,
            *args,
            layers,
            key_heads,
            ratio,
            window,
            channels,
            n_chunks,
            candidate_capacity,
            cutlass.Int32(grid),
            gdn_indirect,
            conv_indirect,
            gdn_base,
            gdn_layer_stride,
            gdn_slot_stride,
            conv_base,
            conv_layer_stride,
            conv_slot_stride,
            stream,
        )
        _CACHE[key] = fn

    fn(*args, cutlass.Int32(grid), stream)
