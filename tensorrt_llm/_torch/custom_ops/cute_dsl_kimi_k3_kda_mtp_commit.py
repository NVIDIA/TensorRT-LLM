# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# One-launch all-layer accepted-prefix commit for KDA MTP replay + Conv4
# checkpoints.
#
# A single persistent grid-strided CuTe DSL kernel performs, per warmed run():
#   * the per-live-item Conv4 three-column shift/commit for every layer, and
#   * the KDA rollover state commit (per-key gate decay) plus history compaction.
#
# Both the state and the Conv backing store are byte addressed through their
# (L, 2) descriptor tensors, so the affine and the indirect Cache-Manager-V2
# layout share one code path and never need a host-side read of device metadata.
# Work-item metadata is staged once per SMEM-bounded batch, so the inner tile
# loops never pay the gWI -> is_dummy -> accepted dependent-load chain.

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cuda.bindings import driver as _drv
from cutlass.cute.nvgpu import warp
from cutlass.cute.typing import AddressSpace

DIM = 128  # K = V = 128 (fixed by the definition)
VEC = 8  # bf16 elements per 128-bit access
NCH = DIM // VEC  # 16 vector chunks per 128-wide row
VS = 4  # state-path elements per thread (16B at fp32)
NCS = DIM // VS  # 32 state chunks per 128-wide row
NT = 256  # threads per CTA
MW = DIM // 32  # warps along M
NW = NT // 32 // MW  # warps along N
STRIPW = 64  # output columns per MMA pass
NSTRIP = DIM // STRIPW
MSTEP = NT // NCS  # 8 checkpoint rows advanced per epilogue step
NJ = DIM // MSTEP  # 16 state chunks of the checkpoint tile per thread
PAD = 136  # padded bf16 stride for every SMEM staging tile
LOG2E = 1.4426950408889634
CTAS_PER_SM = 2
MAXB = 256  # work items staged into SMEM per metadata batch
CVEC = 3 * VEC  # bf16 elements one thread commits per conv unit


def _stype(sdt):
    return cutlass.BFloat16 if sdt == 0 else cutlass.Float32


def _sbytes(sdt):
    return 2 if sdt == 0 else 4


def _xload(X_BASE, xrow, cgu, t, C):
    """One VEC-wide candidate_x chunk of replay row ``t``."""
    g = cute.make_tensor(
        cute.make_ptr(
            cutlass.BFloat16,
            X_BASE + (xrow + cutlass.Int64(t)) * (C * 2) + cutlass.Int64(cgu) * (VEC * 2),
            AddressSpace.gmem,
            assumed_align=16,
        ),
        cute.make_layout(VEC),
    )
    r = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
    cute.autovec_copy(g, r)
    return r


def _load_old(gCv):
    """The three pre-commit Conv columns of VEC channels."""
    ro = [cute.make_rmem_tensor((VEC,), cutlass.BFloat16) for _ in range(3)]
    for q in range(3):
        cute.autovec_copy(gCv[q, None], ro[q])
    return ro


def _emit_conv(dst, spec, ro, xs):
    """Stage the three committed Conv columns of VEC channels (channel major).

    ``spec[j]`` selects column j: ("x", i) takes candidate row ``xs[i]``,
    ("o", m) takes pre-commit column m of the same channel.
    """
    for q in range(3):
        rn = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
        for p in range(VEC):
            flat = VEC * q + p
            e = flat // 3
            kind, idx = spec[flat % 3]
            if kind == "x":
                rn[p] = xs[idx][e]
            else:
                m = 3 * e + idx
                rn[p] = ro[m // VEC][m % VEC]
        cute.autovec_copy(rn, dst[q, None])


def sdt_rounds(sdt):
    return 1 if sdt == 0 else 2


def sdt_ctas(sdt):
    return 3 if sdt == 0 else 2


def _mk_frags(n, dtype, w=VS):
    return [cute.make_rmem_tensor((w,), dtype) for _ in range(n)]


@cute.kernel
def _commit_kernel(
    S_BASE: cutlass.Int64,
    SL_BASE: cutlass.Int64,
    C_BASE: cutlass.Int64,
    CL_BASE: cutlass.Int64,
    X_BASE: cutlass.Int64,
    K_BASE: cutlass.Int64,
    U_BASE: cutlass.Int64,
    G_BASE: cutlass.Int64,
    LEN_BASE: cutlass.Int64,
    WI_BASE: cutlass.Int64,
    AT_BASE: cutlass.Int64,
    DM_BASE: cutlass.Int64,
    B: cutlass.Int32,
    S: cutlass.Constexpr,
    XCAP: cutlass.Constexpr,
    L: cutlass.Constexpr,
    H: cutlass.Constexpr,
    T: cutlass.Constexpr,
    W: cutlass.Constexpr,
    CAP: cutlass.Constexpr,
    C: cutlass.Constexpr,
    SDT: cutlass.Constexpr,
    WIS: cutlass.Constexpr,
    GRID: cutlass.Constexpr,
):
    ST = _stype(SDT)
    SEB = _sbytes(SDT)
    CGU = C // VEC  # conv vector units per (item, layer)
    STEPS = (W * NCS) // NT
    NROUND = sdt_rounds(SDT)
    PER = NJ // NROUND
    f32 = cutlass.Float32
    bf16 = cutlass.BFloat16

    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    sAccP = smem.allocate_tensor(bf16, cute.make_layout(DIM * PAD), byte_alignment=16)
    sUP = smem.allocate_tensor(bf16, cute.make_layout(W * PAD), byte_alignment=16)
    sKP = smem.allocate_tensor(bf16, cute.make_layout(W * PAD), byte_alignment=16)
    mPos = smem.allocate_tensor(cutlass.Int32, cute.make_layout(MAXB), byte_alignment=16)
    mSlot = smem.allocate_tensor(cutlass.Int32, cute.make_layout(MAXB), byte_alignment=16)
    mAcc = smem.allocate_tensor(cutlass.Int32, cute.make_layout(MAXB), byte_alignment=16)
    mTot = smem.allocate_tensor(cutlass.Int32, cute.make_layout(MAXB), byte_alignment=16)

    gWI = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, WI_BASE, AddressSpace.gmem, assumed_align=16),
        cute.make_layout((B, 3), stride=(WIS, 1)),
    )
    gAT = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, AT_BASE, AddressSpace.gmem, assumed_align=4),
        cute.make_layout(B),
    )
    gDM = cute.make_tensor(
        cute.make_ptr(cutlass.Uint8, DM_BASE, AddressSpace.gmem, assumed_align=1),
        cute.make_layout(B),
    )
    gLen = cute.make_tensor(
        cute.make_ptr(cutlass.Int32, LEN_BASE, AddressSpace.gmem, assumed_align=16),
        cute.make_layout(S),
    )
    gSL = cute.make_tensor(
        cute.make_ptr(cutlass.Int64, SL_BASE, AddressSpace.gmem, assumed_align=16),
        cute.make_layout((L, 2), stride=(2, 1)),
    )
    gCL = cute.make_tensor(
        cute.make_ptr(cutlass.Int64, CL_BASE, AddressSpace.gmem, assumed_align=16),
        cute.make_layout((L, 2), stride=(2, 1)),
    )

    sAccS = cute.make_tensor(
        sAccP.iterator, cute.make_layout((NSTRIP, DIM, STRIPW), stride=(STRIPW, PAD, 1))
    )
    sAccV = cute.make_tensor(sAccP.iterator, cute.make_layout((DIM, NCS, VS), stride=(PAD, VS, 1)))
    sUA = cute.make_tensor(sUP.iterator, cute.make_layout((DIM, W), stride=(1, PAD)))
    sUW = cute.make_tensor(sUP.iterator, cute.make_layout((W, NCS, VS), stride=(PAD, VS, 1)))
    sKB = cute.make_tensor(
        sKP.iterator, cute.make_layout((NSTRIP, STRIPW, W), stride=(STRIPW, 1, PAD))
    )
    sKW = cute.make_tensor(sKP.iterator, cute.make_layout((W, NCS, VS), stride=(PAD, VS, 1)))

    mma_op = warp.MmaF16BF16Op(bf16, f32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(
        mma_op, cute.make_layout((MW, NW, 1)), permutation_mnk=(64, 32, 16)
    )
    thr_mma = tiled_mma.get_slice(tid)
    ld_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), bf16)
    tc_a = cute.make_tiled_copy_A(ld_atom, tiled_mma)
    tc_b = cute.make_tiled_copy_B(ld_atom, tiled_mma)
    thr_a = tc_a.get_slice(tid)
    thr_b = tc_b.get_slice(tid)
    st_atom = cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 4), bf16)
    st_c_atom = cute.make_tiled_copy_C_atom(st_atom, tiled_mma)
    st_r2s = cute.make_tiled_copy_S(st_atom, st_c_atom)
    thr_st = st_r2s.get_slice(tid)

    i16 = tid // NCS
    cvec = tid % NCS

    for b0 in cutlass.range(0, B, MAXB):
        rem = B - b0
        nb = MAXB if rem > MAXB else rem

        # -------- stage this batch's work-item metadata into SMEM --------
        for j in cutlass.range(tid, nb, NT):
            wq = b0 + j
            p = gWI[wq, 0]
            sl = gWI[wq, 1]
            av = gAT[p]
            tt = gWI[wq, 2] + av
            dmv = gDM[p].to(cutlass.Int32)
            mPos[j] = p
            mSlot[j] = sl
            mAcc[j] = cutlass.Int32(-1) if dmv != 0 else av
            mTot[j] = tt
            if bid == 0:
                if dmv == 0:
                    gLen[sl] = tt - W if tt > W else tt
        cute.arch.barrier()

        # ---------------- Pass 1 - Conv4 commit ----------------
        # One thread owns VEC channels of one (work item, layer): three
        # contiguous bf16 columns per channel, read and written in place.
        for u in cutlass.range(bid * NT + tid, nb * (L * CGU), GRID * NT):
            cgu = u % CGU
            rc = u // CGU
            layer = rc % L
            jc = rc // L
            a = mAcc[jc]
            if a > 0:
                pos = mPos[jc]
                gCv = cute.make_tensor(
                    cute.make_ptr(
                        bf16,
                        C_BASE
                        + gCL[layer, 0]
                        + cutlass.Int64(mSlot[jc]) * gCL[layer, 1]
                        + cutlass.Int64(cgu) * (CVEC * 2),
                        AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((3, VEC), stride=(VEC, 1)),
                )
                xrow = cutlass.Int64((layer * XCAP + pos) * T)
                if a >= 3:
                    xa = _xload(X_BASE, xrow, cgu, a - 3, C)
                    xb = _xload(X_BASE, xrow, cgu, a - 2, C)
                    xc = _xload(X_BASE, xrow, cgu, a - 1, C)
                    _emit_conv(gCv, (("x", 0), ("x", 1), ("x", 2)), None, (xa, xb, xc))
                else:
                    ro = _load_old(gCv)
                    x0 = _xload(X_BASE, xrow, cgu, 0, C)
                    if a == 1:
                        _emit_conv(gCv, (("o", 1), ("o", 2), ("x", 0)), ro, (x0,))
                    else:
                        x1 = _xload(X_BASE, xrow, cgu, 1, C)
                        _emit_conv(gCv, (("o", 2), ("x", 0), ("x", 1)), ro, (x0, x1))

        # ---------------- Pass 2 - KDA rollover commit ----------------
        for it in cutlass.range(bid, nb * (L * H), GRID):
            h = it % H
            r2 = it // H
            l2 = r2 % L
            j2 = r2 // L
            tot = mTot[j2]
            av2 = mAcc[j2]
            rolx = cutlass.Int32(1) if tot > W else cutlass.Int32(0)
            actx = cutlass.Int32(1) if av2 >= 0 else cutlass.Int32(0)
            if actx * rolx != 0:
                slot2 = mSlot[j2]
                tail = tot - W
                hidx = cutlass.Int64((l2 * S + slot2) * H + h)
                gK = cute.make_tensor(
                    cute.make_ptr(
                        bf16,
                        K_BASE + hidx * (CAP * DIM * 2),
                        AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((CAP, NCS, VS), stride=(DIM, VS, 1)),
                )
                gU = cute.make_tensor(
                    cute.make_ptr(
                        bf16,
                        U_BASE + hidx * (CAP * DIM * 2),
                        AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((CAP, NCS, VS), stride=(DIM, VS, 1)),
                )
                gG = cute.make_tensor(
                    cute.make_ptr(
                        f32,
                        G_BASE + hidx * (CAP * DIM * 4),
                        AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((CAP, NCS, VS), stride=(DIM, VS, 1)),
                )
                gSt = cute.make_tensor(
                    cute.make_ptr(
                        ST,
                        S_BASE
                        + gSL[l2, 0]
                        + cutlass.Int64(slot2) * gSL[l2, 1]
                        + cutlass.Int64(h) * (DIM * DIM * SEB),
                        AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((DIM, NCS, VS), stride=(DIM, VS, 1)),
                )

                rend = cute.make_rmem_tensor((VS,), f32)
                cute.autovec_copy(gG[W - 1, cvec, None], rend)

                # The accepted-overflow compaction reads exactly the rows this
                # same thread staged, so it needs no barrier and its latency can
                # be retired before the MMA instead of after the epilogue.
                rtk = cute.make_rmem_tensor((VS,), bf16)
                rtu = cute.make_rmem_tensor((VS,), bf16)
                rtg = cute.make_rmem_tensor((VS,), f32)
                if i16 < tail:
                    cute.autovec_copy(gK[W + i16, cvec, None], rtk)
                    cute.autovec_copy(gU[W + i16, cvec, None], rtu)
                    cute.autovec_copy(gG[W + i16, cvec, None], rtg)

                rs = _mk_frags(PER, ST)
                for jj in cutlass.range_constexpr(PER):
                    cute.autovec_copy(gSt[i16 + MSTEP * jj, cvec, None], rs[jj])

                for stp in cutlass.range_constexpr(STEPS):
                    irow = i16 + stp * MSTEP
                    rk = cute.make_rmem_tensor((VS,), bf16)
                    ru = cute.make_rmem_tensor((VS,), bf16)
                    rg = cute.make_rmem_tensor((VS,), f32)
                    cute.autovec_copy(gK[irow, cvec, None], rk)
                    cute.autovec_copy(gU[irow, cvec, None], ru)
                    cute.autovec_copy(gG[irow, cvec, None], rg)
                    dec = cute.exp2((rend.load() - rg.load()) * LOG2E, fastmath=True)
                    rkd = cute.make_rmem_tensor((VS,), bf16)
                    rkd.store((rk.load().to(f32) * dec).to(bf16))
                    cute.autovec_copy(rkd, sKW[irow, cvec, None])
                    cute.autovec_copy(ru, sUW[irow, cvec, None])

                rsc = cute.make_rmem_tensor((VS,), f32)
                rsc.store(cute.exp2(rend.load() * LOG2E, fastmath=True))

                if i16 < tail:
                    rgo = cute.make_rmem_tensor((VS,), f32)
                    rgo.store(rtg.load() - rend.load())
                    cute.autovec_copy(rtk, gK[i16, cvec, None])
                    cute.autovec_copy(rtu, gU[i16, cvec, None])
                    cute.autovec_copy(rgo, gG[i16, cvec, None])

                cute.arch.barrier()

                tAs = thr_mma.partition_A(sUA)
                tCrA = tiled_mma.make_fragment_A(tAs)
                cute.copy(tc_a, thr_a.partition_S(sUA), thr_a.retile(tCrA))
                for s in cutlass.range_constexpr(NSTRIP):
                    sB = sKB[s, None, None]
                    tBs = thr_mma.partition_B(sB)
                    tCrB = tiled_mma.make_fragment_B(tBs)
                    cute.copy(tc_b, thr_b.partition_S(sB), thr_b.retile(tCrB))
                    acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C((DIM, STRIPW)), f32)
                    acc.fill(0.0)
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cute.gemm(
                            tiled_mma,
                            acc,
                            tCrA[None, None, kb],
                            tCrB[None, None, kb],
                            acc,
                        )
                    tRS_sAcc = thr_st.partition_D(sAccS[s, None, None])
                    tRS_rAcc = st_r2s.retile(acc)
                    rC = cute.make_rmem_tensor(tRS_rAcc.shape, bf16)
                    rC.store(tRS_rAcc.load().to(bf16))
                    cute.copy(st_r2s, rC, tRS_sAcc)
                cute.arch.barrier()

                for rd in cutlass.range_constexpr(NROUND):
                    for jj in cutlass.range_constexpr(PER):
                        row = i16 + MSTEP * (rd * PER + jj)
                        ra = cute.make_rmem_tensor((VS,), bf16)
                        cute.autovec_copy(sAccV[row, cvec, None], ra)
                        ro2 = cute.make_rmem_tensor((VS,), ST)
                        ro2.store((rs[jj].load().to(f32) * rsc.load() + ra.load().to(f32)).to(ST))
                        cute.autovec_copy(ro2, gSt[row, cvec, None])
                        if cutlass.const_expr(rd + 1 < NROUND):
                            cute.autovec_copy(
                                gSt[i16 + MSTEP * ((rd + 1) * PER + jj), cvec, None],
                                rs[jj],
                            )
        cute.arch.barrier()


@cute.jit
def _launch(
    S_BASE: cutlass.Int64,
    SL_BASE: cutlass.Int64,
    C_BASE: cutlass.Int64,
    CL_BASE: cutlass.Int64,
    X_BASE: cutlass.Int64,
    K_BASE: cutlass.Int64,
    U_BASE: cutlass.Int64,
    G_BASE: cutlass.Int64,
    LEN_BASE: cutlass.Int64,
    WI_BASE: cutlass.Int64,
    AT_BASE: cutlass.Int64,
    DM_BASE: cutlass.Int64,
    B: cutlass.Int32,
    S: cutlass.Constexpr,
    XCAP: cutlass.Constexpr,
    L: cutlass.Constexpr,
    H: cutlass.Constexpr,
    T: cutlass.Constexpr,
    W: cutlass.Constexpr,
    CAP: cutlass.Constexpr,
    C: cutlass.Constexpr,
    SDT: cutlass.Constexpr,
    WIS: cutlass.Constexpr,
    GRID: cutlass.Constexpr,
    stream,
):
    _commit_kernel(
        S_BASE,
        SL_BASE,
        C_BASE,
        CL_BASE,
        X_BASE,
        K_BASE,
        U_BASE,
        G_BASE,
        LEN_BASE,
        WI_BASE,
        AT_BASE,
        DM_BASE,
        B,
        S,
        XCAP,
        L,
        H,
        T,
        W,
        CAP,
        C,
        SDT,
        WIS,
        GRID,
    ).launch(
        grid=[GRID, 1, 1],
        block=[NT, 1, 1],
        stream=stream,
        min_blocks_per_mp=sdt_ctas(SDT),
        preferred_smem_carveout=100,
    )


_CACHE = {}
_SM_COUNTS = {}


def _sm_count(device: torch.device) -> int:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    count = _SM_COUNTS.get(device_index)
    if count is None:
        count = torch.cuda.get_device_properties(device_index).multi_processor_count
        _SM_COUNTS[device_index] = count
    return count


@torch.no_grad()
def kda_mtp_commit(
    ssm_states: torch.Tensor,
    conv_states: torch.Tensor,
    candidate_x: torch.Tensor,
    history_k: torch.Tensor,
    history_u: torch.Tensor,
    history_G: torch.Tensor,
    history_len: torch.Tensor,
    replay_work_items: torch.Tensor,
    accepted_tokens: torch.Tensor,
    is_dummy: torch.Tensor,
    *,
    ssm_state_descriptors: torch.Tensor,
    conv_state_descriptors: torch.Tensor,
) -> None:
    """Commit accepted Conv inputs and rolled-over KDA histories.

    Both affine and indirect Cache Manager V2 pools are described by absolute
    byte-address rows ``[base, slot_stride]``. Conv checkpoints advance every
    invocation; recurrent checkpoints advance only when the history exceeds
    its configured window.
    """
    L, S, H, CAP, _ = history_k.shape
    XCAP = candidate_x.shape[1]
    T = candidate_x.shape[2]
    W = CAP - T
    C = candidate_x.shape[3]
    B = replay_work_items.shape[0]

    if B == 0:
        return
    if not 1 <= L <= 128:
        raise ValueError("KDA MTP commit supports 1 <= layers <= 128")
    if not 1 <= T <= 8:
        raise ValueError("KDA MTP commit requires 1 <= tokens_per_step <= 8")
    if W not in (16, 32):
        raise ValueError("KDA MTP commit requires a history window of 16 or 32")
    if C != 3 * H * DIM:
        raise ValueError("candidate_x has an unexpected channel count")
    if XCAP < B or S < B:
        raise ValueError("batch must fit the candidate and history capacities")
    if candidate_x.shape != (L, XCAP, T, C):
        raise ValueError("candidate_x has an unexpected shape")
    if candidate_x.dtype != torch.bfloat16 or not candidate_x.is_contiguous():
        raise ValueError("candidate_x must be contiguous BF16")
    expected_history_shape = (L, S, H, CAP, DIM)
    if history_u.shape != expected_history_shape or history_G.shape != expected_history_shape:
        raise ValueError("KDA histories have inconsistent shapes")
    if history_k.dtype != torch.bfloat16 or history_u.dtype != torch.bfloat16:
        raise TypeError("KDA K/U histories must be BF16")
    if history_G.dtype != torch.float32:
        raise TypeError("KDA cumulative-G history must be FP32")
    if (
        not history_k.is_contiguous()
        or not history_u.is_contiguous()
        or not history_G.is_contiguous()
    ):
        raise ValueError("KDA histories must be contiguous")
    if history_len.shape != (S,) or history_len.dtype != torch.int32:
        raise ValueError("history_len must be int32 with shape [slots]")
    if replay_work_items.ndim != 2 or replay_work_items.shape[1] < 3:
        raise ValueError("replay_work_items must have at least three columns")
    if replay_work_items.dtype != torch.int32 or replay_work_items.stride(1) != 1:
        raise ValueError("replay_work_items must have dense int32 rows")
    if accepted_tokens.shape != (B,) or accepted_tokens.dtype != torch.int32:
        raise ValueError("accepted_tokens must be int32 with shape [batch]")
    if is_dummy.shape != (B,) or is_dummy.dtype != torch.bool:
        raise ValueError("is_dummy must be bool with shape [batch]")
    if ssm_states.shape != (S, H, DIM, DIM):
        raise ValueError("ssm_states must be the first [slots, heads, 128, 128] layer view")
    if ssm_states.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("KDA recurrent checkpoints must be BF16 or FP32")
    if ssm_states.stride()[1:] != (DIM * DIM, DIM, 1):
        raise ValueError("KDA recurrent checkpoint inner dimensions must be dense")
    if conv_states.shape != (S, C, 3) or conv_states.dtype != torch.bfloat16:
        raise ValueError("conv_states must be BF16 with shape [slots, channels, 3]")
    if conv_states.stride()[1:] != (3, 1):
        raise ValueError("Conv checkpoint inner dimensions must be dense")
    for name, descriptors in (
        ("SSM", ssm_state_descriptors),
        ("Conv", conv_state_descriptors),
    ):
        if descriptors.shape != (L, 2) or descriptors.dtype != torch.int64:
            raise ValueError(f"{name} descriptors must be int64 with shape [layers, 2]")
        if not descriptors.is_contiguous():
            raise ValueError(f"{name} descriptors must be contiguous")

    tensors = (
        ssm_states,
        conv_states,
        candidate_x,
        history_k,
        history_u,
        history_G,
        history_len,
        replay_work_items,
        accepted_tokens,
        is_dummy,
        ssm_state_descriptors,
        conv_state_descriptors,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("KDA MTP commit tensors must be CUDA tensors")
    if any(tensor.device != candidate_x.device for tensor in tensors):
        raise ValueError("KDA MTP commit tensors must be on the same device")

    sdt = int(ssm_states.dtype == torch.float32)
    args = (
        cutlass.Int64(0),
        cutlass.Int64(ssm_state_descriptors.data_ptr()),
        cutlass.Int64(0),
        cutlass.Int64(conv_state_descriptors.data_ptr()),
        cutlass.Int64(candidate_x.data_ptr()),
        cutlass.Int64(history_k.data_ptr()),
        cutlass.Int64(history_u.data_ptr()),
        cutlass.Int64(history_G.data_ptr()),
        cutlass.Int64(history_len.data_ptr()),
        cutlass.Int64(replay_work_items.data_ptr()),
        cutlass.Int64(accepted_tokens.data_ptr()),
        cutlass.Int64(is_dummy.data_ptr()),
    )
    stream = _drv.CUstream(torch.cuda.current_stream().cuda_stream)

    work_item_stride = replay_work_items.stride(0)
    device_index = candidate_x.device.index
    key = (device_index, S, XCAP, L, H, T, W, CAP, C, sdt, work_item_stride)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(
            _launch,
            *args,
            cutlass.Int32(B),
            S,
            XCAP,
            L,
            H,
            T,
            W,
            CAP,
            C,
            sdt,
            work_item_stride,
            _sm_count(candidate_x.device) * sdt_ctas(sdt),
            stream,
        )
        _CACHE[key] = fn

    fn(*args, cutlass.Int32(B), stream)
